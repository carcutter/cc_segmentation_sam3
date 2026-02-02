# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

# pyre-unsafe

"""
Modified trainer with systematic parameter freezing support.

This trainer extends the base SAM3 trainer to allow freezing specific parts
of the network via YAML configuration. 

SAM3 IMAGE MODEL ARCHITECTURE:
==============================
The model has the following components (with actual parameter name patterns):

BACKBONE (pretrained encoders - typically frozen):
  - backbone.vision_backbone.*  : ViT image encoder (~300M params)
  - backbone.language_backbone.*: CLIP text encoder (~150M params)

GEOMETRY ENCODER (encodes spatial prompts):
  - geometry_encoder.*          : Box/point prompt encoder (~7M params)

TRANSFORMER (fusion and decoding):
  - transformer.encoder.*       : Vision-language fusion (~12M params)
  - transformer.decoder.*       : Object query decoder (~15M params)

OUTPUT HEADS:
  - segmentation_head.*         : Pixel decoder + mask predictor (~5M params)
  - dot_prod_scoring.*          : Classification scoring (~2M params)

VIDEO COMPONENTS (only in video models):
  - memory_encoder.*            : Memory encoding
  - memory_attention.*          : Memory attention
  - obj_ptr.*                   : Object pointer

Usage in YAML config:
    trainer:
      freeze_config:
        # Backbone (pretrained)
        vision_backbone: true       # Freeze ViT image encoder
        language_backbone: true     # Freeze CLIP text encoder
        
        # Geometry encoder
        geometry_encoder: false     # Train geometry/prompt encoder
        
        # Transformer
        transformer_encoder: false  # Train vision-language fusion
        transformer_decoder: false  # Train object query decoder
        
        # Output heads
        segmentation_head: false    # Train segmentation head
        dot_prod_scoring: false     # Train classification scoring
        
        # Fine-grained control
        custom_patterns: []         # Additional patterns to freeze
        unfreeze_patterns: []       # Patterns to explicitly unfreeze
"""

import contextlib
import fnmatch
import gc
import json
import logging
import math
import os
import re
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Set

import numpy as np

import torch
import torch.distributed as dist
import torch.nn as nn
from hydra.utils import instantiate
from iopath.common.file_io import g_pathmgr

from sam3.eval.segmentation_metrics import SegmentationMetricsMeter
from sam3.model.data_misc import BatchedDatapoint
from sam3.model.model_misc import SAM3Output
from sam3.model.utils.misc import copy_data_to_device

from sam3.train.optim.optimizer import construct_optimizer

from sam3.train.utils.checkpoint_utils import (
    assert_skipped_parameters_are_frozen,
    exclude_params_matching_unix_pattern,
    load_state_dict_into_model,
    with_check_parameter_frozen,
)

from sam3.train.utils.distributed import all_reduce_max, barrier, get_rank

from sam3.train.utils.logger import Logger, setup_logging
from sam3.train.utils.train_utils import (
    AverageMeter,
    collect_dict_keys,
    DurationMeter,
    get_amp_type,
    get_machine_local_and_dist_rank,
    get_resume_checkpoint,
    human_readable_time,
    is_dist_avail_and_initialized,
    log_env_variables,
    makedir,
    MemMeter,
    Phase,
    ProgressMeter,
    set_seeds,
    setup_distributed_backend,
)


CORE_LOSS_KEY = "core_loss"


def unwrap_ddp_if_wrapped(model):
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        return model.module
    return model


# =============================================================================
# FREEZING CONFIGURATION
# =============================================================================

@dataclass
class FreezeConfig:
    """Configuration for freezing specific parts of the SAM3 model.
    
    This dataclass defines which components of SAM3 should be frozen during training.
    Setting a component to True will freeze all its parameters (requires_grad=False).
    
    SAM3 Image Model Architecture (actual parameter names):
    ========================================================
    
    BACKBONE (Pretrained encoders):
        vision_backbone: ViT image encoder (backbone.vision_backbone.*)
            - trunk.pos_embed, trunk.patch_embed, trunk.blocks.0-31
            - ~300M parameters
        language_backbone: CLIP text encoder (backbone.language_backbone.*)
            - encoder.transformer.resblocks.0-23, encoder.positional_embedding
            - ~150M parameters
    
    GEOMETRY ENCODER (encodes spatial prompts like boxes/points):
        geometry_encoder: Spatial prompt encoder (geometry_encoder.*)
            - label_embed, cls_embed, points/boxes projections
            - encode.0-2 (transformer layers), encode_norm
            - ~7M parameters
    
    TRANSFORMER (fusion and decoding):
        transformer_encoder: Encoder for vision-language fusion (transformer.encoder.*)
            - layers.0-5 (cross-attention to image features)
            - ~12M parameters
        transformer_decoder: Decoder for object queries (transformer.decoder.*)
            - layers.0-5, bbox_embed, query_embed, presence_token
            - ~15M parameters
    
    HEADS (task-specific outputs):
        segmentation_head: Pixel decoder + mask predictor (segmentation_head.*)
            - pixel_decoder.conv_layers, mask_predictor.mask_embed
            - ~5M parameters
        dot_prod_scoring: Classification scoring (dot_prod_scoring.*)
            - prompt_mlp, prompt_proj, hs_proj
            - ~2M parameters
    
    VIDEO/TRACKING (only for video models):
        memory_encoder: Memory encoding for video (memory_encoder.*)
        memory_attention: Memory attention for video (memory_attention.*)
        obj_ptr: Object pointer for tracking (obj_ptr.*)
    
    Args:
        custom_patterns: List of additional Unix glob patterns to freeze
        freeze_all_except: If set, freeze everything EXCEPT these patterns
        unfreeze_patterns: Patterns to explicitly unfreeze (overrides other settings)
    """
    # ==========================================================================
    # BACKBONE COMPONENTS (pretrained, typically frozen for fine-tuning)
    # ==========================================================================
    vision_backbone: bool = False      # backbone.vision_backbone.* (~300M params)
    language_backbone: bool = False    # backbone.language_backbone.* (~150M params)
    
    # ==========================================================================
    # GEOMETRY ENCODER (encodes boxes/points spatial prompts)
    # ==========================================================================
    geometry_encoder: bool = False     # geometry_encoder.* (~7M params)
    
    # ==========================================================================
    # TRANSFORMER COMPONENTS
    # ==========================================================================
    transformer_encoder: bool = False  # transformer.encoder.* (~12M params)
    transformer_decoder: bool = False  # transformer.decoder.* (~15M params)
    
    # ==========================================================================
    # OUTPUT HEADS
    # ==========================================================================
    segmentation_head: bool = False    # segmentation_head.* (~5M params)
    dot_prod_scoring: bool = False     # dot_prod_scoring.* (~2M params)
    
    # ==========================================================================
    # VIDEO/TRACKING COMPONENTS (only present in video models)
    # ==========================================================================
    memory_encoder: bool = False       # memory_encoder.* (video only)
    memory_attention: bool = False     # memory_attention.* (video only)
    obj_ptr: bool = False              # obj_ptr.* (video only)
    
    # ==========================================================================
    # FINE-GRAINED CONTROL
    # ==========================================================================
    custom_patterns: List[str] = field(default_factory=list)
    freeze_all_except: List[str] = field(default_factory=list)
    unfreeze_patterns: List[str] = field(default_factory=list)
    
    # Vision backbone layer-specific freezing
    vision_backbone_num_frozen_layers: Optional[int] = None  # Freeze first N ViT blocks
    
    def get_freeze_patterns(self) -> List[str]:
        """Get all patterns that should be frozen based on configuration.
        
        Returns list of Unix glob patterns matching SAM3 parameter names.
        """
        patterns = []
        
        # Map component flags to actual SAM3 parameter name patterns
        # These patterns are verified against the actual model architecture
        component_patterns = {
            # Backbone
            'vision_backbone': ['backbone.vision_backbone.*'],
            'language_backbone': ['backbone.language_backbone.*'],
            # Geometry encoder  
            'geometry_encoder': ['geometry_encoder.*'],
            # Transformer
            'transformer_encoder': ['transformer.encoder.*'],
            'transformer_decoder': ['transformer.decoder.*'],
            # Heads
            'segmentation_head': ['segmentation_head.*'],
            'dot_prod_scoring': ['dot_prod_scoring.*'],
            # Video components (only present in video models)
            'memory_encoder': ['memory_encoder.*'],
            'memory_attention': ['memory_attention.*'],
            'obj_ptr': ['obj_ptr.*'],
        }
        
        # Add patterns for enabled component freezing
        for component, should_freeze in [
            ('vision_backbone', self.vision_backbone),
            ('language_backbone', self.language_backbone),
            ('geometry_encoder', self.geometry_encoder),
            ('transformer_encoder', self.transformer_encoder),
            ('transformer_decoder', self.transformer_decoder),
            ('segmentation_head', self.segmentation_head),
            ('dot_prod_scoring', self.dot_prod_scoring),
            ('memory_encoder', self.memory_encoder),
            ('memory_attention', self.memory_attention),
            ('obj_ptr', self.obj_ptr),
        ]:
            if should_freeze:
                patterns.extend(component_patterns.get(component, []))
        
        # Add custom patterns
        patterns.extend(self.custom_patterns)
        
        # Handle layer-specific vision backbone freezing
        if self.vision_backbone_num_frozen_layers is not None and not self.vision_backbone:
            for i in range(self.vision_backbone_num_frozen_layers):
                patterns.append(f'backbone.vision_backbone.trunk.blocks.{i}.*')
        
        return patterns


def freeze_parameters_by_patterns(
    model: nn.Module,
    patterns: List[str],
    unfreeze_patterns: Optional[List[str]] = None,
) -> Dict[str, int]:
    """Freeze model parameters matching the given patterns.
    
    Args:
        model: The model whose parameters should be frozen
        patterns: List of Unix glob patterns for parameters to freeze
        unfreeze_patterns: List of patterns to explicitly keep unfrozen
        
    Returns:
        Dictionary with counts of frozen and total parameters
    """
    frozen_count = 0
    total_count = 0
    frozen_params = []
    
    unfreeze_patterns = unfreeze_patterns or []
    
    for name, param in model.named_parameters():
        total_count += 1
        
        # Check if parameter should be unfrozen (highest priority)
        should_unfreeze = any(fnmatch.fnmatch(name, p) for p in unfreeze_patterns)
        if should_unfreeze:
            param.requires_grad = True
            continue
        
        # Check if parameter matches any freeze pattern
        should_freeze = any(fnmatch.fnmatch(name, p) for p in patterns)
        if should_freeze:
            param.requires_grad = False
            frozen_count += 1
            frozen_params.append(name)
    
    return {
        'frozen': frozen_count,
        'total': total_count,
        'trainable': total_count - frozen_count,
        'frozen_params': frozen_params,
    }


def freeze_all_except_patterns(
    model: nn.Module,
    keep_trainable_patterns: List[str],
) -> Dict[str, int]:
    """Freeze all parameters except those matching the given patterns.
    
    This is useful for adapter-style training where you want to freeze
    the base model and only train specific components.
    
    Args:
        model: The model whose parameters should be frozen
        keep_trainable_patterns: List of Unix glob patterns for parameters to keep trainable
        
    Returns:
        Dictionary with counts of frozen and total parameters
    """
    frozen_count = 0
    total_count = 0
    trainable_params = []
    
    for name, param in model.named_parameters():
        total_count += 1
        
        # Check if parameter should remain trainable
        should_train = any(fnmatch.fnmatch(name, p) for p in keep_trainable_patterns)
        
        if should_train:
            param.requires_grad = True
            trainable_params.append(name)
        else:
            param.requires_grad = False
            frozen_count += 1
    
    return {
        'frozen': frozen_count,
        'total': total_count,
        'trainable': total_count - frozen_count,
        'trainable_params': trainable_params,
    }


def log_frozen_parameters(model: nn.Module, log_dir: str = "") -> None:
    """Log information about frozen vs trainable parameters."""
    if get_rank() != 0:
        return
    
    trainable_params = []
    frozen_params = []
    
    for name, param in model.named_parameters():
        if param.requires_grad:
            trainable_params.append((name, param.numel()))
        else:
            frozen_params.append((name, param.numel()))
    
    total_trainable = sum(p[1] for p in trainable_params)
    total_frozen = sum(p[1] for p in frozen_params)
    total = total_trainable + total_frozen
    
    logging.info("=" * 60)
    logging.info("PARAMETER FREEZING SUMMARY")
    logging.info("=" * 60)
    logging.info(f"Total parameters: {get_human_readable_count(total)}")
    logging.info(f"Trainable parameters: {get_human_readable_count(total_trainable)} ({100*total_trainable/total:.1f}%)")
    logging.info(f"Frozen parameters: {get_human_readable_count(total_frozen)} ({100*total_frozen/total:.1f}%)")
    logging.info("-" * 60)
    
    # Log trainable parameter groups
    logging.info("Trainable parameter groups:")
    trainable_groups = {}
    for name, numel in trainable_params:
        # Extract the top-level group name
        group = name.split('.')[0]
        if len(name.split('.')) > 1:
            group = '.'.join(name.split('.')[:2])
        trainable_groups[group] = trainable_groups.get(group, 0) + numel
    
    for group, count in sorted(trainable_groups.items(), key=lambda x: -x[1]):
        logging.info(f"  {group}: {get_human_readable_count(count)}")
    
    logging.info("=" * 60)
    
    # Optionally save to file
    if log_dir:
        output_fpath = os.path.join(log_dir, "frozen_params.txt")
        with g_pathmgr.open(output_fpath, "w") as f:
            f.write("FROZEN PARAMETERS:\n")
            f.write("=" * 60 + "\n")
            for name, numel in frozen_params:
                f.write(f"{name}: {numel}\n")
            f.write("\n\nTRAINABLE PARAMETERS:\n")
            f.write("=" * 60 + "\n")
            for name, numel in trainable_params:
                f.write(f"{name}: {numel}\n")


# =============================================================================
# ORIGINAL TRAINER CODE WITH FREEZING SUPPORT
# =============================================================================

@dataclass
class OptimAMPConf:
    enabled: bool = False
    amp_dtype: str = "float16"


@dataclass
class OptimConf:
    optimizer: torch.optim.Optimizer = None
    options: Optional[Dict[str, Any]] = None
    param_group_modifiers: Optional[List] = None
    amp: Optional[Dict[str, Any]] = None
    gradient_clip: Any = None
    gradient_logger: Any = None

    def __post_init__(self):
        # amp
        if not isinstance(self.amp, OptimAMPConf):
            if self.amp is None:
                self.amp = {}
            assert isinstance(self.amp, Mapping)
            self.amp = OptimAMPConf(**self.amp)


@dataclass
class DistributedConf:
    backend: Optional[str] = None  # inferred from accelerator type
    comms_dtype: Optional[str] = None
    find_unused_parameters: bool = False
    timeout_mins: int = 30
    gradient_as_bucket_view: bool = False  # PyTorch DDP default is False
    static_graph: bool = False  # PyTorch DDP default is False


@dataclass
class CudaConf:
    cudnn_deterministic: bool = False
    cudnn_benchmark: bool = True
    allow_tf32: bool = False
    # if not None, `matmul_allow_tf32` key will override `allow_tf32` for matmul
    matmul_allow_tf32: Optional[bool] = None
    # if not None, `cudnn_allow_tf32` key will override `allow_tf32` for cudnn
    cudnn_allow_tf32: Optional[bool] = None


@dataclass
class CheckpointConf:
    save_dir: str
    save_freq: int
    save_list: List[int] = field(default_factory=list)
    model_weight_initializer: Any = None
    save_best_meters: List[str] = None
    skip_saving_parameters: List[str] = field(default_factory=list)
    initialize_after_preemption: Optional[bool] = None
    # if not None, training will be resumed from this checkpoint
    resume_from: Optional[str] = None

    def infer_missing(self):
        if self.initialize_after_preemption is None:
            with_skip_saving = len(self.skip_saving_parameters) > 0
            self.initialize_after_preemption = with_skip_saving
        return self


@dataclass
class LoggingConf:
    log_dir: str
    log_freq: int  # In iterations
    tensorboard_writer: Any
    log_level_primary: str = "INFO"
    log_level_secondary: str = "ERROR"
    log_scalar_frequency: int = 100
    log_visual_frequency: int = 100
    scalar_keys_to_log: Optional[Dict[str, Any]] = None
    log_batch_stats: bool = False
    wandb_writer: Optional[Any] = None


class TrainerWithFreezing:
    """
    Trainer supporting the DDP training strategies with parameter freezing.
    
    This extends the base Trainer with:
    - Systematic parameter freezing via configuration
    - Detailed logging of frozen vs trainable parameters
    - Support for adapter-style training
    """

    EPSILON = 1e-8

    def __init__(
        self,
        *,  # the order of these args can change at any time, so they are keyword-only
        data: Dict[str, Any],
        model: Dict[str, Any],
        logging: Dict[str, Any],
        checkpoint: Dict[str, Any],
        max_epochs: int,
        mode: str = "train",
        accelerator: str = "cuda",
        seed_value: int = 123,
        val_epoch_freq: int = 1,
        distributed: Dict[str, bool] = None,
        cuda: Dict[str, bool] = None,
        env_variables: Optional[Dict[str, Any]] = None,
        optim: Optional[Dict[str, Any]] = None,
        optim_overrides: Optional[List[Dict[str, Any]]] = None,
        meters: Optional[Dict[str, Any]] = None,
        loss: Optional[Dict[str, Any]] = None,
        skip_first_val: bool = False,
        skip_saving_ckpts: bool = False,
        empty_gpu_mem_cache_after_eval: bool = True,
        gradient_accumulation_steps: int = 1,
        # NEW: Freezing configuration
        freeze_config: Optional[Dict[str, Any]] = None,
    ):
        self._setup_env_variables(env_variables)
        self._setup_timers()

        self.data_conf = data
        self.model_conf = model
        self.logging_conf = LoggingConf(**logging)
        self.checkpoint_conf = CheckpointConf(**checkpoint).infer_missing()
        self.max_epochs = max_epochs
        self.mode = mode
        self.val_epoch_freq = val_epoch_freq
        self.optim_conf = OptimConf(**optim) if optim is not None else OptimConf()
        self.meters_conf = meters
        self.loss_conf = loss
        self.gradient_accumulation_steps = gradient_accumulation_steps
        distributed = DistributedConf(**distributed or {})
        cuda = CudaConf(**cuda or {})
        self.where = 0.0

        self.skip_first_val = skip_first_val
        self.skip_saving_ckpts = skip_saving_ckpts
        self.empty_gpu_mem_cache_after_eval = empty_gpu_mem_cache_after_eval

        # NEW: Parse freezing configuration
        self.freeze_conf = self._parse_freeze_config(freeze_config)

        self._infer_distributed_backend_if_none(distributed, accelerator)

        self._setup_device(accelerator)

        self._setup_torch_dist_and_backend(cuda, distributed)

        makedir(self.logging_conf.log_dir)
        setup_logging(
            __name__,
            output_dir=self.logging_conf.log_dir,
            rank=self.rank,
            log_level_primary=self.logging_conf.log_level_primary,
            log_level_secondary=self.logging_conf.log_level_secondary,
        )

        set_seeds(seed_value, self.max_epochs, self.distributed_rank)
        log_env_variables()

        assert (
            is_dist_avail_and_initialized()
        ), "Torch distributed needs to be initialized before calling the trainer."

        self._setup_components()  # Except Optimizer everything is setup here.
        
        # NEW: Apply freezing after model is created but before moving to device
        self._apply_freezing()
        
        self._move_to_device()
        self._construct_optimizers()
        self._setup_dataloaders()

        self.time_elapsed_meter = DurationMeter("Time Elapsed", self.device, ":.2f")

        if self.checkpoint_conf.resume_from is not None:
            assert os.path.exists(
                self.checkpoint_conf.resume_from
            ), f"The 'resume_from' checkpoint {self.checkpoint_conf.resume_from} does not exist!"
            dst = os.path.join(self.checkpoint_conf.save_dir, "checkpoint.pt")
            if self.distributed_rank == 0 and not os.path.exists(dst):
                # Copy the "resume_from" checkpoint to the checkpoint folder
                # if there is not a checkpoint to resume from already there
                makedir(self.checkpoint_conf.save_dir)
                g_pathmgr.copy(self.checkpoint_conf.resume_from, dst)
            barrier()

        self.load_checkpoint()
        self._setup_ddp_distributed_training(distributed, accelerator)
        barrier()

    def _parse_freeze_config(self, freeze_config: Optional[Dict[str, Any]]) -> FreezeConfig:
        """Parse the freeze configuration from the YAML config."""
        if freeze_config is None:
            return FreezeConfig()
        return FreezeConfig(**freeze_config)

    def _apply_freezing(self) -> None:
        """Apply parameter freezing based on configuration."""
        if self.freeze_conf is None:
            return
        
        logging.info("Applying parameter freezing configuration...")
        
        # Check if we should freeze all except specific patterns
        if self.freeze_conf.freeze_all_except:
            result = freeze_all_except_patterns(
                self.model,
                self.freeze_conf.freeze_all_except,
            )
            logging.info(f"Frozen all except patterns. Trainable: {result['trainable']}, Frozen: {result['frozen']}")
        else:
            # Get patterns to freeze from configuration
            freeze_patterns = self.freeze_conf.get_freeze_patterns()
            
            if freeze_patterns:
                result = freeze_parameters_by_patterns(
                    self.model,
                    freeze_patterns,
                    self.freeze_conf.unfreeze_patterns,
                )
                logging.info(f"Applied freezing patterns. Trainable: {result['trainable']}, Frozen: {result['frozen']}")
        
        # Log detailed freezing information
        log_frozen_parameters(self.model, self.logging_conf.log_dir)
        
        # Update skip_saving_parameters to match frozen parameters
        # This ensures frozen parameters are not saved in checkpoints (optional, saves space)
        frozen_patterns = self.freeze_conf.get_freeze_patterns()
        if frozen_patterns:
            # Merge with existing skip_saving_parameters
            existing = set(self.checkpoint_conf.skip_saving_parameters)
            existing.update(frozen_patterns)
            self.checkpoint_conf.skip_saving_parameters = list(existing)

    def _setup_timers(self):
        """
        Initializes counters for elapsed time and eta.
        """
        self.start_time = time.time()
        self.ckpt_time_elapsed = 0
        self.est_epoch_time = dict.fromkeys([Phase.TRAIN, Phase.VAL], 0)

    def _get_meters(self, phase_filters=None):
        if self.meters is None:
            return {}
        meters = {}
        for phase, phase_meters in self.meters.items():
            if phase_filters is not None and phase not in phase_filters:
                continue
            for key, key_meters in phase_meters.items():
                if key_meters is None:
                    continue
                for name, meter in key_meters.items():
                    meters[f"{phase}_{key}/{name}"] = meter
        return meters

    def _infer_distributed_backend_if_none(self, distributed_conf, accelerator):
        if distributed_conf.backend is None:
            distributed_conf.backend = "nccl" if accelerator == "cuda" else "gloo"

    def _setup_env_variables(self, env_variables_conf) -> None:
        if env_variables_conf is not None:
            for variable_name, value in env_variables_conf.items():
                os.environ[variable_name] = value

    def _setup_torch_dist_and_backend(self, cuda_conf, distributed_conf) -> None:
        if torch.cuda.is_available():
            torch.backends.cudnn.deterministic = cuda_conf.cudnn_deterministic
            torch.backends.cudnn.benchmark = cuda_conf.cudnn_benchmark
            torch.backends.cuda.matmul.allow_tf32 = (
                cuda_conf.matmul_allow_tf32
                if cuda_conf.matmul_allow_tf32 is not None
                else cuda_conf.allow_tf32
            )
            torch.backends.cudnn.allow_tf32 = (
                cuda_conf.cudnn_allow_tf32
                if cuda_conf.cudnn_allow_tf32 is not None
                else cuda_conf.allow_tf32
            )

        self.rank = setup_distributed_backend(
            distributed_conf.backend, distributed_conf.timeout_mins
        )

    def _setup_device(self, accelerator):
        self.local_rank, self.distributed_rank = get_machine_local_and_dist_rank()
        if accelerator == "cuda":
            self.device = torch.device("cuda", self.local_rank)
            torch.cuda.set_device(self.local_rank)
        elif accelerator == "cpu":
            self.device = torch.device("cpu")
        else:
            raise ValueError(f"Unsupported accelerator: {accelerator}")

    def _setup_ddp_distributed_training(self, distributed_conf, accelerator):
        assert isinstance(self.model, torch.nn.Module)

        self.model = nn.parallel.DistributedDataParallel(
            self.model,
            device_ids=[self.local_rank] if accelerator == "cuda" else [],
            find_unused_parameters=distributed_conf.find_unused_parameters,
            gradient_as_bucket_view=distributed_conf.gradient_as_bucket_view,
            static_graph=distributed_conf.static_graph,
        )
        if distributed_conf.comms_dtype is not None:  # noqa
            from torch.distributed.algorithms import ddp_comm_hooks

            amp_type = get_amp_type(distributed_conf.comms_dtype)
            if amp_type == torch.bfloat16:
                hook = ddp_comm_hooks.default_hooks.bf16_compress_hook
                logging.info("Enabling bfloat16 grad communication")
            else:
                hook = ddp_comm_hooks.default_hooks.fp16_compress_hook
                logging.info("Enabling fp16 grad communication")
            process_group = None
            self.model.register_comm_hook(process_group, hook)

    def _move_to_device(self):
        logging.info(
            f"Moving components to device {self.device} and local rank {self.local_rank}."
        )

        self.model.to(self.device)

        logging.info(
            f"Done moving components to device {self.device} and local rank {self.local_rank}."
        )

    def save_checkpoint(self, epoch, checkpoint_names=None):
        if self.skip_saving_ckpts:
            logging.info(
                "skip_saving_ckpts is set to True. So, no checkpoints have been saved."
            )
            return
        checkpoint_folder = self.checkpoint_conf.save_dir
        makedir(checkpoint_folder)
        if checkpoint_names is None:
            checkpoint_names = ["checkpoint"]
            if (
                self.checkpoint_conf.save_freq > 0
                and (int(epoch) % self.checkpoint_conf.save_freq == 0)
            ) or int(epoch) in self.checkpoint_conf.save_list:
                checkpoint_names.append(f"checkpoint_{int(epoch)}")

        checkpoint_paths = []
        for ckpt_name in checkpoint_names:
            checkpoint_paths.append(os.path.join(checkpoint_folder, f"{ckpt_name}.pt"))

        state_dict = unwrap_ddp_if_wrapped(self.model).state_dict()
        state_dict = exclude_params_matching_unix_pattern(
            patterns=self.checkpoint_conf.skip_saving_parameters, state_dict=state_dict
        )

        checkpoint = {
            "model": state_dict,
            "optimizer": self.optim.optimizer.state_dict(),
            "epoch": epoch,
            "loss": self.loss.state_dict(),
            "steps": self.steps,
            "time_elapsed": self.time_elapsed_meter.val,
            "best_meter_values": self.best_meter_values,
            # NEW: Save freeze config for reference
            "freeze_config": self.freeze_conf.__dict__ if self.freeze_conf else None,
        }
        if self.optim_conf.amp.enabled:
            checkpoint["scaler"] = self.scaler.state_dict()

        # DDP checkpoints are only saved on rank 0 (all workers are identical)
        if self.distributed_rank != 0:
            return

        for checkpoint_path in checkpoint_paths:
            self._save_checkpoint(checkpoint, checkpoint_path)

    def _save_checkpoint(self, checkpoint, checkpoint_path):
        """
        Save a checkpoint while guarding against the job being killed in the middle
        of checkpoint saving (which corrupts the checkpoint file and ruins the
        entire training since usually only the last checkpoint is kept per run).

        We first save the new checkpoint to a temp file (with a '.tmp' suffix), and
        and move it to overwrite the old checkpoint_path.
        """
        checkpoint_path_tmp = f"{checkpoint_path}.tmp"
        with g_pathmgr.open(checkpoint_path_tmp, "wb") as f:
            torch.save(checkpoint, f)
        # after torch.save is completed, replace the old checkpoint with the new one
        if g_pathmgr.exists(checkpoint_path):
            # remove the old checkpoint_path file first (otherwise g_pathmgr.mv fails)
            g_pathmgr.rm(checkpoint_path)
        success = g_pathmgr.mv(checkpoint_path_tmp, checkpoint_path)
        assert success

    def load_checkpoint(self):
        ckpt_path = get_resume_checkpoint(self.checkpoint_conf.save_dir)
        if ckpt_path is None:
            self._init_model_state()
        else:
            if self.checkpoint_conf.initialize_after_preemption:
                self._call_model_initializer()
            self._load_resuming_checkpoint(ckpt_path)

    def _init_model_state(self):
        # Checking that parameters that won't be saved are indeed frozen
        # We do this check here before even saving the model to catch errors
        # are early as possible and not at the end of the first epoch
        assert_skipped_parameters_are_frozen(
            patterns=self.checkpoint_conf.skip_saving_parameters,
            model=self.model,
        )

        # Checking that parameters that won't be saved are initialized from
        # within the model definition, unless `initialize_after_preemption`
        # is explicitly set to `True`. If not, this is a bug, and after
        # preemption, the `skip_saving_parameters` will have random values
        allow_init_skip_parameters = self.checkpoint_conf.initialize_after_preemption
        with with_check_parameter_frozen(
            patterns=self.checkpoint_conf.skip_saving_parameters,
            model=self.model,
            disabled=allow_init_skip_parameters,
        ):
            self._call_model_initializer()

    def _call_model_initializer(self):
        model_weight_initializer = instantiate(
            self.checkpoint_conf.model_weight_initializer
        )
        if model_weight_initializer is not None:
            logging.info(
                f"Loading pretrained checkpoint from {self.checkpoint_conf.model_weight_initializer}"
            )
            self.model = model_weight_initializer(model=self.model)

    def _load_resuming_checkpoint(self, ckpt_path: str):
        logging.info(f"Resuming training from {ckpt_path}")

        with g_pathmgr.open(ckpt_path, "rb") as f:
            checkpoint = torch.load(f, map_location="cpu", weights_only=False)
        load_state_dict_into_model(
            model=self.model,
            state_dict=checkpoint["model"],
            ignore_missing_keys=self.checkpoint_conf.skip_saving_parameters,
        )

        self.optim.optimizer.load_state_dict(checkpoint["optimizer"])
        self.loss.load_state_dict(checkpoint["loss"], strict=True)
        self.epoch = checkpoint["epoch"]
        self.steps = checkpoint["steps"]
        self.ckpt_time_elapsed = checkpoint.get("time_elapsed")

        if self.optim_conf.amp.enabled and "scaler" in checkpoint:
            self.scaler.load_state_dict(checkpoint["scaler"])

        self.best_meter_values = checkpoint.get("best_meter_values", {})

        if "train_dataset" in checkpoint and self.train_dataset is not None:
            self.train_dataset.load_checkpoint_state(checkpoint["train_dataset"])

    def is_intermediate_val_epoch(self, epoch):
        skip_epoch = self.skip_first_val and epoch == 0
        return (
            epoch % self.val_epoch_freq == 0
            and epoch < self.max_epochs - 1
            and not skip_epoch
        )

    def _find_loss(self, key: str):
        if key in self.loss:
            return self.loss[key]

        assert key != "all", "Loss must be specified for key='all'"
        assert (
            "default" in self.loss
        ), f"Key {key} not found in losss, and no default provided"
        return self.loss["default"]

    def _find_meter(self, phase: str, key: str):
        if key in self.meters[phase]:
            return self.meters[phase][key]

        for cand_key, meter in self.meters[phase].items():
            if fnmatch.fnmatch(key, cand_key):
                return meter
        return None

    def _step(
        self,
        batch: BatchedDatapoint,
        model: nn.Module,
        phase: str,
    ):
        key, batch = batch.popitem()
        batch = copy_data_to_device(batch, self.device, non_blocking=True)

        find_stages = model(batch)
        find_targets = [
            unwrap_ddp_if_wrapped(model).back_convert(x) for x in batch.find_targets
        ]
        batch_size = len(batch.img_batch)
        loss = self._find_loss(key)(find_stages, find_targets)

        loss_str = f"Losses/{phase}_{key}_loss"

        loss_log_str = os.path.join("Step_Losses", loss_str)

        # loss contains multiple sub-components we wish to log
        step_losses = {}
        if isinstance(loss, dict):
            step_losses.update(
                {f"Losses/{phase}_{key}_{k}": v for k, v in loss.items()}
            )
            loss = self._log_loss_detailed_and_return_core_loss(
                loss, loss_log_str, self.steps[phase]
            )

        if self.steps[phase] % self.logging_conf.log_scalar_frequency == 0:
            self.logger.log(
                loss_log_str,
                loss,
                self.steps[phase],
            )

        self.steps[phase] += 1

        ret_tuple = {loss_str: loss}, batch_size, step_losses

        if phase not in self.meters:
            return ret_tuple

        meters_dict = self._find_meter(phase, key)
        if meters_dict is None:
            return ret_tuple
        if meters_dict is not None:
            for _, meter in meters_dict.items():
                meter.update(
                    find_stages=find_stages,
                    find_metadatas=batch.find_metadatas,
                    model=model,
                    batch=batch,
                    key=key,
                    phase=phase,
                )
            # Cleanup memory
            if isinstance(find_stages, SAM3Output):
                for fs in find_stages:
                    for k in list(fs.keys()):
                        del fs[k]

        return ret_tuple

    def run(self):
        assert self.mode in ["train", "train_only", "val"]
        if self.mode == "train":
            if self.epoch > 0:
                logging.info(f"Resuming training from epoch: {self.epoch}")
                # resuming from a checkpoint
                if self.is_intermediate_val_epoch(self.epoch - 1):
                    logging.info("Running previous val epoch")
                    self.epoch -= 1
                    self.run_val()
                    self.epoch += 1
            self.run_train()
            self.run_val()
        elif self.mode == "val":
            self.run_val()
        elif self.mode == "train_only":
            self.run_train()

    def _setup_dataloaders(self):
        self.train_dataset = None
        self.val_dataset = None

        if self.mode in ["train", "val"]:
            self.val_dataset = instantiate(self.data_conf.get(Phase.VAL, None))

        if self.mode in ["train", "train_only"]:
            self.train_dataset = instantiate(self.data_conf.train)

    def run_train(self):
        while self.epoch < self.max_epochs:
            dataloader = self.train_dataset.get_loader(epoch=int(self.epoch))
            barrier()
            outs = self.train_epoch(dataloader)
            self.logger.log_dict(outs, self.epoch)  # Logged only on rank 0

            # log train to text file.
            if self.distributed_rank == 0:
                with g_pathmgr.open(
                    os.path.join(self.logging_conf.log_dir, "train_stats.json"),
                    "a",
                ) as f:
                    f.write(json.dumps(outs) + "\n")

            # Save checkpoint before validating
            self.save_checkpoint(self.epoch + 1)

            del dataloader
            gc.collect()

            # Run val, not running on last epoch since will run after the
            # loop anyway
            if self.is_intermediate_val_epoch(self.epoch):
                self.run_val()
                if torch.cuda.is_available() and self.empty_gpu_mem_cache_after_eval:
                    # release memory buffers held by the model during eval (which typically
                    # involves a lot more frames in video grounding that during training)
                    torch.cuda.empty_cache()

            if self.distributed_rank == 0:
                self.best_meter_values.update(self._get_trainer_state("train"))
                with g_pathmgr.open(
                    os.path.join(self.logging_conf.log_dir, "best_stats.json"),
                    "a",
                ) as f:
                    f.write(json.dumps(self.best_meter_values) + "\n")

            self.epoch += 1
        # epoch was incremented in the loop but the val step runs out of the loop
        self.epoch -= 1

    def run_val(self):
        if not self.val_dataset:
            return

        dataloader = self.val_dataset.get_loader(epoch=int(self.epoch))
        outs = self.val_epoch(dataloader, phase=Phase.VAL)
        del dataloader
        gc.collect()
        self.logger.log_dict(outs, self.epoch)  # Logged only on rank 0

        if self.distributed_rank == 0:
            with g_pathmgr.open(
                os.path.join(self.logging_conf.log_dir, "val_stats.json"),
                "a",
            ) as f:
                f.write(json.dumps(outs) + "\n")

    def val_epoch(self, val_loader, phase):
        batch_time = AverageMeter("Batch Time", self.device, ":.2f")
        data_time = AverageMeter("Data Time", self.device, ":.2f")
        mem = MemMeter("Mem (GB)", self.device, ":.2f")

        iters_per_epoch = len(val_loader)

        curr_phases = [phase]
        curr_models = [self.model]

        loss_names = []
        for p in curr_phases:
            for key in self.loss.keys():
                loss_names.append(f"Losses/{p}_{key}_loss")

        loss_mts = OrderedDict(
            [(name, AverageMeter(name, self.device, ":.2e")) for name in loss_names]
        )
        extra_loss_mts = {}

        for model in curr_models:
            model.eval()
            if hasattr(unwrap_ddp_if_wrapped(model), "on_validation_epoch_start"):
                unwrap_ddp_if_wrapped(model).on_validation_epoch_start()

        progress = ProgressMeter(
            iters_per_epoch,
            [batch_time, data_time, mem, self.time_elapsed_meter, *loss_mts.values()],
            self._get_meters(curr_phases),
            prefix="Val Epoch: [{}]".format(self.epoch),
        )

        end = time.time()

        for data_iter, batch in enumerate(val_loader):
            # measure data loading time
            data_time.update(time.time() - end)

            # compute output
            with torch.no_grad():
                with torch.amp.autocast(
                    device_type="cuda",
                    enabled=(self.optim_conf.amp.enabled if self.optim_conf else False),
                    dtype=(
                        get_amp_type(self.optim_conf.amp.amp_dtype)
                        if self.optim_conf
                        else None
                    ),
                ):
                    for phase, model in zip(curr_phases, curr_models):
                        loss_dict, batch_size, extra_losses = self._step(
                            batch,
                            model,
                            phase,
                        )
                        for k, v in loss_dict.items():
                            if k in loss_mts:
                                loss_mts[k].update(v.item() if isinstance(v, torch.Tensor) else v, batch_size)
                        for k, v in extra_losses.items():
                            if k not in extra_loss_mts:
                                extra_loss_mts[k] = AverageMeter(k, self.device, ":.2e")
                            extra_loss_mts[k].update(v.item() if isinstance(v, torch.Tensor) else v, batch_size)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            self.time_elapsed_meter.update(
                time.time() - self.start_time + self.ckpt_time_elapsed
            )

            if data_iter % self.logging_conf.log_freq == 0:
                mem.update()
                progress.display(data_iter)

        self.est_epoch_time[phase] = batch_time.avg * iters_per_epoch
        self._log_timers(phase)
        for model in curr_models:
            if hasattr(unwrap_ddp_if_wrapped(model), "on_validation_epoch_end"):
                unwrap_ddp_if_wrapped(model).on_validation_epoch_end()

        logging.info("Synchronizing meters")
        out_dict = self._reduce_and_log_meters(curr_phases)

        for k, v in loss_mts.items():
            out_dict[k] = v.avg
        for k, v in extra_loss_mts.items():
            out_dict[k] = v.avg

        for phase in curr_phases:
            out_dict.update(self._get_trainer_state(phase))
        self._reset_meters(curr_phases)
        logging.info(f"Meters: {out_dict}")
        return out_dict

    def train_epoch(self, train_loader):
        batch_time = AverageMeter("Batch Time", self.device, ":.2f")
        data_time = AverageMeter("Data Time", self.device, ":.2f")
        mem = MemMeter("Mem (GB)", self.device, ":.2f")

        iters_per_epoch = len(train_loader)

        curr_phases = [Phase.TRAIN]
        curr_models = [self.model]

        loss_names = []
        for p in curr_phases:
            for key in self.loss.keys():
                loss_names.append(f"Losses/{p}_{key}_loss")

        loss_mts = OrderedDict(
            [(name, AverageMeter(name, self.device, ":.2e")) for name in loss_names]
        )
        extra_loss_mts = {}

        for model in curr_models:
            model.train()

        progress = ProgressMeter(
            iters_per_epoch,
            [batch_time, data_time, mem, self.time_elapsed_meter, *loss_mts.values()],
            self._get_meters(curr_phases),
            prefix="Epoch: [{}]".format(self.epoch),
        )

        end = time.time()

        for data_iter, batch in enumerate(train_loader):
            # measure data loading time
            data_time.update(time.time() - end)

            # Compute where we are in training and update schedulers
            exact_epoch = self.epoch + float(data_iter) / iters_per_epoch
            self.where = float(exact_epoch) / self.max_epochs
            
            assert self.where <= 1 + self.EPSILON
            if self.where < 1.0:
                self.optim.step_schedulers(
                    self.where, step=int(exact_epoch * iters_per_epoch)
                )
            else:
                logging.warning(
                    f"Skipping scheduler update since training is at the end: {self.where} of [0,1]."
                )

            with torch.amp.autocast(
                device_type="cuda",
                enabled=(self.optim_conf.amp.enabled if self.optim_conf else False),
                dtype=(
                    get_amp_type(self.optim_conf.amp.amp_dtype)
                    if self.optim_conf
                    else None
                ),
            ):
                for phase, model in zip(curr_phases, curr_models):
                    loss_dict, batch_size, extra_losses = self._step(
                        batch,
                        model,
                        phase,
                    )
                    for k, v in loss_dict.items():
                        if k in loss_mts:
                            loss_mts[k].update(v.item() if isinstance(v, torch.Tensor) else v, batch_size)
                    for k, v in extra_losses.items():
                        if k not in extra_loss_mts:
                            extra_loss_mts[k] = AverageMeter(k, self.device, ":.2e")
                        extra_loss_mts[k].update(v.item() if isinstance(v, torch.Tensor) else v, batch_size)

                    # Get the loss value directly from loss_dict (only one key-value pair)
                    loss = list(loss_dict.values())[0]

            # Scale loss for gradient accumulation
            loss = loss / self.gradient_accumulation_steps
            self.scaler.scale(loss).backward()

            # Only update weights after accumulating gradients
            if (data_iter + 1) % self.gradient_accumulation_steps == 0:
                if self.gradient_clipper is not None:
                    self.scaler.unscale_(self.optim.optimizer)
                    self.gradient_clipper(self.model)

                if self.gradient_logger is not None:
                    self.gradient_logger(
                        model=self.model,
                        logger=self.logger,
                        step=self.steps[Phase.TRAIN],
                        log_freq=self.logging_conf.log_scalar_frequency,
                    )

                self.scaler.step(self.optim.optimizer)
                self.scaler.update()
                self.optim.optimizer.zero_grad()

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            self.time_elapsed_meter.update(
                time.time() - self.start_time + self.ckpt_time_elapsed
            )

            if data_iter % self.logging_conf.log_freq == 0:
                mem.update()
                progress.display(data_iter)

        # Est epoch time
        self.est_epoch_time[Phase.TRAIN] = batch_time.avg * iters_per_epoch

        self._log_timers(Phase.TRAIN)

        logging.info("Synchronizing meters")
        out_dict = self._reduce_and_log_meters(curr_phases)

        for k, v in loss_mts.items():
            out_dict[k] = v.avg
        for k, v in extra_loss_mts.items():
            out_dict[k] = v.avg

        # Log LR per param groups
        for i, param_group in enumerate(self.optim.optimizer.param_groups):
            out_dict[f"LR/param_group_{i}"] = param_group["lr"]
            self.logger.log(f"LR/param_group_{i}", param_group["lr"], self.epoch)

        out_dict.update(self._get_trainer_state(Phase.TRAIN))
        logging.info(f"Losses and meters: {out_dict}")
        self._reset_meters([Phase.TRAIN])
        return out_dict

    def _get_trainer_state(self, phase: str) -> Dict[str, float]:
        """Get the current trainer state for logging."""
        return {
            "Trainer/where": self.where,
            "Trainer/epoch": self.epoch,
            f"Trainer/steps_{phase}": self.steps[phase],
        }

    def _reduce_and_log_meters(self, phases: List[str]) -> Dict[str, float]:
        """Reduce meters across processes and log results - matches original _log_meters_and_save_best_ckpts."""
        logging.info("Synchronizing meters")
        out_dict = {}
        checkpoint_save_keys = []
        for key, meter in self._get_meters(phases).items():
            # Try to pass epoch parameter, fall back if not supported
            try:
                meter_output = meter.compute_synced(epoch=self.epoch)
            except TypeError:
                meter_output = meter.compute_synced()
            is_better_check = getattr(meter, "is_better", None)

            for meter_subkey, meter_value in meter_output.items():
                # If metric already has phase prefix followed by underscore (train_*, val_*), use it directly
                # This avoids wrapping custom metrics like train_cc_metrics/iou in Meters_train
                # Otherwise, wrap it in Meters_train for backward compatibility with COCO metrics
                if '_' in meter_subkey and meter_subkey.split('_')[0] in ['train', 'val']:
                    out_dict[meter_subkey] = meter_value
                else:
                    out_dict[os.path.join("Meters_train", key, meter_subkey)] = meter_value

                if is_better_check is None:
                    continue

                tracked_meter_key = os.path.join(key, meter_subkey)
                if tracked_meter_key not in self.best_meter_values or is_better_check(
                    meter_value,
                    self.best_meter_values[tracked_meter_key],
                ):
                    self.best_meter_values[tracked_meter_key] = meter_value

                    if (
                        self.checkpoint_conf.save_best_meters is not None
                        and key in self.checkpoint_conf.save_best_meters
                    ):
                        checkpoint_save_keys.append(tracked_meter_key.replace("/", "_"))

        if len(checkpoint_save_keys) > 0:
            self.save_checkpoint(self.epoch + 1, checkpoint_save_keys)

        return out_dict

    def _log_timers(self, phase):
        time_remaining = 0
        epochs_remaining = self.max_epochs - self.epoch - 1
        val_epochs_remaining = sum(
            n % self.val_epoch_freq == 0 for n in range(self.epoch, self.max_epochs)
        )

        # Adding the guaranteed val run at the end if val_epoch_freq doesn't coincide with
        # the end epoch.
        if (self.max_epochs - 1) % self.val_epoch_freq != 0:
            val_epochs_remaining += 1

        # Remove the current val run from estimate
        if phase == Phase.VAL:
            val_epochs_remaining -= 1

        time_remaining += (
            epochs_remaining * self.est_epoch_time[Phase.TRAIN]
            + val_epochs_remaining * self.est_epoch_time[Phase.VAL]
        )

        self.logger.log(
            os.path.join("Step_Stats", phase, self.time_elapsed_meter.name),
            self.time_elapsed_meter.val,
            self.steps[phase],
        )

        logging.info(f"Estimated time remaining: {human_readable_time(time_remaining)}")

    def _reset_meters(self, phases: str) -> None:
        for meter in self._get_meters(phases).values():
            meter.reset()

    def _check_val_key_match(self, val_keys, phase):
        if val_keys is not None:
            # Check if there are any duplicates
            assert len(val_keys) == len(
                set(val_keys)
            ), f"Duplicate keys in val datasets, keys: {val_keys}"

            # Check that the keys match the meter keys
            if self.meters_conf is not None and phase in self.meters_conf:
                assert set(val_keys) == set(self.meters_conf[phase].keys()), (
                    f"Keys in val datasets do not match the keys in meters."
                    f"\nMissing in meters: {set(val_keys) - set(self.meters_conf[phase].keys())}"
                    f"\nMissing in val datasets: {set(self.meters_conf[phase].keys()) - set(val_keys)}"
                )

            if self.loss_conf is not None:
                loss_keys = set(self.loss_conf.keys()) - set(["all"])
                if "default" not in loss_keys:
                    for k in val_keys:
                        assert (
                            k in loss_keys
                        ), f"Error: key {k} is not defined in the losses, and no default is set"

    def _setup_components(self):
        # Get the keys for all the val datasets, if any
        val_phase = Phase.VAL
        val_keys = None
        if self.data_conf.get(val_phase, None) is not None:
            val_keys = collect_dict_keys(self.data_conf[val_phase])
        # Additional checks on the sanity of the config for val datasets
        self._check_val_key_match(val_keys, phase=val_phase)

        logging.info("Setting up components: Model, loss, optim, meters etc.")
        self.epoch = 0
        self.steps = {Phase.TRAIN: 0, Phase.VAL: 0}

        self.logger = Logger(self.logging_conf)

        self.model = instantiate(self.model_conf, _convert_="all")
        print_model_summary(self.model)

        self.loss = None
        if self.loss_conf:
            self.loss = {
                key: el  # wrap_base_loss(el)
                for (key, el) in instantiate(self.loss_conf, _convert_="all").items()
            }
            self.loss = nn.ModuleDict(self.loss)

        self.meters = {}
        self.best_meter_values = {}
        if self.meters_conf:
            self.meters = instantiate(self.meters_conf, _convert_="all")
            
            # Inject logger into SegmentationMetricsMeter instances
            for phase_meters in self.meters.values():
                if isinstance(phase_meters, dict):
                    for meter in phase_meters.values():
                        if isinstance(meter, dict):
                            for sub_meter in meter.values():
                                if isinstance(sub_meter, SegmentationMetricsMeter):
                                    sub_meter.logger = self.logger
                        elif isinstance(meter, SegmentationMetricsMeter):
                            meter.logger = self.logger
                elif isinstance(phase_meters, SegmentationMetricsMeter):
                    phase_meters.logger = self.logger

        self.scaler = torch.amp.GradScaler(
            self.device,
            enabled=self.optim_conf.amp.enabled if self.optim_conf else False,
        )

        self.gradient_clipper = (
            instantiate(self.optim_conf.gradient_clip) if self.optim_conf else None
        )
        self.gradient_logger = (
            instantiate(self.optim_conf.gradient_logger) if self.optim_conf else None
        )

        logging.info("Finished setting up components: Model, loss, optim, meters etc.")

    def _construct_optimizers(self):
        self.optim = construct_optimizer(
            self.model,
            self.optim_conf.optimizer,
            self.optim_conf.options,
            self.optim_conf.param_group_modifiers,
        )

    def _log_loss_detailed_and_return_core_loss(self, loss, loss_str, step):
        core_loss = loss.pop(CORE_LOSS_KEY)
        if step % self.logging_conf.log_scalar_frequency == 0:
            for k in loss:
                log_str = os.path.join(loss_str, k)
                self.logger.log(log_str, loss[k], step)
        return core_loss


def print_model_summary(model: torch.nn.Module, log_dir: str = ""):
    """
    Prints the model and the number of parameters in the model.
    """
    if get_rank() != 0:
        return
    param_kwargs = {}
    trainable_parameters = sum(
        p.numel() for p in model.parameters(**param_kwargs) if p.requires_grad
    )
    total_parameters = sum(p.numel() for p in model.parameters(**param_kwargs))
    non_trainable_parameters = total_parameters - trainable_parameters
    logging.info("==" * 10)
    logging.info(f"Summary for model {type(model)}")
    logging.info(f"Model is {model}")
    logging.info(f"\tTotal parameters {get_human_readable_count(total_parameters)}")
    logging.info(
        f"\tTrainable parameters {get_human_readable_count(trainable_parameters)}"
    )
    logging.info(
        f"\tNon-Trainable parameters {get_human_readable_count(non_trainable_parameters)}"
    )
    logging.info("==" * 10)

    if log_dir:
        output_fpath = os.path.join(log_dir, "model.txt")
        with g_pathmgr.open(output_fpath, "w") as f:
            print(model, file=f)


PARAMETER_NUM_UNITS = [" ", "K", "M", "B", "T"]


def get_human_readable_count(number: int) -> str:
    """
    Abbreviates an integer number with K, M, B, T for thousands, millions,
    billions and trillions, respectively.
    """
    assert number >= 0
    labels = PARAMETER_NUM_UNITS
    num_digits = int(np.floor(np.log10(number)) + 1 if number > 0 else 1)
    num_groups = int(np.ceil(num_digits / 3))
    num_groups = min(num_groups, len(labels))  # don't abbreviate beyond trillions
    shift = -3 * (num_groups - 1)
    number = number * (10**shift)
    index = num_groups - 1
    if index < 1 or number >= 100:
        return f"{int(number):,d} {labels[index]}"
    else:
        return f"{number:,.1f} {labels[index]}"


# =============================================================================
# CONVENIENCE ALIASES
# =============================================================================

# Alias for backward compatibility - use TrainerWithFreezing as the default Trainer
Trainer = TrainerWithFreezing
