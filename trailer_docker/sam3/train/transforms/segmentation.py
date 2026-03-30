# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

# pyre-unsafe

import random

import numpy as np
import pycocotools.mask as mask_utils
import torch

import torchvision.transforms.functional as F
from PIL import Image as PILImage

from sam3.model.box_ops import masks_to_boxes

from sam3.train.data.sam3_image_dataset import Datapoint
from sam3.train.transforms.basic_for_api import crop as crop_datapoint


class InstanceToSemantic(object):
    """Convert instance segmentation to semantic segmentation."""

    def __init__(self, delete_instance=True, use_rle=False):
        self.delete_instance = delete_instance
        self.use_rle = use_rle

    def __call__(self, datapoint: Datapoint, **kwargs):
        for fquery in datapoint.find_queries:
            h, w = datapoint.images[fquery.image_id].size

            if self.use_rle:
                all_segs = [
                    datapoint.images[fquery.image_id].objects[obj_id].segment
                    for obj_id in fquery.object_ids_output
                ]
                if len(all_segs) > 0:
                    # we need to double check that all rles are the correct size
                    # Otherwise cocotools will fail silently to an empty [0,0] mask
                    for seg in all_segs:
                        assert seg["size"] == all_segs[0]["size"], (
                            "Instance segments have inconsistent sizes. "
                            f"Found sizes {seg['size']} and {all_segs[0]['size']}"
                        )
                    fquery.semantic_target = mask_utils.merge(all_segs)
                else:
                    # There is no good way to create an empty RLE of the correct size
                    # We resort to converting an empty box to RLE
                    fquery.semantic_target = mask_utils.frPyObjects(
                        np.array([[0, 0, 0, 0]], dtype=np.float64), h, w
                    )[0]

            else:
                # `semantic_target` is uint8 and remains uint8 throughout the transforms
                # (it contains binary 0 and 1 values just like `segment` for each object)
                fquery.semantic_target = torch.zeros((h, w), dtype=torch.uint8)
                for obj_id in fquery.object_ids_output:
                    segment = datapoint.images[fquery.image_id].objects[obj_id].segment
                    if segment is not None:
                        assert (
                            isinstance(segment, torch.Tensor)
                            and segment.dtype == torch.uint8
                        )
                        fquery.semantic_target |= segment

        if self.delete_instance:
            for img in datapoint.images:
                for obj in img.objects:
                    del obj.segment
                    obj.segment = None

        return datapoint


class RecomputeBoxesFromMasks:
    """Recompute bounding boxes from masks."""

    def __call__(self, datapoint: Datapoint, **kwargs):
        for img in datapoint.images:
            for obj in img.objects:
                # Note: if the mask is empty, the bounding box will be undefined
                # The empty targets should be subsequently filtered
                obj.bbox = masks_to_boxes(obj.segment)
                obj.area = obj.segment.sum().item()

        return datapoint


class DecodeRle:
    """This transform decodes RLEs into binary segments.
    Implementing it as a transforms allows lazy loading. Some transforms (eg query filters)
    may be deleting masks, so decoding them from the beginning is wasteful.

    This transforms needs to be called before any kind of geometric manipulation
    """

    def __call__(self, datapoint: Datapoint, **kwargs):
        imgId2size = {}
        warning_shown = False
        for imgId, img in enumerate(datapoint.images):
            if isinstance(img.data, PILImage.Image):
                img_w, img_h = img.data.size
            elif isinstance(img.data, torch.Tensor):
                img_w, img_h = img.data.shape[-2:]
            else:
                raise RuntimeError(f"Unexpected image type {type(img.data)}")

            imgId2size[imgId] = (img_h, img_w)

            for obj in img.objects:
                if obj.segment is not None and not isinstance(
                    obj.segment, torch.Tensor
                ):
                    if mask_utils.area(obj.segment) == 0:
                        print("Warning, empty mask found, approximating from box")
                        obj.segment = torch.zeros(img_h, img_w, dtype=torch.uint8)
                        x1, y1, x2, y2 = obj.bbox.int().tolist()
                        obj.segment[y1 : max(y2, y1 + 1), x1 : max(x1 + 1, x2)] = 1
                    else:
                        obj.segment = mask_utils.decode(obj.segment)
                        # segment is uint8 and remains uint8 throughout the transforms
                        obj.segment = torch.tensor(obj.segment).to(torch.uint8)

                    if list(obj.segment.shape) != [img_h, img_w]:
                        # Should not happen often, but adding for security
                        if not warning_shown:
                            print(
                                f"Warning expected instance segmentation size to be {[img_h, img_w]} but found {list(obj.segment.shape)}"
                            )
                            # Printing only once per datapoint to avoid spam
                            warning_shown = True

                        obj.segment = F.resize(
                            obj.segment[None], (img_h, img_w)
                        ).squeeze(0)

                    assert list(obj.segment.shape) == [img_h, img_w]

        warning_shown = False
        for query in datapoint.find_queries:
            if query.semantic_target is not None and not isinstance(
                query.semantic_target, torch.Tensor
            ):
                query.semantic_target = mask_utils.decode(query.semantic_target)
                # segment is uint8 and remains uint8 throughout the transforms
                query.semantic_target = torch.tensor(query.semantic_target).to(
                    torch.uint8
                )
                if tuple(query.semantic_target.shape) != imgId2size[query.image_id]:
                    if not warning_shown:
                        print(
                            f"Warning expected semantic segmentation size to be {imgId2size[query.image_id]} but found {tuple(query.semantic_target.shape)}"
                        )
                        # Printing only once per datapoint to avoid spam
                        warning_shown = True

                    query.semantic_target = F.resize(
                        query.semantic_target[None], imgId2size[query.image_id]
                    ).squeeze(0)

                assert tuple(query.semantic_target.shape) == imgId2size[query.image_id]

            if query.semantic_ignore_mask is not None and not isinstance(
                query.semantic_ignore_mask, torch.Tensor
            ):
                query.semantic_ignore_mask = mask_utils.decode(
                    query.semantic_ignore_mask
                )
                query.semantic_ignore_mask = torch.tensor(
                    query.semantic_ignore_mask
                ).to(torch.uint8)
                if (
                    tuple(query.semantic_ignore_mask.shape)
                    != imgId2size[query.image_id]
                ):
                    if not warning_shown:
                        print(
                            f"Warning expected semantic ignore mask size to be {imgId2size[query.image_id]} but found {tuple(query.semantic_ignore_mask.shape)}"
                        )
                        warning_shown = True

                    query.semantic_ignore_mask = F.resize(
                        query.semantic_ignore_mask[None],
                        imgId2size[query.image_id],
                    ).squeeze(0)

                assert (
                    tuple(query.semantic_ignore_mask.shape)
                    == imgId2size[query.image_id]
                )

        return datapoint


class MaskBoxCropAPI:
    """Crop around the union of instance masks with optional padding.

    Args:
        prob: Probability to apply the crop (ignored if always_apply=True)
        pad_fraction: Padding as a fraction of the mask bbox size
        min_pad_px: Minimum padding in pixels
        always_apply: Apply cropping deterministically (e.g., validation)
    """

    def __init__(
        self,
        prob: float = 0.5,
        pad_fraction: float = 0.1,
        min_pad_px: int = 10,
        always_apply: bool = False,
    ):
        self.prob = prob
        self.pad_fraction = pad_fraction
        self.min_pad_px = min_pad_px
        self.always_apply = always_apply

    def __call__(self, datapoint: Datapoint, **kwargs):
        if not self.always_apply and random.random() > self.prob:
            return datapoint

        for img_idx, img in enumerate(datapoint.images):
            masks = [obj.segment for obj in img.objects if obj.segment is not None]
            if len(masks) == 0:
                continue

            union = torch.zeros_like(masks[0], dtype=torch.uint8)
            for m in masks:
                union |= m

            ys, xs = torch.where(union > 0)
            if ys.numel() == 0 or xs.numel() == 0:
                continue

            y1 = int(ys.min().item())
            y2 = int(ys.max().item())
            x1 = int(xs.min().item())
            x2 = int(xs.max().item())

            if isinstance(img.data, PILImage.Image):
                img_w, img_h = img.data.size
            elif isinstance(img.data, torch.Tensor):
                img_h, img_w = img.data.shape[-2:]
            else:
                raise RuntimeError(f"Unexpected image type {type(img.data)}")

            box_h = max(1, y2 - y1 + 1)
            box_w = max(1, x2 - x1 + 1)
            pad_h = max(self.min_pad_px, int(round(self.pad_fraction * box_h)))
            pad_w = max(self.min_pad_px, int(round(self.pad_fraction * box_w)))

            top = max(0, y1 - pad_h)
            left = max(0, x1 - pad_w)
            bottom = min(img_h, y2 + pad_h + 1)
            right = min(img_w, x2 + pad_w + 1)

            height = max(1, bottom - top)
            width = max(1, right - left)

            datapoint = crop_datapoint(
                datapoint,
                img_idx,
                region=(top, left, height, width),
                v2=False,
                recompute_box_from_mask=True,
            )

        return datapoint
