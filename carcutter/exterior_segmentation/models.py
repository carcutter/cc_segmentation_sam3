"""Model construction and loading for the exterior segmentation pipeline.

THREE networks, four checkpoints:

  car + antenna detector   RF-DETR Medium, 2 classes (0=car, 1=antenna)
                           rfdetr_ca_unified_v1/checkpoint_best_ema.pth        (127 MB)
  spliced BiRefNet         Swin-L encoder run ONCE, two decoders spliced:
                             outline decoder from birefnet_aspect_prod_bucket5.pt  (844 MB)
                             tint+antenna decoder from birefnet_trihead_bucket.pt  (844 MB)
  antenna zoom UNet        EfficientNet-B4 UNet @256
                           unet_antenna_v3/checkpoints/best.pt                  (237 MB)

WHY SPLICED: a single 3-head decoder was trained (outline/tint/antenna), and its tint and antenna
heads beat their standalone models -- but sharing a decoder SOFTENED the outline boundary (BF@1
0.824 -> 0.731). So we keep prod_bucket5's untouched outline decoder and take only ch1/ch2 from the
tri-head decoder. Both were fine-tuned from the same encoder, so the Swin backbone is shared and
runs once.

BIREFNET FORK: this needs carcutter's patched BiRefNet, not upstream ZhengPeng7/BiRefNet. We added
BIREFNET_NHEADS (the tri-head decoder emits 3 channels), plus BIREFNET_SIZE / BIREFNET_BS env
overrides. Point BIREFNET_REPO at that checkout. Once exported to ONNX this dependency disappears,
which is the main argument for shipping ONNX to Triton rather than .pt.
"""
import os
import sys

import torch


def load_birefnet(ckpt_path, nheads, device, repo=None):
    """Build a BiRefNet with `nheads` output channels and load `ckpt_path` into it."""
    repo = repo or os.environ.get("BIREFNET_REPO")
    if not repo or not os.path.isdir(repo):
        raise RuntimeError(
            "Set BIREFNET_REPO (or pass repo=) to the carcutter BiRefNet checkout. "
            "Upstream ZhengPeng7/BiRefNet will NOT load these checkpoints: it lacks the "
            "BIREFNET_NHEADS patch that lets the decoder emit 3 channels."
        )
    if repo not in sys.path:
        sys.path.insert(0, repo)
    os.environ["BIREFNET_NHEADS"] = str(nheads)   # read by config.py at import time
    from models.birefnet import BiRefNet
    from utils import check_state_dict

    raw = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = BiRefNet(bb_pretrained=False).to(device).eval()
    model.load_state_dict(check_state_dict(raw["model"] if "model" in raw else raw))
    return model


class SplicedBiRefNet(torch.nn.Module):
    """Shared frozen Swin encoder + two decoders. Returns (outline, tint, antenna) logits."""

    def __init__(self, prod5, trihead):
        super().__init__()
        self.enc = prod5                        # encoder weights are identical in both checkpoints
        self.sq_o, self.dec_o = prod5.squeeze_module, prod5.decoder
        self.sq_t, self.dec_t = trihead.squeeze_module, trihead.decoder

    @torch.no_grad()
    def forward(self, x):
        (x1, x2, x3, x4), _ = self.enc.forward_enc(x)          # ONE Swin-L pass
        o = self.dec_o([x, x1, x2, x3, self.sq_o(x4)])[-1]     # (B,1) outline
        t = self.dec_t([x, x1, x2, x3, self.sq_t(x4)])[-1]     # (B,3) -> ch1 tint, ch2 antenna
        return o[:, 0:1], t[:, 1:2], t[:, 2:3]


def load_spliced(outline_ckpt, trihead_ckpt, device="cuda", repo=None):
    return SplicedBiRefNet(load_birefnet(outline_ckpt, 1, device, repo),
                           load_birefnet(trihead_ckpt, 3, device, repo)).eval()


def load_detector(ckpt_path):
    import rfdetr
    return rfdetr.RFDETRMedium.from_checkpoint(ckpt_path)


def load_antenna_unet(ckpt_path, device="cuda"):
    import segmentation_models_pytorch as smp
    m = smp.Unet("efficientnet-b4", encoder_weights=None, in_channels=3, classes=1).to(device).eval()
    st = torch.load(ckpt_path, map_location=device)
    m.load_state_dict(st.get("model", st))
    return m
