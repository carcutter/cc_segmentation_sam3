#!/usr/bin/env python3
"""
Targeted hard-negative mining for the small-hole UNet. Run the current model on every
train tile; a tile is a HARD NEGATIVE if the model paints hole pixels where the GT label
has none (a confident false fire — tread/shadow/wheel-well read as a hole). Write the list
of hard-negative tile stems so the retrain can oversample exactly those confusions.

Run from repo root:
  PYTHONPATH=. python carcutter/car_bbox_detector/mine_smallhole_hardneg.py --ckpt <v1 best.pt>
"""
import argparse
from pathlib import Path
import numpy as np, cv2, torch
import segmentation_models_pytorch as smp
MEAN = np.array([0.485, 0.456, 0.406], np.float32); STD = np.array([0.229, 0.224, 0.225], np.float32)
DATA = "carcutter/car_bbox_detector/smallhole_data"


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="carcutter/car_bbox_detector/experiments/unet_smallhole_v1/best.pt")
    ap.add_argument("--out", default=f"{DATA}/hard_negs.txt")
    ap.add_argument("--fp-frac", type=float, default=0.004, help="false-fire pixel frac to call a tile hard")
    args = ap.parse_args()
    dev = "cuda"
    m = smp.Unet("efficientnet-b2", encoder_weights=None, in_channels=3, classes=1).to(dev).eval()
    m.load_state_dict(torch.load(args.ckpt, map_location=dev)["model"])
    items = sorted((Path(DATA) / "train" / "images").glob("*.png"))
    hard = []; nfp = 0
    for i in range(0, len(items), 16):
        batch = items[i:i + 16]
        ims, labs = [], []
        for p in batch:
            im = cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB)
            ims.append(((im / 255.0 - MEAN) / STD).transpose(2, 0, 1))
            labs.append(cv2.imread(str(Path(DATA) / "train" / "labels" / p.name), 0) > 127)
        x = torch.from_numpy(np.stack(ims)).float().to(dev)
        pr = (torch.sigmoid(m(x))[:, 0] > 0.5).cpu().numpy()
        for p, pred, lab in zip(batch, pr, labs):
            fp = (pred & ~lab).sum()
            if fp >= args.fp_frac * pred.size:        # confident false fire, no GT hole there
                hard.append(p.stem); nfp += 1
        if (i + 16) % 3200 == 0: print(f"  {i+16}/{len(items)} hard so far {nfp}", flush=True)
    Path(args.out).write_text("\n".join(hard))
    print(f"hard negatives: {len(hard)} / {len(items)} tiles ({100*len(hard)/len(items):.0f}%) -> {args.out}")


if __name__ == "__main__":
    main()
