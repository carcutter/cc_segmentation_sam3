#!/usr/bin/env python3
"""
Step 2: train the native-res boundary-refinement UNet.

Input = 4ch [native RGB patch + coarse-mask prob channel], sampled as 256px
native-resolution windows centered on the coarse-mask boundary. Target = GT
mask patch. The refiner learns to snap the coarse boundary to the true edge at
native resolution (no downsampling), guided by the coarse channel for which-side-
is-car context. encoder=efficientnet-b2 (light, fast at 256).

Run from repo root:
  PYTHONPATH=. /home/rutger/miniconda3/envs/cc_sam3/bin/python carcutter/car_bbox_detector/train_refine.py
"""
import argparse, csv, random
from pathlib import Path
import numpy as np, cv2, torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import segmentation_models_pytorch as smp

MEAN = np.array([0.485,0.456,0.406], np.float32); STD = np.array([0.229,0.224,0.225], np.float32)
ROOT = "/home/rutger/work/cc_segmentation_sam3_clean/carcutter/car_bbox_detector"


def boundary_band(m, d=1):
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*d+1, 2*d+1))
    return (cv2.dilate(m, k) - cv2.erode(m, k)) > 0


COLORS = {"white": (255, 255, 255), "blue": (0, 0, 255)}


def target_mask(rgb, target):
    return (rgb.sum(2) > 0) if target == "union" else (rgb == COLORS[target]).all(2)


class BoundaryPatchDS(Dataset):
    def __init__(self, rows, coarse_dir, patch=256, augment=False, gt_color="union"):
        self.items = []
        for r in rows:
            cp = Path(coarse_dir)/r["split"]/f"{Path(r['image']).stem}.png"
            if cp.exists():
                self.items.append((r["image"], r["mask"], str(cp)))
        self.patch, self.augment, self.gt_color = patch, augment, gt_color

    def __len__(self): return len(self.items)

    def __getitem__(self, i):
        imgp, mp, cp = self.items[i]
        img = np.array(Image.open(imgp).convert("RGB"))
        gt = target_mask(np.array(Image.open(mp).convert("RGB")), self.gt_color).astype(np.uint8)
        coarse = np.array(Image.open(cp)).astype(np.float32)/255.0
        H, W = gt.shape; p = self.patch
        bnd = boundary_band((coarse > 0.5).astype(np.uint8), 2)
        ys, xs = np.where(bnd)
        if len(ys) == 0:
            ys, xs = np.where(gt)
        if len(ys) == 0:
            cy, cx = H//2, W//2
        else:
            j = random.randrange(len(ys)); cy, cx = ys[j], xs[j]
        cy += random.randint(-p//4, p//4); cx += random.randint(-p//4, p//4)
        top = int(np.clip(cy-p//2, 0, max(0, H-p))); left = int(np.clip(cx-p//2, 0, max(0, W-p)))
        def crop(a):
            c = a[top:top+p, left:left+p]
            if c.shape[0] != p or c.shape[1] != p:  # pad if image smaller than patch
                pad = [(0, p-c.shape[0]), (0, p-c.shape[1])] + ([(0,0)] if c.ndim==3 else [])
                c = np.pad(c, pad)
            return c
        ip, cm, gp = crop(img), crop(coarse), crop(gt)
        if self.augment and random.random() < 0.5:
            ip, cm, gp = ip[:, ::-1], cm[:, ::-1], gp[:, ::-1]
        x = np.concatenate([((ip/255.0-MEAN)/STD).transpose(2,0,1), cm[None]], 0).astype(np.float32)
        return torch.from_numpy(x), torch.from_numpy(gp[None].astype(np.float32))


class BceDice(nn.Module):
    def __init__(self): super().__init__(); self.bce = nn.BCEWithLogitsLoss()
    def forward(self, logit, tgt):
        bce = self.bce(logit, tgt); p = torch.sigmoid(logit)
        dice = 1 - (2*(p*tgt).sum()+1)/((p+tgt).sum()+1)
        return 0.5*bce + 0.5*dice


def biou(logit, tgt):
    p = (torch.sigmoid(logit) > 0.5).float()
    inter = (p*tgt).sum(); union = ((p+tgt) > 0).float().sum()
    return (inter/union).item() if union > 0 else 1.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", default=f"{ROOT}/data/index.csv")
    ap.add_argument("--coarse", default=f"{ROOT}/coarse_masks")
    ap.add_argument("--exp", default=f"{ROOT}/experiments/unet_refine_v1")
    ap.add_argument("--patch", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch-size", type=int, default=24)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--gt-color", default="union", choices=["union", "white", "blue"])
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ck = Path(args.exp)/"checkpoints"; ck.mkdir(parents=True, exist_ok=True)

    rows = list(csv.DictReader(open(args.index)))
    tr = [r for r in rows if r["split"] == "train"]; va = [r for r in rows if r["split"] == "val"]
    tl = DataLoader(BoundaryPatchDS(tr, args.coarse, args.patch, True, args.gt_color), batch_size=args.batch_size,
                    shuffle=True, num_workers=8, pin_memory=True, drop_last=True)
    vl = DataLoader(BoundaryPatchDS(va, args.coarse, args.patch, False, args.gt_color), batch_size=args.batch_size,
                    shuffle=False, num_workers=4, pin_memory=True)
    print(f"Train patches/epoch {len(tl.dataset)}  Val {len(vl.dataset)}", flush=True)

    model = smp.Unet("efficientnet-b2", encoder_weights="imagenet", in_channels=4, classes=1).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=1e-6)
    loss_fn = BceDice(); best = 0.0
    for ep in range(args.epochs):
        model.train()
        for x, y in tl:
            x, y = x.to(device), y.to(device)
            opt.zero_grad(); l = loss_fn(model(x), y); l.backward(); opt.step()
        sch.step()
        model.eval(); vb = 0.0; n = 0
        with torch.no_grad():
            for x, y in vl:
                x, y = x.to(device), y.to(device); vb += biou(model(x), y); n += 1
        vb /= max(1, n)
        torch.save({"model": model.state_dict(), "epoch": ep, "best": best}, ck/"last.pt")
        if vb > best:
            best = vb; torch.save({"model": model.state_dict(), "epoch": ep, "best": best}, ck/"best.pt")
            print(f"Ep{ep:03d} val_biou={vb:.4f}  ↑ New best", flush=True)
        else:
            print(f"Ep{ep:03d} val_biou={vb:.4f}", flush=True)
    print(f"Best checkpoint: {ck/'best.pt'}", flush=True)


if __name__ == "__main__":
    main()
