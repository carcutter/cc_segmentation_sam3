#!/usr/bin/env python3
"""
Dedicated small-hole UNet: segments topological small holes in upsampled hole-prone tiles
(roof/wheel/bumper regions). smp EffNet-B2, 512, BCE+Dice. Saves best by val hole-IoU.

Run from repo root:
  PYTHONPATH=. python carcutter/car_bbox_detector/train_smallhole.py --epochs 30
"""
import argparse, random, os
from pathlib import Path
import numpy as np, cv2, torch
import segmentation_models_pytorch as smp
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torchvision.transforms.functional as TF
MEAN = np.array([0.485, 0.456, 0.406], np.float32); STD = np.array([0.229, 0.224, 0.225], np.float32)
DATA = os.environ.get("SMALLHOLE_DATA", "carcutter/car_bbox_detector/smallhole_data")
EXP = os.environ.get("SMALLHOLE_EXP", "carcutter/car_bbox_detector/experiments/unet_smallhole_v1")


class TileDS(Dataset):
    def __init__(self, split, sz=512, aug=False):
        self.items = sorted((Path(DATA) / split / "images").glob("*.png"))
        self.split, self.sz, self.aug = split, sz, aug

    def __len__(self): return len(self.items)

    def __getitem__(self, i):
        p = self.items[i]
        im = cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB)
        lb = cv2.imread(str(Path(DATA) / self.split / "labels" / p.name), 0)
        if im.shape[0] != self.sz:
            im = cv2.resize(im, (self.sz, self.sz)); lb = cv2.resize(lb, (self.sz, self.sz), interpolation=cv2.INTER_NEAREST)
        t = torch.from_numpy(((im / 255.0 - MEAN) / STD).transpose(2, 0, 1)).float()
        m = torch.from_numpy((lb > 127).astype(np.float32))[None]
        if self.aug:
            if random.random() < 0.5: t, m = TF.hflip(t), TF.hflip(m)
            if random.random() < 0.7:
                t = TF.adjust_brightness(t, 1 + random.uniform(-0.25, 0.25))
                t = TF.adjust_contrast(t, 1 + random.uniform(-0.2, 0.2))
        return t, m


def dice_bce(logit, tgt):
    bce = torch.nn.functional.binary_cross_entropy_with_logits(logit, tgt)
    p = torch.sigmoid(logit); inter = (p * tgt).sum((1, 2, 3)); u = p.sum((1, 2, 3)) + tgt.sum((1, 2, 3))
    dice = (1 - (2 * inter + 1) / (u + 1)).mean()
    return bce + dice


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--epochs", type=int, default=30); ap.add_argument("--bs", type=int, default=12)
    args = ap.parse_args(); dev = "cuda"; Path(EXP).mkdir(parents=True, exist_ok=True)
    trds = TileDS("train", aug=True)
    pos = np.array([cv2.imread(str(Path(DATA) / "train" / "labels" / p.name), 0).max() > 0 for p in trds.items])
    hard_f = Path(DATA) / "hard_negs.txt"
    hard_set = set(hard_f.read_text().split()) if hard_f.exists() else set()
    is_hard = np.array([p.stem in hard_set for p in trds.items])   # mined confusions (incl positives that also false-fire)
    npos, nhard = int(pos.sum()), int(is_hard.sum())
    # 45% positives; mined-hard tiles get a 3x boost; focal loss handles within-tile hard pixels
    fpos = pos.mean()
    w = np.where(pos, 0.45 / max(fpos, 1e-3), 0.55 / max(1 - fpos, 1e-3))
    w[is_hard] *= 3.0
    sampler = WeightedRandomSampler(torch.from_numpy(w).double(), len(w), replacement=True)
    tr = DataLoader(trds, batch_size=args.bs, sampler=sampler, num_workers=10, drop_last=True)
    va = DataLoader(TileDS("val"), batch_size=args.bs, num_workers=10)
    print(f"train tiles {len(trds)} (pos {npos}, mined-hard {nhard} x3) | val {len(va.dataset)} | focal+dice", flush=True)
    model = smp.Unet("efficientnet-b2", encoder_weights="imagenet", in_channels=3, classes=1).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    best = 0.0
    for ep in range(args.epochs):
        model.train()
        for x, y in tr:
            x, y = x.to(dev), y.to(dev); opt.zero_grad(); loss = dice_bce(model(x), y); loss.backward(); opt.step()
        sched.step()
        model.eval(); inter = uni = rec_n = rec_d = 0.0
        with torch.no_grad():
            for x, y in va:
                p = (torch.sigmoid(model(x.to(dev))) > 0.5).float().cpu()
                inter += (p * y).sum().item(); uni += ((p + y) > 0).float().sum().item()
                rec_n += (p * y).sum().item(); rec_d += y.sum().item()
        iou = inter / (uni + 1e-6); rec = rec_n / (rec_d + 1e-6)
        print(f"Ep{ep:02d} val_holeIoU={iou:.4f} val_recall={rec:.4f}" + ("  *" if iou > best else ""), flush=True)
        if iou > best:
            best = iou; torch.save({"model": model.state_dict()}, f"{EXP}/best.pt")
    print(f"Best val_holeIoU={best:.4f} -> {EXP}/best.pt")


if __name__ == "__main__":
    main()
