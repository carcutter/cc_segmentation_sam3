#!/usr/bin/env python3
"""
Antenna discriminator: small binary classifier (EfficientNet-B0) on antenna-box crops,
real-antenna (pos) vs stick/branch/pole (neg). Balanced sampling, BCE. Saves best by val AUC.

Run from repo root:
  PYTHONPATH=. python carcutter/car_bbox_detector/train_antenna_clf.py --epochs 20
"""
import argparse
from pathlib import Path
import numpy as np, cv2, torch, timm
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torchvision.transforms.functional as TF
import random
MEAN = np.array([0.485, 0.456, 0.406], np.float32); STD = np.array([0.229, 0.224, 0.225], np.float32)
DATA = "carcutter/car_bbox_detector/antenna_clf_data"
EXP = "carcutter/car_bbox_detector/experiments/antenna_clf_v1"


class CropDS(Dataset):
    def __init__(self, split, sz=224, aug=False):
        self.items = []
        for lab, c in (("pos", 1), ("neg", 0)):
            for p in (Path(DATA) / split / lab).glob("*.png"):
                self.items.append((str(p), c))
        self.sz = sz; self.aug = aug
        self.labels = [c for _, c in self.items]

    def __len__(self): return len(self.items)

    def __getitem__(self, i):
        p, c = self.items[i]
        im = cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB)
        im = cv2.resize(im, (self.sz, self.sz))
        t = torch.from_numpy(((im / 255.0 - MEAN) / STD).transpose(2, 0, 1)).float()
        if self.aug:
            if random.random() < 0.5: t = TF.hflip(t)
            if random.random() < 0.7:
                t = TF.adjust_brightness(t, 1 + random.uniform(-0.25, 0.25))
                t = TF.adjust_contrast(t, 1 + random.uniform(-0.2, 0.2))
        return t, torch.tensor([c], dtype=torch.float32)


def auc(y, s):
    y = np.asarray(y); s = np.asarray(s)
    pos, neg = s[y == 1], s[y == 0]
    if len(pos) == 0 or len(neg) == 0: return 0.5
    return float((pos[:, None] > neg[None, :]).mean())


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--bs", type=int, default=64); args = ap.parse_args()
    dev = "cuda"; Path(EXP).mkdir(parents=True, exist_ok=True)
    tr = CropDS("train", aug=True); va = CropDS("val")
    print(f"train {len(tr)} (pos {sum(tr.labels)}/neg {len(tr.labels)-sum(tr.labels)}) | val {len(va)}")
    w = np.array([1.0 / max(1, tr.labels.count(c)) for c in tr.labels])
    sampler = WeightedRandomSampler(torch.from_numpy(w).double(), len(w), replacement=True)
    tl = DataLoader(tr, batch_size=args.bs, sampler=sampler, num_workers=8, drop_last=True)
    vl = DataLoader(va, batch_size=args.bs, num_workers=8)
    model = timm.create_model("efficientnet_b0", pretrained=True, num_classes=1).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    lossf = torch.nn.BCEWithLogitsLoss()
    best = 0.0
    for ep in range(args.epochs):
        model.train()
        for x, y in tl:
            x, y = x.to(dev), y.to(dev); opt.zero_grad()
            loss = lossf(model(x), y); loss.backward(); opt.step()
        model.eval(); ys, ss = [], []
        with torch.no_grad():
            for x, y in vl:
                s = torch.sigmoid(model(x.to(dev))).cpu().numpy().ravel()
                ss += s.tolist(); ys += y.numpy().ravel().tolist()
        a = auc(ys, ss); acc = float(((np.array(ss) > 0.5) == np.array(ys)).mean())
        print(f"Ep{ep:02d} val_auc={a:.4f} val_acc={acc:.4f}" + ("  *" if a > best else ""), flush=True)
        if a > best:
            best = a; torch.save({"model": model.state_dict()}, f"{EXP}/best.pt")
    print(f"Best val_auc={best:.4f} -> {EXP}/best.pt")


if __name__ == "__main__":
    main()
