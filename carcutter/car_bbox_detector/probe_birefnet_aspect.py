#!/usr/bin/env python3
"""
PROBE: does landscape aspect-ratio bucketing improve BiRefNet outline (esp. side-view boundary)?
Decoder-only fine-tune of car_outline_v2 on 2 aspect buckets (on-the-fly crop+letterbox routed by
GT-box aspect), homogeneous-but-alternating bucket batches. Then eval BF@1 by view vs the SQUARE
baseline (same weights' parent) on the test set — apples-to-apples.

Buckets (W,H, /32): A=(1152,864)~1.33 for boxaspect<1.57 ; B=(1408,736)~1.91 for >=1.57.
Both ~1.0M px ~= square 1024^2, so this isolates ASPECT at ~equal compute.

Run from repo root:
  PYTHONPATH=.:carcutter/car_bbox_detector/birefnet/BiRefNet HF_HOME=... \
  /home/rutger/miniconda3/envs/cc_sam3/bin/python carcutter/car_bbox_detector/probe_birefnet_aspect.py \
      --n-train 4000 --epochs 4
"""
import argparse, csv, os, random, sys
from pathlib import Path
import numpy as np, cv2, torch
from PIL import Image
sys.path.insert(0, "carcutter/car_bbox_detector/birefnet/BiRefNet")
from torch.utils.data import Dataset, DataLoader, Sampler
from carcutter.car_bbox_detector.seg_eval import boundary_f, mask_iou, MEAN, STD
from carcutter.car_bbox_detector.birefnet_eval import load_birefnet, birefnet_outline

ROOT = "carcutter/car_bbox_detector"
CKPT = f"{ROOT}/birefnet/BiRefNet/ckpts/car_outline_v2/epoch_16.pth"
import math
# 5 aspect buckets incl explicit PORTRAIT (P). (W,H) all /32, ~1.0M px each. Route = nearest center in log-aspect.
BUCKETS = {"P": (960, 1056), "A": (1088, 896), "B": (1248, 800), "C": (1376, 704), "D": (1600, 640)}  # (W,H)
CENTERS = {"P": 0.91, "A": 1.21, "B": 1.56, "C": 1.95, "D": 2.50}
PAD = 0.08
def route(aspect):
    return min(CENTERS, key=lambda k: abs(math.log(aspect / CENTERS[k])))
def bkt(r):
    return route(float(r["aspect"]))


def viewgroup(b):
    b = b.lower()
    if "side" in b: return "side"
    if "trunk" in b or "interior" in b or "engine" in b: return "other"
    import re
    if re.search(r"(34|3-4|quarter)", b): return "corner34"
    fr = ("front" in b) or ("rear" in b) or ("back" in b); lr = ("left" in b) or ("right" in b)
    if fr and lr: return "corner"
    if fr: return "straight"
    return "other"


def gt_box(maskpath):
    rgb = np.array(Image.open(maskpath).convert("RGB")); u = rgb.sum(2) > 0
    if u.sum() < 1500: return None
    ys, xs = np.where(u); return xs.min(), ys.min(), xs.max() + 1, ys.max() + 1


def crop_pad_box(box, H, W):
    x0, y0, x1, y1 = box; bw, bh = x1 - x0, y1 - y0
    px, py = int(bw * PAD), int(bh * PAD)
    return max(0, x0 - px), max(0, y0 - py), min(W, x1 + px), min(H, y1 + py)


def letterbox(arr, W, H, interp):
    h, w = arr.shape[:2]; s = min(W / w, H / h)
    nw, nh = max(1, int(round(w * s))), max(1, int(round(h * s)))
    r = cv2.resize(arr, (nw, nh), interpolation=interp)
    ox, oy = (W - nw) // 2, (H - nh) // 2
    if arr.ndim == 3:
        out = np.zeros((H, W, 3), arr.dtype); out[oy:oy + nh, ox:ox + nw] = r
    else:
        out = np.zeros((H, W), arr.dtype); out[oy:oy + nh, ox:ox + nw] = r
    return out, (ox, oy, nw, nh)


def _manifest_worker(r):
    box = gt_box(r["mask"])
    if box is None: return None
    x0, y0, x1, y1 = box; asp = (x1 - x0) / (y1 - y0)
    return {"image": r["image"], "mask": r["mask"], "split": r["split"],
            "view": viewgroup(os.path.basename(r["image"])),
            "x0": x0, "y0": y0, "x1": x1, "y1": y1, "aspect": f"{asp:.3f}",
            "bucket": route(asp)}


def build_manifest(index, splits, cache):
    if os.path.exists(cache):
        return [r for r in csv.DictReader(open(cache))]
    from multiprocessing import Pool
    rows = [r for r in csv.DictReader(open(index)) if r["split"] in splits]
    with Pool(16) as p:
        out = [x for x in p.map(_manifest_worker, rows, chunksize=32) if x is not None]
    w = csv.DictWriter(open(cache, "w"), fieldnames=list(out[0].keys())); w.writeheader(); w.writerows(out)
    return out


class BucketDS(Dataset):
    def __init__(self, rows, square=False, aug=False):
        self.rows = rows; self.square = square; self.aug = aug
    def __len__(self): return len(self.rows)
    def __getitem__(self, i):
        r = self.rows[i]; W, H = (1024, 1024) if self.square else BUCKETS[bkt(r)]
        img = np.array(Image.open(r["image"]).convert("RGB"))
        rgb = np.array(Image.open(r["mask"]).convert("RGB")); gt = (rgb.sum(2) > 0).astype(np.uint8)
        Hh, Ww = img.shape[:2]
        cx0, cy0, cx1, cy1 = crop_pad_box((int(r["x0"]), int(r["y0"]), int(r["x1"]), int(r["y1"])), Hh, Ww)
        ci = img[cy0:cy1, cx0:cx1]; cg = gt[cy0:cy1, cx0:cx1]
        li, _ = letterbox(ci, W, H, cv2.INTER_LINEAR)
        lg, _ = letterbox(cg, W, H, cv2.INTER_NEAREST)
        if self.aug and random.random() < 0.5:
            li = li[:, ::-1].copy(); lg = lg[:, ::-1].copy()
        t = torch.from_numpy(((li / 255.0 - MEAN) / STD).transpose(2, 0, 1)).float()
        m = torch.from_numpy((lg > 0).astype(np.float32))[None]
        return t, m


class BucketSampler(Sampler):
    """Homogeneous-per-bucket batches, shuffled order across buckets."""
    def __init__(self, rows, bs, seed=0):
        from collections import defaultdict
        self.bs = bs; self.seed = seed
        self.byb = defaultdict(list)
        for i, r in enumerate(rows): self.byb[bkt(r)].append(i)
    def __iter__(self):
        rng = random.Random(self.seed); self.seed += 1
        batches = []
        for b, idxs in self.byb.items():
            idxs = idxs[:]; rng.shuffle(idxs)
            for k in range(0, len(idxs) - self.bs + 1, self.bs):
                batches.append(idxs[k:k + self.bs])
        rng.shuffle(batches)
        for batch in batches:
            yield from batch
    def __len__(self): return sum(len(v) // self.bs * self.bs for v in self.byb.values())


def iou_loss(logit, tgt):
    p = torch.sigmoid(logit); inter = (p * tgt).sum((1, 2, 3)); u = (p + tgt).sum((1, 2, 3)) - inter
    return (1 - (inter + 1) / (u + 1)).mean()


@torch.no_grad()
def outline_bucketed(model, img, box, bucket, dev, square=False):
    W, H = (1024, 1024) if square else BUCKETS[bucket]; Hh, Ww = img.shape[:2]
    cx0, cy0, cx1, cy1 = crop_pad_box(box, Hh, Ww)
    ci = img[cy0:cy1, cx0:cx1]; ch, cw = ci.shape[:2]
    li, (ox, oy, nw, nh) = letterbox(ci, W, H, cv2.INTER_LINEAR)
    t = torch.from_numpy(((li / 255.0 - MEAN) / STD).transpose(2, 0, 1)[None]).float().to(dev)
    with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
        pr = model(t)[-1].sigmoid()[0, 0].float().cpu().numpy()
    pr = pr[oy:oy + nh, ox:ox + nw]; pr = cv2.resize(pr, (cw, ch))
    full = np.zeros((Hh, Ww), np.float32); full[cy0:cy1, cx0:cx1] = pr
    return full > 0.5


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", default=f"{ROOT}/data/index.csv")
    ap.add_argument("--n-train", type=int, default=4000, help="0 = full train split")
    ap.add_argument("--oversample-trunk", type=int, default=1, help="replicate open-trunk views Nx (restores car_outline_v2 fix)")
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--bs", type=int, default=2); ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--n-test", type=int, default=0, help="cap test imgs for fast eval (0=all)")
    ap.add_argument("--full-ft", action="store_true", help="also unfreeze encoder")
    ap.add_argument("--square", action="store_true", help="train/eval at fixed 1024^2 square (control)")
    ap.add_argument("--init", default=CKPT, help="init checkpoint")
    ap.add_argument("--tag", default="probe", help="run tag for outputs")
    ap.add_argument("--cache", default=f"{ROOT}/data/aspect_manifest.csv")
    args = ap.parse_args(); dev = "cuda"
    rows = build_manifest(args.index, {"train", "val", "test"}, args.cache)
    tr = [r for r in rows if r["split"] == "train"]
    rng = random.Random(0); rng.shuffle(tr)
    if args.n_train: tr = tr[:args.n_train]
    if args.oversample_trunk > 1:   # restore the car_outline_v2 open-trunk fix
        trunk = [r for r in tr if "trunk" in os.path.basename(r["image"]).lower()]
        tr = tr + trunk * (args.oversample_trunk - 1); rng.shuffle(tr)
        print(f"oversampled {len(trunk)} trunk views x{args.oversample_trunk} -> +{len(trunk)*(args.oversample_trunk-1)} rows", flush=True)
    te = [r for r in rows if r["split"] == "test"]
    if args.n_test: te = te[:args.n_test]
    from collections import Counter
    print(f"[{args.tag}] mode={'SQUARE' if args.square else 'BUCKETED'} init={Path(args.init).parent.name}/{Path(args.init).name}", flush=True)
    print(f"train {len(tr)} (buckets {Counter(bkt(r) for r in tr)}) | test {len(te)} "
          f"(views {Counter(r['view'] for r in te)})", flush=True)

    # robust init: handle both native BiRefNet ckpts and our per-epoch {"model":sd,"epoch":k} resume ckpts
    _raw = torch.load(args.init, map_location="cpu", weights_only=False)
    if isinstance(_raw, dict) and "model" in _raw and "epoch" in _raw:
        from models.birefnet import BiRefNet
        from utils import check_state_dict
        print(f"[resume] init from {Path(args.init).name} (saved epoch {_raw['epoch']})", flush=True)
        model = BiRefNet(bb_pretrained=False).to(dev).eval()
        model.load_state_dict(check_state_dict(_raw["model"]))
    else:
        model = load_birefnet(args.init, dev)
    for p in model.parameters(): p.requires_grad = False
    dec = model.decoder
    for p in dec.parameters(): p.requires_grad = True
    if args.full_ft:
        for p in model.bb.parameters(): p.requires_grad = True
    model.eval()   # frozen BN/no gdt; grads still flow to decoder
    train_params = [p for p in model.parameters() if p.requires_grad]
    print(f"trainable params {sum(p.numel() for p in train_params)/1e6:.1f}M "
          f"({'full-ft' if args.full_ft else 'decoder-only'})", flush=True)
    opt = torch.optim.AdamW(train_params, lr=args.lr, weight_decay=1e-4)
    ds = BucketDS(tr, square=args.square, aug=True); sampler = BucketSampler(tr, args.bs)
    loader = DataLoader(ds, batch_sampler=None, sampler=sampler, batch_size=args.bs, num_workers=8, drop_last=True)
    ckpt_out = f"{ROOT}/experiments/birefnet_aspect_{args.tag}.pt"

    for ep in range(args.epochs):
        tot = 0.0; nb = 0
        for x, y in loader:
            x, y = x.to(dev), y.to(dev)
            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                logit = model(x)[-1]
                loss = torch.nn.functional.binary_cross_entropy_with_logits(logit, y) + iou_loss(logit, y)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item(); nb += 1
        print(f"  ep{ep} loss {tot/max(1,nb):.4f}", flush=True)
        torch.save({"model": model.state_dict(), "epoch": ep}, ckpt_out)  # per-epoch ckpt

    # eval THIS run's model BF@1 by view (square or bucketed inference, matching how it trained),
    # alongside the shipped car_outline_v2 (square 1024) as a fixed reference. Save per-view CSV so
    # the two queued runs (square-mine vs bucketed-mine) can be compared cleanly.
    base = load_birefnet(CKPT, dev)
    from collections import defaultdict
    agg = {"this": defaultdict(list), "prod": defaultdict(list)}
    for r in te:
        img = np.array(Image.open(r["image"]).convert("RGB"))
        rgb = np.array(Image.open(r["mask"]).convert("RGB")); gt = rgb.sum(2) > 0
        if gt.sum() < 20: continue
        box = (int(r["x0"]), int(r["y0"]), int(r["x1"]), int(r["y1"]))
        mt = outline_bucketed(model, img, box, bkt(r), dev, square=args.square)
        mp = birefnet_outline(base, img, box, dev, size=1024)
        agg["this"][r["view"]].append(boundary_f(gt, mt, 1)); agg["this"]["ALL"].append(boundary_f(gt, mt, 1))
        agg["prod"][r["view"]].append(boundary_f(gt, mp, 1)); agg["prod"]["ALL"].append(boundary_f(gt, mp, 1))

    print(f"\n=== [{args.tag}] outline BF@1 by view: car_outline_v2(prod sq) vs THIS({'square' if args.square else 'bucketed'}) ===")
    print(f"  {'view':<9}{'n':>5}{'prod_sq':>9}{'this':>9}{'delta':>8}")
    out = open(f"{ROOT}/experiments/aspect_byview_{args.tag}.csv", "w"); out.write("view,n,prod_sq,this\n")
    for v in ["straight", "corner", "corner34", "side", "other", "ALL"]:
        s, b = agg["prod"][v], agg["this"][v]
        if not s: continue
        mp_, mt_ = np.mean(s), np.mean(b)
        print(f"  {v:<9}{len(s):>5}{mp_:>9.3f}{mt_:>9.3f}{mt_-mp_:>+8.3f}")
        out.write(f"{v},{len(s)},{mp_:.4f},{mt_:.4f}\n")
    out.close()


if __name__ == "__main__":
    main()
