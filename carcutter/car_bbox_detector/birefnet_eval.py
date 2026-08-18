#!/usr/bin/env python3
"""
Eval a fine-tuned BiRefNet on the SAME test set / framing as our end-to-end run,
and compare outline quality: BiRefNet vs our UNet vs prod, all vs GT union.

Reuses the cached end-to-end artifacts (data/e2e/masks/<stem>.npz): the detector
car box (carbox) and our UNet outline — so this is the true production framing
(detector crop) with no detector re-run. BiRefNet is run on the same car-box crop,
letterboxed to square -> 1024, exactly mirroring how it was trained.

Run from repo root:
  PYTHONPATH=.:carcutter/car_bbox_detector/birefnet/BiRefNet \
  /home/rutger/miniconda3/envs/cc_sam3/bin/python carcutter/car_bbox_detector/birefnet_eval.py \
      --ckpt carcutter/car_bbox_detector/birefnet/BiRefNet/ckpts/car_outline_v1/epoch_1.pth
"""
import argparse, os, sys
from pathlib import Path
import numpy as np, cv2, torch
from PIL import Image

sys.path.insert(0, "carcutter/car_bbox_detector/birefnet/BiRefNet")
from carcutter.car_bbox_detector.seg_eval import boundary_f, mask_iou, MEAN, STD

PAD = 0.15   # deploy crop pad — matches build_birefnet_retrain (covers detector clip)


def load_birefnet(ckpt, dev):
    from models.birefnet import BiRefNet
    from utils import check_state_dict
    m = BiRefNet(bb_pretrained=False).to(dev).eval()
    sd = torch.load(ckpt, map_location="cpu", weights_only=True)
    m.load_state_dict(check_state_dict(sd))
    return m


@torch.no_grad()
def birefnet_outline(model, img, box, dev, size=1024):
    H, W = img.shape[:2]
    x0, y0, x1, y1 = [float(v) for v in box]
    bw, bh = x1 - x0, y1 - y0
    px, py = bw * PAD, bh * PAD
    x0, y0 = max(0, int(x0 - px)), max(0, int(y0 - py))
    x1, y1 = min(W, int(x1 + px)), min(H, int(y1 + py))
    crop = img[y0:y1, x0:x1]
    ch, cw = crop.shape[:2]
    if ch < 2 or cw < 2:
        return np.zeros((H, W), bool)
    side = max(ch, cw); xo, yo = (side - cw) // 2, (side - ch) // 2
    sq = np.zeros((side, side, 3), np.uint8); sq[yo:yo + ch, xo:xo + cw] = crop
    inp = cv2.resize(sq, (size, size))
    t = torch.from_numpy(((inp / 255.0 - MEAN) / STD).transpose(2, 0, 1)[None]).float().to(dev)
    with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
        pr = model(t)[-1].sigmoid()[0, 0].float().cpu().numpy()
    pr = cv2.resize(pr, (side, side))[yo:yo + ch, xo:xo + cw]
    full = np.zeros((H, W), np.float32); full[y0:y1, x0:x1] = pr
    return full > 0.5


def binar(p):
    if not p or not os.path.exists(p):
        return None
    a = np.array(Image.open(p)); return (a.sum(2) > 0) if a.ndim == 3 else (a > 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--e2e", default="carcutter/car_bbox_detector/data/e2e")
    ap.add_argument("--index", default="carcutter/car_bbox_detector/data/index.csv")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--target", default="outline", choices=["outline", "holes"])
    args = ap.parse_args()
    BLUE = (0, 0, 255)
    # prod folder + cached-ours key + GT extractor per target
    PROD_KIND = {"outline": "outline", "holes": "holes_tint"}[args.target]
    OURS_KEY = {"outline": "outline", "holes": "holes"}[args.target]
    dev = "cuda"
    import csv
    idx = {Path(r["image"]).stem: r for r in csv.DictReader(open(args.index))}
    model = load_birefnet(args.ckpt, dev)

    agg = {k: {"iou": [], "bf1": [], "bf2": [], "bf3": []} for k in ("birefnet", "unet", "prod")}
    ant_cover = {k: [] for k in ("birefnet", "unet", "prod")}   # recall of GT antenna inside each outline
    WHITE = (255, 255, 255)
    npzs = sorted((Path(args.e2e) / "masks").glob("*.npz"))[:args.n]
    for i, npz in enumerate(npzs):
        r = idx.get(npz.stem)
        if r is None:
            continue
        d = np.load(npz)
        if "carbox" not in d:
            continue
        img = np.array(Image.open(r["image"]).convert("RGB"))
        rgb = np.array(Image.open(r["mask"]).convert("RGB"))
        gt = (rgb.sum(2) > 0) if args.target == "outline" else (rgb == BLUE).all(2)
        white = (rgb == WHITE).all(2)
        if gt.sum() < 20:
            continue
        masks = {"birefnet": birefnet_outline(model, img, d["carbox"], dev),
                 "unet": d[OURS_KEY].astype(bool)}
        po = r["prod_outline"]
        if po and os.path.exists(po):
            if args.target == "outline":
                ol = binar(po); hp = binar(po.replace("/outline/", "/holes_punchout/"))
                masks["prod"] = (ol & ~hp) if (ol is not None and hp is not None) else ol
            else:
                masks["prod"] = binar(po.replace("/outline/", f"/{PROD_KIND}/"))
        for k, m in masks.items():
            if m is None:
                continue
            agg[k]["iou"].append(mask_iou(gt, m))
            for t in (1, 2, 3):
                agg[k][f"bf{t}"].append(boundary_f(gt, m, t))
            if args.target == "outline" and white.sum() >= 20:
                ant_cover[k].append((white & m).sum() / white.sum())
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(npzs)}", flush=True)

    m = lambda a: float(np.mean(a)) if a else float("nan")
    gtname = {"outline": "GT union", "holes": "GT blue (holes_tint prod)"}[args.target]
    print(f"\n{args.target.upper()} comparison vs {gtname} (ckpt={Path(args.ckpt).name}, n={len(agg['birefnet']['iou'])}):\n")
    print(f"  {'method':<10}{'IoU':>9}{'BF@1':>9}{'BF@2':>9}{'BF@3':>9}")
    for k in ("prod", "unet", "birefnet"):
        a = agg[k]
        if a["iou"]:
            print(f"  {k:<10}" + "".join(f"{m(a[x]):>9.3f}" for x in ("iou", "bf1", "bf2", "bf3")))
    print(f"\nANTENNA coverage (recall of GT antenna inside the outline, n={len(ant_cover['unet'])}):")
    for k in ("prod", "unet", "birefnet"):
        if ant_cover[k]:
            print(f"  {k:<10}{m(ant_cover[k]):>8.3f}")


if __name__ == "__main__":
    main()
