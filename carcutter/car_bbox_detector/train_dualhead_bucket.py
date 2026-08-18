#!/usr/bin/env python3
"""
DUAL-HEAD bucketed BiRefNet: one bucketed Swin-L pass -> 2 channels (ch0 outline, ch1 tint).
Collapses the 3 original Swin-L passes (1024 outline + 1536 holes + 1024 tint) into ONE:
  - ch0 outline -> outline + punchout-holes-via-fill (proven: bucketed silhouette beats the 1536 pass)
  - ch1 tint    -> window see-through
Warm-starts from prod_bucket5 (outline weights -> ch0, ch1 init = ch0). Bucketed input (5-bucket
route/letterbox), 2-ch GT (outline=union, tint=blue). Eval by view: outline BF@1 + tint recall/IoU,
with square car_holes_v2 as the tint reference.

NEEDS env BIREFNET_DUAL_HEAD=1. Run from repo root:
  PYTHONPATH=.:carcutter/car_bbox_detector/birefnet/BiRefNet BIREFNET_DUAL_HEAD=1 HF_HOME=... \
  /home/rutger/miniconda3/envs/cc_sam3/bin/python carcutter/car_bbox_detector/train_dualhead_bucket.py \
      --n-train 0 --oversample-trunk 3 --epochs 8
"""
import argparse, os, random, sys
from pathlib import Path
from collections import Counter, defaultdict
import numpy as np, cv2, torch
from PIL import Image
sys.path.insert(0, "carcutter/car_bbox_detector/birefnet/BiRefNet")
from torch.utils.data import Dataset, DataLoader
from carcutter.car_bbox_detector.seg_eval import boundary_f, mask_iou, MEAN, STD
from carcutter.car_bbox_detector.birefnet_eval import load_birefnet, birefnet_outline
from carcutter.car_bbox_detector.probe_birefnet_aspect import (
    BUCKETS, route, bkt, crop_pad_box, letterbox, build_manifest, BucketSampler, iou_loss)

ROOT = "carcutter/car_bbox_detector"; EXP = f"{ROOT}/experiments"
PROD5 = f"{EXP}/birefnet_aspect_prod_bucket5.pt"
TINT_SQ = f"{ROOT}/birefnet/BiRefNet/ckpts/car_holes_v2/epoch_16.pth"  # square tint reference
BLUE = (0, 0, 255); MINPX = 60
SIZES = [("small", 0, 0.005), ("med", 0.005, 0.02), ("large", 0.02, 1.0)]


class DualDS(Dataset):
    def __init__(self, rows, aug=True): self.rows = rows; self.aug = aug
    def __len__(self): return len(self.rows)
    def __getitem__(self, i):
        r = self.rows[i]; W, H = BUCKETS[bkt(r)]
        img = np.array(Image.open(r["image"]).convert("RGB"))
        rgb = np.array(Image.open(r["mask"]).convert("RGB"))
        outline = (rgb.sum(2) > 0).astype(np.uint8); tint = (rgb == BLUE).all(2).astype(np.uint8)
        Hh, Ww = img.shape[:2]
        cx0, cy0, cx1, cy1 = crop_pad_box((int(r["x0"]), int(r["y0"]), int(r["x1"]), int(r["y1"])), Hh, Ww)
        ci, co, ct = img[cy0:cy1, cx0:cx1], outline[cy0:cy1, cx0:cx1], tint[cy0:cy1, cx0:cx1]
        li, _ = letterbox(ci, W, H, cv2.INTER_LINEAR)
        lo, _ = letterbox(co, W, H, cv2.INTER_NEAREST); lt, _ = letterbox(ct, W, H, cv2.INTER_NEAREST)
        if self.aug and random.random() < 0.5:
            li = li[:, ::-1].copy(); lo = lo[:, ::-1].copy(); lt = lt[:, ::-1].copy()
        t = torch.from_numpy(((li / 255.0 - MEAN) / STD).transpose(2, 0, 1)).float()
        return t, torch.from_numpy((lo > 0).astype(np.float32))[None], torch.from_numpy((lt > 0).astype(np.float32))[None]


def dice_bce(logit, tgt):
    bce = torch.nn.functional.binary_cross_entropy_with_logits(logit, tgt)
    return bce + iou_loss(logit, tgt)


@torch.no_grad()
def dual_infer(model, img, box, bucket, dev):
    """Bucketed forward -> (outline_prob_full, tint_prob_full) at native res."""
    W, H = BUCKETS[bucket]; Hh, Ww = img.shape[:2]
    cx0, cy0, cx1, cy1 = crop_pad_box(box, Hh, Ww); ci = img[cy0:cy1, cx0:cx1]; ch, cw = ci.shape[:2]
    li, (ox, oy, nw, nh) = letterbox(ci, W, H, cv2.INTER_LINEAR)
    t = torch.from_numpy(((li / 255.0 - MEAN) / STD).transpose(2, 0, 1)[None]).float().to(dev)
    with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
        out = model(t)[-1].sigmoid()[0].float().cpu().numpy()  # (2,H,W)
    res = []
    for c in range(2):
        pr = out[c][oy:oy + nh, ox:ox + nw]; pr = cv2.resize(pr, (cw, ch))
        full = np.zeros((Hh, Ww), np.float32); full[cy0:cy1, cx0:cx1] = pr; res.append(full > 0.5)
    return res[0], res[1]


def comps(mask, ca, lo, hi):
    n, lbl, st, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    return [(lbl == i, st[i, cv2.CC_STAT_AREA]) for i in range(1, n)
            if MINPX <= st[i, cv2.CC_STAT_AREA] and lo <= st[i, cv2.CC_STAT_AREA] / ca < hi]


def build_model(dev):
    from models.birefnet import BiRefNet
    from utils import check_state_dict
    assert os.environ.get("BIREFNET_DUAL_HEAD") == "1", "set BIREFNET_DUAL_HEAD=1"
    m = BiRefNet(bb_pretrained=False).to(dev)
    raw = torch.load(PROD5, map_location="cpu", weights_only=False)
    sd = check_state_dict(raw["model"] if "model" in raw else raw)
    # pop the prod5 1-ch outline head (shape-incompatible with the 2-ch dual head), load the rest
    wk = [k for k in sd if k.endswith("conv_out1.0.weight")]; bk = [k for k in sd if k.endswith("conv_out1.0.bias")]
    w = sd.pop(wk[0]) if wk else None; b = sd.pop(bk[0]) if bk else None
    miss = m.load_state_dict(sd, strict=False)
    if w is not None:   # warm-start ch0(outline)+ch1(tint) from prod5's outline head
        with torch.no_grad():
            m.decoder.conv_out1[0].weight[0:1].copy_(w); m.decoder.conv_out1[0].weight[1:2].copy_(w)
            m.decoder.conv_out1[0].bias[0:1].copy_(b); m.decoder.conv_out1[0].bias[1:2].copy_(b)
        print(f"warm-started conv_out1 ch0/ch1 from prod5 outline head; missing={len(miss.missing_keys)} unexpected={len(miss.unexpected_keys)}", flush=True)
    return m.eval()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", default=f"{ROOT}/data/index.csv"); ap.add_argument("--cache", default=f"{ROOT}/data/aspect_manifest.csv")
    ap.add_argument("--n-train", type=int, default=0); ap.add_argument("--oversample-trunk", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=8); ap.add_argument("--bs", type=int, default=2); ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--tag", default="dualhead_bucket"); ap.add_argument("--n-test", type=int, default=0)
    args = ap.parse_args(); dev = "cuda"
    rows = build_manifest(args.index, {"train", "val", "test"}, args.cache)
    tr = [r for r in rows if r["split"] == "train"]; rng = random.Random(0); rng.shuffle(tr)
    if args.n_train: tr = tr[:args.n_train]
    if args.oversample_trunk > 1:
        trunk = [r for r in tr if "trunk" in os.path.basename(r["image"]).lower()]
        tr = tr + trunk * (args.oversample_trunk - 1); rng.shuffle(tr)
    te = [r for r in rows if r["split"] == "test"]
    if args.n_test: te = te[:args.n_test]
    print(f"[{args.tag}] train {len(tr)} (buckets {Counter(bkt(r) for r in tr)}) | test {len(te)}", flush=True)

    model = build_model(dev)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=1e-4)
    loader = DataLoader(DualDS(tr), sampler=BucketSampler(tr, args.bs), batch_size=args.bs, num_workers=8, drop_last=True)
    ckpt = f"{EXP}/birefnet_{args.tag}.pt"
    for ep in range(args.epochs):
        tot = no = nt = 0.0
        for x, yo, yt in loader:
            x, yo, yt = x.to(dev), yo.to(dev), yt.to(dev)
            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                out = model(x)[-1]; lo = dice_bce(out[:, 0:1], yo); lt = dice_bce(out[:, 1:2], yt); loss = lo + lt
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item(); no += lo.item(); nt += lt.item()
        n = max(1, len(loader)); print(f"  ep{ep} loss {tot/n:.4f} (outline {no/n:.4f} tint {nt/n:.4f})", flush=True)
        torch.save({"model": model.state_dict(), "epoch": ep}, ckpt)

    # eval by view: outline BF@1 (vs prod5 sanity) + tint recall/IoU (vs square car_holes_v2 ref).
    # build the 1-ch reference with dual_head OFF (the env flag is global), then restore.
    os.environ["BIREFNET_DUAL_HEAD"] = "0"
    tintsq = load_birefnet(TINT_SQ, dev)
    os.environ["BIREFNET_DUAL_HEAD"] = "1"
    obf = defaultdict(list); tiou = {"dual": defaultdict(list), "sq": defaultdict(list)}
    trec = {"dual": defaultdict(lambda: [0, 0]), "sq": defaultdict(lambda: [0, 0])}
    for r in te:
        img = np.array(Image.open(r["image"]).convert("RGB"))
        rgb = np.array(Image.open(r["mask"]).convert("RGB")); gt_o = rgb.sum(2) > 0; gt_t = (rgb == BLUE).all(2)
        if gt_o.sum() < 20: continue
        box = (int(r["x0"]), int(r["y0"]), int(r["x1"]), int(r["y1"])); v = r["view"]; ca = float(gt_o.sum())
        o, t = dual_infer(model, img, box, bkt(r), dev)
        obf[v].append(boundary_f(gt_o, o, 1)); obf["ALL"].append(boundary_f(gt_o, o, 1))
        tsq = birefnet_outline(tintsq, img, box, dev)
        if gt_t.sum() >= MINPX:
            tiou["dual"][v].append(mask_iou(gt_t, t)); tiou["sq"][v].append(mask_iou(gt_t, tsq))
            tiou["dual"]["ALL"].append(mask_iou(gt_t, t)); tiou["sq"]["ALL"].append(mask_iou(gt_t, tsq))
            for name, lo, hi in SIZES:
                for comp, area in comps(gt_t, ca, lo, hi):
                    for k, pred in (("dual", t), ("sq", tsq)):
                        trec[k][(v, name)][1] += 1; trec[k][(v, name)][0] += int((comp & pred).sum() >= 0.3 * area)
    print("\n=== OUTLINE BF@1 by view (dual ch0; prod5 was side .828 ALL .824) ===")
    for v in ["straight", "corner", "corner34", "side", "other", "ALL"]:
        if obf[v]: print(f"  {v:<9}{len(obf[v]):>5}{np.mean(obf[v]):>9.3f}")
    print("\n=== TINT/window IoU by view: dual ch1 vs square car_holes_v2 ===")
    for v in ["straight", "corner", "corner34", "side", "other", "ALL"]:
        if tiou["dual"][v]:
            print(f"  {v:<9}{len(tiou['dual'][v]):>5}  dual {np.mean(tiou['dual'][v]):.3f}  sq {np.mean(tiou['sq'][v]):.3f}  d {np.mean(tiou['dual'][v])-np.mean(tiou['sq'][v]):+.3f}")
    print("\n=== TINT/window instance recall by size (dual vs sq) ===")
    for name, _, _ in SIZES:
        d = [trec["dual"][(v, name)] for v in ["straight","corner","corner34","side","other"]]
        s = [trec["sq"][(v, name)] for v in ["straight","corner","corner34","side","other"]]
        dn, dd = sum(x[0] for x in d), sum(x[1] for x in d); sn, sd_ = sum(x[0] for x in s), sum(x[1] for x in s)
        print(f"  {name:<7} dual {100*dn/max(1,dd):.0f}% sq {100*sn/max(1,sd_):.0f}%  (n={dd})")


if __name__ == "__main__":
    main()
