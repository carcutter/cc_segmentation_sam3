#!/usr/bin/env python3
"""
3-HEAD bucketed BiRefNet: one bucketed Swin-L pass -> ch0 outline(+holes-via-fill), ch1 tint/windows,
ch2 antenna. Collapses ALL exterior masks into ONE pass (+ the small UNet 2-stage crops for residual
hole/window tails). Warm-start from prod_bucket5 (outline->ch0; ch1,ch2 init from ch0). Bucketed input,
3-ch GT (union / blue / white). Eval by view: outline BF@1, tint IoU (vs square car_holes_v2), antenna
IoU + coverage. Motivated by: bucketing lifted antenna coverage .67->.78 (side .42->.68).

NEEDS env BIREFNET_NHEADS=3. Run from repo root:
  PYTHONPATH=.:.../BiRefNet BIREFNET_NHEADS=3 HF_HOME=... PY train_trihead_bucket.py --n-train 0 --oversample-trunk 3 --epochs 8
"""
import argparse, os, random, sys
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
TINT_SQ = f"{ROOT}/birefnet/BiRefNet/ckpts/car_holes_v2/epoch_16.pth"
BLUE = (0, 0, 255); WHITE = (255, 255, 255); MINPX = 60
SIZES = [("small", 0, 0.005), ("med", 0.005, 0.02), ("large", 0.02, 1.0)]
# head index -> (name, GT extractor from rgb mask)
HEADS = [("outline", lambda rgb: rgb.sum(2) > 0),
         ("tint",    lambda rgb: (rgb == BLUE).all(2)),
         ("antenna", lambda rgb: (rgb == WHITE).all(2))]
NH = len(HEADS)


class TriDS(Dataset):
    def __init__(self, rows, aug=True): self.rows = rows; self.aug = aug
    def __len__(self): return len(self.rows)
    def __getitem__(self, i):
        r = self.rows[i]; W, H = BUCKETS[bkt(r)]
        img = np.array(Image.open(r["image"]).convert("RGB"))
        rgb = np.array(Image.open(r["mask"]).convert("RGB")); Hh, Ww = img.shape[:2]
        cx0, cy0, cx1, cy1 = crop_pad_box((int(r["x0"]), int(r["y0"]), int(r["x1"]), int(r["y1"])), Hh, Ww)
        ci = img[cy0:cy1, cx0:cx1]
        li, _ = letterbox(ci, W, H, cv2.INTER_LINEAR)
        masks = []
        for _, ex in HEADS:
            m = ex(rgb).astype(np.uint8)[cy0:cy1, cx0:cx1]
            lm, _ = letterbox(m, W, H, cv2.INTER_NEAREST); masks.append(lm)
        if self.aug and random.random() < 0.5:
            li = li[:, ::-1].copy(); masks = [m[:, ::-1].copy() for m in masks]
        t = torch.from_numpy(((li / 255.0 - MEAN) / STD).transpose(2, 0, 1)).float()
        y = torch.from_numpy(np.stack([(m > 0).astype(np.float32) for m in masks]))  # (NH,H,W)
        return t, y


def dice_bce(logit, tgt):
    return torch.nn.functional.binary_cross_entropy_with_logits(logit, tgt) + iou_loss(logit, tgt)


@torch.no_grad()
def tri_infer(model, img, box, bucket, dev):
    W, H = BUCKETS[bucket]; Hh, Ww = img.shape[:2]
    cx0, cy0, cx1, cy1 = crop_pad_box(box, Hh, Ww); ci = img[cy0:cy1, cx0:cx1]; ch, cw = ci.shape[:2]
    li, (ox, oy, nw, nh) = letterbox(ci, W, H, cv2.INTER_LINEAR)
    t = torch.from_numpy(((li / 255.0 - MEAN) / STD).transpose(2, 0, 1)[None]).float().to(dev)
    with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
        out = model(t)[-1].sigmoid()[0].float().cpu().numpy()  # (NH,H,W)
    res = []
    for c in range(NH):
        pr = cv2.resize(out[c][oy:oy + nh, ox:ox + nw], (cw, ch))
        full = np.zeros((Hh, Ww), np.float32); full[cy0:cy1, cx0:cx1] = pr; res.append(full > 0.5)
    return res


def build_model(dev):
    from models.birefnet import BiRefNet
    from utils import check_state_dict
    assert os.environ.get("BIREFNET_NHEADS") == "3", "set BIREFNET_NHEADS=3"
    m = BiRefNet(bb_pretrained=False).to(dev)
    raw = torch.load(PROD5, map_location="cpu", weights_only=False)
    sd = check_state_dict(raw["model"] if "model" in raw else raw)
    wk = [k for k in sd if k.endswith("conv_out1.0.weight")]; bk = [k for k in sd if k.endswith("conv_out1.0.bias")]
    w = sd.pop(wk[0]) if wk else None; b = sd.pop(bk[0]) if bk else None
    miss = m.load_state_dict(sd, strict=False)
    if w is not None:
        with torch.no_grad():
            for c in range(NH):
                m.decoder.conv_out1[0].weight[c:c+1].copy_(w); m.decoder.conv_out1[0].bias[c:c+1].copy_(b)
        print(f"warm-started {NH}-ch conv_out1 from prod5 outline; missing={len(miss.missing_keys)} unexpected={len(miss.unexpected_keys)}", flush=True)
    return m.eval()


def comps(mask, ca, lo, hi):
    n, lbl, st, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    return [(lbl == i, st[i, cv2.CC_STAT_AREA]) for i in range(1, n)
            if MINPX <= st[i, cv2.CC_STAT_AREA] and lo <= st[i, cv2.CC_STAT_AREA] / ca < hi]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", default=f"{ROOT}/data/index.csv"); ap.add_argument("--cache", default=f"{ROOT}/data/aspect_manifest.csv")
    ap.add_argument("--n-train", type=int, default=0); ap.add_argument("--oversample-trunk", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=8); ap.add_argument("--bs", type=int, default=2); ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--n-test", type=int, default=0); ap.add_argument("--tag", default="trihead_bucket")
    ap.add_argument("--eval-only", action="store_true", help="skip training; load experiments/birefnet_<tag>.pt and just eval")
    ap.add_argument("--freeze-encoder", action="store_true", help="freeze Swin-L encoder; train decoder+heads only (protects the proven encoder + outline)")
    args = ap.parse_args(); dev = "cuda"
    rows = build_manifest(args.index, {"train", "val", "test"}, args.cache)
    tr = [r for r in rows if r["split"] == "train"]; rng = random.Random(0); rng.shuffle(tr)
    if args.n_train: tr = tr[:args.n_train]
    if args.oversample_trunk > 1:
        trunk = [r for r in tr if "trunk" in os.path.basename(r["image"]).lower()]
        tr = tr + trunk * (args.oversample_trunk - 1); rng.shuffle(tr)
    te = [r for r in rows if r["split"] == "test"]
    if args.n_test: te = te[:args.n_test]
    print(f"[{args.tag}] heads={[h[0] for h in HEADS]} train {len(tr)} (buckets {Counter(bkt(r) for r in tr)}) | test {len(te)}", flush=True)

    ckpt = f"{EXP}/birefnet_{args.tag}.pt"
    if args.eval_only:
        from models.birefnet import BiRefNet
        from utils import check_state_dict
        raw = torch.load(ckpt, map_location="cpu", weights_only=False)
        model = BiRefNet(bb_pretrained=False).to(dev).eval()
        model.load_state_dict(check_state_dict(raw["model"]))
        print(f"[eval-only] loaded {ckpt} (epoch {raw.get('epoch')})", flush=True)
        return _evaluate(model, te, dev)
    model = build_model(dev)
    if args.freeze_encoder:
        nf = 0
        for p in model.bb.parameters(): p.requires_grad = False; nf += p.numel()
        print(f"froze encoder ({nf/1e6:.0f}M params) — training decoder+heads only", flush=True)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=1e-4)
    loader = DataLoader(TriDS(tr), sampler=BucketSampler(tr, args.bs), batch_size=args.bs, num_workers=8, drop_last=True)
    for ep in range(args.epochs):
        tot = np.zeros(NH); nb = 0
        for x, y in loader:
            x, y = x.to(dev), y.to(dev)
            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                out = model(x)[-1]
                losses = [dice_bce(out[:, k:k+1], y[:, k:k+1]) for k in range(NH)]
                loss = sum(losses)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += np.array([l.item() for l in losses]); nb += 1
        nb = max(1, nb); print(f"  ep{ep} " + " ".join(f"{HEADS[k][0]}={tot[k]/nb:.4f}" for k in range(NH)), flush=True)
        torch.save({"model": model.state_dict(), "epoch": ep}, ckpt)
    _evaluate(model, te, dev)


def _evaluate(model, te, dev):
    os.environ["BIREFNET_NHEADS"] = "1"; tintsq = load_birefnet(TINT_SQ, dev); os.environ["BIREFNET_NHEADS"] = "3"
    obf = defaultdict(list); tiou = {"tri": defaultdict(list), "sq": defaultdict(list)}
    aiou = defaultdict(list); acov = defaultdict(list)
    for r in te:
        img = np.array(Image.open(r["image"]).convert("RGB"))
        rgb = np.array(Image.open(r["mask"]).convert("RGB"))
        gt_o, gt_t, gt_a = rgb.sum(2) > 0, (rgb == BLUE).all(2), (rgb == WHITE).all(2)
        if gt_o.sum() < 20: continue
        box = (int(r["x0"]), int(r["y0"]), int(r["x1"]), int(r["y1"])); v = r["view"]
        o, t, a = tri_infer(model, img, box, bkt(r), dev)
        for vv in (v, "ALL"): obf[vv].append(boundary_f(gt_o, o, 1))
        if gt_t.sum() >= MINPX:
            tsq = birefnet_outline(tintsq, img, box, dev)
            for vv in (v, "ALL"): tiou["tri"][vv].append(mask_iou(gt_t, t)); tiou["sq"][vv].append(mask_iou(gt_t, tsq))
        if gt_a.sum() >= 20:
            for vv in (v, "ALL"):
                aiou[vv].append(mask_iou(gt_a, a)); acov[vv].append((gt_a & a).sum() / gt_a.sum())
    V = ["straight", "corner", "corner34", "side", "other", "ALL"]
    print("\n=== OUTLINE ch0 BF@1 by view (prod5: side .828 ALL .824) ===")
    for v in V:
        if obf[v]: print(f"  {v:<9}{len(obf[v]):>5}{np.mean(obf[v]):>9.3f}")
    print("\n=== TINT ch1 IoU by view: tri vs square car_holes_v2 ===")
    for v in V:
        if tiou["tri"][v]: print(f"  {v:<9}{len(tiou['tri'][v]):>5}  tri {np.mean(tiou['tri'][v]):.3f}  sq {np.mean(tiou['sq'][v]):.3f}  d {np.mean(tiou['tri'][v])-np.mean(tiou['sq'][v]):+.3f}")
    print("\n=== ANTENNA ch2 by view: IoU + coverage(recall) (bucket-outline coverage was .784 ALL) ===")
    for v in V:
        if aiou[v]: print(f"  {v:<9}{len(aiou[v]):>5}  IoU {np.mean(aiou[v]):.3f}  cov {np.mean(acov[v]):.3f}")


if __name__ == "__main__":
    main()
