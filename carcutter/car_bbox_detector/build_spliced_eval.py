#!/usr/bin/env python3
"""
SPLICED single-pass model: shared frozen Swin encoder (run ONCE) -> two branches:
  outline branch = prod_bucket5's squeeze_module + decoder        -> ch0 outline (+holes via fill)
  tint/ant branch = trihead's squeeze_module + decoder (3ch)      -> ch1 tint, ch2 antenna
This dodges the shared-decoder outline degradation (.731) by using prod5's untouched outline decoder
(.824) while keeping the trihead's strong tint/antenna heads. One encoder pass + two cheap decoders.
Eval by view (should reproduce prod5 outline .824 + trihead tint .684 + antenna .568) and save the artifact.

Run: PYTHONPATH=.:.../BiRefNet HF_HOME=... PY build_spliced_eval.py --n 930
"""
import argparse, csv, os, sys
from collections import defaultdict
import numpy as np, cv2, torch
from PIL import Image
sys.path.insert(0, "carcutter/car_bbox_detector/birefnet/BiRefNet")
from carcutter.car_bbox_detector.seg_eval import boundary_f, mask_iou, MEAN, STD
from carcutter.car_bbox_detector.probe_birefnet_aspect import BUCKETS, route, crop_pad_box, letterbox

ROOT = "carcutter/car_bbox_detector"; EXP = f"{ROOT}/experiments"
PROD5 = f"{EXP}/birefnet_aspect_prod_bucket5.pt"; TRI = f"{EXP}/birefnet_trihead_bucket.pt"
BLUE = (0, 0, 255); WHITE = (255, 255, 255)
V = ["straight", "corner", "corner34", "side", "other", "ALL"]


def vg(b):
    import re; b = b.lower()
    if "side" in b: return "side"
    if "trunk" in b or "interior" in b or "engine" in b: return "other"
    if re.search(r"(34|3-4|quarter)", b): return "corner34"
    fr = ("front" in b) or ("rear" in b) or ("back" in b); lr = ("left" in b) or ("right" in b)
    if fr and lr: return "corner"
    if fr: return "straight"
    return "other"


def load_bn(path, nheads, dev):
    os.environ["BIREFNET_NHEADS"] = str(nheads)
    from models.birefnet import BiRefNet
    from utils import check_state_dict
    raw = torch.load(path, map_location="cpu", weights_only=False)
    m = BiRefNet(bb_pretrained=False).to(dev).eval()
    m.load_state_dict(check_state_dict(raw["model"] if "model" in raw else raw))
    return m


class Spliced(torch.nn.Module):
    """Shared encoder (prod5.bb == trihead.bb, frozen) run once; prod5 outline branch + trihead tint/ant branch."""
    def __init__(self, prod5, tri):
        super().__init__()
        self.enc = prod5                 # use prod5 for forward_enc (bb identical to tri)
        self.sq_o = prod5.squeeze_module; self.dec_o = prod5.decoder
        self.sq_t = tri.squeeze_module;   self.dec_t = tri.decoder
    @torch.no_grad()
    def forward(self, x):
        (x1, x2, x3, x4), _ = self.enc.forward_enc(x)        # shared Swin, ONE pass
        o = self.dec_o([x, x1, x2, x3, self.sq_o(x4)])[-1]   # (B,1) outline
        t = self.dec_t([x, x1, x2, x3, self.sq_t(x4)])[-1]   # (B,3) -> ch1 tint, ch2 antenna
        return o[:, 0:1], t[:, 1:2], t[:, 2:3]


@torch.no_grad()
def infer(model, img, box, bucket, dev):
    W, H = BUCKETS[bucket]; Hh, Ww = img.shape[:2]
    cx0, cy0, cx1, cy1 = crop_pad_box(box, Hh, Ww); ci = img[cy0:cy1, cx0:cx1]; ch, cw = ci.shape[:2]
    li, (ox, oy, nw, nh) = letterbox(ci, W, H, cv2.INTER_LINEAR)
    t = torch.from_numpy(((li / 255.0 - MEAN) / STD).transpose(2, 0, 1)[None]).float().to(dev)
    with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
        outs = model(t)
    res = []
    for o in outs:
        pr = o.sigmoid()[0, 0].float().cpu().numpy()[oy:oy + nh, ox:ox + nw]
        pr = cv2.resize(pr, (cw, ch)); full = np.zeros((Hh, Ww), np.float32); full[cy0:cy1, cx0:cx1] = pr
        res.append(full > 0.5)
    return res  # outline, tint, antenna


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=930)
    ap.add_argument("--index", default=f"{ROOT}/data/index.csv"); args = ap.parse_args(); dev = "cuda"
    prod5 = load_bn(PROD5, 1, dev); tri = load_bn(TRI, 3, dev)
    model = Spliced(prod5, tri).eval()
    rows = [r for r in csv.DictReader(open(args.index)) if r["split"] == "test"][:args.n]

    obf = defaultdict(list); tiou = defaultdict(list); aiou = defaultdict(list); acov = defaultdict(list)
    for i, r in enumerate(rows):
        img = np.array(Image.open(r["image"]).convert("RGB"))
        rgb = np.array(Image.open(r["mask"]).convert("RGB"))
        gt_o, gt_t, gt_a = rgb.sum(2) > 0, (rgb == BLUE).all(2), (rgb == WHITE).all(2)
        if gt_o.sum() < 20: continue
        bw, bh = int(r["bw"]), int(r["bh"]); box = (int(r["x"]), int(r["y"]), int(r["x"]) + bw, int(r["y"]) + bh)
        v = vg(os.path.basename(r["image"]))
        o, t, a = infer(model, img, box, route(bw / bh), dev)
        for vv in (v, "ALL"): obf[vv].append(boundary_f(gt_o, o, 1))
        if gt_t.sum() >= 60:
            for vv in (v, "ALL"): tiou[vv].append(mask_iou(gt_t, t))
        if gt_a.sum() >= 20:
            for vv in (v, "ALL"): aiou[vv].append(mask_iou(gt_a, a)); acov[vv].append((gt_a & a).sum() / gt_a.sum())
        if (i + 1) % 150 == 0: print(f"  {i+1}/{len(rows)}", flush=True)

    print("\n=== SPLICED single-pass model — by view ===")
    print(f"  {'view':<9}{'n':>5}{'OUTLINE_BF1':>13}{'TINT_IoU':>10}{'ANT_IoU':>9}{'ANT_cov':>9}")
    for v in V:
        if obf[v]:
            print(f"  {v:<9}{len(obf[v]):>5}{np.mean(obf[v]):>13.3f}"
                  f"{(np.mean(tiou[v]) if tiou[v] else float('nan')):>10.3f}"
                  f"{(np.mean(aiou[v]) if aiou[v] else float('nan')):>9.3f}"
                  f"{(np.mean(acov[v]) if acov[v] else float('nan')):>9.3f}")
    print("\n  refs: prod5 outline .824 ALL/.828 side | square tint IoU .662 | antenna UNet IoU ~.41 | REAL prod outline .692")
    # save deploy artifact
    out = f"{EXP}/birefnet_spliced_3head.pt"
    torch.save({"bb": prod5.bb.state_dict(),
                "sq_outline": prod5.squeeze_module.state_dict(), "dec_outline": prod5.decoder.state_dict(),
                "sq_tint": tri.squeeze_module.state_dict(), "dec_tint": tri.decoder.state_dict(),
                "note": "shared bb; outline=prod5 sq+dec(1ch); tint/ant=trihead sq+dec(3ch ch1/ch2)"}, out)
    print(f"\n  saved spliced artifact -> {out}")


if __name__ == "__main__":
    main()
