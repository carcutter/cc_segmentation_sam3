#!/usr/bin/env python3
"""
Profile the unified end-to-end pipeline's per-stage GPU cost (the compute reference).
Times each component over N test images (warmup first, cuda.synchronize), reports mean ms,
share of total, and param counts. Identifies where to attack for a Triton deployment.

Run from repo root:
  PYTHONPATH=. /home/rutger/miniconda3/envs/cc_sam3/bin/python carcutter/car_bbox_detector/profile_e2e.py --n 20
"""
import argparse, csv, time
from pathlib import Path
import numpy as np, torch
from PIL import Image
import segmentation_models_pytorch as smp
import rfdetr
from carcutter.car_bbox_detector.seg_eval import MEAN, STD
from carcutter.car_bbox_detector.birefnet_eval import load_birefnet, birefnet_outline
from carcutter.car_bbox_detector.cascade_eval import predict_union

ROOT = "carcutter/car_bbox_detector"; EXP = f"{ROOT}/experiments"
BIREFNET_CKPT = f"{ROOT}/birefnet/BiRefNet/ckpts/car_outline_v2/epoch_16.pth"
TINT_CKPT = f"{ROOT}/birefnet/BiRefNet/ckpts/car_holes_v2/epoch_16.pth"


def nparams(m):
    for attr in ("", ".model", ".model.model"):
        try:
            obj = eval("m" + attr)
            return sum(p.numel() for p in obj.parameters())
        except Exception:
            continue
    return 0


def load_unet(ckpt, enc, dev):
    m = smp.Unet(enc, encoder_weights=None, in_channels=3, classes=1).to(dev).eval()
    st = torch.load(ckpt, map_location=dev); m.load_state_dict(st.get("model", st)); return m


def sync(): torch.cuda.synchronize()


@torch.no_grad()
def seg_box(seg, img, box, dev, sz):
    x0, y0, x1, y1 = box; import cv2
    tile = cv2.resize(img[y0:y1, x0:x1], (sz, sz))
    t = torch.from_numpy(((tile / 255.0 - MEAN) / STD).transpose(2, 0, 1)[None]).float().to(dev)
    return torch.sigmoid(seg(t))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--index", default=f"{ROOT}/data/index.csv"); args = ap.parse_args()
    dev = "cuda"
    cardet = rfdetr.RFDETRMedium.from_checkpoint(f"{EXP}/rfdetr_car_antenna_v1/checkpoint_best_total.pth")
    unidet = rfdetr.RFDETRMedium.from_checkpoint(f"{EXP}/rfdetr_multiclass_v1/checkpoint_best_ema.pth")
    bfbody = load_birefnet(BIREFNET_CKPT, dev)
    bftint = load_birefnet(TINT_CKPT, dev)
    ant = load_unet(f"{EXP}/unet_antenna_v1/checkpoints/best.pt", "efficientnet-b4", dev)
    holeseg = load_unet(f"{EXP}/unet_holeseg_v2/best.pt", "efficientnet-b2", dev)
    winseg = load_unet(f"{EXP}/unet_windowseg_v2/best.pt", "efficientnet-b2", dev)

    pcount = {"BiRefNet(Swin-L)": nparams(bfbody), "RFDETR-Medium": nparams(unidet),
              "UNet-EffB4(antenna)": nparams(ant), "UNet-EffB2(hole/win)": nparams(holeseg)}

    rows = [r for r in csv.DictReader(open(args.index)) if r["split"] == "test"][:args.n]
    t = {k: [] for k in ["cardet", "bf_body_1024", "bf_body_1536", "bf_tint_1024",
                          "unidet", " unet_box_256", "unet_box_512"]}
    for i, r in enumerate(rows):
        pil = Image.open(r["image"]).convert("RGB"); img = np.array(pil); H, W = img.shape[:2]
        warm = i < 3
        sync(); s = time.time(); carbox = predict_union(cardet, pil, 0.3, 0.15); sync()
        if not warm: t["cardet"].append(time.time() - s)
        if carbox is None: continue
        cx0, cy0, cx1, cy1 = [max(0, int(v)) for v in carbox]
        cx1, cy1 = min(W, cx1), min(H, cy1)
        if cx1 - cx0 < 32 or cy1 - cy0 < 32: continue
        carcrop = img[cy0:cy1, cx0:cx1]
        for tag, sz in [("bf_body_1024", 1024), ("bf_body_1536", 1536)]:
            sync(); s = time.time(); _ = birefnet_outline(bfbody, img, carbox, dev, size=sz); sync()
            if not warm: t[tag].append(time.time() - s)
        sync(); s = time.time(); _ = birefnet_outline(bftint, img, carbox, dev, size=1024); sync()
        if not warm: t["bf_tint_1024"].append(time.time() - s)
        sync(); s = time.time(); _ = unidet.predict(Image.fromarray(carcrop), threshold=0.5); sync()
        if not warm: t["unidet"].append(time.time() - s)
        box = (cx0, cy0, min(W, cx0 + 200), min(H, cy0 + 200))
        sync(); s = time.time(); _ = seg_box(ant, img, box, dev, 256); sync()
        if not warm: t[" unet_box_256"].append(time.time() - s)
        sync(); s = time.time(); _ = seg_box(holeseg, img, box, dev, 512); sync()
        if not warm: t["unet_box_512"].append(time.time() - s)

    print("\n=== param counts ===")
    for k, v in pcount.items(): print(f"  {k:<24}{v/1e6:>7.1f} M")
    print("\n=== per-stage GPU latency (mean ms over {} imgs) ===".format(len(t['cardet'])))
    means = {k: 1000 * np.mean(v) if v else 0 for k, v in t.items()}
    # one full pipeline = cardet + bf_body_1024 + bf_body_1536 + bf_tint_1024 + unidet + (per-box unets, est ~4 boxes)
    per_box = means[" unet_box_256"] + 3 * means["unet_box_512"]   # ~1 antenna + ~3 hole/window boxes
    total = means["cardet"] + means["bf_body_1024"] + means["bf_body_1536"] + means["bf_tint_1024"] + means["unidet"] + per_box
    order = [("cardet (RFDETR full-img)", means["cardet"]),
             ("BiRefNet body @1024", means["bf_body_1024"]),
             ("BiRefNet body @1536", means["bf_body_1536"]),
             ("BiRefNet tint @1024", means["bf_tint_1024"]),
             ("unidet (RFDETR crop)", means["unidet"]),
             ("region UNets (~4 boxes)", per_box)]
    for name, ms in sorted(order, key=lambda x: -x[1]):
        print(f"  {name:<28}{ms:>7.1f} ms{100*ms/total:>7.1f}%")
    print(f"  {'TOTAL / image':<28}{total:>7.1f} ms")
    bf = means["bf_body_1024"] + means["bf_body_1536"] + means["bf_tint_1024"]
    print(f"\n  BiRefNet (3 Swin-L passes) = {bf:.0f} ms = {100*bf/total:.0f}% of pipeline")


if __name__ == "__main__":
    main()
