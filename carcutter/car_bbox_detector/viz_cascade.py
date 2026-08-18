#!/usr/bin/env python3
"""
Visualize cascade predictions: the worst top-clip case + sample antenna cases.

Draws on each image: GT full-extent box (green), PASS1 box (yellow), CASCADE
final box (cyan), predicted antenna boxes (orange), PROD box (red, if present).
Also saves a zoomed crop of the top region so the thin antenna is visible.

Run: /home/rutger/miniconda3/envs/cc_sam3/bin/python viz_cascade.py \
        --weights experiments/rfdetr_car_antenna_v1/checkpoint_best_total.pth
"""
import argparse, csv
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw

CAR, ANT = 0, 1
TOP_PAD, SIDE_PAD = 0.5, 0.15
COL = {"gt": (50, 230, 50), "pass1": (255, 220, 0), "cascade": (0, 220, 255),
       "prod": (235, 40, 40), "ant": (255, 140, 0)}


def union(bs):
    a = np.array(bs); return [float(a[:,0].min()), float(a[:,1].min()), float(a[:,2].max()), float(a[:,3].max())]


def car_and_ant(model, pil, car_thr, ant_thr, ox=0.0, oy=0.0):
    det = model.predict(pil, threshold=ant_thr)
    if not len(det.xyxy):
        return None, []
    cls, conf, xyxy = np.asarray(det.class_id), np.asarray(det.confidence), np.asarray(det.xyxy)
    ci = [i for i in range(len(xyxy)) if cls[i] == CAR and conf[i] >= car_thr]
    if not ci:
        return None, []
    best = ci[int(np.argmax(conf[ci]))]
    car = [xyxy[best][0]+ox, xyxy[best][1]+oy, xyxy[best][2]+ox, xyxy[best][3]+oy]
    ants = [[xyxy[i][0]+ox, xyxy[i][1]+oy, xyxy[i][2]+ox, xyxy[i][3]+oy]
            for i in range(len(xyxy)) if cls[i] == ANT and conf[i] >= ant_thr]
    return car, ants


def prod_bbox(path):
    a = np.array(Image.open(path)); b = a > 0 if a.ndim == 2 else a.sum(2) > 0
    ys, xs = np.where(b)
    return None if len(ys) == 0 else [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]


def box(d, b, c, w=4, label=None):
    d.rectangle([b[0], b[1], b[2], b[3]], outline=c, width=w)
    if label:
        d.rectangle([b[0], max(0, b[1]-16), b[0]+8*len(label)+4, b[1]], fill=c)
        d.text((b[0]+2, max(0, b[1]-15)), label, fill=(0, 0, 0))


def render(rec, out_path):
    pil = Image.open(rec["img"]).convert("RGB"); d = ImageDraw.Draw(pil)
    for a in rec["ants"]:
        box(d, a, COL["ant"], 2)
    if rec.get("prod"): box(d, rec["prod"], COL["prod"], 3, "prod")
    box(d, rec["pass1"], COL["pass1"], 3, "pass1")
    box(d, rec["gt"], COL["gt"], 3, "GT")
    box(d, rec["cascade"], COL["cascade"], 3, f"cascade tc={rec['tc']:.0f}")
    pil.save(out_path, quality=90)
    # zoomed top crop around GT top edge
    g = rec["gt"]; cw = g[2]-g[0]
    z = pil.crop((max(0, int(g[0]-0.1*cw)), max(0, int(g[1]-80)),
                  min(pil.width, int(g[2]+0.1*cw)), min(pil.height, int(g[1]+0.35*(g[3]-g[1])))))
    z.save(out_path.replace(".jpg", "_topzoom.jpg"), quality=92)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="experiments/rfdetr_car_antenna_v1/checkpoint_best_total.pth")
    ap.add_argument("--size", default="medium")
    ap.add_argument("--index", default="data/index.csv")
    ap.add_argument("--out", default="data/viz_cascade")
    ap.add_argument("--n-antenna", type=int, default=6)
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    import rfdetr
    DET = {"nano":"RFDETRNano","small":"RFDETRSmall","medium":"RFDETRMedium","base":"RFDETRBase","large":"RFDETRLarge"}
    model = getattr(rfdetr, DET[args.size]).from_checkpoint(args.weights)
    test = [r for r in csv.DictReader(open(args.index)) if r["split"] == "test"]

    recs = []
    for r in test:
        gx, gy, gw, gh = int(r["x"]), int(r["y"]), int(r["bw"]), int(r["bh"])
        gt = [gx, gy, gx+gw, gy+gh]; W, H = int(r["width"]), int(r["height"])
        pil = Image.open(r["image"]).convert("RGB")
        car1, ant1 = car_and_ant(model, pil, 0.3, 0.15)
        if car1 is None:
            continue
        box1 = union([car1]+ant1)
        bw_, bh_ = box1[2]-box1[0], box1[3]-box1[1]
        c = (max(0, int(box1[0]-SIDE_PAD*bw_)), max(0, int(box1[1]-TOP_PAD*bh_)),
             min(W, int(box1[2]+SIDE_PAD*bw_)), min(H, int(box1[3]+SIDE_PAD*bh_)))
        car2, ant2 = car_and_ant(model, pil.crop(c), 0.3, 0.15, ox=c[0], oy=c[1])
        final = union([box1] + ([car2]+ant2 if car2 else []))
        recs.append({"img": r["image"], "gt": gt, "pass1": box1, "cascade": final,
                     "ants": ant1 + (ant2 if car2 else []), "has_ant": int(r["has_antenna"]),
                     "prod": prod_bbox(r["prod_outline"]) if r["prod_outline"] else None,
                     "tc": max(0, final[1]-gt[1])})

    worst = max(recs, key=lambda x: x["tc"])
    render(worst, str(out / "worst_topclip.jpg"))
    print(f"worst top-clip = {worst['tc']:.0f}px -> {Path(worst['img']).name}")

    ant = sorted([r for r in recs if r["has_ant"] and r is not worst], key=lambda x: -x["tc"])[:args.n_antenna]
    for i, r in enumerate(ant):
        render(r, str(out / f"antenna_{i:02d}_tc{r['tc']:.0f}.jpg"))
        print(f"antenna_{i:02d}: top-clip {r['tc']:.0f}px  {Path(r['img']).name}")
    print(f"\n-> {out}/")


if __name__ == "__main__":
    main()
