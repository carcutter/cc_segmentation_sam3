#!/usr/bin/env python3
"""
Run the production pipeline ONCE over the test set and CACHE per-step predictions to disk, so every
downstream analysis (wheel/non-wheel splits, FP montages, antenna length, thresholds...) is instant.
Per image saves (packed bits): o (spliced bucketed outline), ant_mask (zoom antenna UNet),
hole2 (2-stage hole UNet), carbox, shape. Downstream recomputes outline=keep_main(o|ant) and
punchout=(fill_holes(outline)&~outline)|hole2 cheaply on CPU.
Out: data/e2e_cache/<stem>.npz   Run: PYTHONPATH=.:.../BiRefNet HF_HOME=... PY cache_pipeline.py
"""
import csv, os, sys
from pathlib import Path
import numpy as np, cv2, torch
from PIL import Image
import segmentation_models_pytorch as smp
import rfdetr
sys.path.insert(0, "carcutter/car_bbox_detector/birefnet/BiRefNet")
from carcutter.car_bbox_detector.probe_birefnet_aspect import route
from carcutter.car_bbox_detector.build_spliced_eval import Spliced, load_bn, infer, PROD5, TRI
from carcutter.car_bbox_detector.gen_coarse_masks import coarse_prob
from carcutter.car_bbox_detector.cascade_eval import predict_union
from carcutter.car_bbox_detector.eval_e2e_prod import boxmask, segment_box, DET_THR, ANT_THR, HOLE_THR, SEG_REG, ANT_SZ, ANTENNA, HOLE

ROOT = "carcutter/car_bbox_detector"; EXP = f"{ROOT}/experiments"; dev = "cuda"
OUT = Path(f"{ROOT}/data/e2e_cache"); OUT.mkdir(parents=True, exist_ok=True)


def pack(m): return np.packbits(m.ravel())


def main():
    spliced = Spliced(load_bn(PROD5, 1, dev), load_bn(TRI, 3, dev)).eval()
    cardet = rfdetr.RFDETRMedium.from_checkpoint(f"{EXP}/rfdetr_car_antenna_v1/checkpoint_best_total.pth")
    unidet = rfdetr.RFDETRMedium.from_checkpoint(f"{EXP}/rfdetr_multiclass_v1/checkpoint_best_ema.pth")
    antunet = smp.Unet("efficientnet-b4", encoder_weights=None, in_channels=3, classes=1).to(dev).eval()
    antunet.load_state_dict((lambda s: s.get("model", s))(torch.load(f"{EXP}/unet_antenna_v1/checkpoints/best.pt", map_location=dev)))
    holeunet = smp.Unet("efficientnet-b2", encoder_weights=None, in_channels=3, classes=1).to(dev).eval()
    holeunet.load_state_dict((lambda s: s.get("model", s))(torch.load(f"{EXP}/unet_holeseg_v2/best.pt", map_location=dev)))
    rows = [r for r in csv.DictReader(open(f"{ROOT}/data/index.csv")) if r["split"] == "test"]

    done = 0
    for i, r in enumerate(rows):
        stem = Path(r["image"]).stem; fp = OUT / f"{stem}.npz"
        if fp.exists(): done += 1; continue
        pil = Image.open(r["image"]).convert("RGB"); img = np.array(pil); H, W = img.shape[:2]
        carbox = predict_union(cardet, pil, 0.3, 0.15)
        if carbox is None:
            np.savez_compressed(fp, shape=np.array([H, W]), carbox=np.array([-1,-1,-1,-1]),
                                o=np.zeros(0,np.uint8), ant=np.zeros(0,np.uint8), hole2=np.zeros(0,np.uint8)); continue
        cx0, cy0, cx1, cy1 = [max(0,int(carbox[0])), max(0,int(carbox[1])), min(W,int(carbox[2])), min(H,int(carbox[3]))]
        if cx1-cx0 < 32 or cy1-cy0 < 32:
            np.savez_compressed(fp, shape=np.array([H, W]), carbox=np.array([-1,-1,-1,-1]),
                                o=np.zeros(0,np.uint8), ant=np.zeros(0,np.uint8), hole2=np.zeros(0,np.uint8)); continue
        o, _t, _a = infer(spliced, img, (cx0, cy0, cx1, cy1), route((cx1-cx0)/(cy1-cy0)), dev)
        carcrop = img[cy0:cy1, cx0:cx1]
        dd = unidet.predict(Image.fromarray(carcrop), threshold=DET_THR)
        cls = np.asarray(dd.class_id) if getattr(dd, "class_id", None) is not None else np.zeros(len(dd.xyxy))
        ant_mask = np.zeros((H, W), bool); hole2 = np.zeros((H, W), bool)
        for bi, b in enumerate(np.asarray(dd.xyxy)):
            bx0, by0, bx1, by1 = [int(x) for x in b]; gbox = (cx0+bx0, cy0+by0, cx0+bx1, cy0+by1)
            c = int(cls[bi])
            if c == ANTENNA:
                ant_mask |= coarse_prob(antunet, img, boxmask(gbox, H, W), ANT_SZ, dev) > ANT_THR
            elif c == HOLE:
                m = segment_box(holeunet, img, gbox, HOLE_THR, SEG_REG)
                if m is not None: hole2 |= m
        np.savez_compressed(fp, shape=np.array([H, W]), carbox=np.array([cx0,cy0,cx1,cy1]),
                            o=pack(o), ant=pack(ant_mask), hole2=pack(hole2))
        done += 1
        if (i+1) % 50 == 0: print(f"  {i+1}/{len(rows)} cached", flush=True)
    print(f"Done: {done} cached -> {OUT}")


if __name__ == "__main__":
    main()
