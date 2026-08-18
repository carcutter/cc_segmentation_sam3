#!/usr/bin/env python3
"""
Vanilla SAM3 "antenna" text-prompt — antenna PR curve, same axes as our detector+UNet.

Mirrors carcutter/detail_detection/generate_candidates.py SAM3 infra. Runs SAM3 with
the concept prompt "antenna" on the car crop (detector carbox, cached in e2e_v2), gets
instance masks + scores, places them back to full image, and sweeps the SAM3 score to
trace pixel-level precision/recall vs GT white (3px tol, micro over all test images).

Run from repo root:
  PYTHONPATH=. python carcutter/car_bbox_detector/antenna_sam3_pr.py --n 300 --prompt antenna
"""
import os
os.environ.setdefault("HF_HUB_CACHE", "/home/rutger/work/cc_segmentation_sam3_clean/carcutter/car_bbox_detector/birefnet/hf_cache")
os.environ.setdefault("HF_HOME", os.environ["HF_HUB_CACHE"])
import argparse, csv
from pathlib import Path
import numpy as np, cv2, torch
from PIL import Image
from sam3 import build_sam3_image_model
from sam3.eval.postprocessors import PostProcessImage
from sam3.model.utils.misc import copy_data_to_device
from sam3.train.data.collator import collate_fn_api as collate
from sam3.train.data.sam3_image_dataset import (
    Datapoint, FindQueryLoaded, Image as SAMImage, InferenceMetadata,
)
from sam3.train.transforms.basic_for_api import ComposeAPI, NormalizeAPI, RandomResizeAPI, ToTensorAPI
WHITE = (255, 255, 255)
THRS = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80]


def build_transform():
    return ComposeAPI(transforms=[
        RandomResizeAPI(sizes=1008, max_size=1008, square=True, consistent_transform=False),
        ToTensorAPI(), NormalizeAPI(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])


def make_datapoint(pil, prompt, qid):
    w, h = pil.size
    dp = Datapoint(find_queries=[], images=[])
    dp.images = [SAMImage(data=pil, objects=[], size=[h, w])]
    dp.find_queries.append(FindQueryLoaded(
        query_text=prompt, image_id=0, object_ids_output=[], is_exhaustive=True,
        query_processing_order=0,
        inference_metadata=InferenceMetadata(
            coco_image_id=qid, original_image_id=qid, original_category_id=1,
            original_size=[h, w], object_id=0, frame_index=0)))
    return dp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--e2e", default="carcutter/car_bbox_detector/data/e2e_v2")
    ap.add_argument("--index", default="carcutter/car_bbox_detector/data/index.csv")
    ap.add_argument("--out", default="carcutter/car_bbox_detector/data/antenna_pr_sam3.csv")
    ap.add_argument("--prompt", default="antenna")
    ap.add_argument("--n", type=int, default=300)
    args = ap.parse_args()
    dev = torch.device("cuda")
    idx = {Path(r["image"]).stem: r for r in csv.DictReader(open(args.index))}
    model = build_sam3_image_model(device="cuda", eval_mode=True, load_from_HF=True)
    transform = build_transform()
    post = PostProcessImage(max_dets_per_img=-1, iou_type="segm", use_original_sizes_box=True,
                            use_original_sizes_mask=True, convert_mask_to_rle=False,
                            detection_threshold=0.0, to_cpu=False)
    k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    npzs = sorted((Path(args.e2e) / "masks").glob("*.npz"))[:args.n]

    acc = {t: {"corr": 0, "pred": 0, "cov": 0} for t in THRS}
    white_total = 0; n_used = 0
    for i, npz in enumerate(npzs):
        r = idx.get(npz.stem)
        if r is None:
            continue
        d = np.load(npz)
        if "carbox" not in d:
            continue
        img = np.array(Image.open(r["image"]).convert("RGB")); H, W = img.shape[:2]
        white = (np.array(Image.open(r["mask"]).convert("RGB")) == WHITE).all(2)
        x0, y0, x1, y1 = [int(v) for v in d["carbox"]]
        x0, y0 = max(0, x0), max(0, y0); x1, y1 = min(W, x1), min(H, y1)
        crop = Image.fromarray(img[y0:y1, x0:x1])
        dp = transform(make_datapoint(crop, args.prompt, 0))
        batch = collate([dp], dict_key="inference")["inference"]
        batch = copy_data_to_device(batch, dev, non_blocking=True)
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            out = model(batch)
            proc = post.process_results(out, batch.find_metadatas)
        rr = proc.get(0)
        masks = []; scores = []
        if rr is not None and "masks" in rr and len(rr["masks"]) > 0:
            mk = rr["masks"]
            if mk.dim() == 4:
                mk = mk.squeeze(1)
            masks = (mk > 0.5).cpu().numpy().astype(bool)     # [N, ch, cw]
            scores = rr["scores"].float().cpu().numpy()
        white_total += int(white.sum()); n_used += 1
        wdil = cv2.dilate(white.astype(np.uint8), k3) > 0
        # place each crop mask back into full image
        full = []
        for m in masks:
            f = np.zeros((H, W), bool); f[y0:y1, x0:x1] = m[:y1 - y0, :x1 - x0]; full.append(f)
        for t in THRS:
            pred = np.zeros((H, W), bool)
            for m, s in zip(full, scores):
                if s >= t:
                    pred |= m
            ps = int(pred.sum()); acc[t]["pred"] += ps
            if ps:
                acc[t]["corr"] += int((pred & wdil).sum())
                pdil = cv2.dilate(pred.astype(np.uint8), k3) > 0
                acc[t]["cov"] += int((white & pdil).sum())
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(npzs)}", flush=True)

    pts = [(t, acc[t]["corr"] / acc[t]["pred"] if acc[t]["pred"] else 1.0,
            acc[t]["cov"] / white_total if white_total else 0.0) for t in THRS]
    srt = sorted([(R, P) for _, P, R in pts]); apx = sum(
        (srt[j][0] - srt[j - 1][0]) * (srt[j][1] + srt[j - 1][1]) / 2 for j in range(1, len(srt)))
    with open(args.out, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["method", "thr", "precision", "recall"])
        for t, P, R in pts:
            w.writerow([f"sam3:{args.prompt}", f"{t:.2f}", f"{P:.4f}", f"{R:.4f}"])
    print(f"\nSAM3 '{args.prompt}' antenna PR (pixel, 3px tol, {n_used} imgs):")
    print(f"  {'thr':>5}{'precision':>11}{'recall':>9}")
    for t, P, R in pts:
        print(f"  {t:>5.2f}{P:>11.3f}{R:>9.3f}")
    print(f"  PR-AUC(area) ~ {apx:.3f}\n-> {args.out}")


if __name__ == "__main__":
    main()
