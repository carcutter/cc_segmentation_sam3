#!/usr/bin/env python3
"""
Antenna ensemble test: does combining ours (RF-DETR+UNet) with SAM3 'antenna' help?
Tests the two hedge hypotheses, pixel-level (3px tol, micro over test imgs):
  - UNION (ours ∪ SAM3): does SAM3 catch antennas ours misses? -> recall gain?
  - INTERSECTION (ours ∩ SAM3): can SAM3 veto our stick-FPs? -> precision gain?
ours fixed at det-thr 0.25 (P0.80/R0.88 operating point); SAM3 swept.

Run from repo root:
  PYTHONPATH=. python carcutter/car_bbox_detector/antenna_ensemble_eval.py --n 200
"""
import os
os.environ.setdefault("HF_HUB_CACHE", "/home/rutger/work/cc_segmentation_sam3_clean/carcutter/car_bbox_detector/birefnet/hf_cache")
os.environ.setdefault("HF_HOME", os.environ["HF_HUB_CACHE"])
import argparse, csv
from pathlib import Path
import numpy as np, cv2, torch
from PIL import Image
import segmentation_models_pytorch as smp
import rfdetr
from carcutter.car_bbox_detector.gen_coarse_masks import coarse_prob
from carcutter.car_bbox_detector.antenna_continuity import colfill
from sam3 import build_sam3_image_model
from sam3.eval.postprocessors import PostProcessImage
from sam3.model.utils.misc import copy_data_to_device
from sam3.train.data.collator import collate_fn_api as collate
from sam3.train.data.sam3_image_dataset import Datapoint, FindQueryLoaded, Image as SAMImage, InferenceMetadata
from sam3.train.transforms.basic_for_api import ComposeAPI, NormalizeAPI, RandomResizeAPI, ToTensorAPI
WHITE = (255, 255, 255); ANT = 1
EXP = "carcutter/car_bbox_detector/experiments"
OURS_THR = 0.25
SAM3_THRS = [0.10, 0.30, 0.50]


def load_unet(ck, dev):
    m = smp.Unet("efficientnet-b4", encoder_weights=None, in_channels=3, classes=1).to(dev).eval()
    st = torch.load(ck, map_location=dev); m.load_state_dict(st.get("model", st)); return m


def boxmask(b, H, W):
    m = np.zeros((H, W), np.uint8); x0, y0, x1, y1 = [int(v) for v in b]
    m[max(0, y0):min(H, y1 + 1), max(0, x0):min(W, x1 + 1)] = 255; return m


def sam3_dp(pil, prompt):
    w, h = pil.size
    dp = Datapoint(find_queries=[], images=[])
    dp.images = [SAMImage(data=pil, objects=[], size=[h, w])]
    dp.find_queries.append(FindQueryLoaded(query_text=prompt, image_id=0, object_ids_output=[],
        is_exhaustive=True, query_processing_order=0,
        inference_metadata=InferenceMetadata(coco_image_id=0, original_image_id=0, original_category_id=1,
            original_size=[h, w], object_id=0, frame_index=0)))
    return dp


def pr_acc():
    return {"corr": 0, "pred": 0, "cov": 0}


def update(acc, pred, white, wdil, k3, white_total_holder):
    ps = int(pred.sum()); acc["pred"] += ps
    if ps:
        acc["corr"] += int((pred & wdil).sum())
        pdil = cv2.dilate(pred.astype(np.uint8), k3) > 0
        acc["cov"] += int((white & pdil).sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--e2e", default="carcutter/car_bbox_detector/data/e2e_v2")
    ap.add_argument("--index", default="carcutter/car_bbox_detector/data/index.csv")
    ap.add_argument("--n", type=int, default=200)
    args = ap.parse_args()
    dev = torch.device("cuda")
    idx = {Path(r["image"]).stem: r for r in csv.DictReader(open(args.index))}
    det = rfdetr.RFDETRMedium.from_checkpoint(f"{EXP}/rfdetr_car_antenna_v1/checkpoint_best_total.pth")
    unet = load_unet(f"{EXP}/unet_antenna_v1/checkpoints/best.pt", dev)
    sam = build_sam3_image_model(device="cuda", eval_mode=True, load_from_HF=True)
    transform = ComposeAPI(transforms=[RandomResizeAPI(sizes=1008, max_size=1008, square=True, consistent_transform=False),
                                       ToTensorAPI(), NormalizeAPI(mean=[0.5]*3, std=[0.5]*3)])
    post = PostProcessImage(max_dets_per_img=-1, iou_type="segm", use_original_sizes_box=True,
                            use_original_sizes_mask=True, convert_mask_to_rle=False, detection_threshold=0.0, to_cpu=False)
    k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))

    accs = {"ours": pr_acc()}
    for t in SAM3_THRS:
        accs[f"sam3@{t}"] = pr_acc(); accs[f"union@{t}"] = pr_acc(); accs[f"inter@{t}"] = pr_acc()
    white_total = 0
    npzs = sorted((Path(args.e2e) / "masks").glob("*.npz"))[:args.n]
    for i, npz in enumerate(npzs):
        r = idx.get(npz.stem)
        if r is None:
            continue
        d = np.load(npz)
        if "carbox" not in d:
            continue
        pil = Image.open(r["image"]).convert("RGB"); img = np.array(pil); H, W = img.shape[:2]
        white = (np.array(Image.open(r["mask"]).convert("RGB")) == WHITE).all(2)
        white_total += int(white.sum()); wdil = cv2.dilate(white.astype(np.uint8), k3) > 0
        # ours @ 0.25
        dd = det.predict(pil, threshold=OURS_THR)
        cls, conf, xyxy = np.asarray(dd.class_id), np.asarray(dd.confidence), np.asarray(dd.xyxy)
        ours = np.zeros((H, W), bool)
        for j in range(len(xyxy)):
            if cls[j] == ANT and conf[j] >= OURS_THR:
                ours |= colfill(coarse_prob(unet, img, boxmask(xyxy[j], H, W), 256, dev) > 0.5)
        # sam3 on car crop
        x0, y0, x1, y1 = [int(v) for v in d["carbox"]]; x0, y0 = max(0, x0), max(0, y0); x1, y1 = min(W, x1), min(H, y1)
        crop = Image.fromarray(img[y0:y1, x0:x1])
        batch = collate([transform(sam3_dp(crop, "antenna"))], dict_key="inference")["inference"]
        batch = copy_data_to_device(batch, dev, non_blocking=True)
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            proc = post.process_results(sam(batch), batch.find_metadatas)
        rr = proc.get(0); sm, ss = [], []
        if rr is not None and len(rr.get("masks", [])) > 0:
            mk = rr["masks"];  mk = mk.squeeze(1) if mk.dim() == 4 else mk
            sm = (mk > 0.5).cpu().numpy().astype(bool); ss = rr["scores"].float().cpu().numpy()
        sam_full = []
        for m in sm:
            f = np.zeros((H, W), bool); f[y0:y1, x0:x1] = m[:y1 - y0, :x1 - x0]; sam_full.append(f)
        update(accs["ours"], ours, white, wdil, k3, None)
        for t in SAM3_THRS:
            s = np.zeros((H, W), bool)
            for m, sc in zip(sam_full, ss):
                if sc >= t:
                    s |= m
            update(accs[f"sam3@{t}"], s, white, wdil, k3, None)
            update(accs[f"union@{t}"], ours | s, white, wdil, k3, None)
            update(accs[f"inter@{t}"], ours & s, white, wdil, k3, None)
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(npzs)}", flush=True)

    def PR(a):
        P = a["corr"] / a["pred"] if a["pred"] else 1.0
        R = a["cov"] / white_total if white_total else 0.0
        return P, R
    print(f"\nAntenna ENSEMBLE (ours@{OURS_THR} fixed), pixel 3px tol, {len(npzs)} imgs:\n")
    print(f"  {'method':<12}{'precision':>11}{'recall':>9}")
    P, R = PR(accs["ours"]); print(f"  {'ours':<12}{P:>11.3f}{R:>9.3f}")
    for t in SAM3_THRS:
        for kind in ("sam3", "union", "inter"):
            P, R = PR(accs[f"{kind}@{t}"]); print(f"  {kind+'@'+str(t):<12}{P:>11.3f}{R:>9.3f}")
        print()


if __name__ == "__main__":
    main()
