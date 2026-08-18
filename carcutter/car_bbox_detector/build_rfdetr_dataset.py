#!/usr/bin/env python3
"""
Lay out coco_car_{train,val,test}.json into the RF-DETR directory structure
(symlinked images + _annotations.coco.json per split).

Usage:
    python build_rfdetr_dataset.py
"""
import argparse, json
from pathlib import Path


def write_split(coco_path, split_dir):
    split_dir.mkdir(parents=True, exist_ok=True)
    coco = json.load(open(coco_path))
    out = {"categories": coco["categories"], "images": [], "annotations": coco["annotations"]}
    linked = 0
    for img in coco["images"]:
        src = Path(img["file_name"]); name = f"{img['id']:06d}{src.suffix or '.jpg'}"
        dst = split_dir / name
        if dst.exists() or dst.is_symlink(): dst.unlink()
        try:
            dst.symlink_to(src.resolve()); linked += 1
        except FileNotFoundError:
            print(f"  missing {src}"); continue
        out["images"].append({**img, "file_name": name})
    json.dump(out, open(split_dir / "_annotations.coco.json", "w"))
    return linked, len(out["annotations"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="data/rfdetr_car")
    ap.add_argument("--prefix", default="coco_car", help="COCO file prefix, e.g. coco_ca")
    args = ap.parse_args()
    out = Path(args.out); d = Path(args.data)
    for split, sub in [("train", "train"), ("val", "valid"), ("test", "test")]:
        n = write_split(d / f"{args.prefix}_{split}.json", out / sub)
        print(f"{sub}: {n[0]} imgs, {n[1]} anns")
    print(f"Dataset ready -> {out}")


if __name__ == "__main__":
    main()
