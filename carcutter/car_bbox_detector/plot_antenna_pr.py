#!/usr/bin/env python3
"""Plot antenna PR curves: ours (RF-DETR+UNet) vs SAM3 'antenna' prompt (+ ensemble if present)."""
import argparse, csv
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load(p):
    if not Path(p).exists():
        return None
    rows = list(csv.DictReader(open(p)))
    return (rows[0]["method"],
            [float(r["thr"]) for r in rows],
            [float(r["precision"]) for r in rows],
            [float(r["recall"]) for r in rows])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ours", default="carcutter/car_bbox_detector/data/antenna_pr.csv")
    ap.add_argument("--sam3", default="carcutter/car_bbox_detector/data/antenna_pr_sam3.csv")
    ap.add_argument("--ens", default="carcutter/car_bbox_detector/data/antenna_pr_ensemble.csv")
    ap.add_argument("--out", default="carcutter/car_bbox_detector/data/antenna_pr.png")
    args = ap.parse_args()
    fig, axp = plt.subplots(figsize=(8, 7))
    for path, color in ((args.ours, "tab:blue"), (args.sam3, "tab:red"), (args.ens, "tab:green")):
        c = load(path)
        if c is None:
            continue
        name, thr, P, R = c
        axp.plot(R, P, "-o", color=color, label=name)
        for t, p, r in zip(thr, P, R):
            axp.annotate(f"{t:.2f}", (r, p), fontsize=7, alpha=0.7,
                         textcoords="offset points", xytext=(3, 3))
    axp.set_xlabel("recall"); axp.set_ylabel("precision")
    axp.set_title("Antenna PR — ours vs SAM3 'antenna' (pixel, 3px tol, micro; labels=score thr)")
    axp.set_xlim(0, 1.02); axp.set_ylim(0, 1.02); axp.grid(alpha=0.3); axp.legend()
    fig.tight_layout(); fig.savefig(args.out, dpi=90, bbox_inches="tight")
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
