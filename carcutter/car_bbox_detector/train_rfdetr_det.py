#!/usr/bin/env python3
"""
Train an RF-DETR detection model for the single-class car-extent bbox.

Run with the cc_sam3 env (rfdetr installed):
    /home/rutger/miniconda3/envs/cc_sam3/bin/python train_rfdetr_det.py

Prereq:
    python build_car_bbox_coco.py        # -> data/coco_car_{train,val,test}.json
    python build_rfdetr_dataset.py       # -> data/rfdetr_car/{train,valid,test}
"""
import argparse

DET_MODELS = {"nano": "RFDETRNano", "small": "RFDETRSmall",
              "medium": "RFDETRMedium", "base": "RFDETRBase", "large": "RFDETRLarge"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-dir", default="data/rfdetr_car")
    ap.add_argument("--output-dir", default="experiments/rfdetr_car_bbox_v1")
    ap.add_argument("--size", default="medium", choices=list(DET_MODELS))
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--grad-accum", type=int, default=1)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--num-workers", type=int, default=12)
    ap.add_argument("--classes", nargs="+", default=["car"], help="class names in id order")
    ap.add_argument("--resolution", type=int, default=0, help="0=model default; must be divisible by 56/64")
    ap.add_argument("--no-early-stopping", action="store_true")
    args = ap.parse_args()

    import rfdetr
    ModelCls = getattr(rfdetr, DET_MODELS[args.size])
    print(f"Model: {DET_MODELS[args.size]}  resolution={args.resolution or 'default'}")
    model = ModelCls(resolution=args.resolution) if args.resolution else ModelCls()
    model.train(
        dataset_dir=args.dataset_dir,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        grad_accum_steps=args.grad_accum,
        lr=args.lr,
        num_workers=args.num_workers,
        class_names=args.classes,
        early_stopping=not args.no_early_stopping,
        tensorboard=True,
        run_test=True,
    )
    print(f"\nDone -> {args.output_dir}/")


if __name__ == "__main__":
    main()
