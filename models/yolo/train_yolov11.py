#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import SUBSETS, train_fold


def parse_args():
    parser = argparse.ArgumentParser(description="Train YOLOv11 on a PCB-MC subset/fold")
    parser.add_argument("--subset", required=True, choices=SUBSETS)
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--size", default="s", choices=["n", "s", "m", "l", "x"])
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--batch", type=int, default=-1)
    return parser.parse_args()


def main():
    args = parse_args()

    epochs, imgsz, batch = args.epochs, args.imgsz, args.batch
    if args.config:
        with open(args.config) as f:
            overrides = yaml.safe_load(f) or {}
        epochs = overrides.get("epochs", epochs)
        imgsz = overrides.get("imgsz", imgsz)
        batch = overrides.get("batch", batch)

    train_fold(
        family="yolo11",
        size=args.size,
        subset=args.subset,
        fold=args.fold,
        data_root=args.data_root,
        output_dir=args.output_dir,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
    )


if __name__ == "__main__":
    main()
