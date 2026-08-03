#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

import yaml
from ultralytics import YOLO

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import SUBSETS, fold_yaml_path, load_and_validate_yaml

PRESETS = {
    "A": dict(weights="yolov8s.pt", imgsz=1024, batch=-1, epochs=300),
    "B": dict(weights="yolov8m.pt", imgsz=1280, batch=8, epochs=300),
}

AUG = dict(
    fliplr=0.5,
    flipud=0.2,
    degrees=5.0,
    translate=0.05,
    scale=0.35,
    shear=0.0,
    perspective=0.0,
    hsv_h=0.015,
    hsv_s=0.40,
    hsv_v=0.30,
    mosaic=0.6,
    mixup=0.05,
    copy_paste=0.0,
)

OPTIMIZER = "AdamW"
LR0 = 1e-4
COS_LR = True
PATIENCE = 25


def parse_args():
    parser = argparse.ArgumentParser(description="Train YOLOv8 on a PCB-MC subset/fold")
    parser.add_argument("--subset", required=True, choices=SUBSETS)
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--preset", choices=["A", "B"], default="A")
    return parser.parse_args()


def main():
    args = parse_args()

    hypers = dict(PRESETS[args.preset])
    aug = dict(AUG)
    lr0 = LR0

    if args.config:
        with open(args.config) as f:
            overrides = yaml.safe_load(f) or {}
        hypers.update({k: overrides[k] for k in ("weights", "imgsz", "batch", "epochs") if k in overrides})
        aug.update({k: overrides[k] for k in AUG if k in overrides})
        lr0 = overrides.get("lr0", lr0)

    yaml_path = fold_yaml_path(args.data_root, args.subset, args.fold)
    cfg = load_and_validate_yaml(yaml_path)

    results_path = Path(args.output_dir)
    results_path.mkdir(parents=True, exist_ok=True)

    print(f"\nTraining Fold {args.fold} | PRESET={args.preset}")
    print(f"  weights: {hypers['weights']} | imgsz: {hypers['imgsz']} | batch: {hypers['batch']}")
    print(f"  YAML: {yaml_path}")
    print(f"  nc: {cfg['nc']}")
    print(f"  names[0:5]: {cfg['names'][:5]} ... names[-3:]: {cfg['names'][-3:]}")

    model = YOLO(hypers["weights"])
    model.train(
        data=str(yaml_path),
        epochs=hypers["epochs"],
        imgsz=hypers["imgsz"],
        batch=hypers["batch"],
        project=str(results_path),
        name=f"fold_{args.fold}",
        exist_ok=True,

        optimizer=OPTIMIZER,
        lr0=lr0,
        cos_lr=COS_LR,
        patience=PATIENCE,

        amp=True,
        cache=True,
        workers=8,

        **aug,
    )

    print("\nTraining complete.")


if __name__ == "__main__":
    main()
