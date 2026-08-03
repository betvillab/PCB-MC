from pathlib import Path

import yaml

SUBSETS = ("components_only", "full_dataset", "missing_only", "non_missing")


def load_and_validate_yaml(yaml_path: Path) -> dict:
    if not yaml_path.exists():
        raise FileNotFoundError(f"Missing YAML: {yaml_path}")

    with open(yaml_path, "r") as f:
        data = yaml.safe_load(f)

    if "names" not in data:
        raise ValueError(f"'names' not found in {yaml_path}. Make sure Roboflow YAML includes names.")
    if "nc" not in data:
        data["nc"] = len(data["names"])

    if data["nc"] != len(data["names"]):
        raise ValueError(f"Mismatch in {yaml_path}: nc={data['nc']} but len(names)={len(data['names'])}")

    if "val" not in data and "valid" in data:
        data["val"] = data["valid"]

    if "train" not in data or "val" not in data:
        raise ValueError(f"{yaml_path} must contain 'train' and 'val' (or 'valid'). Keys: {list(data.keys())}")

    return data


def fold_yaml_path(data_root: Path, subset: str, fold: int) -> Path:
    return Path(data_root) / subset / "kfold_data" / f"fold_{fold}" / "data.yaml"


BASE_AUG = dict(
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
    copy_paste=0.0,
    close_mosaic=20,
)

LR0_V8 = 1e-4
LR0_V11 = 3e-4
LR0_V26 = 2.5e-4


def weights_name(family: str, size: str) -> str:
    if family == "yolov8":
        return f"yolov8{size}.pt"
    if family == "yolo11":
        return f"yolo11{size}.pt"
    if family == "yolo26":
        return f"yolo26{size}.pt"
    raise ValueError("family must be: yolov8 | yolo11 | yolo26")


def family_hypers(family: str):
    aug = dict(BASE_AUG)

    if family == "yolov8":
        lr0 = LR0_V8
        aug.update(mosaic=0.6, mixup=0.05)
    elif family == "yolo11":
        lr0 = LR0_V11
        aug.update(mosaic=0.4, mixup=0.02)
    elif family == "yolo26":
        lr0 = LR0_V26
        aug.update(mosaic=0.2, mixup=0.0)
    else:
        raise ValueError("Unknown family")

    return lr0, aug


def train_fold(family: str, size: str, subset: str, fold: int, data_root: Path, output_dir: Path,
               epochs: int = 300, imgsz: int = 1024, batch: int = -1):
    from ultralytics import YOLO

    yaml_path = fold_yaml_path(data_root, subset, fold)
    cfg = load_and_validate_yaml(yaml_path)

    base_weights = weights_name(family, size)
    lr0, aug = family_hypers(family)

    print(f"\nTraining Fold {fold} | {family}{size}")
    print(f"  weights: {base_weights} | imgsz: {imgsz} | batch: {batch} | lr0: {lr0}")
    print(f"  YAML: {yaml_path} | nc: {cfg['nc']}")

    results_path = Path(output_dir)
    results_path.mkdir(parents=True, exist_ok=True)

    model = YOLO(base_weights)
    model.train(
        data=str(yaml_path),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        project=str(results_path / family),
        name=f"{family}{size}_fold_{fold}",
        exist_ok=True,

        optimizer="AdamW",
        lr0=lr0,
        cos_lr=True,
        patience=25,

        amp=True,
        cache=True,
        workers=8,

        **aug,
    )

    print("\nTraining complete.")
