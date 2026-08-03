import argparse
import json
import os
import shutil
from pathlib import Path


def load_json(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def filter_coco(coco: dict, filenames: set[str]) -> dict:
    images = [img for img in coco["images"] if img["file_name"] in filenames]
    image_ids = {img["id"] for img in images}
    annotations = [a for a in coco["annotations"] if a["image_id"] in image_ids]
    return {**coco, "images": images, "annotations": annotations}


def coco_to_yolo_labels(coco: dict) -> dict[int, list[str]]:
    cat_id_to_idx = {c["id"]: i for i, c in enumerate(sorted(coco["categories"], key=lambda c: c["id"]))}
    dims = {img["id"]: (img["width"], img["height"]) for img in coco["images"]}
    by_image: dict[int, list[str]] = {img["id"]: [] for img in coco["images"]}
    for ann in coco["annotations"]:
        iw, ih = dims[ann["image_id"]]
        x, y, w, h = ann["bbox"]
        cx, cy = (x + w / 2) / iw, (y + h / 2) / ih
        by_image[ann["image_id"]].append(
            f"{cat_id_to_idx[ann['category_id']]} {cx:.6f} {cy:.6f} {w / iw:.6f} {h / ih:.6f}"
        )
    return by_image


def materialize_split(images_dir: Path, coco: dict, filenames: list[str], out_dir: Path, link: bool) -> dict:
    images_out = out_dir / "images"
    labels_out = out_dir / "labels"
    images_out.mkdir(parents=True, exist_ok=True)
    labels_out.mkdir(parents=True, exist_ok=True)

    filtered = filter_coco(coco, set(filenames))
    by_image = coco_to_yolo_labels(filtered)
    image_id_by_name = {img["file_name"]: img["id"] for img in filtered["images"]}

    for fname in filenames:
        src, dst = images_dir / fname, images_out / fname
        if link:
            if not dst.exists():
                os.symlink(src.resolve(), dst)
        else:
            shutil.copy2(src, dst)
        lines = by_image.get(image_id_by_name.get(fname), [])
        (labels_out / (Path(fname).stem + ".txt")).write_text("\n".join(lines))

    return filtered


def write_data_yaml(fold_dir: Path, class_names: list[str]) -> None:
    (fold_dir / "data.yaml").write_text(
        f"path: {fold_dir.resolve()}\n"
        "train: train/images\n"
        "val: valid/images\n"
        f"nc: {len(class_names)}\n"
        f"names: {json.dumps(class_names)}\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Materialize the <subset>/kfold_data/fold_N/{train,valid}/{images,labels} "
        "layout models/*/train_*.py expect, from the published images/ + per-task COCO "
        "annotations + splits/<task>_splits.json manifest. Class indices come from the "
        "COCO file's own 'categories' (sorted by id), not from splits/generate_kfold_splits.py's "
        "hardcoded tables, so this is safe to use against the corrected/published class IDs."
    )
    parser.add_argument("--images-dir", required=True, help="Shared images/ folder from the 4TU deposit")
    parser.add_argument("--coco-file", required=True, help="Task-wide COCO annotations, e.g. annotations/PCB-MC-A/annotations.json")
    parser.add_argument("--splits-file", required=True, help="e.g. splits/full_dataset_splits.json")
    parser.add_argument("--output-root", required=True, help="e.g. data/PCB-MC/full_dataset/kfold_data")
    parser.add_argument("--folds", nargs="+", type=int, default=None)
    parser.add_argument("--link", action="store_true", help="Symlink images instead of copying")
    args = parser.parse_args()

    manifest = load_json(args.splits_file)
    coco = load_json(args.coco_file)
    class_names = [c["name"] for c in sorted(coco["categories"], key=lambda c: c["id"])]
    images_dir = Path(args.images_dir)
    output_root = Path(args.output_root)
    wanted_folds = set(args.folds) if args.folds else {f["fold"] for f in manifest["folds"]}

    for fold in manifest["folds"]:
        if fold["fold"] not in wanted_folds:
            continue
        fold_dir = output_root / f"fold_{fold['fold']}"
        materialize_split(images_dir, coco, fold["train_images"], fold_dir / "train", args.link)
        materialize_split(images_dir, coco, fold["valid_images"], fold_dir / "valid", args.link)
        write_data_yaml(fold_dir, class_names)
        print(f"fold_{fold['fold']}: {len(fold['train_images'])} train, {len(fold['valid_images'])} valid")


if __name__ == "__main__":
    main()
