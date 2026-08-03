import argparse
import json
from pathlib import Path

import yaml
from PIL import Image

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".gif", ".bmp")


def get_image_dimensions(image_path: Path) -> tuple[int, int]:
    with Image.open(image_path) as img:
        return img.size


def yolo_to_coco_bbox(img_width: int, img_height: int, yolo_bbox_str: str) -> tuple[int | None, list | None, float | None]:
    parts = yolo_bbox_str.split()
    if len(parts) < 5:
        return None, None, None

    class_id = int(parts[0])
    center_x_norm, center_y_norm, width_norm, height_norm = map(float, parts[1:5])

    abs_center_x = center_x_norm * img_width
    abs_center_y = center_y_norm * img_height
    coco_width = width_norm * img_width
    coco_height = height_norm * img_height
    coco_x_min = abs_center_x - (coco_width / 2)
    coco_y_min = abs_center_y - (coco_height / 2)

    bbox = [coco_x_min, coco_y_min, coco_width, coco_height]
    area = coco_width * coco_height
    return class_id, bbox, area


def load_class_names(yaml_path: Path) -> list[str]:
    with open(yaml_path) as f:
        data = yaml.safe_load(f)
    return data["names"]


def categories_from_yaml(yaml_path: Path) -> list[dict]:
    names = load_class_names(yaml_path)
    return [{"id": idx, "name": name, "supercategory": "none"} for idx, name in enumerate(names)]


def convert_partition(images_dir: Path, labels_dir: Path, categories: list) -> dict:
    coco = {
        "info": {},
        "licenses": [],
        "categories": categories,
        "images": [],
        "annotations": [],
    }
    image_id = annotation_id = 0

    for img_path in sorted(images_dir.iterdir()):
        if img_path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue

        img_width, img_height = get_image_dimensions(img_path)
        coco["images"].append({
            "id": image_id,
            "width": img_width,
            "height": img_height,
            "file_name": img_path.name,
            "license": 0,
            "flickr_url": "",
            "coco_url": "",
            "date_captured": "",
        })

        label_path = labels_dir / f"{img_path.stem}.txt"
        if label_path.exists():
            with open(label_path) as f:
                for line in f:
                    class_id, bbox, area = yolo_to_coco_bbox(img_width, img_height, line.strip())
                    if bbox is None:
                        continue
                    coco["annotations"].append({
                        "id": annotation_id,
                        "image_id": image_id,
                        "category_id": class_id,
                        "bbox": [round(v) for v in bbox],
                        "area": round(area),
                        "iscrowd": 0,
                    })
                    annotation_id += 1
        else:
            print(f"  warning: no label file for {img_path.name} at {label_path}")

        image_id += 1

    return coco


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert a subset's YOLO-format kfold_data splits to COCO annotation JSON.")
    parser.add_argument("--kfold-root", required=True, help="path to <subset>/kfold_data")
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--out-name", default="annotations.json")
    parser.add_argument("--categories-json", default=None,
                         help="fixed COCO categories list (JSON) applied to every fold; "
                              "if omitted, categories are loaded per fold from <kfold-root>/fold_k/data.yaml")
    args = parser.parse_args()

    kfold_root = Path(args.kfold_root)

    fixed_categories = None
    if args.categories_json:
        with open(args.categories_json) as f:
            fixed_categories = json.load(f)

    for fold_idx in range(args.n_folds):
        categories = fixed_categories
        if categories is None:
            categories = categories_from_yaml(kfold_root / f"fold_{fold_idx}" / "data.yaml")

        for partition in ["train", "valid"]:
            part_dir = kfold_root / f"fold_{fold_idx}" / partition
            images_dir = part_dir / "images"
            labels_dir = part_dir / "labels"
            if not images_dir.exists():
                print(f"images path not found: {images_dir}. skipping.")
                continue

            coco = convert_partition(images_dir, labels_dir, categories)
            out_path = part_dir / args.out_name
            with open(out_path, "w") as f:
                json.dump(coco, f, indent=4)
            print(f"{out_path}: {len(coco['images'])} images, {len(coco['annotations'])} annotations")


if __name__ == "__main__":
    main()
