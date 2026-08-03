import argparse
import json
import re
import shutil
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}

# CATEGORIES_FULL matches the published PCB-MC-A category list exactly: one alphabetical
# sort across all 31 names combined (not present-block-then-missing-block), which is why
# the 8 missing classes land at 15-22 rather than a contiguous tail. "Missing Component"
# is the literal class name used in the data (the dataset card's prose calls it "Unknown"
# descriptively, but the actual annotations use "Missing Component").
CATEGORIES_FULL = [
    {"id": 0, "name": "Button", "supercategory": "component"},
    {"id": 1, "name": "Capacitor", "supercategory": "component"},
    {"id": 2, "name": "Clock", "supercategory": "component"},
    {"id": 3, "name": "Connector", "supercategory": "component"},
    {"id": 4, "name": "Diode", "supercategory": "component"},
    {"id": 5, "name": "Display", "supercategory": "component"},
    {"id": 6, "name": "Electrolytic Capacitor", "supercategory": "component"},
    {"id": 7, "name": "EM", "supercategory": "component"},
    {"id": 8, "name": "Ferrite Bead", "supercategory": "component"},
    {"id": 9, "name": "Fuse", "supercategory": "component"},
    {"id": 10, "name": "Heatsink", "supercategory": "component"},
    {"id": 11, "name": "IC", "supercategory": "component"},
    {"id": 12, "name": "Inductor", "supercategory": "component"},
    {"id": 13, "name": "Jumper", "supercategory": "component"},
    {"id": 14, "name": "LED", "supercategory": "component"},
    {"id": 15, "name": "Missing Capacitor", "supercategory": "missing"},
    {"id": 16, "name": "Missing Component", "supercategory": "missing"},
    {"id": 17, "name": "Missing Diode", "supercategory": "missing"},
    {"id": 18, "name": "Missing Ferrite Bead", "supercategory": "missing"},
    {"id": 19, "name": "Missing IC", "supercategory": "missing"},
    {"id": 20, "name": "Missing Inductor", "supercategory": "missing"},
    {"id": 21, "name": "Missing LED", "supercategory": "missing"},
    {"id": 22, "name": "Missing Resistor", "supercategory": "missing"},
    {"id": 23, "name": "Pads", "supercategory": "component"},
    {"id": 24, "name": "Pins", "supercategory": "component"},
    {"id": 25, "name": "Potentiometer", "supercategory": "component"},
    {"id": 26, "name": "Resistor", "supercategory": "component"},
    {"id": 27, "name": "Switch", "supercategory": "component"},
    {"id": 28, "name": "Test Point", "supercategory": "component"},
    {"id": 29, "name": "Transistor", "supercategory": "component"},
    {"id": 30, "name": "Zener Diode", "supercategory": "component"},
]
CATEGORIES_MISSING = [
    {"id": 0, "name": "Missing Capacitor", "supercategory": "missing"},
    {"id": 1, "name": "Missing Component", "supercategory": "missing"},
    {"id": 2, "name": "Missing Diode", "supercategory": "missing"},
    {"id": 3, "name": "Missing Ferrite Bead", "supercategory": "missing"},
    {"id": 4, "name": "Missing IC", "supercategory": "missing"},
    {"id": 5, "name": "Missing Inductor", "supercategory": "missing"},
    {"id": 6, "name": "Missing LED", "supercategory": "missing"},
    {"id": 7, "name": "Missing Resistor", "supercategory": "missing"},
]
CATEGORIES_PRESENT = [
    {"id": 0, "name": "Button", "supercategory": "component"},
    {"id": 1, "name": "Capacitor", "supercategory": "component"},
    {"id": 2, "name": "Clock", "supercategory": "component"},
    {"id": 3, "name": "Connector", "supercategory": "component"},
    {"id": 4, "name": "Diode", "supercategory": "component"},
    {"id": 5, "name": "Display", "supercategory": "component"},
    {"id": 6, "name": "Electrolytic Capacitor", "supercategory": "component"},
    {"id": 7, "name": "EM", "supercategory": "component"},
    {"id": 8, "name": "Ferrite Bead", "supercategory": "component"},
    {"id": 9, "name": "Fuse", "supercategory": "component"},
    {"id": 10, "name": "Heatsink", "supercategory": "component"},
    {"id": 11, "name": "IC", "supercategory": "component"},
    {"id": 12, "name": "Inductor", "supercategory": "component"},
    {"id": 13, "name": "Jumper", "supercategory": "component"},
    {"id": 14, "name": "LED", "supercategory": "component"},
    {"id": 15, "name": "Pads", "supercategory": "component"},
    {"id": 16, "name": "Pins", "supercategory": "component"},
    {"id": 17, "name": "Potentiometer", "supercategory": "component"},
    {"id": 18, "name": "Resistor", "supercategory": "component"},
    {"id": 19, "name": "Switch", "supercategory": "component"},
    {"id": 20, "name": "Test Point", "supercategory": "component"},
    {"id": 21, "name": "Transistor", "supercategory": "component"},
    {"id": 22, "name": "Zener Diode", "supercategory": "component"},
]
SUBSET_CATEGORIES = {
    "full_dataset": CATEGORIES_FULL,
    "missing_only": CATEGORIES_MISSING,
    "components_only": CATEGORIES_PRESENT,
    "non_missing": CATEGORIES_PRESENT,
}


def collect_source_images(subset_path: Path, source_partitions: list[str]) -> dict[str, Path]:
    registry = {}
    for partition in source_partitions:
        img_dir = subset_path / partition / "images"
        if not img_dir.exists():
            continue
        for f in img_dir.iterdir():
            if f.suffix.lower() in IMAGE_EXTENSIONS and f.name not in registry:
                registry[f.name] = img_dir
    return registry


def make_board_id_extractor(strategy: str, a_position: str, b_prefix_parts: int, c_map_path: str | None):
    c_map = {}
    if strategy == "C":
        with open(c_map_path) as f:
            c_map = json.load(f)

    def extract_board_id(filename: str) -> str:
        stem = Path(filename).stem
        if strategy == "A":
            nums = re.findall(r"\d+", stem)
            if not nums:
                return stem
            return nums[0] if a_position == "first" else nums[-1]
        elif strategy == "B":
            parts = stem.split(".")
            bid = ".".join(parts[:b_prefix_parts])
            return bid if bid else stem
        elif strategy == "C":
            return c_map.get(filename, stem)
        raise ValueError(f"unknown strategy: {strategy}")

    return extract_board_id


def group_by_board(subset_path: Path, extract_board_id, source_partitions: list[str]):
    registry = collect_source_images(subset_path, source_partitions)
    groups = defaultdict(list)
    for fn in sorted(registry):
        groups[extract_board_id(fn)].append(fn)
    return dict(groups), registry


def assign_boards_to_folds(groups: dict, n_folds: int) -> tuple[dict, list]:
    sorted_boards = sorted(groups.items(), key=lambda x: (-len(x[1]), x[0]))
    fold_counts = [0] * n_folds
    board_to_fold = {}
    for bid, imgs in sorted_boards:
        t = int(np.argmin(fold_counts))
        board_to_fold[bid] = t
        fold_counts[t] += len(imgs)
    return board_to_fold, fold_counts


def build_train_valid_splits(groups: dict, board_to_fold: dict, n_folds: int, train_ratio: float) -> list[dict]:
    # train_ratio isn't applied as a separate split step: the 70/30 balance falls
    # out of the relative size of the folds themselves (paper's board-identity CV protocol).
    folds = [{"train": [], "valid": []} for _ in range(n_folds)]
    for fold_k in range(n_folds):
        val_boards = [b for b, f in board_to_fold.items() if f == fold_k]
        other_boards = [b for b, f in board_to_fold.items() if f != fold_k]
        for b in val_boards:
            folds[fold_k]["valid"].extend(groups[b])
        for b in other_boards:
            folds[fold_k]["train"].extend(groups[b])
    return folds


def get_label_path(filename: str, registry: dict) -> Path | None:
    src_img_dir = registry.get(filename)
    if src_img_dir is None:
        return None
    lbl = src_img_dir.parent / "labels" / (Path(filename).stem + ".txt")
    return lbl if lbl.exists() else None


def write_fold_structure(fold_splits: list, registry: dict, subset_path: Path, kfold_dir_name: str, clean_existing: bool) -> tuple[int, int, int]:
    kfold_root = subset_path / kfold_dir_name
    total = miss_lbl = miss_img = 0

    if clean_existing and kfold_root.exists():
        shutil.rmtree(kfold_root)

    for fold_k, split in enumerate(fold_splits):
        for partition in ["train", "valid"]:
            img_out = kfold_root / f"fold_{fold_k}" / partition / "images"
            lbl_out = kfold_root / f"fold_{fold_k}" / partition / "labels"
            img_out.mkdir(parents=True, exist_ok=True)
            lbl_out.mkdir(parents=True, exist_ok=True)

            for fname in split[partition]:
                src_dir = registry.get(fname)
                if src_dir is None:
                    miss_img += 1
                    continue

                dst_img = img_out / fname
                if not dst_img.exists():
                    shutil.copy2(src_dir / fname, dst_img)
                total += 1

                lbl = get_label_path(fname, registry)
                if lbl:
                    dst_lbl = lbl_out / lbl.name
                    if not dst_lbl.exists():
                        shutil.copy2(lbl, dst_lbl)
                else:
                    miss_lbl += 1

    return total, miss_lbl, miss_img


def get_img_dims(path: Path) -> tuple[int | None, int | None]:
    try:
        with Image.open(path) as img:
            return img.size
    except Exception as e:
        print(f"  error: {path}: {e}")
        return None, None


def yolo_to_coco(iw: int, ih: int, line: str) -> tuple[int | None, list | None, float | None]:
    parts = line.strip().split()
    if len(parts) < 5:
        return None, None, None
    try:
        cid = int(parts[0])
        cx, cy, bw, bh = map(float, parts[1:5])
    except ValueError:
        return None, None, None
    cw = bw * iw
    ch = bh * ih
    xmin = cx * iw - cw / 2
    ymin = cy * ih - ch / 2
    return cid, [xmin, ymin, cw, ch], cw * ch


def make_coco_json(images_dir: Path, labels_dir: Path, categories: list) -> dict:
    coco = {
        "info": {"description": "PCB-MC board-identity-aware split"},
        "licenses": [],
        "categories": categories,
        "images": [],
        "annotations": [],
    }
    img_id = ann_id = 0
    for img_f in sorted(images_dir.iterdir()):
        if img_f.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        w, h = get_img_dims(img_f)
        if w is None:
            continue
        coco["images"].append({
            "id": img_id, "file_name": img_f.name,
            "width": w, "height": h,
            "license": 0, "flickr_url": "", "coco_url": "", "date_captured": "",
        })
        lbl = labels_dir / f"{img_f.stem}.txt"
        if lbl.exists():
            with open(lbl) as f:
                for line in f:
                    cid, bbox, area = yolo_to_coco(w, h, line)
                    if bbox is None:
                        continue
                    coco["annotations"].append({
                        "id": ann_id, "image_id": img_id,
                        "category_id": cid,
                        "bbox": [round(v, 2) for v in bbox],
                        "area": round(area, 2),
                        "iscrowd": 0, "segmentation": [],
                    })
                    ann_id += 1
        img_id += 1
    return coco


def verify_leakage(subset_paths: dict, kfold_dir_name: str, n_folds: int, extract_board_id) -> bool:
    print("=" * 65)
    print("LEAKAGE VERIFICATION")
    print("=" * 65)
    all_passed = True

    for name, subset_path in subset_paths.items():
        kfold_root = subset_path / kfold_dir_name
        if not kfold_root.exists():
            continue
        print(f"\n{name}:")

        for fold_k in range(n_folds):
            sets, bids = {}, {}
            for p in ["train", "valid"]:
                d = kfold_root / f"fold_{fold_k}" / p / "images"
                files = {f.name for f in d.iterdir() if f.suffix.lower() in IMAGE_EXTENSIONS} if d.exists() else set()
                sets[p] = files
                bids[p] = {extract_board_id(f) for f in files}

            img_overlap = sets["train"] & sets["valid"]
            board_overlap = bids["train"] & bids["valid"]
            status = "PASS" if not img_overlap and not board_overlap else "FAIL"
            if status == "FAIL":
                all_passed = False

            print(f"  fold_{fold_k}  train={len(sets['train']):>4}  valid={len(sets['valid']):>4}  [{status}]")
            if img_overlap:
                print(f"    !! {len(img_overlap)} images shared between train and valid")
            if board_overlap:
                print(f"    !! {len(board_overlap)} boards shared: {sorted(board_overlap)[:5]}")

    print()
    print("all checks passed" if all_passed else "some checks failed")
    return all_passed


def print_summary(subset_paths: dict, kfold_dir_name: str, n_folds: int) -> None:
    for name, subset_path in subset_paths.items():
        kfold_root = subset_path / kfold_dir_name
        if not kfold_root.exists():
            continue
        print(f"\nsubset: {name}")

        fold_data = []
        for fold_k in range(n_folds):
            row = {}
            for p in ["train", "valid"]:
                ap = kfold_root / f"fold_{fold_k}" / p / "annotations.json"
                if ap.exists():
                    with open(ap) as f:
                        d = json.load(f)
                    row[p] = {"imgs": len(d["images"]), "anns": len(d["annotations"])}
                else:
                    row[p] = {"imgs": 0, "anns": 0}
            fold_data.append(row)
            print(f"  fold_{fold_k}  train_imgs={row['train']['imgs']:>4}  valid_imgs={row['valid']['imgs']:>4}  "
                  f"train_anns={row['train']['anns']:>6}  valid_anns={row['valid']['anns']:>6}")

        for p in ["train", "valid"]:
            v = [r[p]["imgs"] for r in fold_data]
            print(f"  {p} mean+/-std: {np.mean(v):.0f} +/- {np.std(v):.1f} imgs")


def main() -> None:
    parser = argparse.ArgumentParser(description="Board-identity-aware 5-fold CV split generator for PCB-MC subsets.")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--subsets", nargs="+", default=["full_dataset", "components_only", "missing_only", "non_missing"])
    parser.add_argument("--source-partitions", nargs="+", default=["train", "valid", "test"])
    parser.add_argument("--kfold-dir-name", default="kfold_data")
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--valid-ratio", type=float, default=0.30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--extraction-strategy", choices=["A", "B", "C"], default="B",
                         help="A: first/last number in filename. B: first N dot-separated prefix parts. C: JSON filename->board_id map.")
    parser.add_argument("--strategy-a-position", choices=["first", "last"], default="first")
    parser.add_argument("--strategy-b-prefix-parts", type=int, default=2)
    parser.add_argument("--strategy-c-map-path", default=None)
    parser.add_argument("--clean-existing", action="store_true")
    args = parser.parse_args()

    np.random.seed(args.seed)

    extract_board_id = make_board_id_extractor(
        args.extraction_strategy, args.strategy_a_position,
        args.strategy_b_prefix_parts, args.strategy_c_map_path,
    )

    data_root = Path(args.data_root)
    subset_paths = {name: data_root / name for name in args.subsets}

    all_groups, all_registries = {}, {}
    for name, path in subset_paths.items():
        groups, registry = group_by_board(path, extract_board_id, args.source_partitions)
        all_groups[name] = groups
        all_registries[name] = registry

    all_fold_splits, all_board_to_fold = {}, {}
    for name, groups in all_groups.items():
        if not groups:
            continue
        board_to_fold, fold_counts = assign_boards_to_folds(groups, args.n_folds)
        fold_splits = build_train_valid_splits(groups, board_to_fold, args.n_folds, args.train_ratio)
        all_board_to_fold[name] = board_to_fold
        all_fold_splits[name] = fold_splits
        print(f"{name}: {sum(fold_counts)} imgs, {len(groups)} boards, fold sizes {fold_counts}")

    for name in all_fold_splits:
        manifest_dir = subset_paths[name] / args.kfold_dir_name / "split_manifests"
        manifest_dir.mkdir(parents=True, exist_ok=True)
        manifest = {
            "subset": name,
            "n_folds": args.n_folds,
            "seed": args.seed,
            "ratios": {"train": args.train_ratio, "valid": args.valid_ratio},
            "strategy": "board_identity_greedy_binpack",
            "board_to_fold": all_board_to_fold[name],
            "folds": [
                {
                    "fold": k,
                    "train_images": sorted(sp["train"]),
                    "valid_images": sorted(sp["valid"]),
                }
                for k, sp in enumerate(all_fold_splits[name])
            ],
        }
        out = manifest_dir / f"{name}_splits.json"
        with open(out, "w") as f:
            json.dump(manifest, f, indent=2)
        print(f"wrote manifest: {out}")

    for name in all_fold_splits:
        t, ml, mi = write_fold_structure(
            all_fold_splits[name], all_registries[name], subset_paths[name],
            args.kfold_dir_name, args.clean_existing,
        )
        msg = f"{name}: {t} files copied"
        if ml:
            msg += f", {ml} missing labels"
        if mi:
            msg += f", {mi} images not found"
        print(msg)

    for name in all_fold_splits:
        cats = SUBSET_CATEGORIES.get(name, CATEGORIES_FULL)
        kfold_root = subset_paths[name] / args.kfold_dir_name
        for fold_k in range(args.n_folds):
            for partition in ["train", "valid"]:
                part_dir = kfold_root / f"fold_{fold_k}" / partition
                images_dir = part_dir / "images"
                labels_dir = part_dir / "labels"
                if not images_dir.exists():
                    continue
                coco = make_coco_json(images_dir, labels_dir, cats)
                out_json = part_dir / "annotations.json"
                with open(out_json, "w") as f:
                    json.dump(coco, f, indent=2)
                print(f"  fold_{fold_k}/{partition:<6} imgs={len(coco['images']):>4}  anns={len(coco['annotations']):>6}")

    verify_leakage(subset_paths, args.kfold_dir_name, args.n_folds, extract_board_id)
    print_summary(subset_paths, args.kfold_dir_name, args.n_folds)


if __name__ == "__main__":
    main()
