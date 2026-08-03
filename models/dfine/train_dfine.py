import argparse
import os
import subprocess

SUBSET_NUM_CLASSES = {
    "components_only": 23,
    "full_dataset": 31,
    "missing_only": 8,
    "non_missing": 23,
}


def train(dfine_repo: str, data_root: str, output_dir: str, subset: str, fold: int, config: str) -> None:
    fold_name = f"fold_{fold}"
    base_data_path = os.path.join(data_root, subset, "kfold_data")

    train_img_folder = os.path.join(base_data_path, fold_name, "train", "images")
    train_ann_file = os.path.join(base_data_path, fold_name, "train", "COCO_train.json")
    val_img_folder = os.path.join(base_data_path, fold_name, "valid", "images")
    val_ann_file = os.path.join(base_data_path, fold_name, "valid", "COCO_valid.json")

    fold_output_dir = os.path.join(output_dir, subset, fold_name)
    os.makedirs(fold_output_dir, exist_ok=True)

    num_classes = SUBSET_NUM_CLASSES[subset]

    cmd = [
        "python", "train.py",
        "-c", config,
        "--use-amp",
        "--seed", "42",
        "-u",
        f"train_dataloader.dataset.img_folder={train_img_folder}",
        f"train_dataloader.dataset.ann_file={train_ann_file}",
        f"val_dataloader.dataset.img_folder={val_img_folder}",
        f"val_dataloader.dataset.ann_file={val_ann_file}",
        "remap_mscoco_category=False",
        f"num_classes={num_classes}",
        "train_dataloader.total_batch_size=16",
        "val_dataloader.total_batch_size=32",
        "--output-dir", fold_output_dir,
    ]

    subprocess.run(cmd, cwd=dfine_repo, check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subset", required=True, choices=list(SUBSET_NUM_CLASSES.keys()))
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--config", default="configs/dfine/dfine_hgnetv2_l_coco.yml")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--dfine-repo", default="D-FINE")
    args = parser.parse_args()

    train(args.dfine_repo, args.data_root, args.output_dir, args.subset, args.fold, args.config)


if __name__ == "__main__":
    main()
