import argparse
import os

from ultralytics import RTDETR

SUBSET_EPOCHS = {
    "components_only": 300,
    "full_dataset": 300,
    "missing_only": 200,
    "non_missing": 200,
}


def train(data_root: str, output_dir: str, subset: str, fold: int, config: str) -> None:
    data_config_path = os.path.join(data_root, subset, "kfold_data", f"fold_{fold}", "data.yaml")
    if not os.path.exists(data_config_path):
        print(f"Error: data.yaml not found for fold {fold} at {data_config_path}.")
        return

    output_base_path = os.path.join(output_dir, subset)

    model = RTDETR(config)
    epochs = SUBSET_EPOCHS[subset]

    model.train(
        data=data_config_path,
        epochs=epochs,
        imgsz=640,
        batch=8,
        name=f"rtdetr_fold_{fold}",
        project=output_base_path,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subset", required=True, choices=list(SUBSET_EPOCHS.keys()))
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--config", default="rtdetr-l.pt")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    train(args.data_root, args.output_dir, args.subset, args.fold, args.config)


if __name__ == "__main__":
    main()
