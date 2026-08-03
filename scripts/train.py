import argparse
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

TRAIN_SCRIPTS = {
    "yolov8": REPO_ROOT / "models/yolo/train_yolov8.py",
    "yolov11": REPO_ROOT / "models/yolo/train_yolov11.py",
    "yolov26": REPO_ROOT / "models/yolo/train_yolov26.py",
    "rtdetr": REPO_ROOT / "models/rtdetr/train_rtdetr.py",
    "dfine": REPO_ROOT / "models/dfine/train_dfine.py",
    "two_stage": REPO_ROOT / "models/two_stage/train_patch_classifier.py",
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Dispatch training to the script registered for --model. "
        "Every other flag (--fold, --subset, --config, --data-root, --output-dir, "
        "plus any model-specific flag such as --preset or --size) is passed through "
        "unchanged to that script — see models/<family>/train_*.py --help for its "
        "actual accepted flags, since they differ per model.",
    )
    parser.add_argument("--model", required=True, choices=sorted(TRAIN_SCRIPTS))
    args, extra = parser.parse_known_args()
    cmd = [sys.executable, str(TRAIN_SCRIPTS[args.model]), *extra]
    sys.exit(subprocess.call(cmd))


if __name__ == "__main__":
    main()
