import argparse
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

EVAL_SCRIPTS = {
    "dfine": REPO_ROOT / "models/dfine/evaluate_dfine.py",
    "sahi": REPO_ROOT / "models/yolo/train_sahi.py",
    "anomaly": REPO_ROOT / "models/anomaly_detection/run_benchmark.py",
    "two_stage": REPO_ROOT / "models/two_stage/two_stage_pipeline.py",
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Dispatch evaluation to the script registered for --model. "
        "Every other flag is passed through unchanged — see the target script's "
        "--help for its actual accepted flags, since they differ per model "
        "(e.g. dfine needs --checkpoint, sahi needs --weights-root and --model-family). "
        "YOLOv8/v11/v26 and RT-DETR have no separate evaluate script: ultralytics "
        "runs validation as part of models/yolo/train_*.py and models/rtdetr/train_rtdetr.py.",
    )
    parser.add_argument("--model", required=True, choices=sorted(EVAL_SCRIPTS))
    args, extra = parser.parse_known_args()
    cmd = [sys.executable, str(EVAL_SCRIPTS[args.model]), *extra]
    sys.exit(subprocess.call(cmd))


if __name__ == "__main__":
    main()
