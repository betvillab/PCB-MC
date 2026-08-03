import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "models" / "dfine"))
from evaluate_dfine import parse_metrics as parse_dfine_metrics  # noqa: E402

FOLDS = [f"fold_{i}" for i in range(5)]
SUBSETS = ("components_only", "full_dataset", "missing_only", "non_missing")

MODEL_DIR_NAMES = {
    "yolov8": "YOLOV8",
    "yolov11": "YOLOV11",
    "yolov26": "YOLOV26",
    "rtdetr": "RT-DETR",
}

ULTRALYTICS_METRIC_KEYS = ["mAP@0.5", "mAP@0.5:0.95", "Precision", "Recall", "F1-Score", "FNR"]
DFINE_METRIC_KEYS = ["mAP@0.5", "Precision (mAP)", "Recall", "F1-Score", "FNR (Missing Class)"]

MISSING_CLASSES = [
    "Missing Capacitor",
    "Missing Component",
    "Missing Diode",
    "Missing Ferrite Bead",
    "Missing IC",
    "Missing Inductor",
    "Missing LED",
    "Missing Resistor",
]


def parse_ultralytics_results_csv(csv_path: Path, fold_name: str):
    """Parses Ultralytics results.csv and returns the last-epoch metrics.

    Works for YOLOv8/v11/v26 and RT-DETR training logs (Results_YOLO_RTDETR.ipynb).
    """
    if not csv_path.exists():
        return None

    df = pd.read_csv(csv_path)
    if df.empty:
        return None

    last = df.iloc[-1].to_dict()

    map50 = last.get("metrics/mAP50(B)", np.nan)
    map5095 = last.get("metrics/mAP50-95(B)", np.nan)
    prec = last.get("metrics/precision(B)", np.nan)
    rec = last.get("metrics/recall(B)", np.nan)

    if np.isnan(map50):
        map50 = last.get("metrics/mAP50", np.nan)
    if np.isnan(map5095):
        map5095 = last.get("metrics/mAP50-95", np.nan)
    if np.isnan(prec):
        prec = last.get("metrics/precision", np.nan)
    if np.isnan(rec):
        rec = last.get("metrics/recall", np.nan)

    if np.isnan(map50) and np.isnan(map5095) and np.isnan(prec) and np.isnan(rec):
        raise ValueError(f"Unexpected columns in {csv_path}. Columns: {list(pd.read_csv(csv_path).columns)}")

    f1 = (2 * prec * rec) / (prec + rec + 1e-9) if (not np.isnan(prec) and not np.isnan(rec)) else np.nan
    fnr = 1 - rec if not np.isnan(rec) else np.nan

    return {
        "Fold": fold_name,
        "mAP@0.5": float(map50) if not np.isnan(map50) else np.nan,
        "mAP@0.5:0.95": float(map5095) if not np.isnan(map5095) else np.nan,
        "Precision": float(prec) if not np.isnan(prec) else np.nan,
        "Recall": float(rec) if not np.isnan(rec) else np.nan,
        "F1-Score": float(f1) if not np.isnan(f1) else np.nan,
        "FNR": float(fnr) if not np.isnan(fnr) else np.nan,
        "Source": "results.csv",
    }


def parse_ultralytics_results_txt(txt_path: Path, fold_name: str):
    """Fallback parser for older Ultralytics results.txt output."""
    if not txt_path.exists():
        return None

    lines = txt_path.read_text().splitlines()
    if not lines:
        return None

    target = None
    for line in reversed(lines):
        if "mAP50" in line and "precision" in line and "recall" in line:
            target = line
            break

    if target is None:
        return None

    try:
        tokens = target.replace(",", " ").split()
        floats = [float(t) for t in tokens if t.replace(".", "", 1).isdigit()]
        prec, rec, map50, map5095 = floats[-4], floats[-3], floats[-2], floats[-1]

        f1 = (2 * prec * rec) / (prec + rec + 1e-9)
        fnr = 1 - rec

        return {
            "Fold": fold_name,
            "mAP@0.5": map50,
            "mAP@0.5:0.95": map5095,
            "Precision": prec,
            "Recall": rec,
            "F1-Score": f1,
            "FNR": fnr,
            "Source": "results.txt",
        }
    except Exception:
        return None


def aggregate_fold_metrics(rows: list, metric_keys: list) -> pd.DataFrame:
    """AVERAGE / STD DEV row aggregation shared by Results_YOLO_RTDETR.ipynb
    and 5.1 D-FINE_Results.ipynb.
    """
    df = pd.DataFrame(rows)
    if df.empty:
        return df

    avg = df.mean(numeric_only=True).to_dict()
    std = df.std(numeric_only=True).to_dict()

    avg["Fold"] = "**AVERAGE**"
    std["Fold"] = "**STD DEV**"
    if "Source" in df.columns:
        avg["Source"] = "-"
        std["Source"] = "-"

    for k in metric_keys:
        if k in avg and not pd.isna(avg[k]):
            avg[k] = round(avg[k], 4)
        if k in std and not pd.isna(std[k]):
            std[k] = round(std[k], 4)

    return pd.concat([df, pd.DataFrame([avg]), pd.DataFrame([std])], ignore_index=True)


def detection_metrics_table(results_base: Path) -> pd.DataFrame:
    """YOLOv8/v11/v26 and RT-DETR: per-fold results.csv/.txt -> 5-fold table
    with AVERAGE/STD DEV rows, as in Results_YOLO_RTDETR.ipynb.
    """
    rows = []
    for fold in FOLDS:
        fold_dir = results_base / fold
        row = parse_ultralytics_results_csv(fold_dir / "results.csv", fold)
        if row is None:
            row = parse_ultralytics_results_txt(fold_dir / "results.txt", fold)
        if row is None:
            print(f"Could not find metrics for {fold} in {fold_dir}", file=sys.stderr)
            continue
        for k in ULTRALYTICS_METRIC_KEYS:
            if k in row and row[k] is not None and not (isinstance(row[k], float) and np.isnan(row[k])):
                row[k] = round(row[k], 4)
        rows.append(row)

    return aggregate_fold_metrics(rows, ULTRALYTICS_METRIC_KEYS)


def dfine_metrics_table(results_base: Path) -> pd.DataFrame:
    """D-FINE: per-fold log.txt (test_coco_eval_bbox) -> 5-fold table with
    AVERAGE/STD DEV rows, as in 5.1 D-FINE_Results.ipynb. Reuses parse_metrics
    from models/dfine/evaluate_dfine.py rather than re-deriving it.
    """
    rows = []
    for fold in FOLDS:
        log_path = results_base / fold / "log.txt"
        row = parse_dfine_metrics(str(log_path), fold)
        if row is None:
            print(f"Could not find metrics for {fold} in {log_path}", file=sys.stderr)
            continue
        rows.append(row)

    return aggregate_fold_metrics(rows, DFINE_METRIC_KEYS)


def read_yolo_labels(label_path: Path):
    boxes = []
    if not os.path.exists(label_path):
        return boxes
    with open(label_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            cls_id = int(float(parts[0]))
            cx, cy, w, h = map(float, parts[1:5])
            boxes.append((cls_id, cx, cy, w, h))
    return boxes


def yolo_to_xyxy(cx, cy, w, h):
    x1 = cx - w / 2
    y1 = cy - h / 2
    x2 = cx + w / 2
    y2 = cy + h / 2
    return x1, y1, x2, y2


def compute_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter

    if union <= 0:
        return 0.0
    return inter / union


def match_predictions_to_gt(gt_boxes, pred_boxes, iou_threshold=0.5):
    """Greedy, confidence-first matching of predicted boxes to ground-truth
    boxes, as in Confusion_Matrix_Missing_Components.ipynb.

    gt_boxes: list of (class_id, x1, y1, x2, y2)
    pred_boxes: list of (class_id, x1, y1, x2, y2, conf)
    """
    pred_boxes = sorted(pred_boxes, key=lambda x: x[5], reverse=True)

    matched_gt_indices = set()
    matches = []
    unmatched_pred = []

    for pred in pred_boxes:
        pred_cls = pred[0]
        pred_xyxy = pred[1:5]

        best_iou = 0
        best_gt_idx = -1

        for gt_idx, gt in enumerate(gt_boxes):
            if gt_idx in matched_gt_indices:
                continue
            gt_xyxy = gt[1:5]
            iou = compute_iou(pred_xyxy, gt_xyxy)
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = gt_idx

        if best_iou >= iou_threshold and best_gt_idx >= 0:
            gt_cls = gt_boxes[best_gt_idx][0]
            matches.append((gt_cls, pred_cls))
            matched_gt_indices.add(best_gt_idx)
        else:
            unmatched_pred.append(pred_cls)

    unmatched_gt = [
        gt_boxes[i][0] for i in range(len(gt_boxes))
        if i not in matched_gt_indices
    ]

    return matches, unmatched_gt, unmatched_pred


def list_images(folder: Path, exts=(".jpg", ".jpeg", ".png", ".bmp", ".webp")):
    if not os.path.exists(folder):
        return []
    return sorted([
        os.path.join(folder, f) for f in os.listdir(folder)
        if f.lower().endswith(exts)
    ])


def build_confusion_matrix(data_root: Path, subset: str, pred_label_dirs: dict,
                            classes: list, iou_threshold: float = 0.5):
    """Cross-fold confusion matrix over the missing-component classes, as in
    Confusion_Matrix_Missing_Components.ipynb cells [4] and [6].

    pred_label_dirs: {fold_index: path to that fold's predicted YOLO labels},
    already produced upstream by running inference with a trained checkpoint
    (ultralytics `model.predict(..., save_txt=True, save_conf=True)`).

    Returns (cm_all, cm_per_fold, per_class_df) where cm_all has shape
    (len(classes)+1, len(classes)+1); the extra row/col is Background
    (unmatched GT = FN row, unmatched predictions = FP column).
    """
    num_classes = len(classes)
    cm_all = np.zeros((num_classes + 1, num_classes + 1), dtype=int)
    cm_per_fold = []

    for fold_idx, pred_label_dir in sorted(pred_label_dirs.items()):
        gt_label_dir = data_root / subset / "kfold_data" / f"fold_{fold_idx}" / "valid" / "labels"
        val_img_dir = data_root / subset / "kfold_data" / f"fold_{fold_idx}" / "valid" / "images"
        val_images = list_images(val_img_dir)

        cm_fold = np.zeros((num_classes + 1, num_classes + 1), dtype=int)

        for img_path in val_images:
            img_stem = Path(img_path).stem

            gt_label_path = os.path.join(gt_label_dir, img_stem + ".txt")
            gt_raw = read_yolo_labels(gt_label_path)
            gt_boxes = [
                (cls_id, *yolo_to_xyxy(cx, cy, w, h))
                for cls_id, cx, cy, w, h in gt_raw
            ]

            pred_label_path = os.path.join(pred_label_dir, img_stem + ".txt")
            pred_raw = []
            if os.path.exists(pred_label_path):
                with open(pred_label_path) as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) < 6:
                            continue
                        cls_id = int(float(parts[0]))
                        cx, cy, w, h = map(float, parts[1:5])
                        conf = float(parts[5])
                        x1, y1, x2, y2 = yolo_to_xyxy(cx, cy, w, h)
                        pred_raw.append((cls_id, x1, y1, x2, y2, conf))

            matches, unmatched_gt, unmatched_pred = match_predictions_to_gt(
                gt_boxes, pred_raw, iou_threshold=iou_threshold
            )

            for gt_cls, pred_cls in matches:
                if gt_cls < num_classes and pred_cls < num_classes:
                    cm_fold[gt_cls, pred_cls] += 1

            for gt_cls in unmatched_gt:
                if gt_cls < num_classes:
                    cm_fold[gt_cls, num_classes] += 1

            for pred_cls in unmatched_pred:
                if pred_cls < num_classes:
                    cm_fold[num_classes, pred_cls] += 1

        cm_all += cm_fold
        cm_per_fold.append(cm_fold)

    per_class_rows = []
    total_tp = total_fp = total_fn = 0
    for i, cls_name in enumerate(classes):
        tp = int(cm_all[i, i])
        fn = int(cm_all[i, :].sum() - tp)
        fp = int(cm_all[:, i].sum() - tp)
        gt_total = int(cm_all[i, :].sum())

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        fnr = 1.0 - recall

        total_tp += tp
        total_fp += fp
        total_fn += fn

        per_class_rows.append({
            "Class": cls_name, "GT": gt_total, "TP": tp, "FP": fp, "FN": fn,
            "Precision": round(precision, 4), "Recall": round(recall, 4), "FNR": round(fnr, 4),
        })

    overall_prec = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    overall_rec = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    per_class_rows.append({
        "Class": "Overall", "GT": None, "TP": total_tp, "FP": total_fp, "FN": total_fn,
        "Precision": round(overall_prec, 4), "Recall": round(overall_rec, 4), "FNR": round(1 - overall_rec, 4),
    })

    return cm_all, cm_per_fold, pd.DataFrame(per_class_rows)


def _print_table(name: str, df: pd.DataFrame, output_dir: Path = None):
    print(f"\n=== {name} ===")
    if df.empty:
        print("(no results found)")
        return
    print(df.to_string(index=False))
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        out_path = output_dir / f"{name}.csv"
        df.to_csv(out_path, index=False)
        print(f"Saved: {out_path}")


def run_detection_metrics(model: str, subset: str, results_dir: Path, output_dir: Path):
    if model == "dfine":
        df = dfine_metrics_table(results_dir)
    else:
        df = detection_metrics_table(results_dir)
    _print_table(f"detection_metrics_{model}_{subset}", df, output_dir)


def run_confusion_matrix(data_root: Path, subset: str, pred_dir_pattern: str,
                          classes: list, iou_threshold: float, output_dir: Path):
    pred_label_dirs = {}
    for i in range(5):
        pred_dir = Path(pred_dir_pattern.format(fold=i))
        if pred_dir.exists():
            pred_label_dirs[i] = pred_dir
        else:
            print(f"Fold {i}: predicted labels not found at {pred_dir}, skipping", file=sys.stderr)

    cm_all, _, per_class_df = build_confusion_matrix(
        data_root, subset, pred_label_dirs, classes, iou_threshold
    )

    print(f"\n=== confusion_matrix_{subset} (counts, {len(classes)} classes + Background) ===")
    labels = classes + ["Background"]
    print(pd.DataFrame(cm_all, index=labels, columns=labels).to_string())
    _print_table(f"confusion_matrix_{subset}_per_class", per_class_df, output_dir)

    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        out_path = output_dir / f"confusion_matrix_{subset}_counts.csv"
        pd.DataFrame(cm_all, index=labels, columns=labels).to_csv(out_path)
        print(f"Saved: {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Reproduce the cross-fold aggregation tables from "
        "Results_YOLO_RTDETR.ipynb, 5.1 D-FINE_Results.ipynb and "
        "Confusion_Matrix_Missing_Components.ipynb from results already "
        "produced on disk by scripts/train.py / scripts/evaluate.py.",
    )
    parser.add_argument("--table", required=True, choices=["detection_metrics", "confusion_matrix", "all"])
    parser.add_argument("--model", choices=["yolov8", "yolov11", "yolov26", "rtdetr", "dfine"],
                         help="required for --table detection_metrics")
    parser.add_argument("--subset", choices=SUBSETS, help="required for --table detection_metrics/confusion_matrix")
    parser.add_argument("--results-dir",
                         help="dir containing fold_0..fold_4 (results.csv/.txt or log.txt); "
                         "defaults to --results-root/<model dir>/<subset>")
    parser.add_argument("--results-root", default=None,
                         help="e.g. .../PCB_MC/Results, laid out as <MODEL_DIR>/<subset>/fold_i "
                         "(YOLOV8, YOLOV11, YOLOV26, RT-DETR, D-Fine); used by --table all")
    parser.add_argument("--data-root", help="root containing <subset>/kfold_data/fold_i/valid/{images,labels}, "
                         "for --table confusion_matrix")
    parser.add_argument("--pred-dir-pattern",
                         help="predicted YOLO label dir per fold, with a {fold} placeholder, "
                         "for --table confusion_matrix")
    parser.add_argument("--classes", nargs="+", default=MISSING_CLASSES)
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    parser.add_argument("--output-dir", help="if set, also write each table to CSV here")
    args = parser.parse_args()

    output_dir = Path(args.output_dir) if args.output_dir else None

    if args.table in ("detection_metrics", "all"):
        models = [args.model] if args.model else list(MODEL_DIR_NAMES) + ["dfine"]
        subsets = [args.subset] if args.subset else list(SUBSETS)

        for model in models:
            for subset in subsets:
                if args.results_dir and args.model and args.subset:
                    results_dir = Path(args.results_dir)
                elif args.results_root:
                    model_dir = "D-Fine" if model == "dfine" else MODEL_DIR_NAMES[model]
                    results_dir = Path(args.results_root) / model_dir / subset
                else:
                    if args.table == "detection_metrics":
                        parser.error("--table detection_metrics requires --results-dir or --results-root")
                    continue
                run_detection_metrics(model, subset, results_dir, output_dir)

    if args.table in ("confusion_matrix", "all"):
        if args.data_root and args.pred_dir_pattern:
            subset = args.subset or "missing_only"
            run_confusion_matrix(
                Path(args.data_root), subset, args.pred_dir_pattern,
                args.classes, args.iou_threshold, output_dir,
            )
        elif args.table == "confusion_matrix":
            parser.error("--table confusion_matrix requires --data-root and --pred-dir-pattern")


if __name__ == "__main__":
    main()
