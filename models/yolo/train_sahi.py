#!/usr/bin/env python3
import argparse
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import yaml
from PIL import Image
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
from sahi.utils.cv import visualize_object_predictions, read_image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import SUBSETS

FAMILIES = ("yolov8", "yolo11", "yolo26")
FAMILY_DIR = {"yolov8": "YOLOV8", "yolo11": "YOLOV11", "yolo26": "YOLOV26"}
SMALL_OBJ_THRESHOLD = 32 * 32
FOLDS = [f"fold_{i}" for i in range(5)]

MAGENTA = (255, 0, 255)
YELLOW = (0, 255, 255)
RED = (0, 0, 255)
ORANGE = (0, 165, 255)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)


def get_image_dimensions(image_path):
    with Image.open(image_path) as img:
        return img.size


def load_yaml(yaml_path):
    with open(yaml_path, "r") as f:
        return yaml.safe_load(f)


def bbox_iou(box1, box2):
    x1, y1, x2, y2 = max(box1[0], box2[0]), max(box1[1], box2[1]), min(box1[2], box2[2]), min(box1[3], box2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    union = (box1[2] - box1[0]) * (box1[3] - box1[1]) + (box2[2] - box2[0]) * (box2[3] - box2[1]) - inter
    return inter / union if union > 0 else 0


def load_yolo_annotations(label_dir, image_filename, image_dims):
    annotations = []
    label_path = os.path.join(label_dir, os.path.splitext(image_filename)[0] + ".txt")
    if not os.path.exists(label_path):
        return annotations
    img_w, img_h = image_dims
    with open(label_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 5:
                c_id, cx, cy, w, h = map(float, parts)
                xmin, ymin = int((cx - w / 2) * img_w), int((cy - h / 2) * img_h)
                xmax, ymax = int((cx + w / 2) * img_w), int((cy + h / 2) * img_h)
                annotations.append({
                    "bbox": [xmin, ymin, xmax, ymax],
                    "category_id": int(c_id),
                    "area": (xmax - xmin) * (ymax - ymin),
                })
    return annotations


def calculate_ap(preds, gts, iou_threshold=0.5):
    if not gts:
        return 0.0
    if not preds:
        return 0.0
    preds = sorted(preds, key=lambda x: x["score"], reverse=True)
    tp, fp = np.zeros(len(preds)), np.zeros(len(preds))
    matched_gt = [False] * len(gts)
    for i, p in enumerate(preds):
        best_iou, best_idx = -1, -1
        for j, g in enumerate(gts):
            if not matched_gt[j] and p["category_id"] == g["category_id"]:
                iou = bbox_iou(p["bbox"], g["bbox"])
                if iou > best_iou:
                    best_iou, best_idx = iou, j
        if best_iou >= iou_threshold:
            tp[i] = 1
            matched_gt[best_idx] = True
        else:
            fp[i] = 1
    tp_cumsum, fp_cumsum = np.cumsum(tp), np.cumsum(fp)
    recalls = tp_cumsum / len(gts)
    precisions = tp_cumsum / (tp_cumsum + fp_cumsum)
    ap = 0.0
    for t in np.arange(0, 1.1, 0.1):
        p = np.max(precisions[recalls >= t]) if any(recalls >= t) else 0.0
        ap += p / 11.0
    return ap


def model_map(weights_root, family, subset, folds):
    family_dir = FAMILY_DIR[family]
    return {
        fold: str(Path(weights_root) / family_dir / subset / fold / "weights" / "best.pt")
        for fold in folds
    }


def run_eval(data_root, weights_root, output_dir, family, subset, folds):
    base_path = Path(data_root) / subset / "kfold_data"
    output_vis_dir = Path(output_dir) / "visuals"
    output_vis_dir.mkdir(parents=True, exist_ok=True)

    mmap = model_map(weights_root, family, subset, folds)
    data_yaml = load_yaml(base_path / folds[0] / "data.yaml")
    num_classes = len(data_yaml.get("names", []))
    results_log = []

    for folder in folds:
        print(f"\nStarting Evaluation with mAP: {folder}")
        img_dir = base_path / folder / "valid" / "images"
        label_dir = base_path / folder / "valid" / "labels"
        image_filenames = [f for f in os.listdir(img_dir) if f.lower().endswith((".png", ".jpg", ".jpeg"))]

        detection_model = AutoDetectionModel.from_pretrained(
            model_type="ultralytics", model_path=mmap[folder],
            confidence_threshold=0.30, device="cuda:0",
        )

        f_tp, f_fp, f_fn = 0, 0, 0
        f_tp_s, f_fn_s = 0, 0
        all_preds_cls = {i: [] for i in range(num_classes)}
        all_gts_cls = {i: [] for i in range(num_classes)}

        for idx, img_name in enumerate(image_filenames):
            image_path = os.path.join(img_dir, img_name)
            dims = get_image_dimensions(image_path)
            gts = load_yolo_annotations(label_dir, img_name, dims)

            result = get_sliced_prediction(
                image_path, detection_model,
                slice_height=1024, slice_width=1024,
                overlap_height_ratio=0.2, overlap_width_ratio=0.2,
                postprocess_type="GREEDYNMM",
                postprocess_match_threshold=0.3,
            )

            preds = [{"bbox": obj.bbox.to_xyxy(), "category_id": obj.category.id, "score": obj.score.value}
                     for obj in result.object_prediction_list]

            matched_gt = [False] * len(gts)
            sorted_p = sorted(preds, key=lambda x: x["score"], reverse=True)
            for p in sorted_p:
                best_iou, best_idx = -1, -1
                for i, g in enumerate(gts):
                    if not matched_gt[i] and p["category_id"] == g["category_id"]:
                        iou = bbox_iou(p["bbox"], g["bbox"])
                        if iou > best_iou:
                            best_iou, best_idx = iou, i
                if best_iou >= 0.5:
                    f_tp += 1
                    matched_gt[best_idx] = True
                else:
                    f_fp += 1
            f_fn += matched_gt.count(False)

            for p in preds:
                all_preds_cls[p["category_id"]].append(p)
            for g in gts:
                all_gts_cls[g["category_id"]].append(g)
                if g["area"] < SMALL_OBJ_THRESHOLD:
                    best_iou = max([bbox_iou(p["bbox"], g["bbox"])
                                     for p in preds if p["category_id"] == g["category_id"]] + [0])
                    if best_iou >= 0.5:
                        f_tp_s += 1
                    else:
                        f_fn_s += 1

            if idx < 3:
                visual_result = visualize_object_predictions(
                    np.array(read_image(image_path)), result.object_prediction_list)
                cv2.imwrite(
                    str(output_vis_dir / f"{folder}_{img_name}"),
                    cv2.cvtColor(visual_result["image"], cv2.COLOR_RGB2BGR),
                )

        p_f = f_tp / (f_tp + f_fp) if (f_tp + f_fp) > 0 else 0
        r_f = f_tp / (f_tp + f_fn) if (f_tp + f_fn) > 0 else 0
        f1_f = 2 * (p_f * r_f) / (p_f + r_f) if (p_f + r_f) > 0 else 0
        map50 = np.mean([calculate_ap(all_preds_cls[i], all_gts_cls[i]) for i in range(num_classes) if all_gts_cls[i]])
        s_rec = f_tp_s / (f_tp_s + f_fn_s) if (f_tp_s + f_fn_s) > 0 else 0

        results_log.append({
            "Folder": folder, "mAP@0.5": round(map50, 4), "Precision": round(p_f, 4),
            "Recall": round(r_f, 4), "F1": round(f1_f, 4), "Small_Recall": round(s_rec, 4),
        })

    df = pd.DataFrame(results_log)
    avg_row = df.mean(numeric_only=True).to_dict()
    avg_row["Folder"] = "AVERAGE"
    std_row = df.std(numeric_only=True).to_dict()
    std_row["Folder"] = "STD DEV"
    df = pd.concat([df, pd.DataFrame([avg_row, std_row])], ignore_index=True)
    print("\nFinal Results (SAHI + mAP)")
    print(df.to_string(index=False))
    return df


def draw_text(img, text, x, y):
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, BLACK, 3, cv2.LINE_AA)
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 1, cv2.LINE_AA)


def load_gt(label_dir, img_name, dims):
    gts = []
    path = os.path.join(label_dir, os.path.splitext(img_name)[0] + ".txt")
    if not os.path.exists(path):
        return gts
    W, H = dims
    with open(path) as f:
        for line in f:
            c, cx, cy, w, h = map(float, line.split())
            x1 = (cx - w / 2) * W
            y1 = (cy - h / 2) * H
            x2 = (cx + w / 2) * W
            y2 = (cy + h / 2) * H
            gts.append({"category_id": int(c), "bbox": [x1, y1, x2, y2]})
    return gts


def match(preds, gts, iou_th):
    preds = sorted(preds, key=lambda x: x["score"], reverse=True)
    used = [False] * len(gts)
    tp, fp = [], []
    for p in preds:
        best_iou, best_j = 0, -1
        for j, g in enumerate(gts):
            if used[j]:
                continue
            if p["category_id"] != g["category_id"]:
                continue
            iou = bbox_iou(p["bbox"], g["bbox"])
            if iou > best_iou:
                best_iou, best_j = iou, j
        if best_iou >= iou_th:
            used[best_j] = True
            tp.append(p)
        else:
            fp.append(p)
    fn = [gts[i] for i in range(len(gts)) if not used[i]]
    return tp, fp, fn


def class_color(cid, mode, connector_ids):
    if mode == "fp":
        return ORANGE
    if mode == "fn":
        return RED
    if cid in connector_ids:
        return YELLOW
    return MAGENTA


def should_label(cid, score, mode, missing_ids, conf_low):
    if cid in missing_ids:
        return True
    if mode in ["fp", "fn"]:
        return True
    if score is not None and score < conf_low:
        return True
    return False


def draw_boxes(img, items, mode, id2name, connector_ids, missing_ids, conf_low):
    for it in items:
        cid = it["category_id"]
        score = it.get("score")
        x1, y1, x2, y2 = map(int, it["bbox"])
        color = class_color(cid, mode, connector_ids)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        if should_label(cid, score, mode, missing_ids, conf_low):
            name = id2name[cid]
            if mode == "fp":
                text = f"FP {name}:{score:.2f}"
            elif mode == "fn":
                text = f"FN {name}"
            else:
                text = f"{name}:{score:.2f}"
            draw_text(img, text, x1, max(15, y1 - 5))


def save_views(img_path, tp, fp, fn, out_base, id2name, connector_ids, missing_ids, conf_low):
    img = cv2.imread(img_path)
    tp_img, fp_img, fn_img = img.copy(), img.copy(), img.copy()
    draw_boxes(tp_img, tp, "tp", id2name, connector_ids, missing_ids, conf_low)
    draw_boxes(fp_img, fp, "fp", id2name, connector_ids, missing_ids, conf_low)
    draw_boxes(fn_img, fn, "fn", id2name, connector_ids, missing_ids, conf_low)
    cv2.imwrite(out_base + "_TP.png", tp_img)
    cv2.imwrite(out_base + "_FP.png", fp_img)
    cv2.imwrite(out_base + "_FN.png", fn_img)


def run_visualize(data_root, weights_root, output_dir, family, subset, folds,
                   slice_size=640, overlap=0.2, conf_model=0.30, conf_low=0.80, iou_th=0.5, save_n=50):
    base_path = Path(data_root) / subset / "kfold_data"
    data_yaml_path = base_path / folds[0] / "data.yaml"
    out_root = Path(output_dir) / "tp_fp_fn_split"
    out_root.mkdir(parents=True, exist_ok=True)

    mmap = model_map(weights_root, family, subset, folds)
    data_yaml = load_yaml(data_yaml_path)

    id2name = {i: n for i, n in enumerate(data_yaml["names"])}
    connector_ids = {i for i, n in id2name.items() if "connector" in n.lower()}
    missing_ids = {i for i, n in id2name.items() if "missing" in n.lower()}

    print("Connector IDs:", connector_ids)
    print("Missing IDs:", missing_ids)

    for fold in folds:
        print(f"\nProcessing {fold}")
        img_dir = base_path / fold / "valid" / "images"
        label_dir = base_path / fold / "valid" / "labels"
        out_dir = out_root / fold
        out_dir.mkdir(parents=True, exist_ok=True)

        model = AutoDetectionModel.from_pretrained(
            model_type="ultralytics", model_path=mmap[fold],
            confidence_threshold=conf_model, device="cuda:0",
        )

        images = sorted(os.listdir(img_dir))[:save_n]

        for img_name in images:
            img_path = os.path.join(img_dir, img_name)
            dims = get_image_dimensions(img_path)
            gts = load_gt(label_dir, img_name, dims)

            result = get_sliced_prediction(
                img_path, model,
                slice_height=slice_size, slice_width=slice_size,
                overlap_height_ratio=overlap, overlap_width_ratio=overlap,
                postprocess_type="GREEDYNMM",
                postprocess_match_threshold=0.5,
            )

            preds = [{"category_id": int(obj.category.id), "score": float(obj.score.value), "bbox": list(obj.bbox.to_xyxy())}
                     for obj in result.object_prediction_list]

            tp, fp, fn = match(preds, gts, iou_th)
            out_base = str(out_dir / os.path.splitext(img_name)[0])
            save_views(img_path, tp, fp, fn, out_base, id2name, connector_ids, missing_ids, conf_low)

        print(f"Saved to {out_dir}")

    print("\nDone.")


def parse_args():
    parser = argparse.ArgumentParser(description="SAHI tiled inference/eval for trained YOLO PCB-MC models")
    parser.add_argument("--subset", required=True, choices=SUBSETS)
    parser.add_argument("--fold", type=int, default=None,
                         help="Evaluate a single fold; omit to run all 5 folds and print the AVERAGE/STD DEV summary")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--weights-root", type=Path, required=True,
                         help="Root containing YOLOV{8,11,26}/<subset>/fold_N/weights/best.pt")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-family", required=True, choices=FAMILIES)
    parser.add_argument("--mode", choices=["eval", "visualize"], default="eval")
    return parser.parse_args()


def main():
    args = parse_args()

    overrides = {}
    if args.config:
        with open(args.config) as f:
            overrides = yaml.safe_load(f) or {}

    folds = [f"fold_{args.fold}"] if args.fold is not None else FOLDS

    if args.mode == "eval":
        run_eval(args.data_root, args.weights_root, args.output_dir, args.model_family, args.subset, folds)
    else:
        run_visualize(
            args.data_root, args.weights_root, args.output_dir, args.model_family, args.subset, folds,
            slice_size=overrides.get("slice_size", 640),
            overlap=overrides.get("overlap", 0.2),
            conf_model=overrides.get("conf_model", 0.30),
            conf_low=overrides.get("conf_low", 0.80),
            iou_th=overrides.get("iou_th", 0.5),
            save_n=overrides.get("save_n", 50),
        )


if __name__ == "__main__":
    main()
