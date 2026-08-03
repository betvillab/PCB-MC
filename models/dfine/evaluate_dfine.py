import argparse
import json
import os
import re
import subprocess
import textwrap

SUBSET_NUM_CLASSES = {
    "components_only": 23,
    "full_dataset": 31,
    "missing_only": 8,
    "non_missing": 23,
}


def patch_det_engine(dfine_repo: str) -> None:
    engine_fp = os.path.join(dfine_repo, "src", "solver", "det_engine.py")
    lines = open(engine_fp, "r", encoding="utf-8").read().splitlines(True)

    if any("DFINE_SAVE_PREDICTIONS_JSON" in ln for ln in lines):
        return

    helper = textwrap.dedent("""
    # --- DFINE_SAVE_PREDICTIONS_JSON (added for PCB visualization) ---
    import json as _json
    import os as _os

    def _dfine_append_coco_preds(buffer, targets, results):
        for tgt, res in zip(targets, results):
            image_id = int(tgt["image_id"])
            boxes = res["boxes"]
            scores = res["scores"]
            labels = res["labels"]

            if hasattr(boxes, "tolist"): boxes = boxes.tolist()
            if hasattr(scores, "tolist"): scores = scores.tolist()
            if hasattr(labels, "tolist"): labels = labels.tolist()

            for (x1, y1, x2, y2), s, lab in zip(boxes, scores, labels):
                buffer.append({
                    "image_id": image_id,
                    "category_id": int(lab),
                    "bbox": [float(x1), float(y1), float(x2-x1), float(y2-y1)],  # xywh
                    "score": float(s),
                })

    def _dfine_dump_coco_preds(buffer, output_dir):
        if output_dir is None:
            output_dir = _os.environ.get("DFINE_PRED_DIR") or _os.getcwd()
        _os.makedirs(output_dir, exist_ok=True)
        out_path = _os.path.join(output_dir, "predictions.json")
        with open(out_path, "w") as f:
            _json.dump(buffer, f)
        print(f"Saved predictions.json: {out_path} ({len(buffer)} detections)")
    # --- end DFINE_SAVE_PREDICTIONS_JSON ---
    """)

    insert_at = 0
    for i in range(min(len(lines), 3000)):
        if lines[i].strip() == "" and i > 0:
            insert_at = i + 1
            break
    lines.insert(insert_at, helper + "\n")

    hook_idx = None
    rx = re.compile(r"^\s*results\s*=\s*postprocessor\s*\(\s*outputs\s*,\s*orig_target_sizes\s*\)\s*$")
    for i, ln in enumerate(lines):
        if rx.match(ln):
            hook_idx = i
            break
    if hook_idx is None:
        raise RuntimeError("Could not find the postprocessor call line in det_engine.py")

    indent = re.match(r"^(\s*)", lines[hook_idx]).group(1)
    hook = (
        f"{indent}# DFINE_SAVE_PREDICTIONS_JSON: collect detections\n"
        f"{indent}if not hasattr(postprocessor, '_dfine_pred_buffer'):\n"
        f"{indent}    postprocessor._dfine_pred_buffer = []\n"
        f"{indent}_dfine_append_coco_preds(postprocessor._dfine_pred_buffer, targets, results)\n"
    )
    lines.insert(hook_idx + 1, hook)

    ret_idx = None
    for i in range(len(lines) - 1, -1, -1):
        if re.match(r"^\s*return\b", lines[i]):
            ret_idx = i
            break
    if ret_idx is None:
        raise RuntimeError("Could not find a return statement to hook dump")

    ret_indent = re.match(r"^(\s*)", lines[ret_idx]).group(1)
    dump = (
        f"{ret_indent}# DFINE_SAVE_PREDICTIONS_JSON: dump once per evaluation\n"
        f"{ret_indent}if hasattr(postprocessor, '_dfine_pred_buffer'):\n"
        f"{ret_indent}    _dfine_dump_coco_preds(postprocessor._dfine_pred_buffer, output_dir)\n"
    )
    lines.insert(ret_idx, dump)

    open(engine_fp, "w", encoding="utf-8").write("".join(lines))


def run_eval(dfine_repo: str, config: str, checkpoint: str, val_img_folder: str,
             val_ann_file: str, num_classes: int, output_dir: str) -> None:
    env = os.environ.copy()
    env["DFINE_PRED_DIR"] = output_dir

    cmd = [
        "python", "train.py",
        "-c", config,
        "--test-only",
        "-r", checkpoint,
        "-u",
        f"val_dataloader.dataset.img_folder={val_img_folder}",
        f"val_dataloader.dataset.ann_file={val_ann_file}",
        "remap_mscoco_category=False",
        f"num_classes={num_classes}",
        "--output-dir", output_dir,
    ]

    subprocess.run(cmd, cwd=dfine_repo, env=env, check=True)


def parse_metrics(log_path: str, fold_name: str):
    if not os.path.exists(log_path):
        return None

    with open(log_path, "r") as f:
        lines = f.readlines()

    for line in reversed(lines):
        if "test_coco_eval_bbox" in line:
            try:
                data = json.loads(line[line.find("{"):])
                stats = data["test_coco_eval_bbox"]

                # COCO Standard Mapping:
                # 0: mAP @0.5:0.95 | 1: mAP @0.5 | 2: mAP @0.75
                # 8: Recall @100 det
                mAP_50 = stats[1]
                recall = stats[8]

                # COCO doesn't provide a single 'Precision' value in this list.
                # mAP@0.5 is used as the precision proxy to compute a meaningful F1.
                precision = mAP_50

                f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
                fnr = 1 - recall

                return {
                    "Fold": fold_name,
                    "mAP@0.5": round(mAP_50, 4),
                    "Precision (mAP)": round(precision, 4),
                    "Recall": round(recall, 4),
                    "F1-Score": round(f1, 4),
                    "FNR (Missing Class)": round(fnr, 4),
                }
            except Exception:
                continue
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--subset", required=True, choices=list(SUBSET_NUM_CLASSES.keys()))
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--config", default="configs/dfine/dfine_hgnetv2_l_coco.yml")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--dfine-repo", default="D-FINE")
    args = parser.parse_args()

    fold_name = f"fold_{args.fold}"
    val_img_folder = os.path.join(args.data_root, args.subset, "kfold_data", fold_name, "valid", "images")
    val_ann_file = os.path.join(args.data_root, args.subset, "kfold_data", fold_name, "valid", "COCO_valid.json")

    os.makedirs(args.output_dir, exist_ok=True)

    patch_det_engine(args.dfine_repo)
    run_eval(
        args.dfine_repo, args.config, args.checkpoint, val_img_folder, val_ann_file,
        SUBSET_NUM_CLASSES[args.subset], args.output_dir,
    )

    metrics = parse_metrics(os.path.join(args.output_dir, "log.txt"), fold_name)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
