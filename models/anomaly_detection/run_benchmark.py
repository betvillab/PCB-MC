"""
PCB-MC Anomaly Detection Benchmark -- 5-Fold Cross Validation.

Two protocols x four methods x five folds = board-identity-aware AD evaluation
matching the detection benchmark.

  Full-image : train on non_missing/fold_k/train, test on missing_only/fold_k/valid
               (box-level mAP, F1, FNR)
  Crop-level : train on present-component crops from non_missing/fold_k/train,
               test on missing-component crops from missing_only/fold_k/valid
               (crop-level AUROC, F1, FNR)

Ported from models/anomaly_detection/notebooks/8_Anomaly_Detection_v4_fixed.ipynb.
"""
import argparse
import gc
import json
import os
import random
import shutil
import time
import warnings
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torchvision.transforms as T
from PIL import Image
from sklearn.metrics import f1_score, recall_score, roc_auc_score
from torch.utils.data import DataLoader, Dataset

warnings.filterwarnings("ignore")

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

METHOD_DISPLAY = {
    "PatchCore": "PatchCore~\\cite{roth2022patchcore}",
    "PaDiM": "PaDiM~\\cite{defard2021padim}",
    "DRAEM": "DRAEM~\\cite{zavrtanik2021draemdiscriminativelytrained}",
    "ReverseDistillation": "Rev.\\ Distill.~\\cite{deng2022reverse}",
}


@dataclass
class Cfg:
    image_size: int = 256
    crop_size: int = 224
    context_factor: float = 1.4
    batch_size: int = 8
    num_workers: int = 2
    seed: int = 42
    epochs_patchcore: int = 1
    epochs_padim: int = 1
    epochs_draem: int = 50
    epochs_rd: int = 50
    thr_quantile: float = 0.995
    min_box_area: int = 24
    # PatchCore memory bank limit: cap training crops to avoid OOM.
    # PatchCore stores embeddings for ALL training patches in GPU memory.
    # With 117k crops this exceeds even 96GB GPUs. 10k is sufficient for
    # a representative memory bank (PatchCore subsamples via coreset anyway).
    max_train_crops_patchcore: int = 10000
    smoke_test: bool = False


def clear_gpu():
    """Aggressively free GPU memory between runs."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def register_safe_globals():
    # Anomalib checkpoints contain custom enum types that torch.load
    # rejects under weights_only=True (the new default in PyTorch 2.6).
    # We allowlist them so load_from_checkpoint works.
    try:
        import anomalib

        safe_types = []
        for attr_name in dir(anomalib):
            obj = getattr(anomalib, attr_name, None)
            if isinstance(obj, type):
                safe_types.append(obj)
        try:
            from anomalib.models.image.reverse_distillation.anomaly_map import (
                AnomalyMapGenerationMode,
            )

            safe_types.append(AnomalyMapGenerationMode)
        except ImportError:
            pass
        if safe_types:
            torch.serialization.add_safe_globals(safe_types)
            print(f"Registered {len(safe_types)} Anomalib safe globals for torch.load")
    except Exception as e:
        print(f"Warning: could not register safe globals: {e}")


def fold_paths(data_root, fold_idx):
    nm = os.path.join(data_root, "non_missing", "kfold_data", f"fold_{fold_idx}")
    mo = os.path.join(data_root, "missing_only", "kfold_data", f"fold_{fold_idx}")
    return {
        "nm_train_img": os.path.join(nm, "train", "images"),
        "nm_train_lab": os.path.join(nm, "train", "labels"),
        "nm_val_img": os.path.join(nm, "valid", "images"),
        "nm_val_lab": os.path.join(nm, "valid", "labels"),
        "mo_train_img": os.path.join(mo, "train", "images"),
        "mo_train_lab": os.path.join(mo, "train", "labels"),
        "mo_val_img": os.path.join(mo, "valid", "images"),
        "mo_val_lab": os.path.join(mo, "valid", "labels"),
    }


# ============================================================
# Helpers
# ============================================================
def list_images(folder):
    return sorted(os.path.join(folder, f) for f in os.listdir(folder) if f.lower().endswith(IMG_EXTS))


def label_path_from_image(image_path, labels_dir):
    return os.path.join(labels_dir, os.path.splitext(os.path.basename(image_path))[0] + ".txt")


def read_yolo_labels(label_path):
    if not os.path.exists(label_path):
        return []
    rows = []
    with open(label_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            rows.append((int(float(parts[0])), *map(float, parts[1:5])))
    return rows


def yolo_to_xyxy(xc, yc, w, h, W, H):
    x1 = int(max(0, (xc - w / 2) * W))
    y1 = int(max(0, (yc - h / 2) * H))
    x2 = int(min(W - 1, (xc + w / 2) * W))
    y2 = int(min(H - 1, (yc + h / 2) * H))
    if x2 <= x1:
        x2 = min(W - 1, x1 + 1)
    if y2 <= y1:
        y2 = min(H - 1, y1 + 1)
    return x1, y1, x2, y2


def expand_box(x1, y1, x2, y2, W, H, factor=1.4):
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    bw, bh = (x2 - x1) * factor, (y2 - y1) * factor
    return int(max(0, cx - bw / 2)), int(max(0, cy - bh / 2)), int(min(W, cx + bw / 2)), int(min(H, cy + bh / 2))


def iou_xyxy(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    ua = max(0, a[2] - a[0]) * max(0, a[3] - a[1])
    ub = max(0, b[2] - b[0]) * max(0, b[3] - b[1])
    return float(inter / (ua + ub - inter + 1e-9))


def nms(boxes, iou_thr=0.3):
    keep = []
    for b in sorted(boxes, key=lambda x: x[4], reverse=True):
        if all(iou_xyxy(b[:4], k[:4]) < iou_thr for k in keep):
            keep.append(b)
    return keep


def heatmap_to_boxes(heatmap, thr, min_area=24):
    mask = (heatmap.astype(np.float32) >= thr).astype(np.uint8) * 255
    if mask.sum() == 0:
        return []
    k = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(cv2.morphologyEx(mask, cv2.MORPH_OPEN, k), cv2.MORPH_CLOSE, k)
    boxes = []
    for c in cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]:
        x, y, w, h = cv2.boundingRect(c)
        if w * h >= min_area:
            boxes.append((x, y, x + w, y + h, float(heatmap[y : y + h, x : x + w].mean())))
    return boxes


def compute_ap_11pt(scored_tp, total_gt):
    if total_gt == 0:
        return 0.0
    scored_tp = sorted(scored_tp, key=lambda x: x[0], reverse=True)
    tp = fp = 0
    precs = []
    recs = []
    for s, is_tp in scored_tp:
        tp += int(is_tp)
        fp += int(1 - is_tp)
        precs.append(tp / max(1, tp + fp))
        recs.append(tp / total_gt)
    return float(
        sum(max([p for p, r in zip(precs, recs) if r >= rt] + [0.0]) for rt in np.linspace(0, 1, 11)) / 11.0
    )


def get_gt_missing_boxes(img_path, labels_dir):
    labs = read_yolo_labels(label_path_from_image(img_path, labels_dir))
    W, H = Image.open(img_path).convert("RGB").size
    return [yolo_to_xyxy(xc, yc, w, h, W, H) for _, xc, yc, w, h in labs]


# ============================================================
# Anomalib model registry
# ============================================================
def get_method(name, cfg):
    from anomalib.models import Padim, Patchcore

    try:
        from anomalib.models import Draem
    except ImportError:
        from anomalib.models.image.draem import Draem

    try:
        from anomalib.models import ReverseDistillation
    except ImportError:
        from anomalib.models.image.reverse_distillation import ReverseDistillation

    smoke = cfg.smoke_test
    reg = {
        "PatchCore": (Patchcore, cfg.epochs_patchcore),
        "PaDiM": (Padim, cfg.epochs_padim),
        "DRAEM": (Draem, cfg.epochs_draem if not smoke else 1),
        "ReverseDistillation": (ReverseDistillation, cfg.epochs_rd if not smoke else 1),
    }
    cls, ep = reg[name]
    return cls(), ep


def make_datamodule(root_dir, cfg):
    from anomalib.data import Folder

    return Folder(
        name="PCB_MC_AD",
        root=root_dir,
        normal_dir=os.path.join("train", "normal"),
        abnormal_dir=os.path.join("test", "abnormal"),
        train_batch_size=cfg.batch_size,
        eval_batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        seed=cfg.seed,
    )


# ============================================================
# Per-fold data preparation
# ============================================================
def symlink_many(src_paths, dst_dir):
    os.makedirs(dst_dir, exist_ok=True)
    for src in src_paths:
        dst = os.path.join(dst_dir, os.path.basename(src))
        if not os.path.exists(dst):
            os.symlink(src, dst)


def prepare_fullimage_folder(data_root, work_root, fold_idx, cfg):
    fp = fold_paths(data_root, fold_idx)
    root = os.path.join(work_root, f"fold_{fold_idx}", "fullimg")
    if os.path.exists(root):
        shutil.rmtree(root)
    train = list_images(fp["nm_train_img"])
    val = list_images(fp["nm_val_img"])
    test = list_images(fp["mo_val_img"])
    if cfg.smoke_test:
        train = train[:20]
        val = val[:10]
        test = test[:10]
    symlink_many(train, os.path.join(root, "train", "normal"))
    symlink_many(val, os.path.join(root, "test", "normal"))
    symlink_many(test, os.path.join(root, "test", "abnormal"))
    return root


def extract_crops(image_dir, labels_dir, output_dir, context_factor=1.4, max_crops=None):
    os.makedirs(output_dir, exist_ok=True)
    count = 0
    for img_path in list_images(image_dir):
        labels = read_yolo_labels(label_path_from_image(img_path, labels_dir))
        if not labels:
            continue
        img = cv2.imread(img_path)
        if img is None:
            continue
        H, W = img.shape[:2]
        stem = os.path.splitext(os.path.basename(img_path))[0]
        for i, (cls, xc, yc, w, h) in enumerate(labels):
            x1, y1, x2, y2 = yolo_to_xyxy(xc, yc, w, h, W, H)
            x1e, y1e, x2e, y2e = expand_box(x1, y1, x2, y2, W, H, context_factor)
            crop = img[y1e:y2e, x1e:x2e]
            if crop.size == 0:
                continue
            cv2.imwrite(os.path.join(output_dir, f"{stem}_c{i:04d}.png"), crop)
            count += 1
            if max_crops and count >= max_crops:
                return count
    return count


def prepare_crop_folder(data_root, work_root, fold_idx, cfg):
    fp = fold_paths(data_root, fold_idx)
    root = os.path.join(work_root, f"fold_{fold_idx}", "crops")
    if os.path.exists(root):
        shutil.rmtree(root)
    max_c = 500 if cfg.smoke_test else None

    n_train = extract_crops(
        fp["nm_train_img"], fp["nm_train_lab"], os.path.join(root, "train", "normal"), cfg.context_factor, max_c
    )
    # Count for reporting; actual subsampling for PatchCore happens at train time
    val_normal_dir = os.path.join(root, "val_normal_raw")
    n_val = extract_crops(fp["nm_val_img"], fp["nm_val_lab"], val_normal_dir, cfg.context_factor, max_c)
    test_normal = os.path.join(root, "test", "normal")
    os.makedirs(test_normal, exist_ok=True)
    for f in os.listdir(val_normal_dir):
        src = os.path.join(val_normal_dir, f)
        dst = os.path.join(test_normal, f)
        if not os.path.exists(dst):
            os.symlink(src, dst)
    n_anom = extract_crops(
        fp["mo_val_img"], fp["mo_val_lab"], os.path.join(root, "test", "abnormal"), cfg.context_factor, max_c
    )
    print(f"  Fold {fold_idx} crops: {n_train} train, {n_val} val-normal, {n_anom} test-anomaly")
    return root, val_normal_dir


# ============================================================
# Training and inference (with memory management + checkpoint fix)
# ============================================================
def train_method(name, data_root_dir, run_dir, cfg, device):
    from anomalib.engine import Engine

    clear_gpu()  # Free memory before each training run
    model, max_epochs = get_method(name, cfg)
    os.makedirs(run_dir, exist_ok=True)

    # For PatchCore on crop-level: subsample training images to avoid OOM
    # when building the memory bank. We create a temp dir with symlinks.
    actual_root = data_root_dir
    if name == "PatchCore":
        train_dir = os.path.join(data_root_dir, "train", "normal")
        if os.path.exists(train_dir):
            all_imgs = sorted([f for f in os.listdir(train_dir) if f.lower().endswith(IMG_EXTS)])
            if len(all_imgs) > cfg.max_train_crops_patchcore:
                print(f"  PatchCore: subsampling {len(all_imgs)} -> {cfg.max_train_crops_patchcore} train crops")
                sub_root = os.path.join(run_dir, "_subsampled")
                sub_train = os.path.join(sub_root, "train", "normal")
                if os.path.exists(sub_root):
                    shutil.rmtree(sub_root)
                os.makedirs(sub_train, exist_ok=True)
                test_dir = os.path.join(data_root_dir, "test")
                sub_test = os.path.join(sub_root, "test")
                if os.path.exists(test_dir):
                    if os.path.exists(sub_test):
                        shutil.rmtree(sub_test)
                    os.symlink(os.path.abspath(test_dir), sub_test)
                rng = random.Random(cfg.seed + 99)
                sampled = rng.sample(all_imgs, cfg.max_train_crops_patchcore)
                for f in sampled:
                    src = os.path.join(train_dir, f)
                    dst = os.path.join(sub_train, f)
                    if not os.path.exists(dst):
                        os.symlink(os.path.abspath(src), dst)
                actual_root = sub_root

    engine = Engine(
        default_root_dir=run_dir,
        accelerator="gpu" if device == "cuda" else "cpu",
        devices=1,
        max_epochs=max_epochs,
        enable_progress_bar=False,
    )
    t0 = time.time()
    engine.fit(model=model, datamodule=make_datamodule(actual_root, cfg))
    elapsed = time.time() - t0

    ckpt = None
    if engine.checkpoint_callback is not None:
        ckpt = engine.checkpoint_callback.best_model_path
    if not ckpt or not os.path.exists(str(ckpt)):
        for p in Path(run_dir).rglob("*.ckpt"):
            ckpt = str(p)
            break

    del model, engine
    clear_gpu()

    return str(ckpt) if ckpt else "", elapsed


class SimpleImageDataset(Dataset):
    def __init__(self, paths, image_size):
        self.paths = paths
        self.tf = T.Compose([T.Resize((image_size, image_size)), T.ToTensor()])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        p = self.paths[idx]
        return {"path": p, "image": self.tf(Image.open(p).convert("RGB"))}


def _extract_amap_score(pred):
    if isinstance(pred, dict):
        amap = pred.get("anomaly_map", pred.get("anomaly_maps"))
        score = pred.get("pred_score", pred.get("pred_scores"))
    else:
        amap = getattr(pred, "anomaly_map", getattr(pred, "anomaly_maps", None))
        score = getattr(pred, "pred_score", getattr(pred, "pred_scores", None))
    return amap, score


@torch.no_grad()
def predict_anomaly_maps(method_name, ckpt_path, image_paths, image_size, cfg, device):
    clear_gpu()
    model_cls = get_method(method_name, cfg)[0].__class__

    # Fix for PyTorch 2.6+: force weights_only=False if safe_globals didn't work
    try:
        model = model_cls.load_from_checkpoint(ckpt_path)
    except Exception:
        ckpt_data = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model = model_cls()
        if "state_dict" in ckpt_data:
            model.load_state_dict(ckpt_data["state_dict"], strict=False)
        else:
            model.load_state_dict(ckpt_data, strict=False)

    model.eval().to(device)

    ds = SimpleImageDataset(image_paths, image_size)
    dl = DataLoader(ds, batch_size=cfg.batch_size, shuffle=False, num_workers=0)
    out = {}
    for batch in dl:
        x = batch["image"].to(device)
        paths = batch["path"]
        pred = model(x)
        amap, score = _extract_amap_score(pred)
        if amap is not None and amap.ndim == 4:
            amap = amap[:, 0]
        if score is None and amap is not None:
            score = amap.flatten(1).max(dim=1).values
        amap_np = amap.detach().cpu().numpy() if amap is not None else np.zeros((len(paths), image_size, image_size))
        score_np = score.detach().cpu().numpy() if score is not None else np.zeros(len(paths))
        for i, p in enumerate(paths):
            W, H = Image.open(p).convert("RGB").size
            out[p] = (W, H, amap_np[i].astype(np.float32), float(score_np[i]))

    del model
    clear_gpu()
    return out


def estimate_thr_from_normals(results_normal, q):
    vals = []
    for (_, _, amap, _) in results_normal.values():
        flat = amap.reshape(-1)
        if flat.size > 20000:
            flat = np.random.choice(flat, 20000, replace=False)
        vals.append(flat)
    return float(np.quantile(np.concatenate(vals), q)) if vals else 0.0


# ============================================================
# Evaluation functions
# ============================================================
def eval_fullimage(results_test, test_img_paths, labels_dir, thr, cfg):
    total_gt = tp = fp = matched_gt = 0
    scored_tp = []
    for img_path in test_img_paths:
        if img_path not in results_test:
            continue
        W, H, amap, _ = results_test[img_path]
        Ha, Wa = amap.shape
        sx, sy = W / Wa, H / Ha
        preds = [
            (int(x1 * sx), int(y1 * sy), int(x2 * sx), int(y2 * sy), s)
            for x1, y1, x2, y2, s in heatmap_to_boxes(amap, thr, cfg.min_box_area)
        ]
        preds = nms(preds, 0.3)
        gt = get_gt_missing_boxes(img_path, labels_dir)
        total_gt += len(gt)
        gt_used = [False] * len(gt)
        for x1, y1, x2, y2, score in preds:
            bi, biou = -1, 0.0
            for i, g in enumerate(gt):
                if gt_used[i]:
                    continue
                v = iou_xyxy((x1, y1, x2, y2), g)
                if v > biou:
                    bi, biou = i, v
            if bi >= 0 and biou >= 0.5:
                gt_used[bi] = True
                tp += 1
                scored_tp.append((score, 1))
            else:
                fp += 1
                scored_tp.append((score, 0))
        matched_gt += sum(gt_used)
    prec = tp / max(1, tp + fp)
    rec = matched_gt / max(1, total_gt)
    f1 = (2 * prec * rec) / max(1e-9, prec + rec)
    return {
        "mAP": round(compute_ap_11pt(scored_tp, total_gt), 4),
        "f1": round(f1, 4),
        "fnr": round(1 - rec, 4),
    }


def eval_crop_level(method_name, ckpt_path, val_normal_dir, test_anomaly_dir, cfg, device):
    normal_paths = sorted(
        [os.path.join(val_normal_dir, f) for f in os.listdir(val_normal_dir) if f.lower().endswith(IMG_EXTS)]
    )
    anomaly_paths = sorted(
        [os.path.join(test_anomaly_dir, f) for f in os.listdir(test_anomaly_dir) if f.lower().endswith(IMG_EXTS)]
    )
    if cfg.smoke_test:
        normal_paths = normal_paths[:200]
        anomaly_paths = anomaly_paths[:200]
    print(f"  Crop eval: {len(normal_paths)} normal, {len(anomaly_paths)} anomaly")

    all_paths = normal_paths + anomaly_paths
    labels = np.array([0] * len(normal_paths) + [1] * len(anomaly_paths))
    results = predict_anomaly_maps(method_name, ckpt_path, all_paths, cfg.crop_size, cfg, device)
    scores = np.array([results[p][3] for p in all_paths if p in results])
    vlabels = np.array([l for p, l in zip(all_paths, labels) if p in results])

    auroc = roc_auc_score(vlabels, scores) if len(np.unique(vlabels)) >= 2 else 0.0
    best_f1, best_thr = 0.0, 0.0
    for q in np.linspace(0.01, 0.99, 200):
        thr = np.quantile(scores, q)
        f = f1_score(vlabels, (scores >= thr).astype(int), zero_division=0)
        if f > best_f1:
            best_f1, best_thr = f, thr
    rec = recall_score(vlabels, (scores >= best_thr).astype(int), zero_division=0)
    return {
        "AUROC": round(float(auroc), 4),
        "f1": round(float(best_f1), 4),
        "fnr": round(float(1 - rec), 4),
        "n_normal": int(sum(vlabels == 0)),
        "n_anomaly": int(sum(vlabels == 1)),
    }


# ============================================================
# EXPERIMENT 1: Full-image AD
# ============================================================
def run_fullimage(data_root, work_root, folds, method_names, cfg, device):
    fullimg_per_fold = {n: [] for n in method_names}
    fullimg_heatmaps_fold0 = {}

    print("=" * 60)
    print("EXPERIMENT 1: FULL-IMAGE AD")
    print("=" * 60)
    for fold_idx in folds:
        clear_gpu()
        print(f"\n{'='*40} FOLD {fold_idx} {'='*40}")
        fp = fold_paths(data_root, fold_idx)
        fi_root = prepare_fullimage_folder(data_root, work_root, fold_idx, cfg)
        val_paths = list_images(fp["nm_val_img"])
        test_paths = list_images(fp["mo_val_img"])
        if cfg.smoke_test:
            val_paths = val_paths[:10]
            test_paths = test_paths[:10]

        for name in method_names:
            try:
                run_dir = os.path.join(work_root, f"fold_{fold_idx}", f"fi_{name.lower()}")
                ckpt, t = train_method(name, fi_root, run_dir, cfg, device)
                print(f"  {name}: {t:.1f}s")
                val_pred = predict_anomaly_maps(name, ckpt, val_paths, cfg.image_size, cfg, device)
                thr = estimate_thr_from_normals(val_pred, cfg.thr_quantile)
                test_pred = predict_anomaly_maps(name, ckpt, test_paths, cfg.image_size, cfg, device)
                if fold_idx == 0:
                    fullimg_heatmaps_fold0[name] = test_pred
                metrics = eval_fullimage(test_pred, test_paths, fp["mo_val_lab"], thr, cfg)
                metrics["fold"] = fold_idx
                metrics["train_s"] = round(t, 1)
                fullimg_per_fold[name].append(metrics)
                print(f"    mAP={metrics['mAP']} F1={metrics['f1']} FNR={metrics['fnr']}")
            except Exception as e:
                print(f"    [ERROR] {name}: {e}")
                import traceback

                traceback.print_exc()

    return fullimg_per_fold, fullimg_heatmaps_fold0


# ============================================================
# EXPERIMENT 2: Crop-level AD
# ============================================================
def run_croplevel(data_root, work_root, folds, method_names, cfg, device):
    crop_per_fold = {n: [] for n in method_names}

    print("=" * 60)
    print("EXPERIMENT 2: CROP-LEVEL AD")
    print("=" * 60)
    for fold_idx in folds:
        clear_gpu()
        print(f"\n{'='*40} FOLD {fold_idx} {'='*40}")
        crop_root, val_normal_dir = prepare_crop_folder(data_root, work_root, fold_idx, cfg)
        test_anomaly_dir = os.path.join(crop_root, "test", "abnormal")

        for name in method_names:
            try:
                run_dir = os.path.join(work_root, f"fold_{fold_idx}", f"cr_{name.lower()}")
                ckpt, t = train_method(name, crop_root, run_dir, cfg, device)
                print(f"  {name}: {t:.1f}s")
                metrics = eval_crop_level(name, ckpt, val_normal_dir, test_anomaly_dir, cfg, device)
                metrics["fold"] = fold_idx
                metrics["train_s"] = round(t, 1)
                crop_per_fold[name].append(metrics)
                print(f"    AUROC={metrics['AUROC']} F1={metrics['f1']} FNR={metrics['fnr']}")
            except Exception as e:
                print(f"    [ERROR] {name}: {e}")
                import traceback

                traceback.print_exc()

    return crop_per_fold


# ============================================================
# Aggregate + LaTeX + save
# ============================================================
def aggregate(per_fold, keys):
    rows = []
    for name, folds in per_fold.items():
        if not folds:
            continue
        row = {"Method": name}
        for k in keys:
            vals = [r[k] for r in folds if k in r]
            row[f"{k}_mean"] = np.mean(vals) if vals else 0.0
            row[f"{k}_std"] = np.std(vals) if vals else 0.0
        rows.append(row)
    return pd.DataFrame(rows)


def fmt_ms(m, s):
    return f"{m:.2f} ± {s:.2f}"


def fmt_si(m, s):
    return f"{m:.2f}({int(s*100):02d})"


def save_results(fullimg_per_fold, crop_per_fold, output_dir):
    df_fi = aggregate(fullimg_per_fold, ["mAP", "f1", "fnr"])
    df_cr = aggregate(crop_per_fold, ["AUROC", "f1", "fnr"])

    if not df_fi.empty:
        print("\n=== FULL-IMAGE AD (mean ± std) ===")
        for _, r in df_fi.iterrows():
            print(
                f"  {r['Method']:22s}  mAP={fmt_ms(r['mAP_mean'],r['mAP_std'])}  "
                f"F1={fmt_ms(r['f1_mean'],r['f1_std'])}  FNR={fmt_ms(r['fnr_mean'],r['fnr_std'])}"
            )

    if not df_cr.empty:
        print("\n=== CROP-LEVEL AD (mean ± std) ===")
        for _, r in df_cr.iterrows():
            print(
                f"  {r['Method']:22s}  AUROC={fmt_ms(r['AUROC_mean'],r['AUROC_std'])}  "
                f"F1={fmt_ms(r['f1_mean'],r['f1_std'])}  FNR={fmt_ms(r['fnr_mean'],r['fnr_std'])}"
            )

    # --- LaTeX ---
    L = []
    if not df_fi.empty:
        L.append("% === Full-image AD rows for Table III (PCB-MC-M) ===")
        L.append("\\midrule")
        L.append("\\multicolumn{5}{l}{\\textit{Anomaly Detection -- full-image (unsupervised)}} \\\\")
        L.append("\\midrule")
        for _, r in df_fi.iterrows():
            d = METHOD_DISPLAY.get(r["Method"], r["Method"])
            L.append(
                f"{d} & PCB-MC-M & {fmt_si(r['mAP_mean'],r['mAP_std'])} & "
                f"{fmt_si(r['f1_mean'],r['f1_std'])} & {fmt_si(r['fnr_mean'],r['fnr_std'])} \\\\"
            )

    if not df_cr.empty:
        L.append("")
        L.append("% === Crop-level AD table (standalone) ===")
        L.append("\\begin{table}[t]")
        L.append("\\caption{Crop-level anomaly detection on component patches")
        L.append("($1.4\\times$ context). Mean $\\pm$ std across 5 folds.}")
        L.append("\\label{tab:ad_crop}")
        L.append("\\centering\\small")
        L.append("\\begin{tabular}{l S[table-format=1.2(2)] S[table-format=1.2(2)] S[table-format=1.2(2)]}")
        L.append("\\toprule")
        L.append("Method & {AUROC} & {F1} & {FNR} \\\\")
        L.append("\\midrule")
        for _, r in df_cr.iterrows():
            d = METHOD_DISPLAY.get(r["Method"], r["Method"])
            L.append(
                f"{d} & {fmt_si(r['AUROC_mean'],r['AUROC_std'])} & "
                f"{fmt_si(r['f1_mean'],r['f1_std'])} & {fmt_si(r['fnr_mean'],r['fnr_std'])} \\\\"
            )
        L.append("\\bottomrule")
        L.append("\\end{tabular}")
        L.append("\\end{table}")

    latex = "\n".join(L)
    out_tex = os.path.join(output_dir, "results_ad_5fold.tex")
    with open(out_tex, "w") as f:
        f.write(latex)
    print("\n=== LaTeX ===")
    print(latex)
    print(f"\nSaved: {out_tex}")

    all_res = {
        "full_image": {n: f for n, f in fullimg_per_fold.items() if f},
        "crop_level": {n: f for n, f in crop_per_fold.items() if f},
    }
    out_json = os.path.join(output_dir, "results_ad_5fold.json")
    with open(out_json, "w") as f:
        json.dump(all_res, f, indent=2)
    print(f"Saved: {out_json}")


# ============================================================
# Qualitative: full-image heatmaps (fold 0)
# ============================================================
def normalize_heatmap(a):
    a = a - a.min()
    return a / a.max() if a.max() > 0 else a


def plot_qualitative_fullimage(data_root, fullimg_heatmaps_fold0, output_dir):
    import matplotlib.pyplot as plt

    fp0 = fold_paths(data_root, 0)
    if not fullimg_heatmaps_fold0:
        print("No heatmaps cached.")
        return
    test_f0 = list_images(fp0["mo_val_img"])
    cands = [
        p
        for p in test_f0
        if len(get_gt_missing_boxes(p, fp0["mo_val_lab"])) > 0
        and all(p in fullimg_heatmaps_fold0[m] for m in fullimg_heatmaps_fold0)
    ]
    random.seed(7)
    random.shuffle(cands)
    demo = cands[:3]
    if not demo:
        return
    methods = list(fullimg_heatmaps_fold0.keys())
    nr, nc = len(demo), 2 + len(methods)
    fig, axes = plt.subplots(nr, nc, figsize=(2.6 * nc, 2.6 * nr))
    if nr == 1:
        axes = axes[None, :]
    for r, ip in enumerate(demo):
        img = np.array(Image.open(ip).convert("RGB"))
        gt = get_gt_missing_boxes(ip, fp0["mo_val_lab"])
        axes[r, 0].imshow(img)
        axes[r, 0].set_title("Input" if r == 0 else "")
        axes[r, 0].axis("off")
        vis = img.copy()
        for x1, y1, x2, y2 in gt:
            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 3)
        axes[r, 1].imshow(vis)
        axes[r, 1].set_title("GT" if r == 0 else "")
        axes[r, 1].axis("off")
        for j, m in enumerate(methods):
            W, H, am, _ = fullimg_heatmaps_fold0[m][ip]
            amr = cv2.resize(am, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_LINEAR)
            ax = axes[r, 2 + j]
            ax.imshow(img)
            ax.imshow(normalize_heatmap(amr), cmap="jet", alpha=0.55)
            for x1, y1, x2, y2 in gt:
                ax.add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor="lime", lw=2))
            ax.set_title(m if r == 0 else "")
            ax.axis("off")
    plt.tight_layout()
    out_pdf = os.path.join(output_dir, "qualitative_fullimage_5fold.pdf")
    plt.savefig(out_pdf, dpi=200, bbox_inches="tight")
    plt.savefig(out_pdf.replace(".pdf", ".png"), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_pdf}")


# ============================================================
# Score distribution plot (crop-level, fold 0)
# ============================================================
def plot_score_distributions(work_root, crop_per_fold, method_names, cfg, device, output_dir):
    import matplotlib.pyplot as plt

    if not (crop_per_fold and any(crop_per_fold.values())):
        print("No crop data.")
        return
    crop_root_f0 = os.path.join(work_root, "fold_0", "crops")
    vnf0 = os.path.join(crop_root_f0, "val_normal_raw")
    taf0 = os.path.join(crop_root_f0, "test", "abnormal")
    if not (os.path.exists(vnf0) and os.path.exists(taf0)):
        return
    np_ = sorted([os.path.join(vnf0, f) for f in os.listdir(vnf0) if f.lower().endswith(IMG_EXTS)])
    ap_ = sorted([os.path.join(taf0, f) for f in os.listdir(taf0) if f.lower().endswith(IMG_EXTS)])
    ma = [n for n in method_names if list(Path(os.path.join(work_root, "fold_0", f"cr_{n.lower()}")).rglob("*.ckpt"))]
    if not ma:
        return
    fig, axes = plt.subplots(1, len(ma), figsize=(4.5 * len(ma), 3.5))
    if len(ma) == 1:
        axes = [axes]
    for ax, name in zip(axes, ma):
        ckpt = str(list(Path(os.path.join(work_root, "fold_0", f"cr_{name.lower()}")).rglob("*.ckpt"))[0])
        preds = predict_anomaly_maps(name, ckpt, np_ + ap_, cfg.crop_size, cfg, device)
        sn = [preds[p][3] for p in np_ if p in preds]
        sa = [preds[p][3] for p in ap_ if p in preds]
        ax.hist(sn, bins=50, alpha=0.6, label="Normal (present)", color="steelblue", density=True)
        ax.hist(sa, bins=50, alpha=0.6, label="Anomaly (missing)", color="coral", density=True)
        ax.set_title(name, fontsize=10)
        ax.set_xlabel("Anomaly score")
        ax.legend(fontsize=8)
    plt.suptitle("Score distributions: Normal vs Missing crops (fold 0)", fontsize=11)
    plt.tight_layout()
    out_pdf = os.path.join(output_dir, "score_distributions_5fold.pdf")
    plt.savefig(out_pdf, dpi=200, bbox_inches="tight")
    plt.savefig(out_pdf.replace(".pdf", ".png"), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_pdf}")


def parse_args():
    p = argparse.ArgumentParser(description="PCB-MC Anomaly Detection Benchmark (5-fold CV, Anomalib)")
    p.add_argument("--data-root", required=True, help="Root containing non_missing/ and missing_only/ kfold_data/")
    p.add_argument("--output-dir", required=True, help="Where results_ad_5fold.{json,tex} and figures are saved")
    p.add_argument("--work-dir", default="./pcbmc_anom_cv", help="Scratch dir for per-fold symlink trees + checkpoints")
    p.add_argument("--protocol", choices=["full_image", "crop_level", "both"], default="both")
    p.add_argument(
        "--methods",
        nargs="+",
        choices=list(METHOD_DISPLAY.keys()),
        default=list(METHOD_DISPLAY.keys()),
    )
    p.add_argument("--folds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    p.add_argument("--smoke-test", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--image-size", type=int, default=256)
    p.add_argument("--crop-size", type=int, default=224)
    p.add_argument("--context-factor", type=float, default=1.4)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--thr-quantile", type=float, default=0.995)
    p.add_argument("--min-box-area", type=int, default=24)
    p.add_argument("--max-train-crops-patchcore", type=int, default=10000)
    return p.parse_args()


def main():
    args = parse_args()

    print(f"torch:           {torch.__version__}")
    print(f"cuda available:  {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"gpu:             {torch.cuda.get_device_name(0)}")
        mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"gpu memory:      {mem:.1f} GB")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    register_safe_globals()

    cfg = Cfg(
        image_size=args.image_size,
        crop_size=args.crop_size,
        context_factor=args.context_factor,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
        thr_quantile=args.thr_quantile,
        min_box_area=args.min_box_area,
        max_train_crops_patchcore=args.max_train_crops_patchcore,
        smoke_test=args.smoke_test,
    )

    fp0 = fold_paths(args.data_root, args.folds[0])
    for key, p in fp0.items():
        assert os.path.exists(p), f"Fold {args.folds[0]} missing: {key} -> {p}"
    print(f"Fold {args.folds[0]} paths validated.")

    os.makedirs(args.work_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Smoke test:  {cfg.smoke_test}")
    print(f"Methods:     {args.methods}")
    print(f"Protocol:    {args.protocol}")
    print(f"Folds:       {args.folds}")

    fullimg_per_fold, fullimg_heatmaps_fold0 = {}, {}
    crop_per_fold = {}

    if args.protocol in ("full_image", "both"):
        fullimg_per_fold, fullimg_heatmaps_fold0 = run_fullimage(
            args.data_root, args.work_dir, args.folds, args.methods, cfg, device
        )
    if args.protocol in ("crop_level", "both"):
        crop_per_fold = run_croplevel(args.data_root, args.work_dir, args.folds, args.methods, cfg, device)

    save_results(fullimg_per_fold, crop_per_fold, args.output_dir)

    if fullimg_heatmaps_fold0:
        plot_qualitative_fullimage(args.data_root, fullimg_heatmaps_fold0, args.output_dir)
    if crop_per_fold:
        plot_score_distributions(args.work_dir, crop_per_fold, args.methods, cfg, device, args.output_dir)


if __name__ == "__main__":
    main()
