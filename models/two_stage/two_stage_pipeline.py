"""Fold-aware two-stage missing-component detection: YOLO proposals + patch classifier.

Ports models/two_stage/notebooks/9_10_PatchClassifier_TwoStage.ipynb's pipeline/eval
sections. Stage 1 (YOLOv11 `full_dataset` fold checkpoint, conf=0.05) proposes candidate
boxes ignoring class label; Stage 2 (that fold's ResNet-18 patch classifier from
train_patch_classifier.py) re-scores each proposal crop as P(missing). Running the
classifier only on Stage-1 proposals — never on raw sliding-window background patches —
is what fixes the false-positive explosion (3903 FP / 45 GT) of the earlier sliding-window
attempt; the classifier only ever has to disambiguate "present" vs "empty" footprint.
"""
import argparse
import json
import os

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T
import yaml
from PIL import Image
from torchvision import models
from tqdm import tqdm
from ultralytics import YOLO

IMG_EXTS = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')

DEFAULTS = dict(
    stage1_conf=0.05,
    stage1_iou=0.45,
    stage1_imgsz=1024,
    clf_thr=0.30,
    nms_iou=0.30,
    context_factor=1.4,
    crop_size=128,
)


def list_images(folder):
    if not os.path.exists(folder):
        return []
    return sorted(
        os.path.join(folder, f) for f in os.listdir(folder)
        if f.lower().endswith(IMG_EXTS)
    )


def read_labels(img_path):
    lp = img_path.replace('/images/', '/labels/').rsplit('.', 1)[0] + '.txt'
    if not os.path.exists(lp):
        return []
    rows = []
    with open(lp) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            rows.append((int(float(parts[0])), *map(float, parts[1:5])))
    return rows


def yolo_to_xyxy(xc, yc, w, h, W, H):
    x1 = int(max(0, (xc - w / 2) * W)); y1 = int(max(0, (yc - h / 2) * H))
    x2 = int(min(W - 1, (xc + w / 2) * W)); y2 = int(min(H - 1, (yc + h / 2) * H))
    if x2 <= x1: x2 = min(W - 1, x1 + 1)
    if y2 <= y1: y2 = min(H - 1, y1 + 1)
    return x1, y1, x2, y2


def expand_bbox(x1, y1, x2, y2, factor, W, H):
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    hw = (x2 - x1) / 2 * factor; hh = (y2 - y1) / 2 * factor
    return (
        int(max(0, cx - hw)), int(max(0, cy - hh)),
        int(min(W - 1, cx + hw)), int(min(H - 1, cy + hh)),
    )


def iou_xyxy(a, b):
    iw = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    ih = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = iw * ih
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return float(inter / max(union, 1e-9))


def nms(boxes, iou_thr):
    keep = []
    for b in sorted(boxes, key=lambda x: x[4], reverse=True):
        if all(iou_xyxy(b[:4], k[:4]) < iou_thr for k in keep):
            keep.append(b)
    return keep


def compute_ap_11pt(scored_tp, total_gt):
    if total_gt == 0:
        return 0.0
    scored_tp = sorted(scored_tp, key=lambda x: x[0], reverse=True)
    tp = fp = 0; precs, recs = [], []
    for s, is_tp in scored_tp:
        tp += int(is_tp); fp += int(1 - is_tp)
        precs.append(tp / max(1, tp + fp))
        recs.append(tp / total_gt)
    return float(sum(
        max([p for p, r in zip(precs, recs) if r >= t] + [0.0])
        for t in np.linspace(0, 1, 11)
    ) / 11.0)


def fold_val_images(data_root, subset, fold):
    d = os.path.join(data_root, subset, 'kfold_data', f'fold_{fold}', 'valid', 'images')
    return list_images(d)


def load_class_ids(data_root, subset):
    yaml_path = os.path.join(data_root, subset, 'kfold_data', 'fold_0', 'data.yaml')
    with open(yaml_path) as f:
        cfg = yaml.safe_load(f)
    class_names = cfg['names']
    return frozenset(i for i, n in enumerate(class_names) if n.lower().startswith('missing'))


def build_classifier():
    m = models.resnet18(weights=None)
    m.fc = nn.Sequential(
        nn.Linear(m.fc.in_features, 256), nn.ReLU(),
        nn.Dropout(0.3), nn.Linear(256, 1),
    )
    return m


def load_classifier(ckpt_dir, fold, device):
    ckpt = os.path.join(ckpt_dir, f'clf_fold_{fold}.pt')
    assert os.path.exists(ckpt), f'Checkpoint not found: {ckpt}'
    m = build_classifier()
    m.load_state_dict(torch.load(ckpt, map_location=device))
    return m.to(device).eval()


def load_yolo(yolo_ckpt_dir, subset, fold):
    ckpt = os.path.join(yolo_ckpt_dir, subset, f'fold_{fold}', 'weights', 'best.pt')
    assert os.path.exists(ckpt), f'Checkpoint not found: {ckpt}'
    return YOLO(ckpt)


@torch.no_grad()
def clf_score_crops(model, crops, eval_tf, device):
    if not crops:
        return []
    tensors = torch.stack([eval_tf(Image.fromarray(c)) for c in crops]).to(device)
    return torch.sigmoid(model(tensors).squeeze(1)).cpu().tolist()


@torch.no_grad()
def get_yolo_proposals(yolo_model, img_path, cfg):
    result = yolo_model.predict(
        img_path, conf=cfg['stage1_conf'], iou=cfg['stage1_iou'],
        imgsz=cfg['stage1_imgsz'], verbose=False)[0]
    if result.boxes is None or len(result.boxes) == 0:
        return []
    boxes = result.boxes.xyxy.cpu().numpy()
    confs = result.boxes.conf.cpu().numpy()
    return [(int(x1), int(y1), int(x2), int(y2), float(c))
            for (x1, y1, x2, y2), c in zip(boxes, confs)]


def two_stage_detect(yolo_model, clf_model, img_path, eval_tf, device, cfg):
    img = np.array(Image.open(img_path).convert('RGB'))
    H, W = img.shape[:2]
    proposals = get_yolo_proposals(yolo_model, img_path, cfg)
    if not proposals:
        return [], 0

    crops, meta = [], []
    for (x1, y1, x2, y2, _) in proposals:
        x1e, y1e, x2e, y2e = expand_bbox(x1, y1, x2, y2, cfg['context_factor'], W, H)
        cr = img[y1e:y2e, x1e:x2e]
        if cr.size == 0:
            continue
        crops.append(cr); meta.append((x1, y1, x2, y2))

    if not crops:
        return [], len(proposals)

    scores = clf_score_crops(clf_model, crops, eval_tf, device)
    detections = [(x1, y1, x2, y2, sc) for (x1, y1, x2, y2), sc in zip(meta, scores)
                  if sc >= cfg['clf_thr']]
    return nms(detections, cfg['nms_iou']), len(proposals)


def evaluate_fold(data_root, subset, fold, yolo_ckpt_dir, clf_ckpt_dir, missing_ids,
                   eval_tf, device, cfg):
    yolo = load_yolo(yolo_ckpt_dir, subset, fold)
    clf = load_classifier(clf_ckpt_dir, fold, device)

    val_imgs = fold_val_images(data_root, subset, fold)
    total_gt = tp = fp = 0
    scored_tp = []
    total_props = 0

    for ip in tqdm(val_imgs, desc=f'Pipeline fold {fold}', leave=False):
        labs = read_labels(ip)
        img = np.array(Image.open(ip).convert('RGB'))
        H, W = img.shape[:2]
        gt = [yolo_to_xyxy(xc, yc, w, h, W, H)
              for (cls, xc, yc, w, h) in labs if cls in missing_ids]
        total_gt += len(gt)
        if not gt:
            continue

        preds, n_props = two_stage_detect(yolo, clf, ip, eval_tf, device, cfg)
        total_props += n_props

        gt_used = [False] * len(gt)
        for (x1, y1, x2, y2, sc) in preds:
            best_i, best_iou = -1, 0.0
            for i, g in enumerate(gt):
                if gt_used[i]:
                    continue
                v = iou_xyxy((x1, y1, x2, y2), g)
                if v > best_iou:
                    best_i, best_iou = i, v
            if best_i >= 0 and best_iou >= 0.5:
                gt_used[best_i] = True; tp += 1
                scored_tp.append((sc, 1))
            else:
                fp += 1; scored_tp.append((sc, 0))

    del yolo, clf
    if device == 'cuda':
        torch.cuda.empty_cache()

    prec = tp / max(1, tp + fp)
    rec = tp / max(1, total_gt)
    f1 = 2 * prec * rec / max(1e-9, prec + rec)
    return {'fold': fold, 'total_gt': total_gt, 'tp': tp, 'fp': fp,
            'total_proposals': total_props, 'precision': prec, 'recall': rec,
            'f1': f1, 'fnr': 1 - rec, 'ap50': compute_ap_11pt(scored_tp, total_gt)}


def main():
    parser = argparse.ArgumentParser(description='Run the fold-aware two-stage detection pipeline')
    parser.add_argument('--data-root', default='data/PCB-MC')
    parser.add_argument('--output-dir', default='models/two_stage/outputs')
    parser.add_argument('--subset', default='full_dataset')
    parser.add_argument('--fold', type=int, default=None, help='single fold; omit to run all 5 folds')
    parser.add_argument('--yolo-checkpoint-dir', default='models/yolo/outputs')
    parser.add_argument('--clf-checkpoint-dir', default=None,
                         help='defaults to <output-dir>/checkpoints (train_patch_classifier.py output)')
    parser.add_argument('--config', default=None)
    args = parser.parse_args()

    cfg = dict(DEFAULTS)
    if args.config:
        with open(args.config) as f:
            cfg.update(yaml.safe_load(f) or {})

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    clf_ckpt_dir = args.clf_checkpoint_dir or os.path.join(args.output_dir, 'checkpoints')
    result_dir = os.path.join(args.output_dir, 'results')
    os.makedirs(result_dir, exist_ok=True)

    missing_ids = load_class_ids(args.data_root, args.subset)
    eval_tf = T.Compose([
        T.Resize((cfg['crop_size'], cfg['crop_size'])),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    folds = [args.fold] if args.fold is not None else list(range(5))
    fold_results = []
    for fi in folds:
        r = evaluate_fold(args.data_root, args.subset, fi, args.yolo_checkpoint_dir,
                           clf_ckpt_dir, missing_ids, eval_tf, device, cfg)
        fold_results.append(r)
        print(f'Fold {fi}: GT={r["total_gt"]} TP={r["tp"]} FP={r["fp"]} '
              f'mAP={r["ap50"]:.4f} F1={r["f1"]:.4f} FNR={r["fnr"]:.4f}')

    ap_vals = [r['ap50'] for r in fold_results]
    f1_vals = [r['f1'] for r in fold_results]
    fnr_vals = [r['fnr'] for r in fold_results]
    summary = {
        'fold_results': fold_results,
        'summary': {
            'mean_ap': float(np.mean(ap_vals)), 'std_ap': float(np.std(ap_vals)),
            'mean_f1': float(np.mean(f1_vals)), 'std_f1': float(np.std(f1_vals)),
            'mean_fnr': float(np.mean(fnr_vals)), 'std_fnr': float(np.std(fnr_vals)),
            'mean_prec': float(np.mean([r['precision'] for r in fold_results])),
        },
    }
    with open(os.path.join(result_dir, 'pipeline_results.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    print(f'MEAN mAP={np.mean(ap_vals):.4f}+/-{np.std(ap_vals):.4f} '
          f'F1={np.mean(f1_vals):.4f}+/-{np.std(f1_vals):.4f} '
          f'FNR={np.mean(fnr_vals):.4f}+/-{np.std(fnr_vals):.4f}')


if __name__ == '__main__':
    main()
