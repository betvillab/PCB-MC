"""Fold-aware ResNet-18 patch classifier training for the PCB-MC two-stage pipeline.

Ports models/two_stage/notebooks/9_10_PatchClassifier_TwoStage.ipynb: for a given fold,
extracts present/missing crops from that fold's own training boards only, then trains
a ResNet-18 binary classifier (focal loss) checkpointed to clf_fold_{fold}.pt.
"""
import argparse
import json
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import yaml
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import models
from tqdm import tqdm

IMG_EXTS = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')

DEFAULTS = dict(
    crop_size=128,
    context_factor=1.4,
    min_crop_px=12,
    present_to_missing_ratio=2.0,
    batch_size=64,
    epochs=40,
    lr=3e-4,
    weight_decay=1e-4,
    focal_alpha=0.75,
    focal_gamma=2.0,
    clf_thr=0.50,
    val_frac=0.15,
    patience=10,
    seed=42,
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


def fold_images_dir(data_root, subset, fold, split):
    return os.path.join(data_root, subset, 'kfold_data', f'fold_{fold}', split, 'images')


def load_class_ids(data_root, subset):
    yaml_path = os.path.join(data_root, subset, 'kfold_data', 'fold_0', 'data.yaml')
    with open(yaml_path) as f:
        cfg = yaml.safe_load(f)
    class_names = cfg['names']
    missing_ids = frozenset(i for i, n in enumerate(class_names) if n.lower().startswith('missing'))
    present_ids = frozenset(i for i, n in enumerate(class_names) if not n.lower().startswith('missing'))
    return missing_ids, present_ids


class PatchDataset(Dataset):
    def __init__(self, paths, labels, transform):
        self.paths = paths; self.labels = labels; self.tf = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        return self.tf(Image.open(self.paths[i]).convert('RGB')), self.labels[i]


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.75, gamma=2.0):
        super().__init__(); self.alpha = alpha; self.gamma = gamma

    def forward(self, logits, targets):
        bce = F.binary_cross_entropy_with_logits(logits, targets.float(), reduction='none')
        pt = torch.exp(-bce)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        return (alpha_t * (1 - pt) ** self.gamma * bce).mean()


def build_model(device):
    m = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    m.fc = nn.Sequential(
        nn.Linear(m.fc.in_features, 256), nn.ReLU(),
        nn.Dropout(0.3), nn.Linear(256, 1),
    )
    return m.to(device)


def extract_crops_for_fold(data_root, subset, fold, crop_dir, missing_ids, present_ids, cfg):
    m_dir = os.path.join(crop_dir, f'fold_{fold}', 'missing')
    p_dir = os.path.join(crop_dir, f'fold_{fold}', 'present')

    n_existing_m = len(list(Path(m_dir).glob('*.jpg'))) if os.path.exists(m_dir) else 0
    n_existing_p = len(list(Path(p_dir).glob('*.jpg'))) if os.path.exists(p_dir) else 0
    if n_existing_m > 100 and n_existing_p > 100:
        print(f'Fold {fold}: crops already extracted (missing={n_existing_m}, present={n_existing_p}) — skipping')
        return n_existing_m, n_existing_p

    os.makedirs(m_dir, exist_ok=True)
    os.makedirs(p_dir, exist_ok=True)
    n_miss = n_pres = 0

    train_imgs = list_images(fold_images_dir(data_root, subset, fold, 'train'))
    for ip in tqdm(train_imgs, desc=f'Fold {fold} crops', leave=False):
        labs = read_labels(ip)
        if not labs:
            continue
        img = Image.open(ip).convert('RGB')
        W, H = img.size
        bn = os.path.splitext(os.path.basename(ip))[0]

        for idx, (cls, xc, yc, w, h) in enumerate(labs):
            x1, y1, x2, y2 = yolo_to_xyxy(xc, yc, w, h, W, H)
            if (x2 - x1) < cfg['min_crop_px'] or (y2 - y1) < cfg['min_crop_px']:
                continue
            x1e, y1e, x2e, y2e = expand_bbox(x1, y1, x2, y2, cfg['context_factor'], W, H)
            crop = img.crop((x1e, y1e, x2e, y2e))
            fname = f'{bn}_{idx:04d}.jpg'
            if cls in missing_ids:
                crop.save(os.path.join(m_dir, fname), quality=92)
                n_miss += 1
            elif cls in present_ids:
                crop.save(os.path.join(p_dir, fname), quality=92)
                n_pres += 1

    # Cap present crops to keep the training ratio bounded — present-class footprints
    # vastly outnumber missing ones on any real board.
    present_cap = int(n_miss * cfg['present_to_missing_ratio'])
    if n_pres > present_cap:
        all_p = list(Path(p_dir).glob('*.jpg'))
        random.shuffle(all_p)
        for f in all_p[present_cap:]:
            os.remove(f)
        n_pres = present_cap

    return n_miss, n_pres


def build_loaders(crop_dir, fold, crop_size, val_frac, batch_size, seed):
    m_dir = os.path.join(crop_dir, f'fold_{fold}', 'missing')
    p_dir = os.path.join(crop_dir, f'fold_{fold}', 'present')
    files = ([(str(f), 1) for f in Path(m_dir).glob('*.jpg')] +
             [(str(f), 0) for f in Path(p_dir).glob('*.jpg')])
    paths = [x[0] for x in files]
    labels = [x[1] for x in files]

    train_tf = T.Compose([
        T.Resize((crop_size, crop_size)),
        T.RandomHorizontalFlip(),
        T.RandomVerticalFlip(),
        T.RandomRotation(15),
        T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    eval_tf = T.Compose([
        T.Resize((crop_size, crop_size)),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    tr_p, va_p, tr_l, va_l = train_test_split(
        paths, labels, test_size=val_frac, stratify=labels, random_state=seed)
    n_m = sum(1 for l in tr_l if l == 1)
    n_p = sum(1 for l in tr_l if l == 0)
    w = [1.0 / n_p if l == 0 else 1.0 / n_m for l in tr_l]
    sampler = WeightedRandomSampler(w, len(tr_l), replacement=True)

    tr_dl = DataLoader(PatchDataset(tr_p, tr_l, train_tf), batch_size=batch_size,
                        sampler=sampler, num_workers=2, pin_memory=True)
    va_dl = DataLoader(PatchDataset(va_p, va_l, eval_tf), batch_size=batch_size,
                        shuffle=False, num_workers=2, pin_memory=True)
    return tr_dl, va_dl


def run_epoch(model, loader, criterion, optimizer, train, device, clf_thr):
    model.train(train)
    tp = fp = fn = total_loss = 0
    with torch.set_grad_enabled(train):
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            logits = model(imgs).squeeze(1)
            loss = criterion(logits, labels)
            if train:
                optimizer.zero_grad(); loss.backward(); optimizer.step()
            total_loss += loss.item() * len(imgs)
            preds = (torch.sigmoid(logits) >= clf_thr).long()
            tp += ((preds == 1) & (labels == 1)).sum().item()
            fp += ((preds == 1) & (labels == 0)).sum().item()
            fn += ((preds == 0) & (labels == 1)).sum().item()
    n = len(loader.dataset)
    rec = tp / max(1, tp + fn); pre = tp / max(1, tp + fp)
    f1 = 2 * pre * rec / max(1e-9, pre + rec)
    return total_loss / n, 1 - rec, f1


def train_fold(data_root, subset, fold, crop_dir, ckpt_dir, missing_ids, present_ids, cfg, device):
    ckpt = os.path.join(ckpt_dir, f'clf_fold_{fold}.pt')
    if os.path.exists(ckpt):
        print(f'Fold {fold}: checkpoint already exists — skipping training')
        return None, []

    nm, npres = extract_crops_for_fold(data_root, subset, fold, crop_dir, missing_ids, present_ids, cfg)
    print(f'Fold {fold}: crops missing={nm} present={npres}')

    tr_dl, va_dl = build_loaders(crop_dir, fold, cfg['crop_size'], cfg['val_frac'], cfg['batch_size'], cfg['seed'])
    n_m = sum(1 for _, l in tr_dl.dataset if l == 1)
    n_p = len(tr_dl.dataset) - n_m
    print(f'  Train crops: missing={n_m}, present={n_p}, ratio={n_p / max(1, n_m):.1f}:1')

    model = build_model(device)
    criterion = FocalLoss(cfg['focal_alpha'], cfg['focal_gamma'])
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg['lr'], weight_decay=cfg['weight_decay'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg['epochs'])

    best_fnr = 1.0; no_imp = 0; history = []
    for ep in range(1, cfg['epochs'] + 1):
        tr_loss, tr_fnr, tr_f1 = run_epoch(model, tr_dl, criterion, optimizer, True, device, cfg['clf_thr'])
        va_loss, va_fnr, va_f1 = run_epoch(model, va_dl, criterion, optimizer, False, device, cfg['clf_thr'])
        scheduler.step()
        history.append({'ep': ep, 'tr_fnr': tr_fnr, 'va_fnr': va_fnr, 'tr_f1': tr_f1, 'va_f1': va_f1})

        if va_fnr < best_fnr:
            best_fnr = va_fnr; no_imp = 0
            torch.save(model.state_dict(), ckpt)
        else:
            no_imp += 1

        if ep % 5 == 0 or ep == 1:
            marker = ' <- best' if no_imp == 0 else ''
            print(f'  Ep {ep:03d} | tr_FNR {tr_fnr:.4f} tr_F1 {tr_f1:.4f} '
                  f'| va_FNR {va_fnr:.4f} va_F1 {va_f1:.4f}{marker}')

        if no_imp >= cfg['patience']:
            print(f'  Early stop at epoch {ep}')
            break

    print(f'  Best val FNR: {best_fnr:.4f} -> {ckpt}')
    del model
    if device == 'cuda':
        torch.cuda.empty_cache()
    return best_fnr, history


def main():
    parser = argparse.ArgumentParser(description='Train the fold-aware ResNet-18 patch classifier')
    parser.add_argument('--data-root', default='data/PCB-MC')
    parser.add_argument('--output-dir', default='models/two_stage/outputs')
    parser.add_argument('--subset', default='full_dataset')
    parser.add_argument('--fold', type=int, required=True)
    parser.add_argument('--config', default=None)
    args = parser.parse_args()

    cfg = dict(DEFAULTS)
    if args.config:
        with open(args.config) as f:
            cfg.update(yaml.safe_load(f) or {})

    random.seed(cfg['seed']); np.random.seed(cfg['seed']); torch.manual_seed(cfg['seed'])
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    crop_dir = os.path.join(args.output_dir, 'crops')
    ckpt_dir = os.path.join(args.output_dir, 'checkpoints')
    os.makedirs(crop_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)

    missing_ids, present_ids = load_class_ids(args.data_root, args.subset)
    best_fnr, history = train_fold(
        args.data_root, args.subset, args.fold, crop_dir, ckpt_dir,
        missing_ids, present_ids, cfg, device)

    if history:
        with open(os.path.join(ckpt_dir, f'history_fold_{args.fold}.json'), 'w') as f:
            json.dump(history, f, indent=2)


if __name__ == '__main__':
    main()
