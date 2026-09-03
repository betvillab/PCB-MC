# PCB-MC: Missing Component Analysis in Printed Circuit Boards

Code accompanying **PCB-MC: Missing Component Analysis in Printed Circuit Boards**, accepted at EUVIP 2026 (IEEE, Luxembourg, September 28 – October 1, 2026).

This repository contains **code only**. The dataset itself is hosted separately at 4TU.ResearchData:

**Dataset DOI:** https://doi.org/10.4121/7c6e5bda-eecf-4feb-8a3a-7f823b1b0140

- **Paper:** "PCB-MC: Missing Component Analysis in Printed Circuit Boards" — DOI: [add once available]
- **Authors:** Betsy Villa Brochero, Ian Gibson, Estefanía Talavera Martínez — University of Twente
- **Contact:** b.j.villabrochero@utwente.nl
- **License (code):** MIT

---

## What's in this repository

PCB-MC targets **missing component detection** on printed circuit boards: identifying empty footprints where a component should be present but is absent. This differs from standard object detection since there is no visible object to learn from, only its absence.

The central methodological contribution is **board identity aware evaluation**: random train/test splits allow models to memorize board layouts rather than learn to detect missing components. This repo includes the exact split generation code used in the paper, so results are reproducible without re-deriving the protocol.

This repository provides:

- Split generation (board identity aware 5 fold cross validation)
- Data loading and preprocessing for images/annotations downloaded from the 4TU deposit
- Training and evaluation code for all baselines reported in the paper (YOLOv8/v11/26, RT-DETR, D-FINE)
- Anomaly detection evaluation (Anomalib-based: PatchCore, PaDiM, DRAEM, Reverse Distillation)
- A two-stage pipeline (YOLO region proposals + a fold-aware ResNet-18 patch classifier),
  built after sliding-window classification failed (3903 false positives for 45 ground-truth
  boxes — the classifier had never seen background patches during training)
- Scripts to reproduce every table and figure in the paper

## Repository structure

```
pcb-mc/
├── README.md
├── LICENSE
├── requirements.txt
├── configs/
│   ├── yolov8.yaml                  # hyperparameter overrides (weights/imgsz/batch/epochs/lr0/aug)
│   ├── yolov11.yaml                 # hyperparameter overrides (epochs/imgsz/batch)
│   └── yolov26.yaml                 # hyperparameter overrides (epochs/imgsz/batch)
│   # RT-DETR, D-FINE, anomaly detection and two-stage don't use this configs/
│   # pattern — see "Training a model" below for why.
├── splits/
│   └── generate_kfold_splits.py     # board-identity-aware split logic
├── data/
│   ├── download.py                  # downloads data from the 4TU DOI (djehuty/Figshare-v2 API)
│   ├── convert_yolo_coco.py         # YOLO -> COCO annotation conversion
│   └── materialize_kfold_layout.py  # published images/+annotations/+splits/ -> kfold_data/fold_N/ layout
├── models/
│   ├── yolo/                        # train_yolov8.py, train_yolov11.py, train_yolov26.py, train_sahi.py
│   ├── rtdetr/                      # train_rtdetr.py
│   ├── dfine/                       # train_dfine.py, evaluate_dfine.py
│   ├── anomaly_detection/           # run_benchmark.py (PatchCore, PaDiM, DRAEM, Reverse Distillation)
│   └── two_stage/                   # train_patch_classifier.py, two_stage_pipeline.py
├── scripts/
│   ├── train.py                     # dispatches to models/*/train_*.py by --model
│   ├── evaluate.py                  # dispatches to the eval scripts by --model
│   └── reproduce_paper_tables.py
├── notebooks/
│   └── figures.ipynb                # exploratory/qualitative plotting kept notebook-side
└── docs/
    ├── dataset_card.md              # dataset card (GitHub-facing)
    ├── 4TU_dataset_card.md          # dataset card (4TU.ResearchData submission)
    └── dataset_package_README.md    # README bundled inside the data package itself
```

Every `models/<family>/notebooks/` folder also keeps the original Colab research notebook(s)
each script was ported from, for provenance.

## Installation

```bash
git clone https://github.com/betvillab/pcb-mc.git
cd pcb-mc
pip install -r requirements.txt
```

D-FINE training (`models/dfine/train_dfine.py`) shells out to a vendored checkout of the
upstream [D-FINE](https://github.com/Peterande/D-FINE) repo rather than a pip package —
clone it and pass its path via `--dfine-repo` (default: `./D-FINE`):

```bash
git clone https://github.com/Peterande/D-FINE.git
```

`anomalib==1.2.0` pins a specific PyTorch Lightning-based API; the benchmark script
(`models/anomaly_detection/run_benchmark.py`) imports `anomalib.engine.Engine` and
`anomalib.models.{Patchcore,Padim,Draem,ReverseDistillation}`, which moved between
anomalib versions, so don't upgrade this pin without re-checking those import paths.

## Getting the data

The dataset is not bundled in this repository. Download it from the 4TU deposit:

```bash
python data/download.py --output ./data/PCB-MC
```

This resolves the dataset DOI via the 4TU.ResearchData / djehuty API (Figshare v2
compatible: `POST /v2/articles/search` by DOI, then `GET /v2/articles/{id}/files`) and
streams each file down with checksum verification. **The dataset is now published**
(4TU.ResearchData, 2026-08-10, DOI [10.4121/7c6e5bda-eecf-4feb-8a3a-7f823b1b0140.v1](https://doi.org/10.4121/7c6e5bda-eecf-4feb-8a3a-7f823b1b0140.v1)),
so `download.py` should resolve and stream it directly. Note the deposit ships as a single
archive, `PCB_MC.zip` (3.77 GB, MD5 `1533d1ca6d54c37b19c1a94297c0745c`) — `download.py`
downloads it as-is but doesn't unzip it yet, so extract it manually into `data/PCB-MC/`
matching the structure below.

Expected structure after download (this is the published 4TU layout — see
`docs/dataset_package_README.md` for the authoritative version bundled with the data):

```
data/PCB-MC/
├── images/               # full PCB-MC image set, shared across tasks
├── annotations/          # COCO-format, per task x per fold x train/valid
│   ├── PCB-MC-A/         # Task A: all 31 classes (present + missing)
│   ├── PCB-MC-M/         # Task M: 8 missing-only classes (primary task)
│   └── PCB-MC-C/         # Task C: 23 present-only classes
├── splits/               # board-type-aware 5-fold manifests
│   ├── full_dataset_splits.json
│   ├── missing_only_splits.json
│   └── components_only_splits.json
└── results/               # table3_per_class_ap.csv, table4_anomaly_detection.csv, stage1_proposal_recall.csv
```

The training/eval scripts in `models/` were ported from the authors' own Colab notebooks and
expect a different, already-materialized layout — `<subset>/kfold_data/fold_N/{train,valid}/{images,labels}`
(YOLO txt labels, subset-specific image copies) — which is what `splits/generate_kfold_splits.py`
produces from a Roboflow YOLO export. To bridge that with what 4TU actually publishes
(shared `images/` + per-task COCO `annotations/` + `splits/*.json` fold manifests), use
`data/materialize_kfold_layout.py`:

```bash
python data/materialize_kfold_layout.py \
    --images-dir data/PCB-MC/images \
    --coco-file data/PCB-MC/annotations/PCB-MC-A/annotations.json \
    --splits-file data/PCB-MC/splits/full_dataset_splits.json \
    --output-root data/PCB-MC/full_dataset/kfold_data
```

Run once per task (`PCB-MC-A` → `full_dataset`, `PCB-MC-M` → `missing_only`, `PCB-MC-C` →
`components_only`), pointing `--coco-file` and `--splits-file` at that task's annotations and
manifest. Class indices come from the COCO file's own `categories` list (sorted by id), so
this always matches whatever ordering the actual annotation files use — the same ordering
`splits/generate_kfold_splits.py`'s `CATEGORIES_FULL`/`CATEGORIES_MISSING` tables now encode:
a single alphabetical sort across all class names combined, which is why the 8 missing
classes (including the catch-all "Missing Component" — the dataset card's "Unknown" is a
descriptive gloss on the same class, not a different name in the data) land at indices
15–22 rather than a contiguous block.

This assumes `annotations/<task>/` ships as one COCO file covering the whole task; if the
4TU deposit instead ships pre-split per-fold COCO files, point `--coco-file` at each one
directly per fold/split invocation instead (the manifest-based filtering is then a harmless
no-op) — the exact packaging wasn't available to verify at the time this was written.

**Do not re-generate splits from scratch.** Use the folds as distributed in `splits/*.json`,
or regenerate them with `splits/generate_kfold_splits.py` (against a Roboflow-style export)
to guarantee board-type separation between train and test partitions. Random splitting
reintroduces the layout-leakage problem this paper identifies — it was found to inflate
reported mAP by up to 55 points.

## Reproducing the paper

The published dataset ships its own reference results under `data/PCB-MC/results/` —
`table3_per_class_ap.csv`, `table4_anomaly_detection.csv`, `stage1_proposal_recall.csv` —
which the numbers in the table below reproduce or extend:


## Training a model

`scripts/train.py --model <name> ...` and `scripts/evaluate.py --model <name> ...` are
thin dispatchers: everything after `--model` is passed straight through to the real
per-model script (`models/<family>/train_*.py` / `evaluate_*.py`), so each family keeps
its own native flags rather than being forced into one shared shape. Run
`python models/<family>/train_*.py --help` to see a given script's actual flags.

```bash
# YOLOv8 — --config is an optional hyperparameter-override YAML (see configs/yolov8.yaml);
# --preset {A,B} picks between the two base configs found in the notebook.
python scripts/train.py --model yolov8 --subset components_only --fold 0 \
    --data-root data/PCB-MC --output-dir results/yolov8 --config configs/yolov8.yaml

# YOLOv11 / YOLOv26 — --config only overrides epochs/imgsz/batch; model size is --size.
python scripts/train.py --model yolov11 --subset full_dataset --fold 0 --size s \
    --data-root data/PCB-MC --output-dir results/yolov11

# RT-DETR — --config is the ultralytics checkpoint to start from, e.g. rtdetr-l.pt
# (not a YAML — ultralytics' RTDETR class takes a weights name/path here).
python scripts/train.py --model rtdetr --subset components_only --fold 0 \
    --data-root data/PCB-MC --output-dir results/rtdetr --config rtdetr-l.pt

# D-FINE — --config is a path into the vendored D-FINE repo's own config schema,
# not a file in this repo's configs/ (D-FINE ships its own configs/dfine/*.yml).
python scripts/train.py --model dfine --subset full_dataset --fold 0 \
    --data-root data/PCB-MC --output-dir results/dfine --dfine-repo ./D-FINE \
    --config ./D-FINE/configs/dfine/dfine_hgnetv2_l_coco.yml
python scripts/evaluate.py --model dfine --checkpoint results/dfine/full_dataset/fold_0/last.pth \
    --subset full_dataset --fold 0 --data-root data/PCB-MC --output-dir results/dfine

# Two-stage (YOLO proposals + fold-aware ResNet-18 patch classifier)
python scripts/train.py --model two_stage --subset full_dataset --fold 0 \
    --data-root data/PCB-MC --output-dir models/two_stage/outputs --config configs/two_stage.yaml
python scripts/evaluate.py --model two_stage --subset full_dataset \
    --data-root data/PCB-MC --yolo-checkpoint-dir results/yolov11

# SAHI-tiled evaluation of already-trained YOLO weights (this notebook never trains —
# see models/yolo/train_sahi.py's own docstring/report for the fold-path convention
# mismatch against the v11/v26 training scripts, not yet reconciled)
python scripts/evaluate.py --model sahi --subset full_dataset --model-family yolov11 \
    --data-root data/PCB-MC --weights-root results/yolov11 --output-dir results/sahi

# Anomaly detection benchmark — no --config; all hyperparameters are CLI flags
# with the notebook's original defaults baked in.
python scripts/evaluate.py --model anomaly --data-root data/PCB-MC --output-dir results/anomaly
```

## Known results and dataset behavior

- **All one-stage/transformer detectors struggle on Task M (missing components).** YOLOv8/v11/26, RT-DETR, and D-FINE all show FNR above 0.87 under board-type-aware 5-fold CV. YOLOv11 is the best of these (mAP 0.09 ± 0.03, FNR 0.87 ± 0.03).
- **The two-stage pipeline (YOLOv11 proposals + fold-aware ResNet-18 classifier) gives the best Task M result overall:** mAP 0.15 ± 0.05, F1 0.13 ± 0.04, FNR 0.75 ± 0.07 — the lowest FNR of any evaluated method, at the cost of lower F1. Stage-1 proposal recall (~0.18) closely tracks this pipeline FNR, meaning detection performance is capped upstream of classification.
- **On the conventional tasks**, YOLOv11 reaches mAP 0.30 ± 0.05 on Task A (all 31 classes) and mAP 0.42 ± 0.09 (F1 0.50 ± 0.08) on Task C (23 present-component classes only).
- **Anomaly detection fails on PCB-MC.** All evaluated methods (PatchCore, PaDiM, DRAEM, Reverse Distillation) score mAP 0.0 / FNR 1.0, due to diffuse activation across heterogeneous board layouts rather than localized activation at missing footprints. This is a reported, expected property of the dataset — not a bug in the evaluation code.

## Citation

If you use this code, please cite both the paper and the dataset:

```bibtex
@inproceedings{villa2026pcbmc,
  title     = {PCB-MC: Missing Component Analysis in Printed Circuit Boards},
  author    = {Villa Brochero, Betsy and Gibson, Ian and Talavera Martínez, Estefanía},
  booktitle = {Proceedings of the IEEE European Workshop on Visual Information Processing (EUVIP)},
  year      = {2026},
  address   = {Luxembourg}
}

@dataset{villa2026pcbmcdata,
  title     = {PCB-MC: Missing Component Analysis in Printed Circuit Boards},
  author    = {Villa Brochero, Betsy and Gibson, Ian and Talavera Martínez, Estefanía},
  year      = {2026},
  publisher = {4TU.ResearchData},
  doi       = {10.4121/7c6e5bda-eecf-4feb-8a3a-7f823b1b0140.v1}
}
```

## License

Code is released under the MIT license (see `LICENSE`). The dataset is released separately under CC BY 4.0 — see the [4TU deposit](https://doi.org/10.4121/7c6e5bda-eecf-4feb-8a3a-7f823b1b0140) for details.
