# PCB-MC: Missing Component Detection in Printed Circuit Boards

Code accompanying **PCB-MC: Missing Component Detection in Printed Circuit Boards**, accepted at EUVIP 2026 (IEEE, Luxembourg, September 28 – October 1, 2026).

This repository contains **code only**. The dataset itself is hosted separately at 4TU.ResearchData:

**Dataset DOI:** https://doi.org/10.4121/7c6e5bda-eecf-4feb-8a3a-7f823b1b0140

- **Paper:** [Paper title] — DOI: [add once available, e.g. after IEEE Xplore indexing]
- **Authors:** Betsy Villa, E. Talavera, I. Gibson — University of Twente
- **Contact:** b.j.villabrochero@utwente.nl
- **License (code):** MIT

---

## What's in this repository

PCB-MC targets **missing component detection** on printed circuit boards: identifying empty footprints where a component should be present but is absent. This differs from standard object detection since there is no visible object to learn from, only its absence.

The central methodological contribution is **board identity aware evaluation**: random train/test splits allow models to memorize board layouts rather than learn to detect missing components, inflating reported mAP by up to 55 points. This repo includes the exact split-generation code used in the paper, so results are reproducible without re-deriving the protocol.

This repository provides:

- Split generation (board identity aware 5 fold cross validation)
- Data loading and preprocessing for images/annotations downloaded from the 4TU deposit
- Training and evaluation code for all baselines reported in the paper (YOLOv8/v11/26, RT-DETR, D-FINE)
- Anomaly detection evaluation (Anomalib-based: PatchCore, PaDiM, DRAEM, Reverse Distillation)
- Scripts to reproduce every table and figure in the paper

## Repository structure

```
pcb-mc/
├── README.md
├── LICENSE
├── requirements.txt
├── configs/
├── splits/
│   └── generate_kfold_splits.py     # board-identity-aware split logic
├── data/
│   └── download.py                  # downloads data from the 4TU DOI
├── models/
│   ├── yolo/
│   ├── rtdetr/
│   ├── dfine/
│   └── anomaly_detection/
├── scripts/
│   ├── train.py
│   ├── evaluate.py
│   └── reproduce_paper_tables.py
├── notebooks/
│   └── figures.ipynb
└── docs/
    ├── dataset_card.md              # dataset card (GitHub-facing)
    ├── 4TU_dataset_card.md          # dataset card (4TU.ResearchData submission)
    └── dataset_package_README.md    # README bundled inside the data package itself
```

Most files above are placeholders (`.gitkeep`) until the corresponding code is added.

## Installation

```bash
git clone https://github.com/betvillab/pcb-mc.git
cd pcb-mc
pip install -r requirements.txt
```

[Add any environment-specific notes here — e.g. required CUDA version, Anomalib version pin (v1.2.0), or a Colab-specific setup path if that's the primary intended environment.]

## Getting the data

The dataset is not bundled in this repository. Download it from the 4TU deposit:

```bash
python data/download.py --output ./data/PCB-MC
```

[If `download.py` isn't automated yet, replace this with manual instructions: download from the DOI link above, then place under `data/PCB-MC/` matching the structure documented in the dataset's own README.]

Expected structure after download:

```
data/PCB-MC/
├── non_missing/kfold_data/fold_{0..4}/
├── missing_only/kfold_data/fold_{0..4}/
├── annotations/
└── class_list.txt
```

**Do not re-generate random splits.** Use the folds as distributed, or regenerate them with `splits/generate_kfold_splits.py` to guarantee board-identity separation between train and test partitions. Random splitting reintroduces the layout-leakage problem this paper identifies.

## Reproducing the paper

| Paper item | Script |
|---|---|
| Table [X]: layout leakage comparison (random vs. board-identity-aware splits) | `scripts/reproduce_paper_tables.py --table X` |
| Table [X]: per-class AP, Task M | `scripts/reproduce_paper_tables.py --table X` |
| Table [X]: anomaly detection results | `scripts/evaluate.py --method anomaly --config configs/anomaly.yaml` |
| Figure [X]: [description] | `notebooks/figures.ipynb` |

*(Fill in the actual table/figure numbers from the camera-ready manuscript — this table is what reviewers and future readers will use first, so it's worth keeping accurate as the paper numbering finalizes.)*

## Training a model

```bash
python scripts/train.py --model yolov8 --fold 0 --config configs/yolov8.yaml
```

```bash
python scripts/evaluate.py --model yolov8 --checkpoint <path> --fold 0
```

[Adjust flags/config names to match your actual CLI once finalized.]

## Known results and dataset behavior

- **Anomaly detection fails on PCB-MC.** All evaluated methods (PatchCore, PaDiM, DRAEM, Reverse Distillation) score mAP 0.0 / FNR 1.0, due to diffuse activation across heterogeneous board layouts rather than localized activation at missing footprints. This is a reported, expected property of the dataset — not a bug in the evaluation code.
- **Stage-1 proposal recall is the bottleneck.** Region proposal recall (~0.18) closely tracks the end-to-end pipeline false negative rate (~0.75), meaning detection performance is capped upstream of classification.

## Citation

If you use this code, please cite both the paper and the dataset:

```bibtex
@inproceedings{villa2026pcbmc,
  title     = {PCB-MC: Missing Component Detection in Printed Circuit Boards},
  author    = {Villa, B. and Talavera, E. and Gibson, I.},
  booktitle = {Proceedings of the IEEE European Workshop on Visual Information Processing (EUVIP)},
  year      = {2026},
  address   = {Luxembourg}
}

@dataset{villa2026pcbmcdata,
  title     = {PCB-MC: Missing Component Detection in Printed Circuit Boards},
  author    = {Villa, B. and Talavera, E. and Gibson, I.},
  year      = {2026},
  publisher = {4TU.ResearchData},
  doi       = {10.4121/7c6e5bda-eecf-4feb-8a3a-7f823b1b0140}
}
```

## License

Code is released under the MIT license (see `LICENSE`). The dataset is released separately under CC BY 4.0 — see the [4TU deposit](https://doi.org/10.4121/7c6e5bda-eecf-4feb-8a3a-7f823b1b0140) for details.
