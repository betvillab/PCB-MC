# PCB-MC: Missing Component Detection in Printed Circuit Boards

## Overview

PCB-MC is a benchmark dataset and evaluation framework for detecting **missing components** on printed circuit boards. Unlike standard object detection, this task requires identifying empty footprints where a component should be present but is absent — there is no visible object appearance to learn from, only the absence of one. This makes PCB-MC a structural absence detection benchmark rather than a conventional detection benchmark.

The dataset is built on top of RF100 (Roboflow), and introduces board-identity-aware evaluation splits to prevent layout memorization from inflating reported performance.

- **Paper:** [Paper title], accepted at EUVIP 2026 (IEEE, Luxembourg, September 28 – October 1, 2026)
- **Authors:** Betsy Villa, E. Talavera, I. Gibson
- **Affiliation:** University of Twente
- **Contact:** b.j.villabrochero@utwente.nl
- **License:** CC BY 4.0 (dataset). Code is released separately under MIT.
- **DOI:** https://doi.org/10.4121/7c6e5bda-eecf-4feb-8a3a-7f823b1b0140

## Dataset statistics

| Property | Value |
|---|---|
| Total images | 615 |
| Unique board layouts | 197 |
| Total classes | 31 (23 present-component classes, 8 missing-component classes) |
| Missing-component instances | ~9,400 |
| Source | Built on RF100 (Roboflow) |

## Subsets

PCB-MC is organized into four subsets used for different evaluation tasks:

- **PCB-MC-A** — [one-line description]
- **PCB-MC-M** — [one-line description, e.g. "missing-component detection task"]
- **PCB-MC-P** — [one-line description]
- **PCB-MC-C** — [one-line description]

*(Fill in the one-line description for each subset from the paper's Section [X] before publishing — this is the part external users will read first to understand which subset fits their use case.)*

## Directory structure

```
PCB-MC/
├── non_missing/
│   └── kfold_data/
│       ├── fold_0/
│       ├── fold_1/
│       ├── fold_2/
│       ├── fold_3/
│       └── fold_4/
├── missing_only/
│   └── kfold_data/
│       ├── fold_0/
│       ├── fold_1/
│       ├── fold_2/
│       ├── fold_3/
│       └── fold_4/
├── annotations/
│   └── [format, e.g. COCO-style JSON per fold]
└── class_list.txt
```

## Splits: board-identity-aware, not random

**This is the most important thing to know before using PCB-MC.**

Random train/test splitting on this dataset inflates reported mAP by up to 55 points, because models memorize board layouts rather than learning to detect missing components. All splits distributed with this dataset are **board-identity-aware 5-fold cross-validation splits** — no board layout appears in both the train and test partition of any fold.

If you re-split this dataset yourself (e.g. randomly), your results will not be comparable to those reported in the PCB-MC paper or any downstream work using the released folds. Please use the folds as distributed, or replicate the board-identity-aware splitting procedure exactly (code provided in the companion GitHub repository, link below).

## Class ID note (versioning)

An earlier internal version of this dataset had a class ID bug where missing-component classes were indexed 23–30 instead of the correct 15–22. This has been corrected in the released version (v1.0 and later). If you obtained an earlier internal copy, re-download the current release before use — results computed against the buggy IDs are not valid.

## Known limitations

- Anomaly-detection-based approaches (PatchCore, PaDiM, DRAEM, Reverse Distillation) were evaluated on this dataset and achieved mAP 0.0 / FNR 1.0, due to diffuse activation across heterogeneous board layouts rather than localized activation at missing footprints. This is a known, reported characteristic of the dataset — not an error in evaluation code.
- Stage-1 region proposal recall is a bottleneck (~0.18), which caps achievable end-to-end detection performance regardless of downstream classifier quality.

## How to cite

```bibtex
@inproceedings{[citekey],
  title     = {[Paper title]},
  author    = {Villa, Betsy and Talavera, E. and Gibson, I.},
  booktitle = {Proceedings of the IEEE European Workshop on Visual Information Processing (EUVIP)},
  year      = {2026},
  address   = {Luxembourg}
}
```

Please also cite the dataset DOI separately if your work uses PCB-MC without using the paper's methods:

```bibtex
@dataset{[dataset_citekey],
  title   = {PCB-MC: Missing Component Detection in Printed Circuit Boards},
  author  = {Villa, Betsy and Talavera, E. and Gibson, I.},
  year    = {2026},
  doi     = {10.4121/7c6e5bda-eecf-4feb-8a3a-7f823b1b0140},
  url     = {https://doi.org/10.4121/7c6e5bda-eecf-4feb-8a3a-7f823b1b0140}
}
```

## Related resources

- **Code repository (training, evaluation, baselines, figure reproduction):** https://github.com/betvillab/pcb-mc
- **Roboflow Universe mirror:** [URL, optional]
