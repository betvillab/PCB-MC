# PCB-MC: Missing Component Analysis in Printed Circuit Boards

## Overview

PCB-MC is a benchmark dataset and evaluation framework for detecting **missing components** on printed circuit boards. Unlike standard object detection, this task requires identifying empty footprints where a component should be present but is absent — there is no visible object appearance to learn from, only the absence of one. This makes PCB-MC a structural absence detection benchmark rather than a conventional detection benchmark.

The dataset is built on top of RF100 (Roboflow), and introduces board-identity-aware evaluation splits to prevent layout memorization from inflating reported performance.

- **Paper:** "PCB-MC: Missing Component Analysis in Printed Circuit Boards", accepted at EUVIP 2026 (IEEE, Luxembourg, September 28 – October 1, 2026). DOI: [add once available, e.g. after IEEE Xplore indexing]
- **Authors:** Betsy Villa Brochero, Ian Gibson, Estefanía Talavera Martínez
- **Affiliation:** University of Twente
- **Contact:** b.j.villabrochero@utwente.nl
- **License:** CC BY 4.0 (dataset). Code is released separately under MIT.
- **Status:** Published at 4TU.ResearchData, 2026-08-10 (Version 1)
- **DOI:** https://doi.org/10.4121/7c6e5bda-eecf-4feb-8a3a-7f823b1b0140 (versioned: [10.4121/7c6e5bda-eecf-4feb-8a3a-7f823b1b0140.v1](https://doi.org/10.4121/7c6e5bda-eecf-4feb-8a3a-7f823b1b0140.v1))

## Dataset statistics

| Property | Value |
|---|---|
| Total images | 615 |
| Unique board layouts | 197 (3.12 ± 1.02 images per design) |
| Total classes | 31 (23 present-component classes, 8 missing-component classes) |
| Present-component instances | 117,091 |
| Missing-component instances | 9,435 |
| Resolution | 1024×1024 |
| Source | Built on RF100 (Roboflow) |

## Subsets

PCB-MC defines three benchmark tasks derived from the same unified annotation set:

| Subset code | Task | Class type | # Classes | # Images |
|---|---|---|---|---|
| **PCB-MC-A** | All classes | Present + missing | 31 | 615 |
| **PCB-MC-M** | Missing only (**primary task**) | Missing only | 8 | 293 |
| **PCB-MC-C** | Components only | Present only | 23 | 615 |

Task M (missing-only) is the primary task of interest — it isolates the structural-absence-detection
problem that motivates this dataset. Task A and Task C follow conventional detection assumptions
and are included for controlled comparison.

## Directory structure

```
PCB-MC/
├── images/               — full PCB-MC image set
├── annotations/          — COCO-format annotations, per task × per fold × train/valid
│   ├── PCB-MC-A/
│   ├── PCB-MC-M/
│   └── PCB-MC-C/
├── splits/                — board-type-aware 5-fold cross-validation split manifests
│                            (full_dataset_splits.json / missing_only_splits.json / components_only_splits.json)
└── results/
    ├── table3_per_class_ap.csv
    ├── table4_anomaly_detection.csv
    └── stage1_proposal_recall.csv
```

This is the layout as published on 4TU.ResearchData. It differs from the
`<subset>/kfold_data/fold_N/{train,valid}/{images,labels}` layout the companion code
repository's training scripts expect (materialized locally from a Roboflow YOLO export
via `splits/generate_kfold_splits.py`) — see the code repo's README for current status
on bridging the two.

## Splits: board-identity-aware, not random

**This is the most important thing to know before using PCB-MC.**

Random train/test splitting on this dataset inflates reported mAP by up to 55 points, because models memorize board layouts rather than learning to detect missing components. All splits distributed with this dataset are **board-identity-aware 5-fold cross-validation splits** — no board layout appears in both the train and test partition of any fold.

If you re-split this dataset yourself (e.g. randomly), your results will not be comparable to those reported in the PCB-MC paper or any downstream work using the released folds. Please use the folds as distributed, or replicate the board-identity-aware splitting procedure exactly (code provided in the companion GitHub repository, link below).

## Class ID note (versioning)

An earlier internal version of this dataset had a class ID bug where missing-component classes were indexed 23–30 instead of the correct 15–22. This has been corrected in the released version (v1.0 and later). If you obtained an earlier internal copy, re-download the current release before use — results computed against the buggy IDs are not valid.

## Benchmark results

On the primary task (M — missing components), all one-stage and transformer-based detectors
(YOLOv8/v11/26, RT-DETR, D-FINE) show FNR consistently above 0.87; YOLOv11 is the best
one-stage model (mAP 0.09 ± 0.03, FNR 0.87 ± 0.03). The two-stage pipeline (YOLOv11 region
proposals + ResNet-18 classifier) achieves the best result on Task M overall — mAP 0.15 ± 0.05,
F1 0.13 ± 0.04, FNR 0.75 ± 0.07 — the lowest FNR of any evaluated method, at the cost of lower F1.
On the conventional tasks, YOLOv11 reaches mAP 0.30 ± 0.05 on Task A and mAP 0.42 ± 0.09
(F1 0.50 ± 0.08) on Task C.

## Known limitations

- Anomaly-detection-based approaches (PatchCore, PaDiM, DRAEM, Reverse Distillation) were evaluated on this dataset and achieved mAP 0.0 / FNR 1.0, due to diffuse activation across heterogeneous board layouts rather than localized activation at missing footprints. This is a known, reported characteristic of the dataset — not an error in evaluation code.
- Stage-1 region proposal recall is a bottleneck (~0.18), which caps achievable end-to-end detection performance regardless of downstream classifier quality.

## How to cite

```bibtex
@inproceedings{villa2026pcbmc,
  title     = {PCB-MC: Missing Component Analysis in Printed Circuit Boards},
  author    = {Villa Brochero, Betsy and Gibson, Ian and Talavera Martínez, Estefanía},
  booktitle = {Proceedings of the IEEE European Workshop on Visual Information Processing (EUVIP)},
  year      = {2026},
  address   = {Luxembourg}
}
```

Please also cite the dataset DOI separately if your work uses PCB-MC without using the paper's methods:

```bibtex
@dataset{villa2026pcbmcdata,
  title     = {PCB-MC: Missing Component Analysis in Printed Circuit Boards},
  author    = {Villa Brochero, Betsy and Gibson, Ian and Talavera Martínez, Estefanía},
  year      = {2026},
  publisher = {4TU.ResearchData},
  doi       = {10.4121/7c6e5bda-eecf-4feb-8a3a-7f823b1b0140.v1}
}
```

## Related resources

- **Code repository (training, evaluation, baselines, figure reproduction):** https://github.com/betvillab/pcb-mc
- **Roboflow Universe mirror:** https://universe.roboflow.com/roboflow-100/printed-circuit-board
