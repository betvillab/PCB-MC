# PCB-MC: Missing Component Detection in Printed Circuit Boards

## Overview

PCB-MC is a benchmark dataset and evaluation framework for detecting **missing components** on printed circuit boards (PCBs) — identifying empty footprints where a component should be but isn't. This differs fundamentally from conventional object detection because the model must localize components that are *not* present, using weak structural cues such as exposed solder pads and silkscreen outlines rather than visible object appearance.

The dataset is built on top of the RF100 (Roboflow 100) "Printed Circuit Board" dataset. PCB-MC keeps the present-component boxes from RF100 (manually corrected and standardized) and adds newly, manually annotated missing-component footprints, cross-validated by multiple annotators.

This deposit accompanies the paper:

> B. Villa Brochero, E. Talavera, and I. Gibson, "PCB-MC: Missing Component Detection in Printed Circuit Boards," in *Proceedings of the IEEE European Workshop on Visual Information Processing (EUVIP)*, Luxembourg, Sept 28–Oct 1, 2026.

Code used to generate these results is available at: [github.com/betvillab/PCB-MC](https://github.com/betvillab/PCB-MC)

## Dataset summary

| | |
|---|---|
| Images | 615 |
| Unique board designs (layouts) | 197 (3.12 ± 1.02 images per design) |
| Total classes | 31 (23 present-component classes + 8 missing-component classes) |
| Present component instances | 117,091 |
| Missing component instances | 9,435 |
| Annotation format | YOLO bounding box (`class x_center y_center width height`, normalized) |
| Evaluation protocol | Board-type-aware 5-fold cross-validation (70% of board designs for training, 30% for validation per fold) |
| Resolution | 1024×1024 (preserves fine-grained footprint cues) |

**Missing-component classes (8):** Missing Capacitor, Missing Diode, Missing Ferrite Bead, Missing IC, Missing Inductor, Missing LED, Missing Resistor, and **Unknown** (a missing footprint whose component family cannot be reliably inferred from the remaining visual cues — this is the one missing class with no present-component counterpart).

**Present-component classes (23):** Button, Capacitor, Clock, Connector, Diode, Display, Electrolytic Capacitor, EM, Ferrite Bead, Fuse, Heatsink, IC, Inductor, Jumper, LED, Pads, Pins, Potentiometer, Resistor, Switch, Test Point, Transistor, Zener Diode.

### Benchmark tasks (dataset subsets)

PCB-MC defines three benchmark tasks derived from the same unified annotation set:

| Subset code | Task | Class type | # Classes | # Images |
|---|---|---|---|---|
| **PCB-MC-A** | All classes | Present + missing | 31 | 615 |
| **PCB-MC-M** | Missing only (**primary task**) | Missing only | 8 | 293 |
| **PCB-MC-C** | Components only | Present only | 23 | 615 |

Task M (missing-only) is the primary task of interest — it isolates the structural-absence-detection problem that motivates this dataset. Task A and Task C follow conventional detection assumptions and are included for controlled comparison.

### Why board-type-aware splits?

A random train/test split allows the same physical board design to appear in both training and test sets, which inflates results because models can memorize layouts rather than learning to detect genuine component absence. PCB-MC contains 197 physical board designs; all images from the same design are assigned to exactly one fold, so no board layout appears in more than one fold. We use 5-fold cross-validation with 70% of board designs for training and 30% for validation per fold.

## Folder structure

```
PCB_MC/
├── images/               — full PCB-MC image set
├── annotations/          — COCO-format annotations, per task × per fold × train/valid
│   ├── PCB-MC-A/         — Task A: all 31 classes (present + missing)
│   ├── PCB-MC-M/         — Task M: 8 missing-only classes (primary task)
│   └── PCB-MC-C/         — Task C: 23 present-only classes
├── splits/               — board-type-aware 5-fold cross-validation split manifests
│                           (full_dataset_splits.json = PCB-MC-A, missing_only_splits.json = PCB-MC-M,
│                            components_only_splits.json = PCB-MC-C)
└── results/
    ├── table3_per_class_ap.csv        — per-class AP, supervised detectors (YOLOv8/v11/26, RT-DETR, D-FINE, SAHI)
    ├── table4_anomaly_detection.csv   — anomaly detection benchmark (PatchCore, PaDiM, DRAEM, Reverse Distillation)
    ├── stage1_proposal_recall.csv     — proposal-stage recall analysis (two-stage pipeline)
    └── README.md                     — this file
```

**This differs from the directory layout the training/evaluation scripts in the companion
code repository expect** (`<subset>/kfold_data/fold_N/{train,valid}/{images,labels}`,
materialized during the authors' own experiments via `splits/generate_kfold_splits.py`
against a Roboflow YOLO export). Mapping this published layout — shared `images/` plus
per-task COCO `annotations/` plus `splits/*.json` fold manifests — into that per-fold
directory structure is not yet automated in the code repository; see its README for
current status before running training end-to-end against a fresh download.

## Results summary

**Supervised detection:** YOLOv8, YOLOv11, YOLO26, RT-DETR, and D-FINE were evaluated under board-type-aware 5-fold cross-validation, with SAHI tiled inference also tested on each YOLO variant. On the primary task (M — missing components), all one-stage and transformer-based detectors show high false negative rates (FNR consistently above 0.87); YOLOv11 is the best one-stage model (mAP 0.09 ± 0.03, FNR 0.87 ± 0.03). The proposed **two-stage pipeline** (YOLOv11 region proposals + ResNet-18 classifier) achieves the best result on Task M overall: **mAP 0.15 ± 0.05, F1 0.13 ± 0.04, FNR 0.75 ± 0.07** — the lowest FNR of any evaluated method, at the cost of lower F1. On the conventional tasks, YOLOv11 reaches mAP 0.30 ± 0.05 on Task A and mAP 0.42 ± 0.09 (F1 0.50 ± 0.08) on Task C.

**Anomaly detection (unsupervised):** PatchCore, PaDiM, DRAEM, and Reverse Distillation were evaluated as an alternative to supervised detection on Task M, trained on boards containing only present components. All four methods produced zero box-level mAP and FNR of 1.00 under the evaluated anomaly-map-to-box conversion protocol — a complete failure to localize missing footprints on diverse, unaligned board layouts. The failure mode reflects diffuse activation across heterogeneous board layouts rather than localized activation at missing-component footprints; evaluating native anomaly metrics and threshold sensitivity is noted as future work. Full per-fold results are in `table4_anomaly_detection.csv`.

## Auxiliary information / abbreviations

- **mAP** — mean Average Precision (reported at IoU@0.5)
- **F1** — F1 score
- **FNR** — False Negative Rate
- **SAHI** — Slicing Aided Hyper Inference (tiled inference for small-object detection; 1024×1024 slices, 20% overlap, Greedy Non-Maximum Merging at IoU 0.30)
- **PCB-MC-A** — Task A: All classes (present + missing), 31 classes, 615 images
- **PCB-MC-M** — Task M: Missing components only (primary task), 8 classes, 293 images
- **PCB-MC-C** — Task C: Present components only, 23 classes, 615 images

## License

CC BY 4.0

## Citation

If you use PCB-MC, please cite both the paper and the dataset:

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

## Contact

Betsy Villa Brochero, University of Twente (EEMCS)
Supervisors: E. Talavera, I. Gibson
