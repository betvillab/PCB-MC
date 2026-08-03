# PCB-MC Dataset Card — 4TU.ResearchData Submission

4TU.ResearchData (built on Figshare) collects metadata through structured fields at upload time rather than a single freeform README. This document is organized to match those fields directly — copy each section into the corresponding box in the submission form.

---

## Title

PCB-MC: Missing Component Detection in Printed Circuit Boards

## Authors

List each author with ORCID if available (4TU requires ORCID for the submitting/corresponding author, and strongly encourages it for all authors):

- Betsy Villa — University of Twente — ORCID: [xxxx-xxxx-xxxx-xxxx]
- E. Talavera — University of Twente — ORCID: [xxxx-xxxx-xxxx-xxxx]
- I. Gibson — University of Twente — ORCID: [xxxx-xxxx-xxxx-xxxx]

## Description (abstract field)

*4TU displays this as the main dataset description — keep it self-contained, since it's often read without the paper.*

PCB-MC is a benchmark dataset for detecting missing components on printed circuit boards (PCBs). Unlike conventional object detection, this task requires identifying empty footprints where a component should be present but is absent, so there is no visible object appearance to learn from. The dataset is built on RF100 (Roboflow) and contains 615 images across 197 unique board layouts (3.12 ± 1.02 images per design), annotated with 31 classes (23 present-component classes and 8 missing-component classes) — 117,091 present-component instances and 9,435 missing-component instances.

PCB-MC is distributed with board-type-aware 5-fold cross-validation splits (70% of board designs for training, 30% for validation per fold). Random train/test splitting on this dataset allows models to memorize board layouts rather than learn to detect missing components, inflating reported mAP by up to 55 points. The released folds guarantee no board layout appears in both the train and test partition of the same fold, and should be used as distributed for results to be comparable with the associated publication.

The dataset supports three evaluation subsets, all derived from the same unified annotation set:
- **PCB-MC-A** — all 31 classes (present + missing), 615 images
- **PCB-MC-M** — missing-component classes only (8 classes, 293 images); the primary task, isolating structural-absence detection
- **PCB-MC-C** — present-component classes only (23 classes, 615 images); included for conventional-detection comparison

## Keywords

printed circuit boards; PCB inspection; object detection; anomaly detection; missing component detection; industrial inspection; benchmark dataset; computer vision; layout leakage; cross-validation

## Categories / Subject classification

*(4TU uses ANZSRC Fields of Research codes — pick the closest matches during submission, typically under Information and Computing Sciences)*

- Computer Vision
- Machine Learning
- Image Processing

## Format

- Images: 1024×1024, full PCB-MC image set under `images/`
- Annotations: COCO-format JSON, one set per task (PCB-MC-A/M/C) × fold × train/valid, under `annotations/`
- Splits: board-type-aware 5-fold manifests under `splits/` (`full_dataset_splits.json`,
  `missing_only_splits.json`, `components_only_splits.json`)
- Folder structure: see "Data structure" section below

## License

CC BY 4.0

## Dataset DOI

10.4121/7c6e5bda-eecf-4feb-8a3a-7f823b1b0140

## Related publication

B. Villa Brochero, E. Talavera, and I. Gibson, "PCB-MC: Missing Component Detection in Printed Circuit Boards," accepted at the IEEE European Workshop on Visual Information Processing (EUVIP 2026), Luxembourg, September 28 – October 1, 2026.
DOI: [add once available, e.g. after IEEE Xplore indexing]

## Related materials / code

Companion code repository (training, evaluation, baselines, figure reproduction): https://github.com/betvillab/pcb-mc

## Funding (if applicable)

[Grant number / funding body, if the project was funded — 4TU has a dedicated funder field]

## Language

English

## Geographical coverage (if requested)

Not applicable (synthetic/industrial imagery, no geographic association)

## Time period covered

[Data collection period, e.g. 2024–2026]

---

## Data structure

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
    └── stage1_proposal_recall.csv     — proposal-stage recall analysis (two-stage pipeline)
```

## Dataset statistics

| Property | Value |
|---|---|
| Total images | 615 |
| Unique board layouts | 197 (3.12 ± 1.02 images per design) |
| Total classes | 31 (23 present, 8 missing) |
| Present-component instances | 117,091 |
| Missing-component instances | 9,435 |
| Resolution | 1024×1024 |
| Source | Built on RF100 (Roboflow) |

## Splits: board-type-aware, not random

This is the single most important usage note. All folds are board-type-aware: no board layout appears in both the train and test partition of any fold (70% of board designs for training, 30% for validation per fold). Re-splitting the dataset randomly will produce results that are not comparable to those reported in the associated paper, because random splits allow board-layout memorization to inflate performance.

## Class ID versioning note

An earlier internal version of this dataset had missing-component classes indexed 23–30 due to a labeling bug; this has been corrected to 15–22 in the version deposited here (v1.0). Any prior internal copies with the old indexing should be discarded.

## Benchmark results

On the primary task (M — missing components), all one-stage and transformer-based detectors
(YOLOv8/v11/26, RT-DETR, D-FINE) show FNR consistently above 0.87; YOLOv11 is the best
one-stage model (mAP 0.09 ± 0.03, FNR 0.87 ± 0.03). The two-stage pipeline (YOLOv11 region
proposals + ResNet-18 classifier) achieves the best result on Task M overall — mAP 0.15 ± 0.05,
F1 0.13 ± 0.04, FNR 0.75 ± 0.07 — the lowest FNR of any evaluated method. On the conventional
tasks, YOLOv11 reaches mAP 0.30 ± 0.05 on Task A and mAP 0.42 ± 0.09 (F1 0.50 ± 0.08) on Task C.

## Known limitations

- Anomaly-detection methods (PatchCore, PaDiM, DRAEM, Reverse Distillation) achieve mAP 0.0 / FNR 1.0 on this dataset due to diffuse activation across heterogeneous board layouts, rather than localized activation at missing footprints. This is an expected, reported property of the dataset, not an evaluation error.
- Stage-1 region proposal recall (~0.18) is the primary performance bottleneck; end-to-end detection performance is capped by this regardless of downstream classifier choice.

## How to cite

```bibtex
@dataset{villa2026pcbmcdata,
  title     = {PCB-MC: Missing Component Detection in Printed Circuit Boards},
  author    = {Villa, B. and Talavera, E. and Gibson, I.},
  year      = {2026},
  publisher = {4TU.ResearchData},
  doi       = {10.4121/7c6e5bda-eecf-4feb-8a3a-7f823b1b0140}
}
```

```bibtex
@inproceedings{villa2026pcbmc,
  title     = {PCB-MC: Missing Component Detection in Printed Circuit Boards},
  author    = {Villa, B. and Talavera, E. and Gibson, I.},
  booktitle = {Proceedings of the IEEE European Workshop on Visual Information Processing (EUVIP)},
  year      = {2026},
  address   = {Luxembourg}
}
```

---

### Notes on submitting to 4TU specifically

- 4TU requires the submitting author to have (or create) an ORCID and typically a University of Twente affiliation login for deposit under the institutional collection — check with your data steward/library contact if you don't already have deposit access.
- 4TU assigns the DOI automatically upon publication of the record; embargo periods are supported if you want the DOI reserved before the data is publicly downloadable (useful if camera-ready deadline is earlier than data-cleaning completion).
- File size limits and any required data management plan (DMP) reference should be confirmed with your faculty's data steward before upload, since these can be discipline- or faculty-specific at Twente.
