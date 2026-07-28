# README — PCB-MC Dataset

This README is included inside the data package itself, so that anyone who downloads the files retains this information even without access to the 4TU.ResearchData web page.

---

## Dataset name

PCB-MC: Missing Component Detection in Printed Circuit Boards

## Authors

- Betsy Villa, University of Twente
- E. Talavera, University of Twente
- I. Gibson, University of Twente

## Contact

b.j.villabrochero@utwente.nl

## Date of deposit

[YYYY-MM-DD]

## DOI

https://doi.org/10.4121/7c6e5bda-eecf-4feb-8a3a-7f823b1b0140

## License

CC BY 4.0. You are free to share and adapt this dataset for any purpose, including commercially, provided you give appropriate credit (see "How to cite" below).

---

## What this dataset is

PCB-MC supports the task of detecting **missing components** on printed circuit boards: identifying empty footprints where a component should be present but is not. This differs from standard object detection because there is no visible object to learn from — only its absence.

The dataset is built on RF100 (Roboflow) and contains:

- 615 images
- 197 unique board layouts
- 31 classes total: 23 present-component classes, 8 missing-component classes
- ~9,400 missing-component instances

## Folder structure

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
├── class_list.txt
└── README.md   (this file)
```

Each `fold_k/` directory contains the images and annotations assigned to that fold, already split into train/test partitions according to the board-identity-aware protocol described below.

## File formats

- Images: [e.g. JPG/PNG, resolution range]
- Annotations: [e.g. COCO-style JSON, one file per fold]
- Class list: `class_list.txt` maps class IDs to class names

## IMPORTANT — how to split this dataset correctly

**Do not re-split this dataset at random.** The folds provided here are **board-identity-aware**: no board layout appears in both the train and test partition within the same fold. This matters because random splitting allows a model to memorize specific board layouts rather than learning to detect missing components — this alone was found to inflate reported mAP by up to 55 points.

If you need to regenerate splits (e.g. for a different fold count), you must preserve board-identity separation between train and test. The split-generation logic is provided in the companion code repository (see below), not in this data package, so that the exact procedure stays version-controlled and auditable.

## Class ID note

Missing-component classes are indexed **15–22**. An earlier internal version of this dataset mistakenly indexed them as 23–30; that version has been fully superseded. If any file, script, or cached copy you have uses IDs 23–30 for missing components, discard it and use only the version in this deposit.

## Known behavior of anomaly detection methods on this dataset

If you plan to benchmark anomaly detection methods (e.g. PatchCore, PaDiM, DRAEM, Reverse Distillation) on PCB-MC: in our own evaluation, all four methods scored mAP 0.0 / FNR 1.0. This is not a bug in your evaluation pipeline — it reflects diffuse activation across heterogeneous board layouts rather than localized activation at the missing footprint. This failure mode is discussed in the associated paper.

## Related resources

- **Paper:** [Paper title], EUVIP 2026 (IEEE), Luxembourg
- **Code repository:** https://github.com/betvillab/pcb-mc — training, evaluation, baselines, and scripts to reproduce every table and figure in the paper
- **Roboflow Universe mirror (optional):** [URL]

## How to cite

If you use this dataset, please cite both the dataset deposit and the paper:

```bibtex
@dataset{[dataset_citekey],
  title     = {PCB-MC: Missing Component Detection in Printed Circuit Boards},
  author    = {Villa, Betsy and Talavera, E. and Gibson, I.},
  year      = {2026},
  publisher = {4TU.ResearchData},
  doi       = {10.4121/7c6e5bda-eecf-4feb-8a3a-7f823b1b0140}
}

@inproceedings{[paper_citekey],
  title     = {[Paper title]},
  author    = {Villa, Betsy and Talavera, E. and Gibson, I.},
  booktitle = {Proceedings of the IEEE European Workshop on Visual Information Processing (EUVIP)},
  year      = {2026},
  address   = {Luxembourg}
}
```

## Changelog

- **v1.0** — Initial public release. Corrected missing-component class indexing (15–22, superseding an internal version that used 23–30).
