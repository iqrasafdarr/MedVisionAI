# Data

No raw or processed data is committed to this repository (`.gitignore` excludes
everything under `data/` except this file). Set the two datasets up as follows.

## 1. Medical Segmentation Decathlon — Task01_BrainTumour (segmentation)

Downloaded automatically via MONAI's `DecathlonDataset` API:

```bash
python scripts/download_data.py --config configs/segmentation.yaml
```

This populates `data/msd_task01/`. The dataset is provided by the Medical
Segmentation Decathlon for research use — see http://medicaldecathlon.com for
current licensing terms before any redistribution or non-research use.

## 2. Existing 4-class brain tumor MRI dataset (classification)

This project reuses the dataset from prior work (see `docs/experiments.md` for
provenance). It is **not** re-downloaded or committed here — set
`data.root_dir` in `configs/classification.yaml` to wherever your local copy
already lives.

## Why nothing is committed

Both datasets carry research-use licensing terms that don't permit blanket
redistribution, and MSD alone is multiple GB — committing either would bloat
the repository and risk licensing violations. Anyone cloning this repo
reproduces the data locally via the steps above.
