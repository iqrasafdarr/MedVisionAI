# Limitations

This is a research prototype built by an undergraduate researcher on limited
Colab compute. It is **not** a clinical tool and must not be used for
diagnosis. Specific limitations:

## Scope and integration

- Segmentation, classification, and the vision-language module are
  **independently trained and evaluated**. This repository does not
  implement or claim an integrated segmentation-to-classification pipeline
  (e.g., using predicted tumor masks to inform or crop input to the
  classifier). If that integration is built and evaluated in the future,
  it will be documented as its own explicit experiment — not implied by the
  architecture diagram.

## Dataset limitations

- Segmentation: MSD Task01_BrainTumour is a single, curated dataset; no
  cross-institution or cross-scanner domain-shift evaluation is performed.
- Classification: the 4-class dataset (7,023 images) is modest in scale by
  clinical-deployment standards; class balance and any residual imbalance
  are documented in `docs/experiments.md` once training runs.
- Neither dataset's demographic or acquisition-site diversity is
  characterized in this project — a real limitation for any claim of
  generalizability.

## Computational limitations

- Segmentation uses 2D slices from 3D volumes (not full 3D U-Net training)
  due to Colab session/VRAM constraints — this trades some spatial context
  for feasibility.
- Hyperparameter search is limited in scope; reported results reflect a
  small number of configurations, not an exhaustive search.

## Domain shift and class imbalance

- No external test set is used for the segmentation module in this phase.
- Classification class imbalance (if present) is handled via automatic
  class weighting (`configs/classification.yaml: data.class_weighting`),
  not resampling or synthetic data generation — a simpler but less
  aggressive mitigation.

## Vision-language module

- BLIP-base is a general-purpose captioning model, not medically
  fine-tuned by default in this project's zero-shot mode. Its raw captions
  may contain imprecise or non-clinical phrasing — this is exactly why
  output is constrained to a fixed structured template rather than used
  as free-form text, and why a disclaimer is a non-model-generated,
  always-present section of every report.
- No hallucination-rate evaluation has been performed on the VLM's output;
  the template constraint is a mitigation, not a guarantee.

## No clinical validation

- No results in this repository have been validated against clinical
  ground truth by a radiologist or clinician.
- No claim of FDA approval, clinical-grade performance, or diagnostic
  capability is made or implied anywhere in this project.
