"""Loads BLIP-base (Salesforce/blip-image-captioning-base) for structured
image description generation.

TODO(Phase 5): load via transformers.BlipProcessor / BlipForConditionalGeneration.
Zero-shot captioning first; fine-tuning only if zero-shot output quality
needs it (per configs/vlm.yaml mode setting).
"""


def load_blip_model(model_config: dict):
    raise NotImplementedError("Implemented in Phase 5.")
