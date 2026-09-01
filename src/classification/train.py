"""Fine-tunes google/vit-base-patch16-224 (Hugging Face) on the 4-class
brain tumor MRI dataset via the Trainer API.

TODO(Phase 4): implement fine-tuning wired to src/classification/dataset.py,
src/classification/metrics.py, and src/utils/{seed,logging}.py.
Prior MobileNetV2/CNN results are NOT reproduced here — they're documented
separately in docs/experiments.md as prior work.
"""


def train(config: dict) -> dict:
    raise NotImplementedError("Implemented in Phase 4.")
