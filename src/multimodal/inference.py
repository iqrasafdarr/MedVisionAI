"""Runs BLIP captioning on a single image and returns a raw description
string. No free-form clinical claims are generated here — this is pure
visual captioning, kept separate from the structured report template in
report_generator.py which is what enforces the disclaimer/format contract.

TODO(Phase 5): implement generate_caption(image, model, processor) -> str.
"""


def generate_caption(image, model, processor) -> str:
    raise NotImplementedError("Implemented in Phase 5.")
