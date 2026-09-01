"""Combines: (1) the BLIP caption, (2) the classifier's prediction +
confidence, and (3) a fixed disclaimer, into the controlled structured
report template defined in configs/vlm.yaml.

This is the guardrail against VLM hallucination: the template's section
boundaries and the disclaimer text are NOT model-generated — only the
"image_description" section content comes from BLIP; classification
context comes from real classifier output; the disclaimer is fixed text.

TODO(Phase 5): implement build_report(caption, classifier_output, config) -> str.
"""


def build_report(caption: str, classifier_prediction: str,
                  classifier_confidence: float, config: dict) -> str:
    raise NotImplementedError("Implemented in Phase 5.")
