"""Gradio demo: upload an image -> view classifier prediction + confidence
-> view generated description -> view the structured report with disclaimer.

TODO(Phase 6): wire together src/classification/{...} inference,
src/multimodal/{inference,report_generator}.py into a Gradio Blocks/Interface.
Entry point is scripts/run_demo.py.
"""


def build_interface():
    raise NotImplementedError("Implemented in Phase 6.")
