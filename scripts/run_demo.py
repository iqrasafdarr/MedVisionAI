"""CLI entry point: launches the Gradio demo.

Usage:
    python scripts/run_demo.py
"""

from src.multimodal.app import build_interface


def main() -> None:
    demo = build_interface()
    demo.launch()


if __name__ == "__main__":
    main()
