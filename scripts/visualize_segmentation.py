"""CLI entry point for generating MRI / Ground Truth / Prediction / Overlay
figures for representative test-set cases.

Usage:
    python scripts/visualize_segmentation.py --config configs/segmentation.yaml
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import yaml

from src.segmentation.visualize import visualize_predictions


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--n-examples", type=int, default=8)
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    paths = visualize_predictions(
        config, checkpoint_path=args.checkpoint, n_examples=args.n_examples
    )
    print(f"Saved {len(paths)} visualization(s):")
    for p in paths:
        print(f"  {p}")


if __name__ == "__main__":
    main()
