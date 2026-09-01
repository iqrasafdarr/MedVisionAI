"""CLI entry point for segmentation evaluation on the held-out test split.

Usage:
    python scripts/evaluate_segmentation.py --config configs/segmentation.yaml
    python scripts/evaluate_segmentation.py --config configs/segmentation.yaml \
        --checkpoint results/segmentation/checkpoints/best_model.pt
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import yaml

from src.segmentation.evaluate import evaluate


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Path to a checkpoint. Defaults to output.checkpoint_dir/best_model.pt from the config.",
    )
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    metrics = evaluate(config, checkpoint_path=args.checkpoint)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
