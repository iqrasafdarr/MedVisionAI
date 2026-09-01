"""CLI entry point for segmentation training.

Usage:
    python scripts/train_segmentation.py --config configs/segmentation.yaml
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import yaml

from src.utils.seed import set_seed
from src.segmentation.train import train


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    set_seed(config["seed"])
    train(config)


if __name__ == "__main__":
    main()
