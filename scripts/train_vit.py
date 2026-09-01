"""CLI entry point for ViT classification fine-tuning.

Usage:
    python scripts/train_vit.py --config configs/classification.yaml
"""

import argparse

import yaml

from src.utils.seed import set_seed
from src.classification.train import train


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
