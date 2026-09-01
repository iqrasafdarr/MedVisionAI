"""Global seeding utility — call this first in every training/eval script
so results are reproducible given the same config."""

import os
import random

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Seed Python, NumPy, and torch (CPU + CUDA) RNGs.

    Args:
        seed: seed value, read from the run's config file.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    # Trade a little speed for determinism — acceptable at this project's scale.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
