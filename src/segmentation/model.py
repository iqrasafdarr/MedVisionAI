"""MONAI U-Net model construction and device selection for brain tumor
segmentation. Uses MONAI's own U-Net implementation directly — no custom
architecture — per the project plan."""

from __future__ import annotations

import torch
from monai.networks.nets import UNet


def get_device(requested_device: str = "cuda") -> torch.device:
    """Resolve the requested device, falling back to CPU automatically if
    CUDA was requested but is unavailable."""
    if requested_device == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(requested_device)


def build_model(model_config: dict, spatial_dims: int = 2) -> UNet:
    """Construct a monai.networks.nets.UNet from a resolved model config
    block (configs/segmentation.yaml: model.*).

    Args:
        model_config: dict with keys in_channels, out_channels, channels,
            strides, num_res_units.
        spatial_dims: 2 for the 2D-slice pipeline used in this project,
            3 if a future phase moves to full volumetric training.
    """
    required_keys = {"in_channels", "out_channels", "channels", "strides", "num_res_units"}
    missing = required_keys - model_config.keys()
    if missing:
        raise ValueError(f"model config is missing required keys: {missing}")

    if len(model_config["channels"]) != len(model_config["strides"]) + 1:
        raise ValueError(
            f"channels must have exactly one more entry than strides "
            f"(got {len(model_config['channels'])} channels, "
            f"{len(model_config['strides'])} strides)."
        )

    model = UNet(
        spatial_dims=spatial_dims,
        in_channels=model_config["in_channels"],
        out_channels=model_config["out_channels"],
        channels=model_config["channels"],
        strides=model_config["strides"],
        num_res_units=model_config["num_res_units"],
    )
    return model
