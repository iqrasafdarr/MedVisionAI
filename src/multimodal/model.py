"""BLIP model loading for zero-shot image description."""

import torch
from transformers import BlipProcessor, BlipForConditionalGeneration


def load_blip_model(model_config: dict):
    model_name = model_config["name"]
    requested_device = model_config.get("device", "cpu")

    device = torch.device(
        "cuda" if requested_device == "cuda" and torch.cuda.is_available() else "cpu"
    )

    processor = BlipProcessor.from_pretrained(model_name)
    model = BlipForConditionalGeneration.from_pretrained(model_name)
    model.to(device)
    model.eval()

    return model, processor, device
