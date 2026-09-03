"""Generate a visual description using BLIP."""

import torch
from PIL import Image


def generate_caption(image, model, processor, device=None) -> str:
    if not isinstance(image, Image.Image):
        image = Image.fromarray(image)

    image = image.convert("RGB")

    inputs = processor(images=image, return_tensors="pt")

    if device is None:
        device = next(model.parameters()).device

    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=40,
            num_beams=3,
        )

    caption = processor.decode(output[0], skip_special_tokens=True).strip()
    return caption
