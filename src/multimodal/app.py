"""MedVisionAI Gradio multimodal demonstration.

Pipeline:
MRI image
    -> ViT tumor classification
    -> BLIP visual description
    -> controlled structured report

Research prototype only. Outputs are not clinical diagnoses.
"""

from pathlib import Path

import gradio as gr
import torch
import yaml
from PIL import Image
from transformers import (
    ViTForImageClassification,
    ViTImageProcessor,
)

from src.multimodal.model import load_blip_model
from src.multimodal.inference import generate_caption
from src.multimodal.report_generator import build_report


ROOT = Path(__file__).resolve().parents[2]

with open(ROOT / "configs" / "classification.yaml", encoding="utf-8") as f:
    CLASS_CONFIG = yaml.safe_load(f)

with open(ROOT / "configs" / "vlm.yaml", encoding="utf-8") as f:
    VLM_CONFIG = yaml.safe_load(f)


DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ---------------------------------------------------------
# Load BLIP
# ---------------------------------------------------------

print("Loading BLIP...")
BLIP_MODEL, BLIP_PROCESSOR, BLIP_DEVICE = load_blip_model(
    VLM_CONFIG["model"]
)

print(f"BLIP device: {BLIP_DEVICE}")


# ---------------------------------------------------------
# Load ViT classifier
# ---------------------------------------------------------

print("Loading ViT classifier...")

CLASS_NAMES = CLASS_CONFIG["data"]["classes"]

CLASSIFIER_PROCESSOR = ViTImageProcessor.from_pretrained(
    CLASS_CONFIG["model"]["name"]
)

CLASSIFIER_MODEL = ViTForImageClassification.from_pretrained(
    CLASS_CONFIG["model"]["name"],
    num_labels=len(CLASS_NAMES),
    ignore_mismatched_sizes=True,
)

CHECKPOINT = (
    ROOT
    / "results"
    / "classification"
    / "checkpoints"
    / "best_model.pt"
)

if CHECKPOINT.exists():
    checkpoint = torch.load(
        CHECKPOINT,
        map_location="cpu",
        weights_only=False,
    )

    if isinstance(checkpoint, dict):
        state_dict = checkpoint.get(
            "model_state_dict",
            checkpoint.get("state_dict", checkpoint)
        )

        try:
            CLASSIFIER_MODEL.load_state_dict(
                state_dict,
                strict=False,
            )
            print("Loaded trained ViT checkpoint.")
        except Exception as exc:
            print("Checkpoint loading warning:", exc)
            print("Using initialized classifier weights.")
else:
    print("WARNING: trained ViT checkpoint not found.")
    print("Expected:", CHECKPOINT)

CLASSIFIER_MODEL.to(DEVICE)
CLASSIFIER_MODEL.eval()

print(f"ViT device: {DEVICE}")


# ---------------------------------------------------------
# ViT inference
# ---------------------------------------------------------

def classify_image(image):
    if image is None:
        return "No image provided.", 0.0, None

    if not isinstance(image, Image.Image):
        image = Image.fromarray(image)

    image = image.convert("RGB")

    inputs = CLASSIFIER_PROCESSOR(
        images=image,
        return_tensors="pt",
    )

    inputs = {
        key: value.to(DEVICE)
        for key, value in inputs.items()
    }

    with torch.no_grad():
        outputs = CLASSIFIER_MODEL(**inputs)
        probabilities = torch.softmax(
            outputs.logits,
            dim=-1,
        )[0]

    index = int(torch.argmax(probabilities))
    confidence = float(probabilities[index])

    prediction = CLASS_NAMES[index]

    scores = {
        CLASS_NAMES[i]: float(probabilities[i])
        for i in range(len(CLASS_NAMES))
    }

    return prediction, confidence, scores


# ---------------------------------------------------------
# Complete multimodal pipeline
# ---------------------------------------------------------

def analyze_mri(image):

    if image is None:
        return (
            "Please upload a brain MRI image.",
            0.0,
            {},
            "",
            "",
        )

    prediction, confidence, scores = classify_image(image)

    caption = generate_caption(
        image,
        BLIP_MODEL,
        BLIP_PROCESSOR,
        BLIP_DEVICE,
    )

    report = build_report(
        caption=caption,
        classifier_prediction=prediction,
        classifier_confidence=confidence,
        config=VLM_CONFIG,
    )

    return (
        prediction,
        confidence,
        scores,
        caption,
        report,
    )


# ---------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------

def build_interface():

    with gr.Blocks(
        title="MedVisionAI",
    ) as demo:

        gr.Markdown(
            """
# MedVisionAI
### Trustworthy Multimodal Brain MRI Analysis

A research prototype combining:

- **MONAI U-Net** for tumor segmentation
- **Vision Transformer (ViT)** for tumor classification
- **BLIP** for generic visual description
- **Controlled report generation** with explicit safety boundaries

**Important:** This system is for research and demonstration only.
It is not intended for clinical diagnosis or medical decision-making.
"""
        )

        with gr.Row():

            with gr.Column():

                image_input = gr.Image(
                    type="pil",
                    label="Upload Brain MRI",
                )

                analyze_button = gr.Button(
                    "Analyze MRI",
                    variant="primary",
                )

            with gr.Column():

                prediction_output = gr.Textbox(
                    label="Predicted Tumor Class",
                )

                confidence_output = gr.Number(
                    label="Classification Confidence",
                )

                scores_output = gr.Label(
                    label="Class Probabilities",
                    num_top_classes=4,
                )

        caption_output = gr.Textbox(
            label="BLIP Visual Description",
            lines=3,
        )

        report_output = gr.Markdown(
            label="Structured Multimodal Report",
        )

        analyze_button.click(
            fn=analyze_mri,
            inputs=image_input,
            outputs=[
                prediction_output,
                confidence_output,
                scores_output,
                caption_output,
                report_output,
            ],
        )

        gr.Markdown(
            """
---
**Research prototype only.**

The BLIP component provides generic visual descriptions and should not
be interpreted as a medical finding. The tumor classification result
comes from the trained ViT classifier. Always consult qualified medical
professionals for clinical decisions.
"""
        )

    return demo
