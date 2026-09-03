"""Controlled structured report generation."""

def build_report(
    caption: str,
    classifier_prediction: str,
    classifier_confidence: float,
    config: dict,
) -> str:

    disclaimer = config.get(
        "disclaimer_text",
        "Research prototype only. This output is not intended for clinical diagnosis."
    )

    sections = config.get("report_template", {}).get("sections", [])

    parts = []

    if "image_description" in sections:
        parts.append(
            "## Image Description\n"
            f"{caption}"
        )

    if "ai_classification_context" in sections:
        parts.append(
            "## AI Classification Context\n"
            f"Predicted class: {classifier_prediction}\n"
            f"Model confidence: {classifier_confidence:.4f}"
        )

    if "safety_disclaimer" in sections:
        parts.append(
            "## Safety Disclaimer\n"
            f"{disclaimer}"
        )

    return "\n\n".join(parts)
