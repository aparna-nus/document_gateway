"""End-to-end CV + Ollama VLM document gateway.

Usage:
    python document_gateway.py image.jpg \
        --config config/final_locked_pipeline_config.json

The output is a single JSON record containing CV features, routing decisions,
the optional VLM response, and the final operational action.
"""

from __future__ import annotations

import argparse
import json
import time
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
import ollama
from PIL import Image, ImageOps

from cv_features import score_image
from gateway_config import (
    APPROVED_THRESHOLDS,
    CLASS_NAMES,
    DEFAULT_LOCKED_CONFIG_PATH,
    LOW_DOCUMENT_PASS_THRESHOLD,
    MAX_TOKENS,
    MODEL_ID,
    OUTPUT_SCHEMA,
    RANDOM_SEED,
    ROUTING_ACTIONS,
    TEMPERATURE,
    VLM_PROMPT_TEMPLATE,
)


def load_locked_config(path: Path | None) -> dict[str, Any]:
    """
    Load the exported notebook configuration.

    The scaler is required because Notebook 4 fitted it on the tuning split.
    """
    config_path = path or DEFAULT_LOCKED_CONFIG_PATH

    if not config_path.is_file():
        raise FileNotFoundError(
            f"Locked config not found: {config_path}. "
            "Export config/final_locked_pipeline_config.json from Notebook 4."
        )

    with config_path.open("r", encoding="utf-8") as file:
        return json.load(file)


def apply_locked_cv_router(
    scores: dict[str, float],
    thresholds: dict[str, float],
) -> dict[str, Any]:
    document_score = float(scores["document_score"])
    crop_score = float(scores["crop_score"])
    quality_risk = float(scores["quality_risk"])

    if document_score <= thresholds["class4_document_max"]:
        prediction, route = 4, "cv_class4"
    elif crop_score >= thresholds["class2_crop_min"]:
        prediction, route = 2, "cv_class2"
    elif quality_risk >= thresholds["class3_quality_risk_min"]:
        prediction, route = 3, "cv_class3"
    elif (
        document_score >= thresholds["class1_document_min"]
        and crop_score <= thresholds["class1_crop_max"]
        and quality_risk <= thresholds["class1_quality_risk_max"]
    ):
        prediction, route = 1, "cv_class1"
    else:
        prediction, route = None, "vlm_fallback"

    return {
        "cv_prediction": prediction,
        "cv_route": route,
        "requires_vlm": route == "vlm_fallback",
    }


def normalize_image_to_png_bytes(image_path: Path) -> bytes:
    image_path = image_path.expanduser().resolve()
    if not image_path.is_file():
        raise FileNotFoundError(f"Image not found: {image_path}")

    with Image.open(image_path) as image:
        image = ImageOps.exif_transpose(image).convert("RGB")
        buffer = BytesIO()
        image.save(buffer, format="PNG")
        return buffer.getvalue()


def validate_prediction(prediction: dict[str, Any]) -> dict[str, Any]:
    required = {
        "class_prediction",
        "confidence_score",
        "document_present",
        "meaningful_content_cropped",
        "materially_unreadable",
        "indeterminate",
        "justification",
    }
    missing = required - set(prediction)
    if missing:
        raise ValueError(f"Prediction missing fields: {sorted(missing)}")

    prediction = dict(prediction)
    prediction["class_prediction"] = int(prediction["class_prediction"])
    prediction["confidence_score"] = float(prediction["confidence_score"])

    if prediction["class_prediction"] not in {1, 2, 3, 4}:
        raise ValueError("class_prediction must be one of 1, 2, 3, 4")
    if not 0.0 <= prediction["confidence_score"] <= 1.0:
        raise ValueError("confidence_score must be between 0 and 1")

    for field in (
        "document_present",
        "meaningful_content_cropped",
        "materially_unreadable",
        "indeterminate",
    ):
        if not isinstance(prediction[field], bool):
            raise ValueError(f"{field} must be a boolean")

    prediction["justification"] = str(prediction["justification"]).strip()
    return prediction


def call_ollama_vlm(
    image_path: Path,
    model_config: dict[str, Any],
) -> dict[str, Any]:
    start = time.perf_counter()
    raw_output = None

    try:
        response = ollama.chat(
            model=model_config.get("model_id", MODEL_ID),
            messages=[
                {
                    "role": "user",
                    "content": model_config.get(
                        "prompt_template", VLM_PROMPT_TEMPLATE
                    ),
                    "images": [normalize_image_to_png_bytes(image_path)],
                }
            ],
            format=model_config.get("output_schema", OUTPUT_SCHEMA),
            options={
                "temperature": model_config.get("temperature", TEMPERATURE),
                "num_predict": model_config.get("max_tokens", MAX_TOKENS),
                "seed": RANDOM_SEED,
            },
            stream=False,
            keep_alive="10m",
        )
        raw_output = response.message.content
        prediction = validate_prediction(json.loads(raw_output))

        return {
            "success": True,
            "prediction": prediction,
            "raw_output": raw_output,
            "latency_seconds": time.perf_counter() - start,
            "error": None,
        }
    except Exception as error:
        return {
            "success": False,
            "prediction": None,
            "raw_output": raw_output,
            "latency_seconds": time.perf_counter() - start,
            "error": f"{type(error).__name__}: {error}",
        }


def apply_final_guardrail(
    prediction: int,
    decision_source: str,
    document_score: float,
    threshold: float,
) -> tuple[int, str, str | None]:
    """
    Final guardrail from the frozen Notebook 4.

    It applies only to a VLM-produced Class 1. A very low document score does
    not prove Class 4, so the image is held for manual review instead of blocked.
    """
    if (
        decision_source == "vlm"
        and prediction == 1
        and document_score < threshold
    ):
        return (
            0,
            "guardrail_review",
            "vlm_class1_rejected_by_low_document_score",
        )

    return prediction, decision_source, None


def run_gateway(
    image_path: Path,
    config_path: Path | None = None,
) -> dict[str, Any]:
    config = load_locked_config(config_path)
    thresholds = config.get("thresholds", APPROVED_THRESHOLDS)
    model_config = config.get("model", {})
    scaler = config["scaler"]

    scores = score_image(str(image_path), scaler)
    cv_result = apply_locked_cv_router(scores, thresholds)

    vlm_result: dict[str, Any] | None = None

    if cv_result["requires_vlm"]:
        vlm_result = call_ollama_vlm(image_path, model_config)
        prediction_payload = vlm_result["prediction"] or {}

        if (
            vlm_result["success"]
            and not bool(prediction_payload.get("indeterminate", True))
            and prediction_payload.get("class_prediction") is not None
        ):
            final_prediction = int(prediction_payload["class_prediction"])
            decision_source = "vlm"
        else:
            final_prediction = 0
            decision_source = "indeterminate"
    else:
        final_prediction = int(cv_result["cv_prediction"])
        decision_source = "cv"

    guardrail_threshold = float(
        config.get(
            "low_document_pass_threshold",
            LOW_DOCUMENT_PASS_THRESHOLD,
        )
    )

    final_prediction, decision_source, safety_gate_reason = (
        apply_final_guardrail(
            final_prediction,
            decision_source,
            float(scores["document_score"]),
            guardrail_threshold,
        )
    )

    return {
        "image_path": str(image_path),
        "cv": {
            **cv_result,
            "document_score": float(scores["document_score"]),
            "crop_score": float(scores["crop_score"]),
            "quality_risk": float(scores["quality_risk"]),
            "quality_score": float(scores["quality_score"]),
        },
        "vlm": vlm_result,
        "final": {
            "class_id": final_prediction,
            "class_name": CLASS_NAMES[final_prediction],
            "routing_action": ROUTING_ACTIONS[final_prediction],
            "decision_source": decision_source,
            "safety_gate_reason": safety_gate_reason,
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the document pre-validation gateway on one image."
    )
    parser.add_argument("image_path", type=Path)
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_LOCKED_CONFIG_PATH,
        help="Notebook 4 exported locked configuration JSON.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional path to save the resulting JSON.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_gateway(args.image_path, args.config)
    rendered = json.dumps(result, indent=2, ensure_ascii=False)
    print(rendered)

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
