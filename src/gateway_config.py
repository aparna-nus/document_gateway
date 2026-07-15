"""Locked configuration for the document pre-validation gateway."""

from __future__ import annotations

from pathlib import Path

MODEL_ID = "qwen2.5vl:3b"
PROMPT_VERSION = "crop_sensitive_v5"
TEMPERATURE = 0.0
MAX_TOKENS = 300
RANDOM_SEED = 42

# Notebook 3 locked routing thresholds.
APPROVED_THRESHOLDS = {
    "class4_document_max": 0.199772,
    "class2_crop_min": 1.010000,
    "class3_quality_risk_min": 0.600999,
    "class1_document_min": 0.727049,
    "class1_crop_max": 0.291236,
    "class1_quality_risk_max": 0.569187,
}

# Final VLM-only PASS guardrail.
# A VLM Class 1 result below this document score is sent to manual review.
LOW_DOCUMENT_PASS_THRESHOLD = 0.30

CLASS_NAMES = {
    0: "Indeterminate",
    1: "Clear and Readable",
    2: "Content Got Cut",
    3: "Unclear / Unreadable",
    4: "Random Image",
}

ROUTING_ACTIONS = {
    0: "MANUAL_REVIEW",
    1: "PASS",
    2: "FLAG",
    3: "DENOISE",
    4: "BLOCK",
}

SCALER_FEATURES = [
    "white_pixel_ratio",
    "dark_pixel_ratio",
    "mean_saturation",
    "otsu_separation",
    "crop_cc_touch_ratio",
    "crop_cc_touch_area_ratio",
    "crop_boundary_ink_max",
    "crop_projection_delta_max",
    "local_laplacian_p25",
    "tenengrad_mean",
    "quality_fft_local_p25",
    "intensity_dynamic_range",
    "underexposed_ratio",
    "overexposed_ratio",
    "near_binary_ratio",
]

LOG_SCALE_FEATURES = {
    "local_laplacian_p25",
    "tenengrad_mean",
}

VLM_PROMPT_TEMPLATE = """
You are the fallback classifier in a document pre-validation pipeline.
Classify the image into exactly one category:

<guidelines>
1. Clear and Readable
   A text-bearing document is the primary subject, all meaningful content
   is visible, and the text is sufficiently readable.

2. Content of the document got cut
   A text-bearing document is present, but part of the document layout is
   visibly clipped, cut off, missing beyond the image frame, or materially
   occluded.

   Inspect the TOP, BOTTOM, LEFT, and RIGHT image boundaries separately.

   Strong evidence for Class 2 includes any of the following:
   - text, a table, a form field, a signature area, a photograph, a header,
     a footer, or a printed border continues beyond the image boundary;
   - a line of text or table row is visibly truncated;
   - part of an ID card, receipt, page, or form is outside the frame;
   - the document layout clearly appears incomplete on one or more sides.

   Do not choose Class 1 merely because the remaining visible text is readable.
   Tight framing alone is not enough for Class 2 when the complete layout is still visible.

3. Unclear / Unreadable
   The document is substantially complete, but blur, motion shake,
   darkness, low contrast, glare, compression, noise, or artefacts
   make it unreadable. Slight blur without loss of readability redirects to class 1.

4. Random image
   The image is not primarily a text-bearing document. This includes
   landscapes, people, animals, rooms, objects, fields, machinery,
   blank notebooks, and documents appearing only as incidental scene objects.
</guidelines>

<task>
Decision order:
A. If a text-bearing document is not the primary subject, choose Class 4.
B. Otherwise inspect all four image boundaries. If any document content or
   layout is visibly clipped, truncated, missing beyond the frame, or
   materially occluded, choose Class 2.
C. Otherwise, if the document is materially difficult to read, choose Class 3.
D. Otherwise, choose Class 1.

If both cropping and degradation are present, choose Class 2.
</task>

Use confidence conservatively. Set indeterminate=true when the image cannot
be assigned safely. Never classify a non-document image as a valid document.

Return only the requested JSON object.
"""

OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "class_prediction": {"type": "integer", "enum": [1, 2, 3, 4]},
        "confidence_score": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        "document_present": {"type": "boolean"},
        "meaningful_content_cropped": {"type": "boolean"},
        "materially_unreadable": {"type": "boolean"},
        "indeterminate": {"type": "boolean"},
        "justification": {"type": "string"},
    },
    "required": [
        "class_prediction",
        "confidence_score",
        "document_present",
        "meaningful_content_cropped",
        "materially_unreadable",
        "indeterminate",
        "justification",
    ],
    "additionalProperties": False,
}

DEFAULT_LOCKED_CONFIG_PATH = Path("config/final_locked_pipeline_config.json")
