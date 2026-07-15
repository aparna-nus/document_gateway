"""OpenCV feature extraction and composite CV scoring."""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np
import pandas as pd

from gateway_config import LOG_SCALE_FEATURES, SCALER_FEATURES


def _read_image(image_path: str) -> tuple[np.ndarray, np.ndarray]:
    image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    return image_bgr, gray


def _safe_ratio(mask: np.ndarray) -> float:
    return float(np.mean(mask.astype(np.float32)))


def extract_raw_features(image_path: str) -> dict[str, float]:
    """
    Extract the raw CV signals consumed by the locked scoring layer.

    The formulas mirror the feature families used by the final notebook:
    document appearance, crop/boundary evidence, and quality/readability.
    """
    bgr, gray = _read_image(image_path)
    height, width = gray.shape
    pixels = max(height * width, 1)

    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    saturation = hsv[:, :, 1].astype(np.float32) / 255.0

    white_pixel_ratio = _safe_ratio(gray >= 235)
    dark_pixel_ratio = _safe_ratio(gray <= 45)
    mean_saturation = float(np.mean(saturation))

    otsu_threshold, binary = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    foreground = gray < otsu_threshold
    bg_values = gray[~foreground]
    fg_values = gray[foreground]
    if len(bg_values) and len(fg_values):
        otsu_separation = float(
            abs(float(bg_values.mean()) - float(fg_values.mean())) / 255.0
        )
    else:
        otsu_separation = 0.0

    # Foreground mask for text/ink and connected-component boundary checks.
    ink = (255 - binary).astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        (ink > 0).astype(np.uint8), connectivity=8
    )

    component_count = max(num_labels - 1, 0)
    touching_count = 0
    touching_area = 0
    total_component_area = 0

    for label_id in range(1, num_labels):
        x, y, w, h, area = stats[label_id]
        total_component_area += int(area)
        touches = x <= 0 or y <= 0 or (x + w) >= width or (y + h) >= height
        if touches:
            touching_count += 1
            touching_area += int(area)

    crop_cc_touch_ratio = (
        touching_count / component_count if component_count else 0.0
    )
    crop_cc_touch_area_ratio = (
        touching_area / total_component_area if total_component_area else 0.0
    )

    band_y = max(1, int(round(height * 0.03)))
    band_x = max(1, int(round(width * 0.03)))
    ink_bool = ink > 0
    boundary_ratios = [
        float(ink_bool[:band_y, :].mean()),
        float(ink_bool[-band_y:, :].mean()),
        float(ink_bool[:, :band_x].mean()),
        float(ink_bool[:, -band_x:].mean()),
    ]
    crop_boundary_ink_max = max(boundary_ratios)

    row_projection = ink_bool.mean(axis=1)
    col_projection = ink_bool.mean(axis=0)
    crop_projection_delta_max = float(
        max(
            abs(row_projection[0] - row_projection[min(band_y, height - 1)]),
            abs(row_projection[-1] - row_projection[max(0, height - band_y - 1)]),
            abs(col_projection[0] - col_projection[min(band_x, width - 1)]),
            abs(col_projection[-1] - col_projection[max(0, width - band_x - 1)]),
        )
    )

    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    local_window = 32
    local_laplacian_values = []
    for y in range(0, height, local_window):
        for x in range(0, width, local_window):
            patch = laplacian[y:y + local_window, x:x + local_window]
            if patch.size:
                local_laplacian_values.append(float(np.var(patch)))
    local_laplacian_p25 = float(
        np.percentile(local_laplacian_values, 25)
        if local_laplacian_values
        else 0.0
    )

    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    tenengrad_mean = float(np.mean(gx * gx + gy * gy))

    fft = np.fft.fftshift(np.fft.fft2(gray.astype(np.float32)))
    magnitude = np.log1p(np.abs(fft))
    fft_local_values = []
    for y in range(0, height, local_window):
        for x in range(0, width, local_window):
            patch = magnitude[y:y + local_window, x:x + local_window]
            if patch.size:
                fft_local_values.append(float(np.mean(patch)))
    quality_fft_local_p25 = float(
        np.percentile(fft_local_values, 25)
        if fft_local_values
        else 0.0
    )

    intensity_dynamic_range = float(
        (np.percentile(gray, 95) - np.percentile(gray, 5)) / 255.0
    )
    underexposed_ratio = _safe_ratio(gray <= 35)
    overexposed_ratio = _safe_ratio(gray >= 245)
    near_binary_ratio = _safe_ratio((gray <= 20) | (gray >= 235))

    return {
        "white_pixel_ratio": white_pixel_ratio,
        "dark_pixel_ratio": dark_pixel_ratio,
        "mean_saturation": mean_saturation,
        "otsu_separation": otsu_separation,
        "crop_cc_touch_ratio": float(crop_cc_touch_ratio),
        "crop_cc_touch_area_ratio": float(crop_cc_touch_area_ratio),
        "crop_boundary_ink_max": float(crop_boundary_ink_max),
        "crop_projection_delta_max": float(crop_projection_delta_max),
        "local_laplacian_p25": local_laplacian_p25,
        "tenengrad_mean": tenengrad_mean,
        "quality_fft_local_p25": quality_fft_local_p25,
        "intensity_dynamic_range": intensity_dynamic_range,
        "underexposed_ratio": underexposed_ratio,
        "overexposed_ratio": overexposed_ratio,
        "near_binary_ratio": near_binary_ratio,
    }


def apply_locked_scaler(
    features: dict[str, float],
    scaler: dict[str, dict[str, float]],
) -> dict[str, float]:
    output = dict(features)

    missing = [name for name in SCALER_FEATURES if name not in scaler]
    if missing:
        raise KeyError(
            "Locked scaler is missing feature definitions: "
            + ", ".join(missing)
        )

    for feature_name in SCALER_FEATURES:
        value = float(output[feature_name])
        limits = scaler[feature_name]

        if limits.get("log1p", feature_name in LOG_SCALE_FEATURES):
            value = float(np.log1p(max(value, 0.0)))

        lower = float(limits["lower"])
        upper = float(limits["upper"])
        denominator = max(upper - lower, 1e-9)

        scaled = (min(max(value, lower), upper) - lower) / denominator
        output[f"{feature_name}_scaled"] = float(scaled)

    return output


def add_cv_scores(features: dict[str, float]) -> dict[str, float]:
    output = dict(features)

    output["document_score"] = (
        0.40 * output["white_pixel_ratio_scaled"]
        + 0.25 * (1.0 - output["mean_saturation_scaled"])
        + 0.15 * (1.0 - output["dark_pixel_ratio_scaled"])
        + 0.20 * output["otsu_separation_scaled"]
    )

    output["crop_score"] = (
        0.40 * output["crop_cc_touch_area_ratio_scaled"]
        + 0.25 * output["crop_cc_touch_ratio_scaled"]
        + 0.20 * output["crop_projection_delta_max_scaled"]
        + 0.15 * output["crop_boundary_ink_max_scaled"]
    )

    output["sharpness_score"] = (
        0.45 * output["local_laplacian_p25_scaled"]
        + 0.35 * output["tenengrad_mean_scaled"]
        + 0.20 * output["quality_fft_local_p25_scaled"]
    )

    output["blur_risk"] = 1.0 - output["sharpness_score"]
    output["contrast_risk"] = 1.0 - output["intensity_dynamic_range_scaled"]
    output["illumination_risk"] = max(
        output["underexposed_ratio_scaled"],
        output["overexposed_ratio_scaled"],
    )
    output["threshold_artifact_risk"] = output["near_binary_ratio_scaled"]

    output["quality_risk"] = float(
        np.clip(
            0.70 * output["blur_risk"]
            + 0.15 * output["contrast_risk"]
            + 0.10 * output["illumination_risk"]
            + 0.05 * output["threshold_artifact_risk"],
            0.0,
            1.0,
        )
    )
    output["quality_score"] = 1.0 - output["quality_risk"]

    return output


def score_image(
    image_path: str,
    scaler: dict[str, dict[str, float]],
) -> dict[str, float]:
    raw = extract_raw_features(image_path)
    scaled = apply_locked_scaler(raw, scaler)
    return add_cv_scores(scaled)
