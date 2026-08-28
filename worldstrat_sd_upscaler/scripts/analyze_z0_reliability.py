#!/usr/bin/env python3
"""Diagnose clean-latent predictions for HR-noised probes and pure-noise inference."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import sys
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.diffusion_prediction import (
    mix_x0_and_convert_model_output,
    model_output_to_x0,
    normalized_timesteps,
    timestep_range_mask,
    validate_timestep_range,
)
from src.latent_phi import LatentPhi

LOGGER = logging.getLogger("analyze_z0_reliability")
REPRESENTATIVE_FRACTIONS = (0.0, 0.01, 0.025, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 0.95, 1.0)
METRIC_FIELDS = (
    "sample_id",
    "filename",
    "timestep",
    "normalized_timestep",
    "alpha_bar",
    "snr",
    "log_snr",
    "zt_to_z0_mse",
    "noise_prediction_mse",
    "model_target_mse",
    "latent_mse",
    "latent_nmse",
    "psnr",
    "ssim",
    "lpips",
    "lr_reconstruction_mse",
    "predicted_z0_abs_mean",
    "predicted_z0_abs_max",
    "decoded_x0_raw_min",
    "decoded_x0_raw_max",
    "decoded_x0_below_zero_fraction",
    "decoded_x0_above_one_fraction",
    "base_decoded_x0_psnr",
    "base_decoded_x0_ssim",
    "base_decoded_x0_lpips",
    "phi_decoded_x0_psnr",
    "phi_decoded_x0_ssim",
    "phi_decoded_x0_lpips",
    "mixed_decoded_x0_psnr",
    "mixed_decoded_x0_ssim",
    "mixed_decoded_x0_lpips",
    "phi_z0_abs_mean",
    "mixed_z0_abs_mean",
    "phi_active",
    "phi_weight",
)
REVERSE_METRIC_FIELDS = (
    "sample_id",
    "filename",
    "inference_step",
    "inference_step_fraction",
    "timestep",
    "alpha_bar",
    "snr",
    "log_snr",
    "predicted_z0_abs_mean",
    "predicted_z0_abs_max",
    "predicted_clean_psnr",
    "predicted_clean_ssim",
    "base_decoded_x0_psnr",
    "base_decoded_x0_ssim",
    "predicted_clean_latent_mse_to_hr_z0",
    "decoded_x0_below_zero_fraction",
    "decoded_x0_above_one_fraction",
    "base_decoded_x0_lpips",
    "phi_decoded_x0_psnr",
    "phi_decoded_x0_ssim",
    "phi_decoded_x0_lpips",
    "mixed_decoded_x0_psnr",
    "mixed_decoded_x0_ssim",
    "mixed_decoded_x0_lpips",
    "phi_z0_abs_mean",
    "mixed_z0_abs_mean",
    "phi_active",
    "phi_weight",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("stage1_synthetic.yaml"))
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="LoRA/ConditionAdapter artifact directory; defaults to <config output_dir>/final.",
    )
    parser.add_argument("--output_dir", type=Path, default=None)
    parser.add_argument("--data_root", type=Path, default=None)
    parser.add_argument("--split", default="val")
    parser.add_argument("--num_samples", type=int, default=1)
    parser.add_argument("--noise_seed", type=int, default=1234)
    parser.add_argument("--vae_seed", type=int, default=4321)
    parser.add_argument("--timestep_stride", type=int, default=1)
    parser.add_argument(
        "--num_inference_steps",
        type=int,
        default=None,
        help="Pure-noise reverse trajectory length; defaults to validation_num_inference_steps.",
    )
    parser.add_argument(
        "--skip_training_timestep_analysis",
        action="store_true",
        help="Only run the pure-noise inference trajectory; skip HR-derived q(z_t|z_0).",
    )
    parser.add_argument("--low_res_noise_level", type=int, default=None)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--mixed_precision", choices=("no", "fp16", "bf16"), default=None)
    parser.add_argument("--skip_lpips", action="store_true")
    parser.add_argument("--phi_path", type=Path, default=None)
    parser.add_argument("--phi_timestep_range", type=float, nargs=2, default=None, metavar=("MIN", "MAX"))
    parser.add_argument("--phi_weight", type=float, default=None)
    parser.add_argument("--save_all_timestep_images", action="store_true")
    parser.add_argument(
        "--error_display_max",
        type=float,
        default=0.25,
        help="Fixed upper bound shared by all absolute-error heatmaps (RGB [0,1]).",
    )
    return parser.parse_args()


def resolve_config_path(path: Path) -> Path:
    candidates = (path, PROJECT_ROOT / path, PROJECT_ROOT / "configs" / path)
    for candidate in candidates:
        resolved = candidate.expanduser().resolve()
        if resolved.is_file():
            return resolved
    raise FileNotFoundError(
        "Configuration was not found. Checked: " + ", ".join(str(item.expanduser().resolve()) for item in candidates)
    )


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false.")
    return torch.device(name)


def resolve_dtype(name: str | None, config: dict[str, Any], device: torch.device) -> torch.dtype:
    precision = name or str(config.get("mixed_precision", "fp16" if device.type == "cuda" else "no"))
    if device.type == "cpu":
        return torch.float32
    return {"fp16": torch.float16, "bf16": torch.bfloat16, "no": torch.float32}[precision]


def prediction_to_x0(
    model_output: torch.Tensor,
    z_t: torch.Tensor,
    alpha_bar: torch.Tensor,
    prediction_type: str,
) -> torch.Tensor:
    """Backward-compatible wrapper around the shared float32 conversion."""
    return model_output_to_x0(model_output, z_t, alpha_bar, prediction_type)


def prediction_to_epsilon(
    model_output: torch.Tensor,
    z_t: torch.Tensor,
    alpha_bar: torch.Tensor,
    prediction_type: str,
) -> torch.Tensor:
    """Convert scheduler parameterization to the implied epsilon prediction."""
    sqrt_alpha = alpha_bar.sqrt()
    sqrt_beta = (1.0 - alpha_bar).clamp_min(0.0).sqrt()
    if prediction_type == "epsilon":
        return model_output
    if prediction_type == "v_prediction":
        return sqrt_alpha * model_output + sqrt_beta * z_t
    if prediction_type in {"sample", "x0", "x_start"}:
        return (z_t - sqrt_alpha * model_output) / sqrt_beta.clamp_min(1e-12)
    raise ValueError(f"Unsupported scheduler prediction_type={prediction_type!r}")


def true_model_target(
    scheduler: Any,
    z0: torch.Tensor,
    noise: torch.Tensor,
    timestep: torch.Tensor,
    prediction_type: str,
) -> torch.Tensor:
    if prediction_type == "epsilon":
        return noise
    if prediction_type == "v_prediction":
        return scheduler.get_velocity(z0, noise, timestep)
    if prediction_type in {"sample", "x0", "x_start"}:
        return z0
    raise ValueError(f"Unsupported scheduler prediction_type={prediction_type!r}")


def evaluated_timesteps(total: int, stride: int) -> list[int]:
    if total <= 0 or stride <= 0:
        raise ValueError(f"total and stride must be positive, got total={total}, stride={stride}")
    values = list(range(0, total, stride))
    if values[-1] != total - 1:
        values.append(total - 1)
    return values


def representative_timesteps(total: int, available: Iterable[int]) -> list[int]:
    candidates = sorted(set(int(value) for value in available))
    if not candidates:
        return []
    requested = [round(fraction * (total - 1)) for fraction in REPRESENTATIVE_FRACTIONS]
    return sorted(set(min(candidates, key=lambda value: (abs(value - target), value)) for target in requested))


def representative_inference_steps(total: int) -> set[int]:
    """Select trajectory positions by fraction, independent of scheduler length."""
    if total <= 0:
        raise ValueError(f"total must be positive, got {total}")
    return {round(fraction * (total - 1)) for fraction in REPRESENTATIVE_FRACTIONS}


def decode_latent_raw(vae: Any, latent: torch.Tensor, scaling_factor: float) -> torch.Tensor:
    """Decode to display-RGB units without hiding out-of-range values by clipping."""
    decoded = vae.decode(latent / scaling_factor, return_dict=False)[0]
    return decoded.float().add(1.0).div(2.0)


def decode_latent(vae: Any, latent: torch.Tensor, scaling_factor: float) -> torch.Tensor:
    """Use the inverse of the exact VAE scaling employed during training."""
    return decode_latent_raw(vae, latent, scaling_factor).clamp(0.0, 1.0)


def image_tensor_to_array(image: torch.Tensor) -> np.ndarray:
    tensor = image[0] if image.ndim == 4 else image
    return tensor.detach().float().cpu().permute(1, 2, 0).numpy().clip(0.0, 1.0)


def optional_phi_prediction(
    phi: LatentPhi | None,
    base_z0: torch.Tensor,
    normalized_t: torch.Tensor,
    timestep_range: tuple[float, float],
    weight: float,
) -> tuple[torch.Tensor | None, torch.Tensor, bool]:
    """Apply φ only inside the configured hard range and only for nonzero weight."""
    active = (
        phi is not None
        and weight != 0.0
        and bool(timestep_range_mask(normalized_t, timestep_range).all().item())
    )
    if not active:
        return None, base_z0, False
    phi_z0 = phi(base_z0.detach().float(), normalized_t.float())
    mixed_z0 = (1.0 - weight) * base_z0.float() + weight * phi_z0.float()
    return phi_z0, mixed_z0, True


def decoded_quality(
    decoded: torch.Tensor, hr: torch.Tensor, lpips_metric: Any | None
) -> tuple[float, float, float | str]:
    from src.metrics import psnr, ssim

    prediction = image_tensor_to_array(decoded)
    target = image_tensor_to_array(hr)
    lpips_value: float | str = ""
    if lpips_metric is not None:
        lpips_value = lpips_metric(decoded[0], hr[0])
    return psnr(prediction, target), ssim(prediction, target), lpips_value


def fixed_error_heatmap(
    prediction: torch.Tensor, target: torch.Tensor, display_max: float = 0.25
) -> Image.Image:
    """Render RGB MAE with one fixed scale shared by every timestep."""
    if display_max <= 0:
        raise ValueError(f"display_max must be positive, got {display_max}")
    error = torch.mean(torch.abs(prediction.float() - target.float()), dim=1)[0]
    scaled = error.div(display_max).clamp(0.0, 1.0).cpu().numpy()
    red = np.clip(2.0 * scaled, 0.0, 1.0)
    green = np.clip(2.0 * scaled - 1.0, 0.0, 1.0)
    blue = np.zeros_like(scaled)
    rgb = np.stack((red, green, blue), axis=-1)
    return Image.fromarray(np.rint(rgb * 255.0).astype(np.uint8), mode="RGB")


def label_tile(image: Image.Image, label: str, label_height: int = 36) -> Image.Image:
    result = Image.new("RGB", (image.width, image.height + label_height), "white")
    result.paste(image.convert("RGB"), (0, label_height))
    draw = ImageDraw.Draw(result)
    draw.multiline_text((5, 4), label, fill="black", spacing=2)
    return result


def make_timestep_row(
    timestep: int,
    log_snr: float,
    latent_nmse: float,
    psnr_value: float,
    clipped_fraction: float,
    hr: torch.Tensor,
    raw_lr_up: torch.Tensor,
    adapted_lr_up: torch.Tensor,
    decoded_zt: torch.Tensor,
    decoded_x0: torch.Tensor,
    error_display_max: float,
    phi_decoded_x0: torch.Tensor | None = None,
    mixed_decoded_x0: torch.Tensor | None = None,
) -> Image.Image:
    from src.utils import tensor_to_pil

    size = (hr.shape[-1], hr.shape[-2])
    label_width = 210
    label_height = 36
    label = Image.new("RGB", (label_width, size[1] + label_height), "white")
    ImageDraw.Draw(label).multiline_text(
        (8, 5),
        f"t={timestep}  log-SNR={log_snr:.2f}\n"
        f"NMSE={latent_nmse:.3g}  PSNR={psnr_value:.2f}\n"
        f"decoded clip={clipped_fraction:.2%}",
        fill="black",
        spacing=1,
    )
    images = [
        label_tile(tensor_to_pil(hr), "HR ground truth (fixed target)", label_height),
        label_tile(tensor_to_pil(raw_lr_up), "raw LR bicubic", label_height),
        label_tile(tensor_to_pil(adapted_lr_up), "ConditionAdapter(LR), pre-noise", label_height),
        label_tile(tensor_to_pil(decoded_zt), "decoded z_t", label_height),
        label_tile(tensor_to_pil(decoded_x0), "predicted clean", label_height),
        label_tile(
            fixed_error_heatmap(decoded_x0, hr, error_display_max),
            f"absolute RGB error (fixed 0-{error_display_max:g})",
            label_height,
        ),
    ]
    if phi_decoded_x0 is not None:
        images.append(label_tile(tensor_to_pil(phi_decoded_x0), "phi predicted clean", label_height))
    if mixed_decoded_x0 is not None:
        images.append(label_tile(tensor_to_pil(mixed_decoded_x0), "mixed predicted clean", label_height))
    row = Image.new("RGB", (label_width + len(images) * size[0], size[1] + label_height), "white")
    row.paste(label, (0, 0))
    for index, image in enumerate(images):
        row.paste(image, (label_width + index * size[0], 0))
    return row


def make_reverse_trajectory_row(
    inference_step: int,
    total_steps: int,
    timestep: int,
    log_snr: float,
    psnr_value: float,
    clipped_fraction: float,
    hr: torch.Tensor,
    raw_lr_up: torch.Tensor,
    current_latent_decoded: torch.Tensor,
    predicted_clean: torch.Tensor,
    next_latent_decoded: torch.Tensor,
    error_display_max: float,
    phi_decoded_x0: torch.Tensor | None = None,
    mixed_decoded_x0: torch.Tensor | None = None,
) -> Image.Image:
    """Visualize one real scheduler step starting from an HR-independent noise latent."""
    from src.utils import tensor_to_pil

    size = (hr.shape[-1], hr.shape[-2])
    label_width = 220
    label_height = 36
    label = Image.new("RGB", (label_width, size[1] + label_height), "white")
    ImageDraw.Draw(label).multiline_text(
        (8, 5),
        f"reverse step={inference_step}/{total_steps - 1}\n"
        f"scheduler t={timestep}  log-SNR={log_snr:.2f}\n"
        f"x0 PSNR={psnr_value:.2f}  clip={clipped_fraction:.2%}",
        fill="black",
        spacing=1,
    )
    images = [
        label_tile(tensor_to_pil(hr), "HR reference only (not model input)", label_height),
        label_tile(tensor_to_pil(raw_lr_up), "raw LR condition", label_height),
        label_tile(tensor_to_pil(current_latent_decoded), "current latent before step", label_height),
        label_tile(tensor_to_pil(predicted_clean), "formula x0 from model_output", label_height),
        label_tile(tensor_to_pil(next_latent_decoded), "latent after scheduler.step", label_height),
        label_tile(
            fixed_error_heatmap(predicted_clean, hr, error_display_max),
            f"x0 error vs HR (fixed 0-{error_display_max:g})",
            label_height,
        ),
    ]
    if phi_decoded_x0 is not None:
        images.append(label_tile(tensor_to_pil(phi_decoded_x0), "phi predicted clean", label_height))
    if mixed_decoded_x0 is not None:
        images.append(label_tile(tensor_to_pil(mixed_decoded_x0), "mixed x0 used by scheduler", label_height))
    row = Image.new("RGB", (label_width + len(images) * size[0], size[1] + label_height), "white")
    row.paste(label, (0, 0))
    for index, image in enumerate(images):
        row.paste(image, (label_width + index * size[0], 0))
    return row


def stack_rows(rows: list[Image.Image]) -> Image.Image:
    width = max(row.width for row in rows)
    canvas = Image.new("RGB", (width, sum(row.height for row in rows)), "white")
    top = 0
    for row in rows:
        canvas.paste(row, (0, top))
        top += row.height
    return canvas


def save_metrics(
    path: Path, rows: list[dict[str, Any]], fieldnames: Iterable[str] = METRIC_FIELDS
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(fieldnames))
        writer.writeheader()
        writer.writerows(rows)


def aggregate_metrics(
    rows: list[dict[str, Any]], fields: Iterable[str] = METRIC_FIELDS
) -> list[dict[str, float]]:
    aggregated: list[dict[str, float]] = []
    for timestep in sorted({int(row["timestep"]) for row in rows}):
        selected = [row for row in rows if int(row["timestep"]) == timestep]
        item: dict[str, float] = {"timestep": float(timestep)}
        for key in fields:
            if key in {"sample_id", "filename", "timestep"}:
                continue
            numeric = [float(row[key]) for row in selected if row.get(key) not in (None, "")]
            if numeric:
                item[key] = float(np.mean(numeric))
        aggregated.append(item)
    return aggregated


def save_phi_comparison_curves(path: Path, rows: list[dict[str, float]], x_label: str) -> None:
    """Plot base/phi/mixed PSNR, SSIM, and LPIPS with fixed series semantics."""
    panels = (
        ("PSNR", ("base_decoded_x0_psnr", "phi_decoded_x0_psnr", "mixed_decoded_x0_psnr")),
        ("SSIM", ("base_decoded_x0_ssim", "phi_decoded_x0_ssim", "mixed_decoded_x0_ssim")),
        ("LPIPS", ("base_decoded_x0_lpips", "phi_decoded_x0_lpips", "mixed_decoded_x0_lpips")),
    )
    colors = ((40, 90, 190), (210, 80, 50), (40, 150, 80))
    labels = ("base", "phi", "mixed")
    width, panel_height = 1200, 230
    left, right, top, bottom = 100, 25, 35, 40
    canvas = Image.new("RGB", (width, panel_height * len(panels)), "white")
    draw = ImageDraw.Draw(canvas)
    x_values = np.asarray([row["timestep"] for row in rows], dtype=np.float64)
    x_min, x_max = float(x_values.min()), float(x_values.max())
    x_denominator = max(x_max - x_min, 1.0)
    for panel_index, (panel_name, keys) in enumerate(panels):
        offset = panel_index * panel_height
        all_values = [
            float(row[key])
            for row in rows
            for key in keys
            if key in row and math.isfinite(float(row[key]))
        ]
        if not all_values:
            continue
        minimum, maximum = min(all_values), max(all_values)
        if maximum <= minimum:
            maximum = minimum + 1.0
        draw.rectangle((left, offset + top, width - right, offset + panel_height - bottom), outline="black")
        for series_index, key in enumerate(keys):
            points: list[tuple[int, int]] = []
            for x_value, row in zip(x_values, rows):
                if key not in row or not math.isfinite(float(row[key])):
                    continue
                x_pixel = left + (width - left - right) * (float(x_value) - x_min) / x_denominator
                y_pixel = offset + top + (panel_height - top - bottom) * (
                    maximum - float(row[key])
                ) / (maximum - minimum)
                points.append((round(x_pixel), round(y_pixel)))
            if len(points) >= 2:
                draw.line(points, fill=colors[series_index], width=3)
            draw.text((left + 100 * series_index, offset + 8), labels[series_index], fill=colors[series_index])
        draw.text((8, offset + 8), panel_name, fill="black")
    draw.text((left, canvas.height - 18), x_label, fill="black")
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


def save_curves(path: Path, rows: list[dict[str, float]]) -> None:
    """Render diagnostic curves without adding a plotting-library dependency."""
    save_curves_pil(path, rows)


def save_curves_pil(path: Path, rows: list[dict[str, float]]) -> None:
    """Dependency-free curve renderer used only when matplotlib is unavailable."""
    curve_specs = (
        ("latent_nmse", "latent NMSE", True),
        ("psnr", "PSNR (dB)", False),
        ("ssim", "SSIM", False),
        ("lpips", "LPIPS", False),
        ("noise_prediction_mse", "noise prediction MSE", True),
    )
    available = [spec for spec in curve_specs if all(spec[0] in row for row in rows)]
    width, panel_height = 1200, 220
    left, right, top, bottom = 110, 25, 35, 35
    canvas = Image.new("RGB", (width, panel_height * len(available)), "white")
    draw = ImageDraw.Draw(canvas)
    x_values = np.asarray([row["timestep"] for row in rows], dtype=np.float64)
    for panel, (key, label, logarithmic) in enumerate(available):
        offset = panel * panel_height
        values = np.asarray([row[key] for row in rows], dtype=np.float64)
        plotted = np.log10(np.maximum(values, 1e-30)) if logarithmic else values
        finite = np.isfinite(plotted)
        if not finite.any():
            continue
        minimum, maximum = float(plotted[finite].min()), float(plotted[finite].max())
        if maximum <= minimum:
            maximum = minimum + 1.0
        x_denominator = max(float(x_values[-1] - x_values[0]), 1.0)
        points = []
        for x_value, y_value in zip(x_values, plotted):
            if not math.isfinite(float(y_value)):
                continue
            x_pixel = left + (width - left - right) * float(x_value - x_values[0]) / x_denominator
            y_pixel = offset + top + (panel_height - top - bottom) * (maximum - float(y_value)) / (maximum - minimum)
            points.append((round(x_pixel), round(y_pixel)))
        draw.rectangle((left, offset + top, width - right, offset + panel_height - bottom), outline="black")
        if len(points) >= 2:
            draw.line(points, fill=(30, 90, 190), width=2)
        scale_note = "log10" if logarithmic else "linear"
        draw.text((8, offset + 8), f"{label} ({scale_note})", fill="black")
        draw.text((8, offset + top), f"max {maximum:.4g}", fill="black")
        draw.text((8, offset + panel_height - bottom - 12), f"min {minimum:.4g}", fill="black")
    draw.text((left, canvas.height - 18), "training timestep t (larger t = higher noise)", fill="black")
    canvas.save(path)


def sustained_boundary(flags: list[bool], timesteps: list[int], start: int, run_length: int = 3) -> int | None:
    for index in range(start, max(start, len(flags) - run_length + 1)):
        if all(flags[index : index + run_length]):
            return timesteps[index]
    return None


def reliability_summary(rows: list[dict[str, float]]) -> dict[str, Any]:
    """Apply a transparent plateau-relative rule without assuming fixed timestep boundaries."""
    baseline_count = min(len(rows), max(5, math.ceil(0.05 * len(rows))))
    baseline_rows = rows[:baseline_count]
    baseline = {
        "psnr": float(np.median([row["psnr"] for row in baseline_rows])),
        "ssim": float(np.median([row["ssim"] for row in baseline_rows])),
        "latent_nmse": float(np.median([row["latent_nmse"] for row in baseline_rows])),
    }
    moderate_nmse = max(4.0 * baseline["latent_nmse"], 1e-3)
    severe_nmse = max(16.0 * baseline["latent_nmse"], 1e-2)
    moderate: list[bool] = []
    severe: list[bool] = []
    for row in rows:
        moderate.append(
            sum(
                (
                    row["psnr"] < baseline["psnr"] - 3.0,
                    row["ssim"] < baseline["ssim"] - 0.05,
                    row["latent_nmse"] > moderate_nmse,
                )
            )
            >= 2
        )
        severe.append(
            sum(
                (
                    row["psnr"] < baseline["psnr"] - 8.0,
                    row["ssim"] < baseline["ssim"] - 0.15,
                    row["latent_nmse"] > severe_nmse,
                )
            )
            >= 2
        )
    timesteps = [int(row["timestep"]) for row in rows]
    transition_start = sustained_boundary(moderate, timesteps, 0)
    transition_index = timesteps.index(transition_start) if transition_start is not None else len(timesteps)
    unreliable_start = sustained_boundary(severe, timesteps, transition_index)
    last = timesteps[-1]
    reliable_end = (
        last
        if transition_start is None
        else next((t for t in reversed(timesteps) if t < transition_start), None)
    )
    transition_end = (
        last
        if unreliable_start is None
        else next((t for t in reversed(timesteps) if t < unreliable_start), None)
    )
    return {
        "rule": (
            "Baseline is the median of the lowest-noise max(5,5%) evaluated timesteps. Transition starts after "
            "3 consecutive points where at least 2/3 hold: PSNR drops >3 dB, SSIM drops >0.05, latent NMSE "
            "exceeds max(4x baseline,1e-3). Unreliable starts similarly with >8 dB, >0.15, and "
            "max(16x baseline,1e-2)."
        ),
        "baseline": baseline,
        "low_noise_reliable_region": None
        if reliable_end is None
        else {"start": timesteps[0], "end": reliable_end},
        "transition_region": None
        if transition_start is None or transition_end is None
        else {"start": transition_start, "end": transition_end},
        "high_noise_unreliable_region": None
        if unreliable_start is None
        else {"start": unreliable_start, "end": last},
    }


def load_models(
    config: dict[str, Any], checkpoint: Path, device: torch.device, dtype: torch.dtype
) -> tuple[Any, Any]:
    from diffusers import StableDiffusionUpscalePipeline
    from src.condition_adapter import ConditionAdapter
    from src.utils import normalize_tokenizer_max_length

    pipe = StableDiffusionUpscalePipeline.from_pretrained(
        str(config["model_id"]), torch_dtype=dtype, safety_checker=None
    )
    normalize_tokenizer_max_length(pipe.tokenizer, pipe.text_encoder)
    lora_file = checkpoint / "pytorch_lora_weights.safetensors"
    if not lora_file.is_file():
        raise FileNotFoundError(f"LoRA checkpoint is missing: {lora_file}")
    pipe.load_lora_weights(str(checkpoint))
    adapter = ConditionAdapter.from_pretrained(
        checkpoint, adapter_scale=float(config.get("adapter_scale", 1.0)), device=device
    ).to(device=device, dtype=dtype)
    pipe.to(device)
    pipe.vae.to(device=device, dtype=torch.float32)
    pipe.text_encoder.to(device=device, dtype=dtype)
    pipe.unet.eval()
    pipe.vae.eval()
    pipe.text_encoder.eval()
    adapter.eval()
    for module in (pipe.unet, pipe.vae, pipe.text_encoder, adapter):
        module.requires_grad_(False)
    return pipe, adapter


def build_dataset(config: dict[str, Any], split: str, num_samples: int) -> Any:
    from src.dataset import PairedSatelliteDataset

    return PairedSatelliteDataset(
        data_root=config["data_root"],
        split=split,
        lr_subdir=str(config["val_lr_subdir"]),
        gt_subdir=str(config.get("gt_subdir", "GT")),
        gt_crop_size=int(config.get("gt_crop_size", 512)),
        scale=int(config.get("scale", 4)),
        training=False,
        strict_pairs=bool(config.get("strict_pairs", False)),
        prompt_mode=str(config.get("prompt_mode", "fixed")),
        metadata_path=config.get("metadata_path"),
        prompt_dropout_probability=0.0,
        augment=False,
        validation_limit=num_samples,
    )


@torch.no_grad()
def analyze_sample(
    sample: dict[str, Any],
    pipe: Any,
    adapter: Any,
    config: dict[str, Any],
    device: torch.device,
    dtype: torch.dtype,
    timesteps: list[int],
    selected: set[int],
    noise_seed: int,
    vae_seed: int,
    low_res_noise_level: int,
    lpips_metric: Any | None,
    output_dir: Path,
    save_all_images: bool,
    error_display_max: float,
    phi: LatentPhi | None,
    phi_timestep_range: tuple[float, float],
    phi_weight: float,
) -> tuple[list[dict[str, Any]], list[Image.Image], float, dict[str, float]]:
    from src.metrics import psnr, ssim
    from src.train_lora_upscaler import encode_prompts

    gt = sample["gt"].unsqueeze(0).to(device=device, dtype=torch.float32)
    lr = sample["lr"].unsqueeze(0).to(device=device, dtype=dtype)
    scaling_factor = float(pipe.vae.config.scaling_factor)
    vae_generator = torch.Generator(device=device).manual_seed(vae_seed)
    posterior = pipe.vae.encode(gt).latent_dist
    z0 = posterior.sample(generator=vae_generator) * scaling_factor
    z0 = z0.to(dtype=dtype)

    noise_generator = torch.Generator(device=device).manual_seed(noise_seed)
    epsilon = torch.randn(z0.shape, generator=noise_generator, device=device, dtype=dtype)
    condition_noise = torch.randn(lr.shape, generator=noise_generator, device=device, dtype=dtype)
    adapted_lr = adapter(lr)
    low_level = torch.tensor([low_res_noise_level], device=device, dtype=torch.long)
    noisy_low = pipe.low_res_scheduler.add_noise(adapted_lr, condition_noise, low_level)
    if noisy_low.shape[-2:] != z0.shape[-2:]:
        if not bool(config.get("resize_low_res_condition_if_needed", False)):
            raise AssertionError(
                f"Training-equivalent condition/latent mismatch: condition={noisy_low.shape}, latent={z0.shape}"
            )
        noisy_low = F.interpolate(noisy_low, size=z0.shape[-2:], mode="bicubic", align_corners=False)
    prompt_embeds = encode_prompts(pipe.tokenizer, pipe.text_encoder, [sample["prompt"]], device)
    prediction_type = str(pipe.scheduler.config.prediction_type)
    alphas = pipe.scheduler.alphas_cumprod.to(device=device, dtype=torch.float32)
    hr_01 = gt.add(1.0).div(2.0).clamp(0.0, 1.0)
    lr_01 = lr.float().add(1.0).div(2.0).clamp(0.0, 1.0)
    adapted_lr_01 = adapted_lr.float().add(1.0).div(2.0).clamp(0.0, 1.0)
    raw_lr_up = F.interpolate(lr_01, size=hr_01.shape[-2:], mode="bicubic", align_corners=False)
    adapted_lr_up = F.interpolate(
        adapted_lr_01, size=hr_01.shape[-2:], mode="bicubic", align_corners=False
    )
    condition_diagnostics = {
        "raw_lr_mean": float(lr_01.mean().item()),
        "raw_lr_std": float(lr_01.std().item()),
        "adapted_lr_mean": float(adapted_lr_01.mean().item()),
        "adapted_lr_std": float(adapted_lr_01.std().item()),
        "adapter_mean_absolute_delta": float(
            torch.mean(torch.abs(adapted_lr.float() - lr.float())).item()
        ),
        "adapter_lower_clamp_fraction": float(
            (adapted_lr.float() <= -0.999).float().mean().item()
        ),
        "adapter_upper_clamp_fraction": float(
            (adapted_lr.float() >= 0.999).float().mean().item()
        ),
    }
    latent_power = float(torch.mean(z0.float().square()).item())
    rows: list[dict[str, Any]] = []
    visual_rows: list[Image.Image] = []
    oracle_max_error = 0.0

    for timestep_value in timesteps:
        timestep = torch.tensor([timestep_value], device=device, dtype=torch.long)
        zt = pipe.scheduler.add_noise(z0, epsilon, timestep)
        model_input = torch.cat((zt, noisy_low), dim=1)
        model_output = pipe.unet(
            model_input,
            timestep,
            encoder_hidden_states=prompt_embeds,
            class_labels=low_level,
            return_dict=False,
        )[0]
        alpha_bar = alphas[timestep_value].reshape(1, 1, 1, 1).to(dtype=torch.float32)
        zt_float = zt.float()
        model_float = model_output.float()
        predicted_z0 = prediction_to_x0(model_float, zt_float, alpha_bar, prediction_type)
        normalized_t = normalized_timesteps(timestep, len(alphas))
        phi_z0, mixed_z0, phi_active = optional_phi_prediction(
            phi, predicted_z0, normalized_t, phi_timestep_range, phi_weight
        )
        predicted_epsilon = prediction_to_epsilon(model_float, zt_float, alpha_bar, prediction_type)
        target = true_model_target(pipe.scheduler, z0.float(), epsilon.float(), timestep, prediction_type)

        # Use scheduler q-sampling in float32 for the oracle formula check so
        # high-timestep inversion is not dominated by fp16 quantization.
        oracle_zt = pipe.scheduler.add_noise(z0.float(), epsilon.float(), timestep)
        oracle_z0 = prediction_to_x0(epsilon.float(), oracle_zt, alpha_bar, "epsilon")
        oracle_max_error = max(oracle_max_error, float(torch.max(torch.abs(oracle_z0 - z0.float())).item()))
        decoded_zt = decode_latent(pipe.vae, zt_float, scaling_factor)
        decoded_x0_raw = decode_latent_raw(pipe.vae, predicted_z0, scaling_factor)
        decoded_x0 = decoded_x0_raw.clamp(0.0, 1.0)
        phi_decoded = decode_latent(pipe.vae, phi_z0, scaling_factor) if phi_z0 is not None else None
        mixed_decoded = (
            decode_latent(pipe.vae, mixed_z0, scaling_factor) if phi_active else decoded_x0
        )
        prediction_array = image_tensor_to_array(decoded_x0)
        target_array = image_tensor_to_array(hr_01)
        alpha_value = float(alpha_bar.item())
        snr_value = alpha_value / max(1.0 - alpha_value, 1e-12)
        latent_mse = float(F.mse_loss(predicted_z0, z0.float()).item())
        base_psnr, base_ssim, base_lpips = decoded_quality(decoded_x0, hr_01, lpips_metric)
        if phi_decoded is not None:
            phi_psnr, phi_ssim, phi_lpips = decoded_quality(phi_decoded, hr_01, lpips_metric)
        else:
            phi_psnr, phi_ssim, phi_lpips = "", "", ""
        mixed_psnr, mixed_ssim, mixed_lpips = decoded_quality(
            mixed_decoded, hr_01, lpips_metric
        )
        below_fraction = float((decoded_x0_raw < 0.0).float().mean().item())
        above_fraction = float((decoded_x0_raw > 1.0).float().mean().item())
        lr_reconstruction = F.interpolate(
            decoded_x0, size=lr_01.shape[-2:], mode="bicubic", align_corners=False, antialias=True
        )
        row: dict[str, Any] = {
            "sample_id": sample["sample_id"],
            "filename": sample["filename"],
            "timestep": timestep_value,
            "normalized_timestep": timestep_value / max(1, len(alphas) - 1),
            "alpha_bar": alpha_value,
            "snr": snr_value,
            "log_snr": math.log(max(snr_value, 1e-30)),
            "zt_to_z0_mse": float(F.mse_loss(zt_float, z0.float()).item()),
            "noise_prediction_mse": float(F.mse_loss(predicted_epsilon, epsilon.float()).item()),
            "model_target_mse": float(F.mse_loss(model_float, target.float()).item()),
            "latent_mse": latent_mse,
            "latent_nmse": latent_mse / max(latent_power, 1e-12),
            "psnr": base_psnr,
            "ssim": base_ssim,
            "lpips": base_lpips,
            "lr_reconstruction_mse": float(F.mse_loss(lr_reconstruction, lr_01.float()).item()),
            "predicted_z0_abs_mean": float(predicted_z0.abs().mean().item()),
            "predicted_z0_abs_max": float(predicted_z0.abs().max().item()),
            "decoded_x0_raw_min": float(decoded_x0_raw.min().item()),
            "decoded_x0_raw_max": float(decoded_x0_raw.max().item()),
            "decoded_x0_below_zero_fraction": below_fraction,
            "decoded_x0_above_one_fraction": above_fraction,
            "base_decoded_x0_psnr": base_psnr,
            "base_decoded_x0_ssim": base_ssim,
            "base_decoded_x0_lpips": base_lpips,
            "phi_decoded_x0_psnr": phi_psnr,
            "phi_decoded_x0_ssim": phi_ssim,
            "phi_decoded_x0_lpips": phi_lpips,
            "mixed_decoded_x0_psnr": mixed_psnr,
            "mixed_decoded_x0_ssim": mixed_ssim,
            "mixed_decoded_x0_lpips": mixed_lpips,
            "phi_z0_abs_mean": float(phi_z0.abs().mean().item()) if phi_z0 is not None else "",
            "mixed_z0_abs_mean": float(mixed_z0.abs().mean().item()),
            "phi_active": int(phi_active),
            "phi_weight": float(phi_weight) if phi_active else 0.0,
        }
        rows.append(row)

        if timestep_value in selected or save_all_images:
            visual = make_timestep_row(
                timestep_value,
                float(row["log_snr"]),
                float(row["latent_nmse"]),
                float(row["psnr"]),
                below_fraction + above_fraction,
                hr_01,
                raw_lr_up,
                adapted_lr_up,
                decoded_zt,
                decoded_x0,
                error_display_max,
                phi_decoded,
                mixed_decoded if phi_active else None,
            )
            sample_dir = output_dir / "per_timestep" / str(sample["sample_id"])
            sample_dir.mkdir(parents=True, exist_ok=True)
            visual.save(sample_dir / f"t_{timestep_value:04d}.png")
            if timestep_value in selected:
                visual_rows.append(visual)
    return rows, visual_rows, oracle_max_error, condition_diagnostics


@torch.no_grad()
def analyze_pure_noise_trajectory(
    sample: dict[str, Any],
    pipe: Any,
    adapter: Any,
    device: torch.device,
    dtype: torch.dtype,
    noise_seed: int,
    vae_seed: int,
    low_res_noise_level: int,
    num_inference_steps: int,
    output_dir: Path,
    error_display_max: float,
    phi: LatentPhi | None,
    phi_timestep_range: tuple[float, float],
    phi_weight: float,
    lpips_metric: Any | None,
) -> tuple[list[dict[str, Any]], list[Image.Image]]:
    """Run real reverse inference from pure noise, without constructing z_t from HR."""
    from src.metrics import psnr, ssim
    from src.train_lora_upscaler import encode_prompts
    from src.utils import tensor_to_pil

    gt = sample["gt"].unsqueeze(0).to(device=device, dtype=torch.float32)
    lr = sample["lr"].unsqueeze(0).to(device=device, dtype=dtype)
    hr_01 = gt.add(1.0).div(2.0).clamp(0.0, 1.0)
    lr_01 = lr.float().add(1.0).div(2.0).clamp(0.0, 1.0)
    raw_lr_up = F.interpolate(lr_01, size=hr_01.shape[-2:], mode="bicubic", align_corners=False)

    # Match the repository's formal inference boundary: adapter tensor -> PIL ->
    # pipeline image preprocessing. HR is deliberately absent from this path.
    adapted_tensor = adapter(lr)
    adapted_pil = tensor_to_pil(adapted_tensor[0].float().cpu())
    condition = pipe.image_processor.preprocess(adapted_pil).to(device=device, dtype=dtype)
    low_level = torch.tensor([low_res_noise_level], device=device, dtype=torch.long)
    prompt_embeds = encode_prompts(pipe.tokenizer, pipe.text_encoder, [sample["prompt"]], device)

    pipe.scheduler.set_timesteps(num_inference_steps, device=device)
    scheduler_timesteps = pipe.scheduler.timesteps
    generator = torch.Generator(device=device).manual_seed(noise_seed)

    # Keep the draw order identical to StableDiffusionUpscalePipeline: condition
    # noise first, then the initial latent. Neither draw uses the HR target.
    condition_noise = torch.randn(
        condition.shape, generator=generator, device=device, dtype=dtype
    )
    noisy_condition = pipe.low_res_scheduler.add_noise(condition, condition_noise, low_level)
    latent_shape = (
        1,
        int(pipe.vae.config.latent_channels),
        int(noisy_condition.shape[-2]),
        int(noisy_condition.shape[-1]),
    )
    latents = torch.randn(latent_shape, generator=generator, device=device, dtype=dtype)
    latents = latents * float(pipe.scheduler.init_noise_sigma)

    # The reference encoding is metrics-only. It is never used to initialize or
    # update `latents`, nor is it passed to the UNet or scheduler.
    scaling_factor = float(pipe.vae.config.scaling_factor)
    vae_generator = torch.Generator(device=device).manual_seed(vae_seed)
    reference_z0 = pipe.vae.encode(gt).latent_dist.sample(generator=vae_generator)
    reference_z0 = reference_z0.float() * scaling_factor
    prediction_type = str(pipe.scheduler.config.prediction_type)
    alphas = pipe.scheduler.alphas_cumprod.to(device=device, dtype=torch.float32)
    selected_steps = representative_inference_steps(len(scheduler_timesteps))
    extra_step_kwargs = pipe.prepare_extra_step_kwargs(generator, eta=0.0)
    rows: list[dict[str, Any]] = []
    visuals: list[Image.Image] = []

    for step_index, timestep in enumerate(scheduler_timesteps):
        timestep_value = int(round(float(timestep.detach().cpu().item())))
        latent_before = latents
        latent_model_input = pipe.scheduler.scale_model_input(latent_before, timestep)
        model_input = torch.cat((latent_model_input, noisy_condition), dim=1)
        model_output = pipe.unet(
            model_input,
            timestep,
            encoder_hidden_states=prompt_embeds,
            class_labels=low_level,
            return_dict=False,
        )[0]
        alpha_bar = alphas[timestep_value].reshape(1, 1, 1, 1)
        predicted_z0 = prediction_to_x0(
            model_output.float(), latent_before.float(), alpha_bar, prediction_type
        )
        normalized_t = normalized_timesteps(timestep.reshape(1).long(), len(alphas))
        phi_z0, mixed_z0, phi_active = optional_phi_prediction(
            phi, predicted_z0, normalized_t, phi_timestep_range, phi_weight
        )
        step_model_output = model_output
        if phi_active:
            _, converted_output = mix_x0_and_convert_model_output(
                predicted_z0,
                phi_z0,
                phi_weight,
                latent_before,
                alpha_bar,
                prediction_type,
            )
            step_model_output = converted_output.to(dtype=model_output.dtype)
        latents = pipe.scheduler.step(
            step_model_output, timestep, latent_before, **extra_step_kwargs, return_dict=False
        )[0]

        predicted_clean_raw = decode_latent_raw(pipe.vae, predicted_z0, scaling_factor)
        predicted_clean = predicted_clean_raw.clamp(0.0, 1.0)
        phi_decoded = decode_latent(pipe.vae, phi_z0, scaling_factor) if phi_z0 is not None else None
        mixed_decoded = (
            decode_latent(pipe.vae, mixed_z0, scaling_factor) if phi_active else predicted_clean
        )
        predicted_array = image_tensor_to_array(predicted_clean)
        target_array = image_tensor_to_array(hr_01)
        alpha_value = float(alpha_bar.item())
        snr_value = alpha_value / max(1.0 - alpha_value, 1e-12)
        below_fraction = float((predicted_clean_raw < 0.0).float().mean().item())
        above_fraction = float((predicted_clean_raw > 1.0).float().mean().item())
        base_psnr, base_ssim, base_lpips = decoded_quality(
            predicted_clean, hr_01, lpips_metric
        )
        if phi_decoded is not None:
            phi_psnr, phi_ssim, phi_lpips = decoded_quality(phi_decoded, hr_01, lpips_metric)
        else:
            phi_psnr, phi_ssim, phi_lpips = "", "", ""
        mixed_psnr, mixed_ssim, mixed_lpips = decoded_quality(
            mixed_decoded, hr_01, lpips_metric
        )
        row: dict[str, Any] = {
            "sample_id": sample["sample_id"],
            "filename": sample["filename"],
            "inference_step": step_index,
            "inference_step_fraction": step_index / max(1, len(scheduler_timesteps) - 1),
            "timestep": timestep_value,
            "alpha_bar": alpha_value,
            "snr": snr_value,
            "log_snr": math.log(max(snr_value, 1e-30)),
            "predicted_z0_abs_mean": float(predicted_z0.abs().mean().item()),
            "predicted_z0_abs_max": float(predicted_z0.abs().max().item()),
            "predicted_clean_psnr": base_psnr,
            "predicted_clean_ssim": base_ssim,
            "base_decoded_x0_psnr": base_psnr,
            "base_decoded_x0_ssim": base_ssim,
            "predicted_clean_latent_mse_to_hr_z0": float(
                F.mse_loss(predicted_z0, reference_z0).item()
            ),
            "decoded_x0_below_zero_fraction": below_fraction,
            "decoded_x0_above_one_fraction": above_fraction,
            "base_decoded_x0_lpips": base_lpips,
            "phi_decoded_x0_psnr": phi_psnr,
            "phi_decoded_x0_ssim": phi_ssim,
            "phi_decoded_x0_lpips": phi_lpips,
            "mixed_decoded_x0_psnr": mixed_psnr,
            "mixed_decoded_x0_ssim": mixed_ssim,
            "mixed_decoded_x0_lpips": mixed_lpips,
            "phi_z0_abs_mean": float(phi_z0.abs().mean().item()) if phi_z0 is not None else "",
            "mixed_z0_abs_mean": float(mixed_z0.abs().mean().item()),
            "phi_active": int(phi_active),
            "phi_weight": float(phi_weight) if phi_active else 0.0,
        }
        rows.append(row)

        if step_index in selected_steps:
            current_decoded = decode_latent(pipe.vae, latent_before.float(), scaling_factor)
            next_decoded = decode_latent(pipe.vae, latents.float(), scaling_factor)
            visual = make_reverse_trajectory_row(
                step_index,
                len(scheduler_timesteps),
                timestep_value,
                float(row["log_snr"]),
                float(row["predicted_clean_psnr"]),
                below_fraction + above_fraction,
                hr_01,
                raw_lr_up,
                current_decoded,
                predicted_clean,
                next_decoded,
                error_display_max,
                phi_decoded,
                mixed_decoded if phi_active else None,
            )
            sample_dir = output_dir / "pure_noise_trajectory" / "per_step" / str(sample["sample_id"])
            sample_dir.mkdir(parents=True, exist_ok=True)
            visual.save(sample_dir / f"step_{step_index:04d}_t_{timestep_value:04d}.png")
            visuals.append(visual)
    return rows, visuals


def main() -> None:
    from src.metrics import OptionalLPIPS
    from src.utils import (
        configure_logging,
        load_yaml_config,
        require_config,
        resolve_project_path,
        save_json,
        save_yaml,
    )

    args = parse_args()
    configure_logging()
    if args.num_samples <= 0 or args.timestep_stride <= 0 or args.error_display_max <= 0:
        raise ValueError(
            "--num_samples, --timestep_stride, and --error_display_max must be positive."
        )
    config_path = resolve_config_path(args.config)
    config = load_yaml_config(config_path)
    require_config(config, "model_id", "data_root", "val_lr_subdir", "output_dir")
    if args.data_root is not None:
        config["data_root"] = str(args.data_root.expanduser().resolve())
    checkpoint = (
        resolve_project_path(args.checkpoint, PROJECT_ROOT)
        if args.checkpoint is not None
        else resolve_project_path(config["output_dir"], PROJECT_ROOT) / "final"
    )
    if not checkpoint.is_dir():
        raise FileNotFoundError(f"Checkpoint artifact directory does not exist: {checkpoint}")
    experiment_name = f"{config_path.stem}_{checkpoint.name}"
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else PROJECT_ROOT / "outputs" / "z0_reliability" / experiment_name
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device)
    dtype = resolve_dtype(args.mixed_precision, config, device)
    pipe, adapter = load_models(config, checkpoint, device, dtype)
    phi: LatentPhi | None = None
    if args.phi_path is not None:
        phi_path = resolve_project_path(args.phi_path, PROJECT_ROOT)
        phi = LatentPhi.from_pretrained(phi_path, device=device).eval()
        phi.requires_grad_(False)
    phi_timestep_range = validate_timestep_range(
        args.phi_timestep_range or config.get("phi_infer_timestep_range", [0.0, 1.0]),
        "phi_timestep_range",
    )
    phi_weight = float(
        args.phi_weight if args.phi_weight is not None else config.get("phi_weight", 1.0)
    )
    if not 0.0 <= phi_weight <= 1.0:
        raise ValueError("--phi_weight must be in [0,1]")
    prediction_type = str(pipe.scheduler.config.prediction_type)
    total_timesteps = int(pipe.scheduler.config.num_train_timesteps)
    timesteps = evaluated_timesteps(total_timesteps, args.timestep_stride)
    selected_values = representative_timesteps(total_timesteps, timesteps)
    low_res_noise_level = (
        args.low_res_noise_level
        if args.low_res_noise_level is not None
        else int(config.get("validation_noise_level", 10))
    )
    low_res_total = int(pipe.low_res_scheduler.config.num_train_timesteps)
    if not 0 <= low_res_noise_level < low_res_total:
        raise ValueError(
            f"--low_res_noise_level must be in [0,{low_res_total - 1}], got {low_res_noise_level}."
        )
    num_inference_steps = (
        args.num_inference_steps
        if args.num_inference_steps is not None
        else int(config.get("validation_num_inference_steps", 40))
    )
    if num_inference_steps <= 0:
        raise ValueError(f"--num_inference_steps must be positive, got {num_inference_steps}.")
    dataset = build_dataset(config, args.split, args.num_samples)

    lpips_metric: Any | None = None
    lpips_status = "disabled by --skip_lpips"
    if not args.skip_lpips:
        try:
            lpips_metric = OptionalLPIPS(device)
            lpips_status = "enabled"
        except Exception as error:  # optional dependency or unavailable pretrained backbone
            lpips_status = f"skipped: {type(error).__name__}: {error}"
            LOGGER.warning("LPIPS %s", lpips_status)

    snapshot = dict(config)
    snapshot.update(
        {
            "analysis_config": str(config_path),
            "analysis_checkpoint": str(checkpoint),
            "analysis_output_dir": str(output_dir),
            "analysis_split": args.split,
            "analysis_num_samples": min(args.num_samples, len(dataset)),
            "analysis_noise_seed": args.noise_seed,
            "analysis_vae_seed": args.vae_seed,
            "analysis_timestep_stride": args.timestep_stride,
            "analysis_low_res_noise_level": low_res_noise_level,
            "analysis_prediction_type": prediction_type,
            "analysis_latent_scaling_factor": float(pipe.vae.config.scaling_factor),
            "analysis_device": str(device),
            "analysis_dtype": str(dtype),
            "analysis_lpips": lpips_status,
            "analysis_error_display_max": args.error_display_max,
            "analysis_phi_path": str(args.phi_path) if args.phi_path is not None else None,
            "analysis_phi_timestep_range": list(phi_timestep_range),
            "analysis_phi_weight": phi_weight,
            "analysis_num_inference_steps": num_inference_steps,
            "analysis_skip_training_timestep_analysis": args.skip_training_timestep_analysis,
        }
    )
    save_yaml(snapshot, output_dir / "config_snapshot.yaml")
    LOGGER.info(
        "Analyzing %d samples, %d/%d timesteps, prediction_type=%s, device=%s",
        min(args.num_samples, len(dataset)), len(timesteps), total_timesteps, prediction_type, device,
    )

    all_rows: list[dict[str, Any]] = []
    all_reverse_rows: list[dict[str, Any]] = []
    oracle_errors: dict[str, float] = {}
    condition_diagnostics: dict[str, dict[str, float]] = {}
    selected_grids: dict[str, str] = {}
    reverse_grids: dict[str, str] = {}
    selected_set = set(selected_values)
    for index in range(min(args.num_samples, len(dataset))):
        sample = dataset[index]
        if sample.get("source_type") != "bicubic":
            raise ValueError(
                "This diagnostic requires the strictly aligned synthetic condition, but "
                f"sample {sample['filename']} has source_type={sample.get('source_type')!r}. "
                "Use the Stage 1 config whose val_lr_subdir points to LR_bicubic."
            )
        if not args.skip_training_timestep_analysis:
            rows, visuals, oracle_error, sample_condition_diagnostics = analyze_sample(
                sample, pipe, adapter, config, device, dtype, timesteps, selected_set,
                args.noise_seed, args.vae_seed, low_res_noise_level, lpips_metric,
                output_dir, args.save_all_timestep_images, args.error_display_max,
                phi, phi_timestep_range, phi_weight,
            )
            all_rows.extend(rows)
            oracle_errors[str(sample["sample_id"])] = oracle_error
            condition_diagnostics[str(sample["sample_id"])] = sample_condition_diagnostics
            grid = stack_rows(visuals)
            grid_path = output_dir / f"selected_timesteps_grid_{sample['sample_id']}.png"
            grid.save(grid_path)
            selected_grids[str(sample["sample_id"])] = str(grid_path)
            if index == 0:
                grid.save(output_dir / "selected_timesteps_grid.png")

        reverse_rows, reverse_visuals = analyze_pure_noise_trajectory(
            sample,
            pipe,
            adapter,
            device,
            dtype,
            args.noise_seed,
            args.vae_seed,
            low_res_noise_level,
            num_inference_steps,
            output_dir,
            args.error_display_max,
            phi,
            phi_timestep_range,
            phi_weight,
            lpips_metric,
        )
        all_reverse_rows.extend(reverse_rows)
        reverse_grid = stack_rows(reverse_visuals)
        reverse_grid_path = (
            output_dir / "pure_noise_trajectory" / f"trajectory_grid_{sample['sample_id']}.png"
        )
        reverse_grid_path.parent.mkdir(parents=True, exist_ok=True)
        reverse_grid.save(reverse_grid_path)
        reverse_grids[str(sample["sample_id"])] = str(reverse_grid_path)
        if index == 0:
            reverse_grid.save(output_dir / "pure_noise_trajectory_grid.png")

    aggregated: list[dict[str, float]] = []
    reliability: dict[str, Any] | None = None
    if all_rows:
        save_metrics(output_dir / "metrics.csv", all_rows)
        aggregated = aggregate_metrics(all_rows)
        save_curves(output_dir / "curves.png", aggregated)
        if phi is not None:
            save_phi_comparison_curves(
                output_dir / "phi_comparison_curves.png",
                aggregated,
                "training timestep t (larger t = higher noise)",
            )
        reliability = reliability_summary(aggregated)
    save_metrics(
        output_dir / "pure_noise_trajectory" / "metrics.csv",
        all_reverse_rows,
        REVERSE_METRIC_FIELDS,
    )
    if phi is not None and all_reverse_rows:
        reverse_aggregated = aggregate_metrics(all_reverse_rows, REVERSE_METRIC_FIELDS)
        save_phi_comparison_curves(
            output_dir / "pure_noise_trajectory" / "phi_comparison_curves.png",
            reverse_aggregated,
            "scheduler timestep t (trajectory runs from high to low t)",
        )
    summary = {
        "parameterization": prediction_type,
        "phi": {
            "enabled": phi is not None,
            "path": str(args.phi_path) if args.phi_path is not None else None,
            "timestep_range": list(phi_timestep_range),
            "weight": phi_weight,
        },
        "num_train_timesteps": total_timesteps,
        "evaluated_timestep_count": len(timesteps),
        "representative_timesteps": selected_values,
        "hr_derived_training_timestep_analysis_enabled": not args.skip_training_timestep_analysis,
        "pure_noise_trajectory": {
            "enabled": True,
            "num_inference_steps": num_inference_steps,
            "starts_from_hr_independent_gaussian_latent": True,
            "hr_used_only_for_reference_metrics": True,
            "uses_scheduler_scale_model_input_and_step": True,
            "grids": reverse_grids,
            "metrics": str(output_dir / "pure_noise_trajectory" / "metrics.csv"),
        },
        "same_diffusion_noise_for_all_timesteps": True,
        "same_lr_condition_noise_for_all_timesteps": True,
        "noise_seed": args.noise_seed,
        "vae_seed": args.vae_seed,
        "low_res_noise_level": low_res_noise_level,
        "latent_scaling_factor_from_vae_config": float(pipe.vae.config.scaling_factor),
        "oracle_max_abs_z0_recovery_error": oracle_errors,
        "condition_adapter_diagnostics": condition_diagnostics,
        "error_heatmap_fixed_display_max": args.error_display_max,
        "oracle_z0_recovery_check_passed": (
            all(error <= 1e-4 for error in oracle_errors.values()) if oracle_errors else None
        ),
        "lowest_timestep_zt_to_z0_mse": (
            aggregated[0]["zt_to_z0_mse"] if aggregated else None
        ),
        "lpips": lpips_status,
        "selected_grids": selected_grids,
        "reliability_regions": reliability,
        "notes": [
            "decoded z_t visualizes forward-noise severity and is not a restoration metric.",
            "Image metrics compare decoded predicted z0 against the original RGB HR target in [0,1].",
            "Raw LR and ConditionAdapter(LR) are shown separately; the adapter output is not a natural RGB image and may look pale.",
            "HR ground truth is intentionally identical in every row; predicted clean is the timestep-dependent estimate.",
            "Decoded-clip fractions expose values hidden by the required [0,1] display/metric clipping.",
            "The pure-noise trajectory never constructs z_t from HR; HR is reference-only.",
            "At each reverse step, formula x0 is computed from that step's UNet model_output and current latent.",
            "Latent NMSE is latent MSE divided by mean(z0^2).",
        ],
    }
    save_json(summary, output_dir / "summary.json")
    LOGGER.info("Finished. Results: %s", output_dir)


if __name__ == "__main__":
    main()
