#!/usr/bin/env python
"""Evaluate SR fidelity, spectral error, and LR observation consistency."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image, UnidentifiedImageError
from tqdm.auto import tqdm

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.dataset import SUPPORTED_EXTENSIONS
from src.metrics import OptionalLPIPS, basic_metrics, bicubic_downsample, psnr, spectral_angle_deg
from src.utils import configure_logging, pil_to_tensor

LOGGER = logging.getLogger("evaluate")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sr_dir", type=Path, required=True)
    parser.add_argument("--gt_dir", type=Path, required=True)
    parser.add_argument("--lr_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--skip_lpips", action="store_true")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return parser.parse_args()


def summarize(frame: pd.DataFrame) -> dict[str, dict[str, float]]:
    """Summarize every numeric metric with requested robust statistics."""
    result: dict[str, dict[str, float]] = {}
    for column in frame.select_dtypes(include=[np.number]).columns:
        values = frame[column].dropna().to_numpy(dtype=np.float64)
        if values.size:
            result[column] = {
                "mean": float(np.mean(values)),
                "median": float(np.median(values)),
                "std": float(np.std(values, ddof=0)),
                "p5": float(np.percentile(values, 5)),
                "p95": float(np.percentile(values, 95)),
                "count": int(values.size),
            }
    return result


def main() -> None:
    args = parse_args()
    configure_logging()
    sr_dir, gt_dir, lr_dir = (path.expanduser().resolve() for path in (args.sr_dir, args.gt_dir, args.lr_dir))
    for label, directory in (("SR", sr_dir), ("GT", gt_dir), ("LR", lr_dir)):
        if not directory.is_dir():
            raise FileNotFoundError(f"{label} directory does not exist: {directory}")
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available() else "cpu" if args.device == "auto" else args.device
    )
    lpips_metric: OptionalLPIPS | None = None
    if not args.skip_lpips:
        try:
            lpips_metric = OptionalLPIPS(device)
        except Exception as error:  # LPIPS may also fail while fetching its backbone weights.
            LOGGER.warning(
                "LPIPS initialization failed (%s: %s). Continuing without LPIPS; values will be blank.",
                type(error).__name__,
                error,
            )

    sr_files = [path for path in sorted(sr_dir.iterdir()) if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS]
    if not sr_files:
        raise RuntimeError(f"No supported SR images found in {sr_dir}")
    rows: list[dict[str, float | str]] = []
    for sr_path in tqdm(sr_files, desc="evaluation"):
        gt_path, lr_path = gt_dir / sr_path.name, lr_dir / sr_path.name
        if not gt_path.is_file() or not lr_path.is_file():
            raise FileNotFoundError(
                f"Exact-name pair missing for {sr_path.name}: GT exists={gt_path.is_file()}, LR exists={lr_path.is_file()}"
            )
        try:
            with Image.open(sr_path) as image:
                sr = pil_to_tensor(image.convert("RGB"))
            with Image.open(gt_path) as image:
                gt = pil_to_tensor(image.convert("RGB"))
            with Image.open(lr_path) as image:
                lr = pil_to_tensor(image.convert("RGB"))
        except (OSError, UnidentifiedImageError) as error:
            raise RuntimeError(f"Could not read metric triplet for {sr_path.name}: {error}") from error
        expected = (lr.shape[-2] * 4, lr.shape[-1] * 4)
        if tuple(sr.shape[-2:]) != expected or sr.shape != gt.shape:
            raise ValueError(
                f"Size mismatch for {sr_path.name}: SR={tuple(sr.shape)}, GT={tuple(gt.shape)}, "
                f"LR={tuple(lr.shape)}, expected SR HxW={expected}"
            )
        metrics = basic_metrics(sr, gt)
        if lpips_metric is not None:
            metrics["lpips"] = lpips_metric(sr, gt)
        else:
            metrics["lpips"] = float("nan")
        low_sr = bicubic_downsample(sr, (lr.shape[-2], lr.shape[-1]))
        low_difference = low_sr - lr
        metrics.update(
            {
                "low_frequency_mae": float(low_difference.abs().mean().item()),
                "low_frequency_psnr": psnr(low_sr, lr),
                "low_frequency_spectral_angle": spectral_angle_deg(low_sr, lr),
            }
        )
        rows.append({"filename": sr_path.name, **metrics})

    frame = pd.DataFrame(rows)
    frame.to_csv(output_dir / "per_image_metrics.csv", index=False)
    summary = summarize(frame)
    with (output_dir / "summary_metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, allow_nan=False)
    summary_rows = [{"metric": metric, **statistics} for metric, statistics in summary.items()]
    pd.DataFrame(summary_rows).to_csv(output_dir / "summary_metrics.csv", index=False)
    LOGGER.info("Evaluated %d images; metrics written to %s", len(frame), output_dir)


if __name__ == "__main__":
    main()
