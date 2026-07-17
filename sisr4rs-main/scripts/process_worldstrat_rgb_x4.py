#!/usr/bin/env python3
"""
Process RGB WorldStrat x4 LR/GT pairs with the geometric-registration and
low-frequency residual strategy from sisr4rs.

Expected input layout
---------------------
<root>/
    LR/
        image_001.png
        ...
    GT/
        image_001.png
        ...

The LR and GT files must have identical relative paths and names.
GT width/height must be exactly 4x LR width/height by default.

Important
---------
This script accepts visually stretched uint8 RGB images. Geometric correction
is still useful for visual inspection. The optional low-frequency radiometric
correction then operates in display RGB space, not in physical reflectance
space; its output must therefore be treated as visualization-oriented.

Run from the sisr4rs repository root, for example:

pixi run python scripts/process_worldstrat_rgb_x4.py \
    --input-root /data/zhengay/EDiffSR-main/data/EDiffSR_worldstrat_rgb_x4_per_image/val \
    --checkpoint /path/to/wsx4_registration_model.ckpt \
    --output-root /data/zhengay/EDiffSR-main/data/EDiffSR_worldstrat_rgb_x4_per_image/val_registered
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import math
import platform
import subprocess
import sys
import textwrap
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw

# Make the repository package importable when this script is placed in scripts/.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from torchsisr.dataset import generate_psf_kernel, generic_downscale
from torchsisr.registration import UnetOpticalFlowEstimation, warp


SUPPORTED_EXTENSIONS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".tif",
    ".tiff",
    ".bmp",
    ".webp",
}

LOGGER = logging.getLogger("process_worldstrat_rgb_x4")
OFFICIAL_CHECKPOINT_ARCHIVE_URL = (
    "https://zenodo.org/records/14734095/files/"
    "pretrained_registration_models.tar.gz?download=1"
)
METRIC_FIELDS = (
    "relative_path",
    "lr_width",
    "lr_height",
    "gt_width",
    "gt_height",
    "mae_before",
    "mae_after_geo",
    "mae_after_geo_rad",
    "flow_mean_lr_px",
    "flow_p95_lr_px",
    "flow_max_lr_px",
    "status",
    "error_message",
)


def configure_logging(log_path: Path) -> None:
    """Log to both the terminal and the persistent processing log."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    # Each invocation writes a self-contained run log; otherwise a retry after
    # fixing an error would mix stale failures into the new diagnostics.
    handlers.append(logging.FileHandler(log_path, mode="w", encoding="utf-8"))
    for handler in handlers:
        handler.setFormatter(formatter)
    logging.basicConfig(level=logging.INFO, handlers=handlers, force=True)


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def current_git_commit() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def write_json(path: Path, value: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Apply sisr4rs optical-flow geometric correction and optional "
            "low-frequency display-RGB correction to paired WorldStrat RGB x4 data."
        )
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        required=True,
        help="Folder containing LR/ and GT/ subfolders.",
    )
    parser.add_argument(
        "--lr-dir-name",
        default="LR",
        help="LR subfolder name under --input-root. Default: LR",
    )
    parser.add_argument(
        "--gt-dir-name",
        default="GT",
        help="GT subfolder name under --input-root. Default: GT",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Path to the pretrained wsx4 registration Lightning checkpoint.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        required=True,
        help="Output folder.",
    )
    parser.add_argument(
        "--scale",
        type=int,
        default=4,
        help="GT/LR spatial scale factor. Default: 4",
    )
    parser.add_argument(
        "--registration-channel",
        type=int,
        default=0,
        choices=(0, 1, 2),
        help=(
            "RGB channel used for flow estimation: 0=R, 1=G, 2=B. "
            "Default 0 because the original wsx4 checkpoint used Sentinel-2 B4/red."
        ),
    )
    parser.add_argument(
        "--mtf",
        type=float,
        default=0.4,
        help="MTF parameter used to generate pseudo-LR images. Default: 0.4",
    )
    parser.add_argument(
        "--max-offset",
        type=float,
        default=20.0,
        help="Maximum registration-network flow range in LR pixels. Default: 20",
    )
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--min-skip-depth", type=int, default=2)
    parser.add_argument("--num-features", type=int, default=64)
    parser.add_argument(
        "--device",
        default="auto",
        choices=("auto", "cuda", "cpu", "mps"),
        help="Processing device. Default: auto",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process only the first N pairs for a quick visual check.",
    )
    parser.add_argument(
        "--summary-count",
        type=int,
        default=12,
        help="Number of comparison panels placed in summary_grid.png.",
    )
    parser.add_argument(
        "--skip-radiometric",
        action="store_true",
        help="Skip low-frequency display-RGB residual injection.",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Search LR files recursively and preserve relative subfolders.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing result files.",
    )
    return parser.parse_args()


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    device = torch.device(name)
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    if name == "mps" and not (
        hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    ):
        raise RuntimeError("MPS was requested but is not available.")
    return device


def load_rgb(path: Path) -> torch.Tensor:
    """
    Load an image as a float tensor in [0, 1], shape [1, 3, H, W].
    """
    with Image.open(path) as image:
        rgb = image.convert("RGB")
        array = np.asarray(rgb, dtype=np.float32) / 255.0

    return torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0)


def tensor_to_pil_rgb(tensor: torch.Tensor) -> Image.Image:
    """
    Convert [1,3,H,W] or [3,H,W] float tensor to uint8 PIL RGB.
    """
    if tensor.ndim == 4:
        tensor = tensor[0]
    if tensor.ndim != 3 or tensor.shape[0] != 3:
        raise ValueError(f"Expected RGB CHW tensor, received {tuple(tensor.shape)}")

    array = (
        tensor.detach()
        .float()
        .cpu()
        .clamp(0.0, 1.0)
        .permute(1, 2, 0)
        .numpy()
    )
    array = np.rint(array * 255.0).astype(np.uint8)
    return Image.fromarray(array, mode="RGB")


def scalar_map_to_pil(
    value: torch.Tensor,
    percentile: float = 99.0,
) -> Image.Image:
    """
    Save a scalar [H,W] or [1,H,W] map as a robustly normalized grayscale image.
    """
    value = value.detach().float().cpu()
    while value.ndim > 2:
        value = value[0]

    array = value.numpy()
    finite = np.isfinite(array)
    if not finite.any():
        return Image.fromarray(np.zeros(array.shape, dtype=np.uint8), mode="L")

    high = float(np.percentile(array[finite], percentile))
    if not math.isfinite(high) or high <= 1.0e-12:
        high = float(np.max(array[finite]))
    if not math.isfinite(high) or high <= 1.0e-12:
        normalized = np.zeros_like(array, dtype=np.float32)
    else:
        normalized = np.clip(array / high, 0.0, 1.0)

    return Image.fromarray(np.rint(normalized * 255).astype(np.uint8), mode="L")


def depthwise_filter(image: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
    if image.ndim != 4:
        raise ValueError("Expected BCHW tensor.")

    channels = image.shape[1]
    weight = kernel[None, None, :, :].expand(channels, 1, -1, -1)
    return F.conv2d(
        image,
        weight,
        groups=channels,
        padding="same",
    )


def registration_grid_multiple(model: torch.nn.Module) -> int:
    """Return the spatial multiple required by the registration U-Net pools."""
    depth = getattr(getattr(model, "unet", None), "depth", None)
    if isinstance(depth, int) and depth >= 1:
        # A depth-N UNet contains N-1 stride-2 pooling operations.
        return 2 ** (depth - 1)
    return 1


def pad_registration_bands(
    source: torch.Tensor,
    target: torch.Tensor,
    multiple: int,
) -> tuple[torch.Tensor, torch.Tensor, tuple[int, int, int, int]]:
    """Pad equally sized BHW bands to a U-Net-compatible spatial grid."""
    if source.shape != target.shape:
        raise ValueError(
            f"Registration bands must have equal shapes, got {tuple(source.shape)} "
            f"and {tuple(target.shape)}."
        )
    if source.ndim != 3:
        raise ValueError(f"Registration bands must be BHW, got {tuple(source.shape)}.")
    if multiple <= 0:
        raise ValueError(f"Registration grid multiple must be positive, got {multiple}.")

    height, width = source.shape[-2:]
    pad_height = (-height) % multiple
    pad_width = (-width) % multiple
    pad_left = pad_width // 2
    pad_right = pad_width - pad_left
    pad_top = pad_height // 2
    pad_bottom = pad_height - pad_top
    padding = (pad_left, pad_right, pad_top, pad_bottom)
    if not any(padding):
        return source, target, padding

    # Reflection avoids introducing a constant border. Very small inputs can
    # violate reflection-padding limits, so use replication for that edge case.
    pad_mode = (
        "reflect"
        if pad_left < width
        and pad_right < width
        and pad_top < height
        and pad_bottom < height
        else "replicate"
    )
    return F.pad(source, padding, mode=pad_mode), F.pad(
        target, padding, mode=pad_mode
    ), padding


def load_registration_model(
    checkpoint_path: Path,
    *,
    max_offset: float,
    depth: int,
    min_skip_depth: int,
    num_features: int,
    device: torch.device,
) -> UnetOpticalFlowEstimation:
    model = UnetOpticalFlowEstimation(
        max_range=max_offset,
        depth=depth,
        min_skip_depth=min_skip_depth,
        start_filts=num_features,
    )

    try:
        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=False,
        )
    except TypeError:
        # Compatibility with older PyTorch releases.
        checkpoint = torch.load(checkpoint_path, map_location="cpu")

    raw_state = checkpoint.get("state_dict", checkpoint)
    if not isinstance(raw_state, dict) or not all(isinstance(key, str) for key in raw_state):
        raise TypeError("Checkpoint does not contain a string-keyed state_dict mapping.")

    prefixes = (
        "registration_module.",
        "model.registration_module.",
        "module.registration_module.",
    )
    candidate_states: list[tuple[str, dict[str, torch.Tensor]]] = []
    if raw_state and all(
        key.startswith(("unet.", "final_conv.")) for key in raw_state
    ):
        candidate_states.append(("direct registration state_dict", dict(raw_state)))
    for prefix in prefixes:
        stripped = {
            key[len(prefix) :]: value
            for key, value in raw_state.items()
            if key.startswith(prefix)
        }
        if stripped:
            candidate_states.append((f"prefix {prefix!r}", stripped))

    expected_state = model.state_dict()
    diagnostics: list[str] = []
    for label, state in candidate_states:
        missing = sorted(set(expected_state) - set(state))
        unexpected = sorted(set(state) - set(expected_state))
        shape_mismatches = sorted(
            f"{key}: checkpoint={tuple(state[key].shape)} model={tuple(expected_state[key].shape)}"
            for key in set(expected_state) & set(state)
            if tuple(state[key].shape) != tuple(expected_state[key].shape)
        )
        if missing or unexpected or shape_mismatches:
            diagnostics.append(
                f"{label}: missing={missing}; unexpected={unexpected}; "
                f"shape_mismatches={shape_mismatches}"
            )
            continue
        model.load_state_dict(state, strict=True)
        model.eval().to(device)
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        LOGGER.info("Strictly loaded registration weights using %s", label)
        return model

    config_text = (
        f"max_offset={max_offset}, depth={depth}, min_skip_depth={min_skip_depth}, "
        f"num_features={num_features}"
    )
    if not candidate_states:
        prefixes_seen = sorted({key.split(".", 1)[0] for key in raw_state})
        diagnostics.append(f"No registration state found; top-level key prefixes={prefixes_seen}")
    raise RuntimeError(
        "Registration checkpoint is incompatible with strict loading. "
        f"Network configuration: {config_text}. " + " | ".join(diagnostics)
    )


@torch.inference_mode()
def correct_pair(
    lr: torch.Tensor,
    gt: torch.Tensor,
    model: UnetOpticalFlowEstimation,
    *,
    scale: int,
    registration_channel: int,
    mtf: float,
    compute_radiometric: bool,
) -> dict[str, torch.Tensor]:
    """
    Reproduce the core logic of DoubleSISRTrainingModule.register_target().
    Input tensors are display RGB in [0, 1].
    """
    if lr.ndim != 4 or gt.ndim != 4:
        raise ValueError("LR and GT must be BCHW tensors.")
    if lr.shape[0] != 1 or gt.shape[0] != 1:
        raise ValueError("This visual-check script processes one pair at a time.")
    if lr.shape[1] != 3 or gt.shape[1] != 3:
        raise ValueError("This script expects three RGB channels.")

    expected_gt = (lr.shape[-2] * scale, lr.shape[-1] * scale)
    if gt.shape[-2:] != expected_gt:
        raise ValueError(
            f"GT size {tuple(gt.shape[-2:])} does not equal "
            f"{scale}x LR size {tuple(lr.shape[-2:])}; expected {expected_gt}."
        )

    device = next(model.parameters()).device
    lr = lr.to(device=device, dtype=torch.float32)
    gt = gt.to(device=device, dtype=torch.float32)

    # HR/GT -> pseudo-LR using the same MTF-aware helper as the repository.
    pseudo_lr_raw = generic_downscale(
        gt,
        factor=float(scale),
        mtf=mtf,
        padding="valid",
        mode="bicubic",
    )

    if pseudo_lr_raw.shape[-2:] != lr.shape[-2:]:
        raise RuntimeError(
            f"Pseudo-LR size {tuple(pseudo_lr_raw.shape[-2:])} does not match "
            f"LR size {tuple(lr.shape[-2:])}."
        )

    # The pretrained WorldStrat model used the red band. In RGB files this is index 0.
    source_band = pseudo_lr_raw[:, registration_channel, :, :]
    target_band = lr[:, registration_channel, :, :]

    # This U-Net has depth-1 pooling operations and therefore needs H/W to be
    # divisible by 2**(depth-1). Pad only the registration inputs, then crop
    # the predicted flow back to the exact LR grid. No LR or GT is resized.
    registration_multiple = registration_grid_multiple(model)
    source_padded, target_padded, padding = pad_registration_bands(
        source_band,
        target_band,
        registration_multiple,
    )
    flow_padded = model(source_padded, target_padded)
    pad_left, _, pad_top, _ = padding
    height, width = source_band.shape[-2:]
    flow_lr = flow_padded[
        ...,
        pad_top : pad_top + height,
        pad_left : pad_left + width,
    ]
    if flow_lr.shape != (source_band.shape[0], 2, height, width):
        raise RuntimeError(
            "Registration model returned an unexpected flow shape after padding/cropping: "
            f"padded_input={tuple(source_padded.shape)}, "
            f"padded_flow={tuple(flow_padded.shape)}, cropped_flow={tuple(flow_lr.shape)}."
        )

    # Smooth the estimated flow exactly as register_target() does.
    flow_kernel = torch.as_tensor(
        generate_psf_kernel(1.0, 1.0, 1.0e-6, 7),
        device=device,
        dtype=lr.dtype,
    )
    flow_lr = depthwise_filter(flow_lr, flow_kernel)

    # Convert LR-pixel flow to the x4 GT grid.
    flow_gt = F.interpolate(
        flow_lr,
        scale_factor=float(scale),
        mode="bicubic",
        align_corners=False,
    )
    flow_gt = float(scale) * flow_gt

    gt_geo = warp(gt, flow_gt).clamp(0.0, 1.0)
    pseudo_lr_aligned_raw = warp(pseudo_lr_raw, flow_lr)

    # Default radiometric output is equal to geometry-only output.
    gt_geo_rad = gt_geo
    residual_lr = lr - pseudo_lr_aligned_raw
    residual_low_lr = torch.zeros_like(residual_lr)
    residual_gt = torch.zeros_like(gt_geo)

    if compute_radiometric:
        # This is a display-space correction because the input was stretched to uint8.
        residual_kernel = torch.as_tensor(
            generate_psf_kernel(1.0, 1.0, 1.0e-5, 7),
            device=device,
            dtype=lr.dtype,
        )
        residual_low_lr = depthwise_filter(residual_lr, residual_kernel)
        residual_gt = F.interpolate(
            residual_low_lr,
            scale_factor=float(scale),
            mode="bicubic",
            align_corners=False,
        )
        gt_geo_rad = (gt_geo + residual_gt).clamp(0.0, 1.0)

    # Downscale corrected GTs for LR-domain consistency metrics.
    gt_geo_lr = generic_downscale(
        gt_geo,
        factor=float(scale),
        mtf=mtf,
        padding="valid",
        mode="bicubic",
    )
    gt_geo_rad_lr = generic_downscale(
        gt_geo_rad,
        factor=float(scale),
        mtf=mtf,
        padding="valid",
        mode="bicubic",
    )

    return {
        "lr": lr,
        "gt": gt,
        "pseudo_lr": pseudo_lr_raw.clamp(0.0, 1.0),
        "pseudo_lr_aligned": pseudo_lr_aligned_raw.clamp(0.0, 1.0),
        "flow_lr": flow_lr,
        "flow_gt": flow_gt,
        "gt_geo": gt_geo,
        "residual_lr": residual_lr,
        "residual_low_lr": residual_low_lr,
        "residual_gt": residual_gt,
        "gt_geo_rad": gt_geo_rad,
        "gt_geo_lr": gt_geo_lr.clamp(0.0, 1.0),
        "gt_geo_rad_lr": gt_geo_rad_lr.clamp(0.0, 1.0),
        "radiometric_visual_enabled": torch.tensor(compute_radiometric, device=device),
    }


def mae(left: torch.Tensor, right: torch.Tensor) -> float:
    return float(torch.mean(torch.abs(left - right)).detach().cpu())


def save_image(image: Image.Image, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)


def wrap_label(label: str, image_width: int) -> list[str]:
    """Wrap a panel label so it remains readable on small image chips."""
    approximate_character_width = 6
    characters_per_line = max(8, (image_width - 12) // approximate_character_width)
    return textwrap.wrap(label, width=characters_per_line) or [label]


def add_label(image: Image.Image, lines: list[str], label_height: int) -> Image.Image:
    canvas = Image.new("RGB", (image.width, image.height + label_height), "white")
    canvas.paste(image.convert("RGB"), (0, label_height))
    draw = ImageDraw.Draw(canvas)
    for line_number, line in enumerate(lines):
        draw.text((6, 4 + 11 * line_number), line, fill="black")
    return canvas


def make_comparison_panel(result: dict[str, torch.Tensor]) -> Image.Image:
    gt_size = (result["gt"].shape[-1], result["gt"].shape[-2])

    lr_up = tensor_to_pil_rgb(
        F.interpolate(
            result["lr"],
            size=result["gt"].shape[-2:],
            mode="bicubic",
            align_corners=False,
        )
    )
    pseudo_up = tensor_to_pil_rgb(
        F.interpolate(
            result["pseudo_lr"],
            size=result["gt"].shape[-2:],
            mode="bicubic",
            align_corners=False,
        )
    )
    aligned_up = tensor_to_pil_rgb(
        F.interpolate(
            result["pseudo_lr_aligned"],
            size=result["gt"].shape[-2:],
            mode="bicubic",
            align_corners=False,
        )
    )

    radiometric_enabled = bool(result["radiometric_visual_enabled"].item())
    radiometric_label = (
        "GT geometric + low-frequency RGB correction (visual)"
        if radiometric_enabled
        else "Radiometric visual correction skipped"
    )
    images_and_labels = [
        (lr_up, "LR bicubic x4"),
        (tensor_to_pil_rgb(result["gt"]), "Original GT"),
        (tensor_to_pil_rgb(result["gt_geo"]), "GT geometric"),
        (tensor_to_pil_rgb(result["gt_geo_rad"]), radiometric_label),
        (pseudo_up, "Pseudo-LR from GT"),
        (aligned_up, "Pseudo-LR aligned"),
    ]
    wrapped_labels = [wrap_label(label, gt_size[0]) for _, label in images_and_labels]
    label_height = 8 + 11 * max(len(lines) for lines in wrapped_labels)
    tiles = [
        add_label(image, lines, label_height)
        for (image, _), lines in zip(images_and_labels, wrapped_labels)
    ]

    tile_w = gt_size[0]
    tile_h = gt_size[1] + label_height
    panel = Image.new("RGB", (3 * tile_w, 2 * tile_h), "white")

    for index, tile in enumerate(tiles):
        row = index // 3
        col = index % 3
        panel.paste(tile, (col * tile_w, row * tile_h))

    return panel


def build_summary_grid(panel_paths: list[Path], output_path: Path) -> None:
    if not panel_paths:
        return

    panels = [Image.open(path).convert("RGB") for path in panel_paths]
    try:
        columns = 2 if len(panels) > 1 else 1
        rows = math.ceil(len(panels) / columns)
        panel_w = max(image.width for image in panels)
        panel_h = max(image.height for image in panels)

        summary = Image.new("RGB", (columns * panel_w, rows * panel_h), "white")
        for index, image in enumerate(panels):
            row = index // columns
            col = index % columns
            summary.paste(image, (col * panel_w, row * panel_h))

        output_path.parent.mkdir(parents=True, exist_ok=True)
        summary.save(output_path)
    finally:
        for image in panels:
            image.close()


def find_lr_files(lr_root: Path, recursive: bool) -> list[Path]:
    iterator: Iterable[Path]
    iterator = lr_root.rglob("*") if recursive else lr_root.glob("*")
    return sorted(
        path
        for path in iterator
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )


def output_path(output_root: Path, category: str, relative_path: Path) -> Path:
    return output_root / category / relative_path.with_suffix(".png")


def metric_row(relative_path: Path, status: str, error_message: str = "") -> dict[str, object]:
    row: dict[str, object] = {field: "" for field in METRIC_FIELDS}
    row.update(
        {
            "relative_path": relative_path.as_posix(),
            "status": status,
            "error_message": error_message,
        }
    )
    return row


def write_metrics(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=METRIC_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def validate_args(args: argparse.Namespace) -> None:
    if args.scale <= 0:
        raise ValueError(f"--scale must be positive, got {args.scale}")
    if not 0.0 < args.mtf <= 1.0:
        raise ValueError(f"--mtf must be in (0, 1], got {args.mtf}")
    if args.max_offset <= 0:
        raise ValueError(f"--max-offset must be positive, got {args.max_offset}")
    if args.limit is not None and args.limit <= 0:
        raise ValueError(f"--limit must be positive, got {args.limit}")
    if args.summary_count < 0:
        raise ValueError(f"--summary-count must be non-negative, got {args.summary_count}")


def build_run_config(
    args: argparse.Namespace,
    device: torch.device,
    checkpoint_sha256: str,
    started_at: datetime,
) -> dict[str, object]:
    return {
        "input_root": str(args.input_root.resolve()),
        "lr_dir_name": args.lr_dir_name,
        "gt_dir_name": args.gt_dir_name,
        "output_root": str(args.output_root.resolve()),
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": checkpoint_sha256,
        "checkpoint_size_bytes": args.checkpoint.stat().st_size,
        "official_checkpoint_archive_url": OFFICIAL_CHECKPOINT_ARCHIVE_URL,
        "scale": args.scale,
        "mtf": args.mtf,
        "registration_channel": args.registration_channel,
        "max_offset": args.max_offset,
        "model_structure": {
            "class": "torchsisr.registration.UnetOpticalFlowEstimation",
            "depth": args.depth,
            "min_skip_depth": args.min_skip_depth,
            "num_features": args.num_features,
        },
        "python_version": platform.python_version(),
        "python_executable": sys.executable,
        "pytorch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "device": str(device),
        "started_at_utc": started_at.isoformat(),
        "git_commit": current_git_commit(),
        "radiometric_visual_correction_enabled": not args.skip_radiometric,
        "radiometric_note": (
            "Low-frequency correction is performed in stretched display-RGB space; "
            "it is not a physical reflectance correction."
        ),
        "recursive": args.recursive,
        "limit": args.limit,
    }


def log_diagnostics(rows: list[dict[str, object]], max_offset: float) -> dict[str, object]:
    successful = [row for row in rows if row["status"] == "success"]
    improved = [
        row for row in successful if float(row["mae_after_geo"]) < float(row["mae_before"])
    ]
    saturated_p95 = [
        row for row in successful if float(row["flow_p95_lr_px"]) > 0.9 * max_offset
    ]
    near_limit = [
        row for row in successful if float(row["flow_max_lr_px"]) >= 0.99 * max_offset
    ]
    degraded = sorted(
        successful,
        key=lambda row: float(row["mae_after_geo"]) - float(row["mae_before"]),
        reverse=True,
    )[:20]

    denominator = len(successful)
    summary: dict[str, object] = {
        "successful": denominator,
        "skipped": sum(str(row["status"]).startswith("skipped") for row in rows),
        "missing_gt": sum(row["status"] == "skipped_missing_gt" for row in rows),
        "size_mismatch": sum(row["status"] == "failed_size_mismatch" for row in rows),
        "processing_errors": sum(row["status"] == "failed_processing" for row in rows),
        "failed": sum(str(row["status"]).startswith("failed") for row in rows),
        "mae_after_geo_improved_fraction": len(improved) / denominator if denominator else None,
        "flow_p95_saturated_fraction": len(saturated_p95) / denominator if denominator else None,
        "flow_max_near_limit_samples": [str(row["relative_path"]) for row in near_limit],
        "most_degraded_after_geo": [
            {
                "relative_path": row["relative_path"],
                "mae_delta": float(row["mae_after_geo"]) - float(row["mae_before"]),
            }
            for row in degraded
        ],
    }
    LOGGER.info(
        "Summary: success=%d skipped=%d missing_gt=%d size_mismatch=%d processing_errors=%d",
        summary["successful"],
        summary["skipped"],
        summary["missing_gt"],
        summary["size_mismatch"],
        summary["processing_errors"],
    )
    if denominator:
        LOGGER.info(
            "MAE improved after geometry: %.1f%%; flow p95 > 0.9*max_offset: %.1f%%",
            100.0 * len(improved) / denominator,
            100.0 * len(saturated_p95) / denominator,
        )
    if near_limit:
        LOGGER.warning(
            "Flow maximum reached >=99%% of max_offset for: %s",
            ", ".join(str(row["relative_path"]) for row in near_limit),
        )
    for item in summary["most_degraded_after_geo"]:
        LOGGER.info(
            "Geometry MAE degradation: %s delta=%+.6f",
            item["relative_path"],
            item["mae_delta"],
        )
    return summary


def main() -> None:
    args = parse_args()
    validate_args(args)
    args.input_root = args.input_root.expanduser().resolve()
    args.checkpoint = args.checkpoint.expanduser().resolve()
    args.output_root = args.output_root.expanduser().resolve()
    device = resolve_device(args.device)

    lr_root = args.input_root / args.lr_dir_name
    gt_root = args.input_root / args.gt_dir_name

    if not lr_root.is_dir():
        raise FileNotFoundError(f"LR directory does not exist: {lr_root}")
    if not gt_root.is_dir():
        raise FileNotFoundError(f"GT directory does not exist: {gt_root}")
    if not args.checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {args.checkpoint}")
    if args.output_root == gt_root or gt_root in args.output_root.parents:
        raise ValueError(
            f"--output-root must not be the original GT directory or one of its children: {gt_root}"
        )

    args.output_root.mkdir(parents=True, exist_ok=True)
    output_categories = (
        "GT_geo",
        "GT_geo_rad_visual",
        "pseudo_LR",
        "pseudo_LR_aligned",
        "flow_magnitude",
        "error_before",
        "error_after_geo",
        "error_after_rad",
        "comparisons",
    )
    for category in output_categories:
        (args.output_root / category).mkdir(parents=True, exist_ok=True)
    configure_logging(args.output_root / "processing.log")
    started_at = datetime.now(timezone.utc)
    monotonic_start = time.monotonic()
    checkpoint_sha256 = sha256_file(args.checkpoint)
    run_config = build_run_config(args, device, checkpoint_sha256, started_at)
    write_json(args.output_root / "run_config.json", run_config)

    LOGGER.info("Input LR: %s", lr_root)
    LOGGER.info("Input GT: %s", gt_root)
    LOGGER.info("Output: %s", args.output_root)
    LOGGER.info("Checkpoint: %s", args.checkpoint)
    LOGGER.info("Checkpoint SHA256: %s", checkpoint_sha256)
    LOGGER.info("Device: %s", device)
    LOGGER.info(
        "Radiometric visual correction: %s (stretched display RGB; not physical reflectance)",
        "disabled" if args.skip_radiometric else "enabled",
    )

    model = load_registration_model(
        args.checkpoint,
        max_offset=args.max_offset,
        depth=args.depth,
        min_skip_depth=args.min_skip_depth,
        num_features=args.num_features,
        device=device,
    )

    lr_files = find_lr_files(lr_root, args.recursive)
    if args.limit is not None:
        lr_files = lr_files[: args.limit]

    if not lr_files:
        raise RuntimeError(f"No supported images were found in {lr_root}")

    metrics_path = args.output_root / "metrics.csv"
    comparison_paths: list[Path] = []
    rows: list[dict[str, object]] = []

    LOGGER.info("LR files selected: %d", len(lr_files))

    for index, lr_path in enumerate(lr_files, start=1):
        relative = lr_path.relative_to(lr_root)
        gt_path = gt_root / relative

        if not gt_path.is_file():
            message = f"Missing same-relative-path GT: {gt_path}"
            LOGGER.warning("[skip] %s", message)
            rows.append(metric_row(relative, "skipped_missing_gt", message))
            continue

        comparison_path = output_path(args.output_root, "comparisons", relative)
        if comparison_path.exists() and not args.overwrite:
            message = f"Existing result (use --overwrite): {comparison_path}"
            LOGGER.info("[skip] %s", message)
            rows.append(metric_row(relative, "skipped_existing", message))
            continue

        try:
            lr = load_rgb(lr_path)
            gt = load_rgb(gt_path)

            expected_gt = (lr.shape[-2] * args.scale, lr.shape[-1] * args.scale)
            if gt.shape[-2:] != expected_gt:
                message = (
                    f"GT HxW={tuple(gt.shape[-2:])} does not equal {args.scale}x "
                    f"LR HxW={tuple(lr.shape[-2:])}; expected {expected_gt}; no resize was applied"
                )
                row = metric_row(relative, "failed_size_mismatch", message)
                row.update(
                    {
                        "lr_width": int(lr.shape[-1]),
                        "lr_height": int(lr.shape[-2]),
                        "gt_width": int(gt.shape[-1]),
                        "gt_height": int(gt.shape[-2]),
                    }
                )
                rows.append(row)
                LOGGER.error("[%d/%d] %s: %s", index, len(lr_files), relative, message)
                continue

            result = correct_pair(
                lr,
                gt,
                model,
                scale=args.scale,
                registration_channel=args.registration_channel,
                mtf=args.mtf,
                compute_radiometric=not args.skip_radiometric,
            )

            # Core outputs.
            save_image(
                tensor_to_pil_rgb(result["gt_geo"]),
                output_path(args.output_root, "GT_geo", relative),
            )
            if not args.skip_radiometric:
                save_image(
                    tensor_to_pil_rgb(result["gt_geo_rad"]),
                    output_path(args.output_root, "GT_geo_rad_visual", relative),
                )
            save_image(
                tensor_to_pil_rgb(result["pseudo_lr"]),
                output_path(args.output_root, "pseudo_LR", relative),
            )
            save_image(
                tensor_to_pil_rgb(result["pseudo_lr_aligned"]),
                output_path(args.output_root, "pseudo_LR_aligned", relative),
            )

            # Diagnostic maps.
            flow_magnitude = torch.linalg.vector_norm(result["flow_lr"], dim=1)
            save_image(
                scalar_map_to_pil(flow_magnitude).convert("RGB"),
                output_path(args.output_root, "flow_magnitude", relative),
            )

            error_before = torch.mean(
                torch.abs(result["lr"] - result["pseudo_lr"]),
                dim=1,
            )
            error_after_geo = torch.mean(
                torch.abs(result["lr"] - result["gt_geo_lr"]),
                dim=1,
            )
            error_after_rad = torch.mean(
                torch.abs(result["lr"] - result["gt_geo_rad_lr"]),
                dim=1,
            )

            save_image(
                scalar_map_to_pil(error_before).convert("RGB"),
                output_path(args.output_root, "error_before", relative),
            )
            save_image(
                scalar_map_to_pil(error_after_geo).convert("RGB"),
                output_path(args.output_root, "error_after_geo", relative),
            )
            if not args.skip_radiometric:
                save_image(
                    scalar_map_to_pil(error_after_rad).convert("RGB"),
                    output_path(args.output_root, "error_after_rad", relative),
                )

            panel = make_comparison_panel(result)
            save_image(panel, comparison_path)
            if len(comparison_paths) < args.summary_count:
                comparison_paths.append(comparison_path)

            flow_flat = flow_magnitude.detach().float().cpu().flatten()
            row = metric_row(relative, "success")
            row.update({
                "lr_width": int(lr.shape[-1]),
                "lr_height": int(lr.shape[-2]),
                "gt_width": int(gt.shape[-1]),
                "gt_height": int(gt.shape[-2]),
                "mae_before": mae(result["lr"], result["pseudo_lr"]),
                "mae_after_geo": mae(result["lr"], result["gt_geo_lr"]),
                "mae_after_geo_rad": (
                    "" if args.skip_radiometric else mae(result["lr"], result["gt_geo_rad_lr"])
                ),
                "flow_mean_lr_px": float(flow_flat.mean()),
                "flow_p95_lr_px": float(torch.quantile(flow_flat, 0.95)),
                "flow_max_lr_px": float(flow_flat.max()),
            })
            rows.append(row)

            LOGGER.info(
                f"[{index}/{len(lr_files)}] {relative} | "
                f"MAE {row['mae_before']:.4f} -> "
                f"{row['mae_after_geo']:.4f} | "
                f"flow p95={row['flow_p95_lr_px']:.2f}px"
            )

        except Exception as exc:  # noqa: BLE001 - continue batch and report pair
            message = f"{type(exc).__name__}: {exc}"
            LOGGER.exception("[error] %s: %s", relative, message)
            rows.append(metric_row(relative, "failed_processing", message))

    write_metrics(metrics_path, rows)

    build_summary_grid(
        comparison_paths,
        args.output_root / "summary_grid.png",
    )

    diagnostics = log_diagnostics(rows, args.max_offset)
    finished_at = datetime.now(timezone.utc)
    run_config.update(
        {
            "finished_at_utc": finished_at.isoformat(),
            "elapsed_seconds": time.monotonic() - monotonic_start,
            "diagnostics": diagnostics,
        }
    )
    write_json(args.output_root / "run_config.json", run_config)
    LOGGER.info("Metrics: %s", metrics_path)
    LOGGER.info("Visual summary: %s", args.output_root / "summary_grid.png")


if __name__ == "__main__":
    main()
