#!/usr/bin/env python
"""Run whole-image or Hann-blended tiled x4 upscaler inference."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm.auto import tqdm

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.condition_adapter import ConditionAdapter
from src.dataset import SUPPORTED_EXTENSIONS
from src.tri_input_bridge import (
    TriConditionedUNet,
    TriInputBridge,
    pipeline_cross_attention_kwargs,
)
from src.tri_input_conditioner import (
    TRI_INPUT_CONFIG_NAME,
    TRI_INPUT_WEIGHTS_NAME,
    TriInputConditioner,
)
from src.tri_input_data import (
    AuxiliaryPair,
    RawBandStats,
    RawValueConversion,
    build_auxiliary_pairs,
    load_auxiliary_pair,
    load_raw_band_stats,
    validate_sentinel2_l2a_band_names,
)
from src.tri_input_runtime import prepare_tri_condition, tri_diagnostic_metrics
from src.utils import (
    FIXED_PROMPT,
    NEGATIVE_PROMPT,
    configure_logging,
    load_yaml_config,
    low_frequency_projection,
    normalize_tokenizer_max_length,
    pil_to_tensor,
    require_diffusers_version,
    resolve_project_path,
    tensor_to_pil,
)

LOGGER = logging.getLogger("infer")
PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class InferenceAuxiliary:
    raw_ms: torch.Tensor
    unmixing: torch.Tensor
    raw_valid: torch.Tensor
    raw_ms_path: Path
    unmixing_path: Path


@dataclass(frozen=True)
class TriInferenceRuntime:
    conditioner: TriInputConditioner
    bridge: TriInputBridge
    band_names: tuple[str, ...]
    conversion: RawValueConversion
    stats: RawBandStats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=None, help="Training YAML used for model_id and data_root")
    parser.add_argument("--model_id", default=None)
    parser.add_argument(
        "--checkpoint_path",
        "--artifact_path",
        dest="artifact_path",
        type=Path,
        default=None,
        help="Any checkpoint-* or final artifact directory containing both LoRA and ConditionAdapter weights",
    )
    parser.add_argument("--lora_path", type=Path, default=None, help="Advanced override for the LoRA file/directory")
    parser.add_argument("--adapter_path", type=Path, default=None, help="Advanced override for the adapter file/directory")
    parser.add_argument("--input_dir", type=Path, default=None, help="Arbitrary LR directory; independent of training data")
    parser.add_argument("--gt_dir", type=Path, default=None, help="Optional matching GT directory")
    parser.add_argument(
        "--split",
        choices=("train", "val", "test"),
        default=None,
        help="Explicit auxiliary-data split when a tri-input checkpoint is enabled",
    )
    parser.add_argument("--raw_ms_dir", type=Path, default=None)
    parser.add_argument("--unmixing_dir", type=Path, default=None)
    parser.add_argument("--aux_manifest_path", type=Path, default=None)
    parser.add_argument("--raw_stats_path", type=Path, default=None)
    parser.add_argument("--aux_recursive", action="store_true")
    parser.add_argument(
        "--disable_tri_input",
        action="store_true",
        help="Explicit RGB-only ablation even when the artifact declares tri-input conditioning",
    )
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--prompt_mode", choices=("fixed", "metadata"), default="fixed")
    parser.add_argument("--metadata_path", type=Path, default=None)
    parser.add_argument("--guidance_scale", type=float, default=1.0)
    parser.add_argument("--noise_level", type=int, default=10)
    parser.add_argument("--num_inference_steps", type=int, default=40)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--low_freq_projection_alpha", type=float, default=0.5)
    parser.add_argument("--tiled", action="store_true")
    parser.add_argument("--tile_size", type=int, default=128)
    parser.add_argument("--tile_overlap", type=int, default=32)
    parser.add_argument("--mixed_precision", choices=("no", "fp16", "bf16"), default=None)
    parser.add_argument(
        "--sample",
        action="append",
        default=[],
        help="Infer one filename or filename stem; repeat this option to select multiple samples",
    )
    parser.add_argument("--sample_file", type=Path, default=None, help="Text file with one filename or stem per line")
    parser.add_argument("--start_index", type=int, default=0, help="Skip this many sorted/selected input samples")
    parser.add_argument("--limit", type=int, default=None, help="Maximum number of samples after filtering and skipping")
    return parser.parse_args()


def _prompt_lookup(mode: str, metadata_path: Path | None) -> Callable[[str], str]:
    if mode != "metadata" or metadata_path is None or not metadata_path.is_file():
        if mode == "metadata":
            LOGGER.warning("Metadata prompt mode requested but metadata CSV was not found; using fixed prompt")
        return lambda _sample_id: FIXED_PROMPT
    frame = pd.read_csv(metadata_path, dtype=str).fillna("")
    rows: dict[str, dict[str, str]] = {}
    for _, row in frame.iterrows():
        key = str(row.get("sample_id") or Path(str(row.get("filename", ""))).stem)
        values = {str(column): str(value) for column, value in row.items()}
        rows[key] = values
        filename = str(row.get("filename", "")).strip()
        if filename:
            rows[Path(filename).stem] = values

    def lookup(sample_id: str) -> str:
        row = rows.get(sample_id, {})
        ipcc, smod = row.get("IPCC Class", "").strip(), row.get("SMOD Class", "").strip()
        if ipcc and smod:
            return (
                f"a high-resolution overhead satellite image of {ipcc}, {smod}, "
                "with accurate geographic structures and natural colors"
            )
        return FIXED_PROMPT

    return lookup


def _positions(length: int, tile: int, overlap: int) -> list[int]:
    if length < tile:
        raise ValueError(f"LR dimension {length} is smaller than tile_size={tile}; use whole-image inference")
    stride = tile - overlap
    if stride <= 0:
        raise ValueError(f"tile_overlap={overlap} must be smaller than tile_size={tile}")
    positions = list(range(0, max(1, length - tile + 1), stride))
    final = length - tile
    if positions[-1] != final:
        positions.append(final)
    return positions


def _select_input_files(
    input_dir: Path,
    requested_samples: list[str],
    sample_file: Path | None,
    start_index: int,
    limit: int | None,
) -> list[Path]:
    """Select deterministic input files by exact filename or unambiguous stem."""
    if start_index < 0:
        raise ValueError(f"--start_index must be non-negative, got {start_index}")
    if limit is not None and limit <= 0:
        raise ValueError(f"--limit must be positive, got {limit}")

    selectors = [value.strip() for value in requested_samples if value.strip()]
    if sample_file is not None:
        list_path = sample_file.expanduser().resolve()
        if not list_path.is_file():
            raise FileNotFoundError(f"Sample list does not exist: {list_path}")
        selectors.extend(
            line.strip()
            for line in list_path.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        )

    available = [
        path for path in sorted(input_dir.iterdir()) if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    ]
    if selectors:
        by_name = {path.name: path for path in available}
        by_stem: dict[str, list[Path]] = {}
        for path in available:
            by_stem.setdefault(path.stem, []).append(path)
        selected: list[Path] = []
        seen: set[Path] = set()
        for selector in selectors:
            match = by_name.get(Path(selector).name)
            if match is None:
                stem_matches = by_stem.get(Path(selector).stem, [])
                if len(stem_matches) > 1:
                    names = ", ".join(path.name for path in stem_matches)
                    raise ValueError(f"Sample stem {selector!r} is ambiguous; use an exact filename: {names}")
                match = stem_matches[0] if stem_matches else None
            if match is None:
                raise FileNotFoundError(f"Requested sample {selector!r} was not found under {input_dir}")
            if match not in seen:
                selected.append(match)
                seen.add(match)
        available = selected

    files = available[start_index:]
    if limit is not None:
        files = files[:limit]
    return files


def _adapt_image(adapter: ConditionAdapter, image: Image.Image, device: torch.device, dtype: torch.dtype) -> Image.Image:
    tensor = pil_to_tensor(image, "minus_one_one").unsqueeze(0).to(device=device, dtype=dtype)
    with torch.no_grad():
        adapted = adapter(tensor).float().cpu()[0]
    return tensor_to_pil(adapted)


def _pad_to_multiple(image: Image.Image, multiple: int) -> tuple[Image.Image, tuple[int, int]]:
    """Edge-pad an LR image so the upscaler does not silently shrink its dimensions."""
    if multiple <= 0:
        raise ValueError(f"Padding multiple must be positive, got {multiple}")
    original_size = image.size
    pad_width = (-image.width) % multiple
    pad_height = (-image.height) % multiple
    if pad_width == 0 and pad_height == 0:
        return image, original_size
    array = np.asarray(image)
    padded = np.pad(array, ((0, pad_height), (0, pad_width), (0, 0)), mode="edge")
    return Image.fromarray(padded, mode="RGB"), original_size


def _pad_auxiliary(
    auxiliary: InferenceAuxiliary, target_height: int, target_width: int
) -> InferenceAuxiliary:
    """Pad auxiliary inputs without inventing observations in padded pixels."""

    height, width = auxiliary.raw_ms.shape[-2:]
    if auxiliary.unmixing.shape != (10, height, width):
        raise ValueError(
            f"Unmixing tensor shape {tuple(auxiliary.unmixing.shape)} does not match "
            f"raw grid {(height, width)} for {auxiliary.unmixing_path}"
        )
    if auxiliary.raw_valid.shape != (1, height, width):
        raise ValueError(
            f"raw_valid shape {tuple(auxiliary.raw_valid.shape)} does not match raw grid "
            f"{(height, width)} for {auxiliary.raw_ms_path}"
        )
    if target_height < height or target_width < width:
        raise ValueError("Auxiliary padding target cannot be smaller than the source grid")
    if (target_height, target_width) == (height, width):
        return auxiliary
    raw = torch.zeros(12, target_height, target_width, dtype=torch.float32)
    raw[:, :height, :width] = auxiliary.raw_ms
    valid = torch.zeros(1, target_height, target_width, dtype=torch.bool)
    valid[:, :height, :width] = auxiliary.raw_valid
    # F=0,U=1 is an explicit unknown prior in padding, not zero cover.
    prior = torch.cat(
        (
            torch.zeros(5, target_height, target_width, dtype=torch.float32),
            torch.ones(5, target_height, target_width, dtype=torch.float32),
        ),
        dim=0,
    )
    prior[:, :height, :width] = auxiliary.unmixing
    return InferenceAuxiliary(
        raw_ms=raw,
        unmixing=prior,
        raw_valid=valid,
        raw_ms_path=auxiliary.raw_ms_path,
        unmixing_path=auxiliary.unmixing_path,
    )


def _crop_auxiliary(
    auxiliary: InferenceAuxiliary, left: int, top: int, width: int, height: int
) -> InferenceAuxiliary:
    right, bottom = left + width, top + height
    raw_height, raw_width = auxiliary.raw_ms.shape[-2:]
    if left < 0 or top < 0 or right > raw_width or bottom > raw_height:
        raise ValueError(
            f"Auxiliary tile {(left, top, right, bottom)} exceeds raw grid "
            f"{(raw_width, raw_height)} for {auxiliary.raw_ms_path}"
        )
    return InferenceAuxiliary(
        raw_ms=auxiliary.raw_ms[:, top:bottom, left:right],
        unmixing=auxiliary.unmixing[:, top:bottom, left:right],
        raw_valid=auxiliary.raw_valid[:, top:bottom, left:right],
        raw_ms_path=auxiliary.raw_ms_path,
        unmixing_path=auxiliary.unmixing_path,
    )


def _load_inference_auxiliary(
    pair: AuxiliaryPair,
    runtime: TriInferenceRuntime,
    expected_size: tuple[int, int],
) -> InferenceAuxiliary:
    raw, prior = load_auxiliary_pair(pair, runtime.band_names, runtime.conversion)
    expected_hw = (expected_size[1], expected_size[0])
    if tuple(raw.raw_ms.shape[-2:]) != expected_hw:
        raise ValueError(
            f"Tri-input grid mismatch for sample {pair.sample_id!r}: RGB={expected_hw}, "
            f"raw={tuple(raw.raw_ms.shape[-2:])} ({pair.raw_ms_path}); silent resize is forbidden"
        )
    return InferenceAuxiliary(
        raw_ms=raw.raw_ms,
        unmixing=prior,
        raw_valid=raw.raw_valid,
        raw_ms_path=pair.raw_ms_path,
        unmixing_path=pair.unmixing_path,
    )


def _tri_pipeline_kwargs(
    pipe: Any,
    runtime: TriInferenceRuntime,
    image: Image.Image,
    auxiliary: InferenceAuxiliary,
    *,
    device: torch.device,
    dtype: torch.dtype,
    guidance_scale: float,
) -> dict[str, Any]:
    if auxiliary.raw_ms.shape[-2:] != (image.height, image.width):
        raise ValueError(
            f"RGB/auxiliary condition shapes differ: RGB={(image.height, image.width)}, "
            f"raw={tuple(auxiliary.raw_ms.shape[-2:])}"
        )
    rgb = pil_to_tensor(image, "minus_one_one").unsqueeze(0).to(
        device=device, dtype=torch.float32
    )
    dummy = torch.empty(
        (1, int(pipe.unet.config.in_channels), image.height, image.width),
        device=device,
        dtype=dtype,
    )
    prepared = prepare_tri_condition(
        runtime.conditioner,
        runtime.bridge,
        rgb_minus_one_one=rgb,
        raw_ms=auxiliary.raw_ms.unsqueeze(0).to(device=device, dtype=torch.float32),
        unmixing=auxiliary.unmixing.unsqueeze(0).to(device=device, dtype=torch.float32),
        raw_valid=auxiliary.raw_valid.unsqueeze(0).to(device=device).bool(),
        aux_present=torch.ones(1, device=device, dtype=torch.bool),
        unet_sample=dummy,
        do_classifier_free_guidance=guidance_scale > 1.0,
    )
    if prepared is None:  # impossible for explicit inference observations
        raise RuntimeError("Tri-input inference unexpectedly produced an inactive condition")
    diagnostics = tri_diagnostic_metrics(prepared)
    LOGGER.info(
        "Tri-input checker for %s: valid_windows=%.1f accepted_fraction=%.4f "
        "mean_shift_hr=%.4f internal_gain=%.6g reason=%s "
        "(internal heuristic, not calibrated truth confidence)",
        auxiliary.raw_ms_path.name,
        float(diagnostics["checker_valid_windows"]),
        float(diagnostics["checker_accepted_fraction"]),
        float(diagnostics["checker_mean_shift_hr"]),
        float(diagnostics["checker_internal_gain"]),
        ",".join(prepared.output.checker_diagnostics.fallback_reasons),
    )
    return {
        "cross_attention_kwargs": pipeline_cross_attention_kwargs(prepared.residuals)
    }


def _run_pipeline(
    pipe: Any,
    adapter: ConditionAdapter,
    image: Image.Image,
    prompt: str,
    args: argparse.Namespace,
    generator: torch.Generator,
    device: torch.device,
    dtype: torch.dtype,
    tri_runtime: TriInferenceRuntime | None = None,
    auxiliary: InferenceAuxiliary | None = None,
) -> Image.Image:
    padded, original_size = _pad_to_multiple(image, int(pipe.vae_scale_factor))
    padded_auxiliary: InferenceAuxiliary | None = None
    if tri_runtime is not None:
        if auxiliary is None:
            raise ValueError("Tri-input checkpoint requires raw multispectral and unmixing inputs")
        padded_auxiliary = _pad_auxiliary(auxiliary, padded.height, padded.width)
    elif auxiliary is not None:
        raise ValueError("Auxiliary input was supplied while tri-input inference is disabled")
    adapted = _adapt_image(adapter, padded, device, dtype)
    call: dict[str, Any] = {
        "prompt": prompt,
        "image": adapted,
        "noise_level": args.noise_level,
        "guidance_scale": args.guidance_scale,
        "num_inference_steps": args.num_inference_steps,
        "generator": generator,
    }
    if args.guidance_scale > 1.0:
        call["negative_prompt"] = NEGATIVE_PROMPT
    if tri_runtime is not None and padded_auxiliary is not None:
        call.update(
            _tri_pipeline_kwargs(
                pipe,
                tri_runtime,
                padded,
                padded_auxiliary,
                device=device,
                dtype=dtype,
                guidance_scale=float(args.guidance_scale),
            )
        )
    result = pipe(**call).images[0]
    padded_expected = (padded.width * 4, padded.height * 4)
    if result.size != padded_expected:
        raise AssertionError(
            f"Upscaler returned {result.size}, expected padded LR size {padded.size} x4 = {padded_expected}"
        )
    expected = (original_size[0] * 4, original_size[1] * 4)
    if result.size != expected:
        result = result.crop((0, 0, expected[0], expected[1]))
    return result


def tiled_inference(
    pipe: Any,
    adapter: ConditionAdapter,
    image: Image.Image,
    prompt: str,
    args: argparse.Namespace,
    device: torch.device,
    dtype: torch.dtype,
    image_seed: int,
    tri_runtime: TriInferenceRuntime | None = None,
    auxiliary: InferenceAuxiliary | None = None,
) -> Image.Image:
    """Infer overlapping LR tiles and blend their x4 outputs with a 2-D Hann window."""
    scale = 4
    if image.width < args.tile_size or image.height < args.tile_size:
        LOGGER.info(
            "Image %s is smaller than tile_size=%d in at least one dimension; "
            "using explicit whole-image inference without resizing",
            image.size,
            args.tile_size,
        )
        generator = torch.Generator(device=device).manual_seed(image_seed)
        return _run_pipeline(
            pipe,
            adapter,
            image,
            prompt,
            args,
            generator,
            device,
            dtype,
            tri_runtime,
            auxiliary,
        )
    xs = _positions(image.width, args.tile_size, args.tile_overlap)
    ys = _positions(image.height, args.tile_size, args.tile_overlap)
    hr_tile = args.tile_size * scale
    one_d = torch.hann_window(hr_tile, periodic=False, dtype=torch.float32).clamp_min(1e-3)
    weight = torch.outer(one_d, one_d).unsqueeze(-1)
    accumulation = torch.zeros((image.height * scale, image.width * scale, 3), dtype=torch.float32)
    weights = torch.zeros((image.height * scale, image.width * scale, 1), dtype=torch.float32)
    tile_index = 0
    for top in ys:
        for left in xs:
            tile = image.crop((left, top, left + args.tile_size, top + args.tile_size))
            auxiliary_tile = (
                _crop_auxiliary(auxiliary, left, top, args.tile_size, args.tile_size)
                if auxiliary is not None
                else None
            )
            generator = torch.Generator(device=device).manual_seed(image_seed + tile_index)
            sr_tile = _run_pipeline(
                pipe,
                adapter,
                tile,
                prompt,
                args,
                generator,
                device,
                dtype,
                tri_runtime,
                auxiliary_tile,
            )
            sr_tensor = pil_to_tensor(sr_tile).permute(1, 2, 0)
            hr_left, hr_top = left * scale, top * scale
            accumulation[hr_top : hr_top + hr_tile, hr_left : hr_left + hr_tile] += sr_tensor * weight
            weights[hr_top : hr_top + hr_tile, hr_left : hr_left + hr_tile] += weight
            tile_index += 1
    blended = (accumulation / weights.clamp_min(1e-8)).clamp(0, 1).permute(2, 0, 1)
    return tensor_to_pil(blended)


def _artifact_training_config(artifact_path: Path | None) -> dict[str, Any]:
    if artifact_path is None:
        return {}
    path = artifact_path / "training_config.yaml"
    return load_yaml_config(path) if path.is_file() else {}


def _declares_tri_input(
    artifact_path: Path | None,
    cli_config: dict[str, Any],
    artifact_config: dict[str, Any],
) -> bool:
    for source in (artifact_config, cli_config):
        tri = source.get("tri_input")
        if isinstance(tri, dict) and bool(tri.get("enabled", False)):
            return True
    if artifact_path is None:
        return False
    info_path = artifact_path / "model_info.json"
    if info_path.is_file():
        payload = json.loads(info_path.read_text(encoding="utf-8"))
        if bool(payload.get("tri_input_enabled", False)):
            return True
    return (artifact_path / TRI_INPUT_CONFIG_NAME).is_file()


def _load_tri_runtime(
    artifact_path: Path,
    pipe: Any,
    *,
    raw_stats_override: Path | None,
    cli_config: dict[str, Any],
    artifact_config: dict[str, Any],
    device: torch.device,
) -> TriInferenceRuntime:
    required = (
        artifact_path / TRI_INPUT_WEIGHTS_NAME,
        artifact_path / TRI_INPUT_CONFIG_NAME,
        artifact_path / "tri_input_bridges.safetensors",
        artifact_path / "tri_input_bridge_config.json",
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Checkpoint declares tri-input conditioning but required weights/configs are missing: "
            + ", ".join(missing)
        )
    conditioner = TriInputConditioner.from_pretrained(artifact_path, device=device).eval()
    bridge = TriInputBridge.from_pretrained(
        artifact_path, unet=pipe.unet, device=device
    ).eval()
    metadata = conditioner.artifact_metadata
    source_tri: dict[str, Any] = {}
    for source in (artifact_config, cli_config):
        candidate = source.get("tri_input")
        if isinstance(candidate, dict):
            source_tri.update(candidate)
    band_names = metadata.get("band_names") or source_tri.get("band_names")
    conversion_config = metadata.get("raw_value_conversion") or source_tri.get(
        "raw_value_conversion"
    )
    layout = validate_sentinel2_l2a_band_names(band_names)
    conversion = RawValueConversion.from_config(conversion_config)
    stats_path = (
        raw_stats_override.expanduser().resolve()
        if raw_stats_override is not None
        else artifact_path / "raw_band_stats.json"
    )
    stats = load_raw_band_stats(
        stats_path,
        expected_band_names=layout.band_names,
        expected_conversion=conversion,
    )
    if not torch.equal(
        conditioner.raw_mean.detach().cpu().flatten(),
        torch.tensor(stats.mean, dtype=torch.float32),
    ) or not torch.equal(
        conditioner.raw_std.detach().cpu().flatten(),
        torch.tensor(stats.std, dtype=torch.float32),
    ):
        raise ValueError(
            f"Raw statistics {stats_path} do not match the conditioner checkpoint buffers"
        )
    pipe.unet = TriConditionedUNet(pipe.unet, bridge)
    LOGGER.info(
        "Loaded tri-input conditioner/bridge from %s with train-only stats %s",
        artifact_path,
        stats_path,
    )
    return TriInferenceRuntime(
        conditioner=conditioner,
        bridge=bridge,
        band_names=layout.band_names,
        conversion=conversion,
        stats=stats,
    )


def _resolve_auxiliary_pairs(
    args: argparse.Namespace,
    files: list[Path],
    cli_config: dict[str, Any],
    artifact_config: dict[str, Any],
) -> dict[str, AuxiliaryPair]:
    if args.split is None:
        raise ValueError(
            "Tri-input inference requires explicit --split train|val|test; the split is never "
            "guessed from input or checkpoint names"
        )
    split_config: dict[str, Any] = {}
    for source in (artifact_config, cli_config):
        tri = source.get("tri_input")
        data = tri.get("data") if isinstance(tri, dict) else None
        candidate = data.get(args.split) if isinstance(data, dict) else None
        if isinstance(candidate, dict):
            split_config.update(candidate)
    raw_dir = args.raw_ms_dir or split_config.get("raw_ms_dir")
    prior_dir = args.unmixing_dir or split_config.get("unmixing_dir")
    if raw_dir is None or prior_dir is None:
        raise ValueError(
            f"Tri-input split={args.split!r} requires explicit --raw_ms_dir and "
            "--unmixing_dir (or non-null paths for that exact split in YAML); no path is guessed"
        )
    manifest = args.aux_manifest_path or split_config.get("manifest_path")
    recursive = bool(args.aux_recursive or split_config.get("recursive", False))
    pairs = build_auxiliary_pairs(
        [path.stem for path in files],
        raw_dir,
        prior_dir,
        manifest_path=manifest,
        recursive=recursive,
    )
    return {pair.sample_id: pair for pair in pairs}


def main() -> None:
    args = parse_args()
    configure_logging()
    config = load_yaml_config(args.config) if args.config else {}
    model_id = args.model_id or config.get("model_id", "stabilityai/stable-diffusion-x4-upscaler")
    data_root = Path(config.get("data_root", "."))
    if args.input_dir is None:
        raise ValueError("--input_dir is required (for example DATA_ROOT/test/LR)")
    input_dir = args.input_dir.expanduser().resolve()
    if not input_dir.is_dir():
        raise FileNotFoundError(f"Input LR directory does not exist: {input_dir}")
    gt_dir = args.gt_dir.expanduser().resolve() if args.gt_dir else None
    output_dir = args.output_dir.expanduser().resolve()
    output_dirs = {name: output_dir / name for name in ("sr_raw", "sr_projected", "lr_bicubic", "gt", "previews")}
    for directory in output_dirs.values():
        directory.mkdir(parents=True, exist_ok=True)

    if not 0.0 <= args.low_freq_projection_alpha <= 1.0:
        raise ValueError("--low_freq_projection_alpha must be in [0,1]")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    precision = args.mixed_precision or config.get("mixed_precision", "fp16" if device.type == "cuda" else "no")
    dtype = torch.float16 if precision == "fp16" else torch.bfloat16 if precision == "bf16" else torch.float32
    if device.type == "cpu" and dtype != torch.float32:
        LOGGER.warning("CPU inference does not reliably support %s; using float32", dtype)
        dtype = torch.float32

    require_diffusers_version()
    from diffusers import StableDiffusionUpscalePipeline

    pipe = StableDiffusionUpscalePipeline.from_pretrained(str(model_id), torch_dtype=dtype, safety_checker=None)
    tokenizer_max_length = normalize_tokenizer_max_length(pipe.tokenizer, pipe.text_encoder)
    LOGGER.info("Using tokenizer max length %d from the text encoder configuration", tokenizer_max_length)
    pipe.to(device)
    pipe.set_progress_bar_config(disable=False)
    if args.artifact_path is not None and (args.lora_path is not None or args.adapter_path is not None):
        raise ValueError("Use either --checkpoint_path/--artifact_path or the separate --lora_path/--adapter_path options")
    artifact_path = (
        resolve_project_path(args.artifact_path, PROJECT_ROOT) if args.artifact_path is not None else None
    )
    if artifact_path is not None and not artifact_path.is_dir():
        raise FileNotFoundError(f"Checkpoint/artifact directory does not exist: {artifact_path}")
    lora_path = artifact_path or args.lora_path
    if lora_path is not None:
        lora_path = resolve_project_path(lora_path, PROJECT_ROOT)
        expected_lora = lora_path / "pytorch_lora_weights.safetensors" if lora_path.is_dir() else lora_path
        if not expected_lora.is_file():
            raise FileNotFoundError(f"LoRA weights not found: {expected_lora}")
        pipe.load_lora_weights(str(lora_path if lora_path.is_dir() else lora_path.parent))
        LOGGER.info("Loaded LoRA weights from %s", expected_lora)
    adapter_path = artifact_path or args.adapter_path
    if adapter_path is not None:
        adapter_path = resolve_project_path(adapter_path, PROJECT_ROOT)
        adapter = ConditionAdapter.from_pretrained(
            adapter_path, adapter_scale=float(config.get("adapter_scale", 1.0)), device=device
        ).to(dtype=dtype).eval()
        LOGGER.info("Loaded ConditionAdapter from %s", adapter_path)
    else:
        LOGGER.warning("No --adapter_path supplied; using zero-initialized identity ConditionAdapter")
        adapter = ConditionAdapter(adapter_scale=float(config.get("adapter_scale", 1.0))).to(device=device, dtype=dtype).eval()

    artifact_config = _artifact_training_config(artifact_path)
    tri_declared = _declares_tri_input(artifact_path, config, artifact_config)
    tri_runtime: TriInferenceRuntime | None = None
    if tri_declared and not args.disable_tri_input:
        if artifact_path is None:
            raise ValueError(
                "The configuration enables tri-input inference, but no --checkpoint_path artifact "
                "was supplied for strict conditioner/bridge loading"
            )
        tri_runtime = _load_tri_runtime(
            artifact_path,
            pipe,
            raw_stats_override=args.raw_stats_path,
            cli_config=config,
            artifact_config=artifact_config,
            device=device,
        )
    elif tri_declared:
        LOGGER.warning(
            "Tri-input artifact explicitly disabled by --disable_tri_input; running the RGB-only ablation"
        )
    elif any(
        value is not None
        for value in (args.raw_ms_dir, args.unmixing_dir, args.aux_manifest_path, args.raw_stats_path)
    ):
        raise ValueError(
            "Auxiliary paths were supplied, but the checkpoint does not declare tri-input weights"
        )

    metadata_path = args.metadata_path
    if metadata_path is None:
        candidate = data_root / "metadata_final" / "final_metadata.csv"
        metadata_path = candidate if candidate.is_file() else None
    prompt_for = _prompt_lookup(args.prompt_mode, metadata_path)
    files = _select_input_files(input_dir, args.sample, args.sample_file, args.start_index, args.limit)
    if not files:
        raise RuntimeError(f"No supported images remain after sample selection under {input_dir}")
    LOGGER.info(
        "Selected %d inference samples from %s%s",
        len(files),
        input_dir,
        f" using checkpoint {artifact_path}" if artifact_path is not None else "",
    )
    auxiliary_pairs = (
        _resolve_auxiliary_pairs(args, files, config, artifact_config)
        if tri_runtime is not None
        else {}
    )

    for image_index, path in enumerate(tqdm(files, desc="inference")):
        with Image.open(path) as opened:
            lr_image = opened.convert("RGB")
        prompt = prompt_for(path.stem)
        image_seed = args.seed + image_index * 100_000
        auxiliary = None
        if tri_runtime is not None:
            pair = auxiliary_pairs.get(path.stem)
            if pair is None:
                raise FileNotFoundError(
                    f"No exact auxiliary mapping for inference sample {path.stem!r}: RGB={path}"
                )
            auxiliary = _load_inference_auxiliary(pair, tri_runtime, lr_image.size)
        if args.tiled:
            sr = tiled_inference(
                pipe,
                adapter,
                lr_image,
                prompt,
                args,
                device,
                dtype,
                image_seed,
                tri_runtime,
                auxiliary,
            )
        else:
            generator = torch.Generator(device=device).manual_seed(image_seed)
            sr = _run_pipeline(
                pipe,
                adapter,
                lr_image,
                prompt,
                args,
                generator,
                device,
                dtype,
                tri_runtime,
                auxiliary,
            )
        expected = (lr_image.width * 4, lr_image.height * 4)
        if sr.size != expected:
            raise AssertionError(f"Output size check failed for {path.name}: SR={sr.size}, expected={expected}")
        sr.save(output_dirs["sr_raw"] / path.name)
        lr_bicubic = lr_image.resize(expected, Image.Resampling.BICUBIC)
        lr_bicubic.save(output_dirs["lr_bicubic"] / path.name)
        projected_tensor = low_frequency_projection(
            pil_to_tensor(sr), pil_to_tensor(lr_image), args.low_freq_projection_alpha, scale=4
        )
        projected = tensor_to_pil(projected_tensor)
        projected.save(output_dirs["sr_projected"] / path.name)
        gt_image: Image.Image | None = None
        if gt_dir is not None:
            gt_path = gt_dir / path.name
            if not gt_path.is_file():
                raise FileNotFoundError(f"Matching GT is missing for inference image {path.name}: {gt_path}")
            with Image.open(gt_path) as opened_gt:
                gt_image = opened_gt.convert("RGB")
            if gt_image.size != expected:
                raise ValueError(f"GT size mismatch for {path.name}: GT={gt_image.size}, expected={expected}")
            gt_image.save(output_dirs["gt"] / path.name)
        panels = [lr_bicubic, sr, projected] + ([gt_image] if gt_image is not None else [])
        preview = Image.new("RGB", (expected[0] * len(panels), expected[1]))
        for panel_index, panel in enumerate(panels):
            assert panel is not None
            preview.paste(panel, (panel_index * expected[0], 0))
        preview.save(output_dirs["previews"] / f"{path.stem}_lr_raw_projected_gt.png")
    LOGGER.info("Saved %d inference results under %s", len(files), output_dir)


if __name__ == "__main__":
    main()
