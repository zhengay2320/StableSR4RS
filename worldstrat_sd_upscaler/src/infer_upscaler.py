#!/usr/bin/env python
"""Run whole-image or Hann-blended tiled x4 upscaler inference."""

from __future__ import annotations

import argparse
import logging
import sys
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


def _run_pipeline(
    pipe: Any,
    adapter: ConditionAdapter,
    image: Image.Image,
    prompt: str,
    args: argparse.Namespace,
    generator: torch.Generator,
    device: torch.device,
    dtype: torch.dtype,
) -> Image.Image:
    padded, original_size = _pad_to_multiple(image, int(pipe.vae_scale_factor))
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
) -> Image.Image:
    """Infer overlapping LR tiles and blend their x4 outputs with a 2-D Hann window."""
    scale = 4
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
            generator = torch.Generator(device=device).manual_seed(image_seed + tile_index)
            sr_tile = _run_pipeline(pipe, adapter, tile, prompt, args, generator, device, dtype)
            sr_tensor = pil_to_tensor(sr_tile).permute(1, 2, 0)
            hr_left, hr_top = left * scale, top * scale
            accumulation[hr_top : hr_top + hr_tile, hr_left : hr_left + hr_tile] += sr_tensor * weight
            weights[hr_top : hr_top + hr_tile, hr_left : hr_left + hr_tile] += weight
            tile_index += 1
    blended = (accumulation / weights.clamp_min(1e-8)).clamp(0, 1).permute(2, 0, 1)
    return tensor_to_pil(blended)


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

    for image_index, path in enumerate(tqdm(files, desc="inference")):
        with Image.open(path) as opened:
            lr_image = opened.convert("RGB")
        prompt = prompt_for(path.stem)
        image_seed = args.seed + image_index * 100_000
        if args.tiled:
            sr = tiled_inference(pipe, adapter, lr_image, prompt, args, device, dtype, image_seed)
        else:
            generator = torch.Generator(device=device).manual_seed(image_seed)
            sr = _run_pipeline(pipe, adapter, lr_image, prompt, args, generator, device, dtype)
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
