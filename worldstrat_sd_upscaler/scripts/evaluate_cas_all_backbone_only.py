#!/usr/bin/env python3
"""Evaluate a Cas-All checkpoint with its latent Cas branch disabled.

The script loads the Cas-All UNet LoRA and ConditionAdapter, but never loads or
calls LatentPhi. It runs inference and metrics together for every synthetic and
real test pair. Edit only the fixed constants below when paths change.
"""

from __future__ import annotations

import json
import logging
import platform
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm.auto import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.condition_adapter import ConditionAdapter
from src.dataset import validate_pair_directories
from src.metrics import (
    OptionalLPIPS,
    basic_metrics,
    bicubic_downsample,
    psnr,
    spectral_angle_deg,
)
from src.utils import (
    FIXED_PROMPT,
    configure_logging,
    normalize_tokenizer_max_length,
    pil_to_tensor,
    tensor_to_pil,
)


# ---------------------------------------------------------------------------
# Fixed ablation settings. This standalone script intentionally has no CLI.
# ---------------------------------------------------------------------------
MODEL_ID = Path("/data/zhengay/models/stabilityaistable-diffusion-x4-upscaler")
CAS_ALL_CHECKPOINT = PROJECT_ROOT / "outputs" / "stage1_synthetic_cas_all" / "final"
DATA_ROOT = Path("/data/zhengay/EDiffSR-main/data/EDiffSR_worldstrat_rgb_x4_per_image")
OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "cas_all_backbone_only_benchmark"
TEST_SPLIT = "test"
GT_SUBDIR = "GT_geo_rad_visual"
CONDITIONS = {
    "synthetic": "LR_bicubic",
    "real": "LR",
}
SCALE = 4
ADAPTER_SCALE = 1.0
SEED = 42
NUM_INFERENCE_STEPS = 40
NOISE_LEVEL = 10
GUIDANCE_SCALE = 1.0
MIXED_PRECISION = "fp16"
ENABLE_LPIPS = True
SAVE_PREVIEWS = True


LOGGER = logging.getLogger("evaluate_cas_all_backbone_only")


def summarize(frame: pd.DataFrame) -> dict[str, dict[str, float | int]]:
    result: dict[str, dict[str, float | int]] = {}
    for column in frame.select_dtypes(include=[np.number]).columns:
        if column == "seed":
            continue
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


def pad_to_multiple(image: Image.Image, multiple: int) -> tuple[Image.Image, tuple[int, int]]:
    original_size = image.size
    pad_width = (-image.width) % multiple
    pad_height = (-image.height) % multiple
    if pad_width == 0 and pad_height == 0:
        return image, original_size
    array = np.asarray(image.convert("RGB"))
    padded = np.pad(array, ((0, pad_height), (0, pad_width), (0, 0)), mode="edge")
    return Image.fromarray(padded, mode="RGB"), original_size


@torch.no_grad()
def run_backbone_only_pipeline(
    pipe: Any,
    adapter: ConditionAdapter,
    lr_image: Image.Image,
    generator: torch.Generator,
    device: torch.device,
    dtype: torch.dtype,
) -> Image.Image:
    """Run trained theta/adapter through the original pipeline, without phi/r_t."""
    padded, original_size = pad_to_multiple(lr_image, int(pipe.vae_scale_factor))
    lr_tensor = pil_to_tensor(padded, "minus_one_one").unsqueeze(0).to(
        device=device, dtype=dtype
    )
    adapted = adapter(lr_tensor)
    adapted_image = tensor_to_pil(adapted[0].float().cpu())
    result = pipe(
        prompt=FIXED_PROMPT,
        image=adapted_image,
        noise_level=NOISE_LEVEL,
        guidance_scale=GUIDANCE_SCALE,
        num_inference_steps=NUM_INFERENCE_STEPS,
        generator=generator,
    ).images[0]
    padded_expected = (padded.width * SCALE, padded.height * SCALE)
    if result.size != padded_expected:
        raise AssertionError(
            f"Backbone-only pipeline returned {result.size}, expected {padded_expected}"
        )
    expected = (original_size[0] * SCALE, original_size[1] * SCALE)
    return result if result.size == expected else result.crop((0, 0, *expected))


def add_lr_consistency_metrics(
    metrics: dict[str, float], sr: torch.Tensor, lr: torch.Tensor
) -> None:
    low_sr = bicubic_downsample(sr, (lr.shape[-2], lr.shape[-1]))
    difference = low_sr - lr
    metrics.update(
        {
            "low_frequency_mae": float(difference.abs().mean().item()),
            "low_frequency_psnr": psnr(low_sr, lr),
            "low_frequency_spectral_angle": spectral_angle_deg(low_sr, lr),
        }
    )


def evaluate_condition(
    pipe: Any,
    adapter: ConditionAdapter,
    condition_name: str,
    lr_subdir: str,
    device: torch.device,
    dtype: torch.dtype,
    lpips_metric: OptionalLPIPS | None,
) -> tuple[pd.DataFrame, dict[str, dict[str, float | int]]]:
    condition_dir = OUTPUT_ROOT / condition_name
    sr_dir = condition_dir / "sr_raw"
    preview_dir = condition_dir / "previews"
    sr_dir.mkdir(parents=True, exist_ok=True)
    if SAVE_PREVIEWS:
        preview_dir.mkdir(parents=True, exist_ok=True)

    records, invalid = validate_pair_directories(
        gt_dir=DATA_ROOT / TEST_SPLIT / GT_SUBDIR,
        lr_dir=DATA_ROOT / TEST_SPLIT / lr_subdir,
        scale=SCALE,
        strict_pairs=False,
        invalid_log_path=condition_dir / "invalid_pairs.csv",
    )
    LOGGER.info(
        "%s: evaluating all %d valid pairs, rejected=%d, LR=%s",
        condition_name,
        len(records),
        len(invalid),
        lr_subdir,
    )

    rows: list[dict[str, float | str | int]] = []
    for index, record in enumerate(
        tqdm(records, desc=f"Cas-All backbone-only / {condition_name}")
    ):
        with Image.open(record.lr_path) as opened_lr:
            lr_image = opened_lr.convert("RGB")
        with Image.open(record.gt_path) as opened_gt:
            gt_image = opened_gt.convert("RGB")

        # Match infer_stage1_latent_phi.py so full-Cas and backbone-only runs use
        # the same diffusion noise for a given position in the sorted test set.
        image_seed = SEED + index
        generator = torch.Generator(device=device).manual_seed(image_seed)
        started = time.perf_counter()
        sr_image = run_backbone_only_pipeline(
            pipe, adapter, lr_image, generator, device, dtype
        )
        elapsed = time.perf_counter() - started
        if sr_image.size != gt_image.size:
            raise ValueError(
                f"Output/GT mismatch for {record.filename}: SR={sr_image.size}, GT={gt_image.size}"
            )
        sr_image.save(sr_dir / record.filename)

        sr_tensor = pil_to_tensor(sr_image)
        gt_tensor = pil_to_tensor(gt_image)
        lr_tensor = pil_to_tensor(lr_image)
        metrics = basic_metrics(sr_tensor, gt_tensor)
        metrics["lpips"] = (
            lpips_metric(sr_tensor, gt_tensor) if lpips_metric is not None else float("nan")
        )
        add_lr_consistency_metrics(metrics, sr_tensor, lr_tensor)
        rows.append(
            {
                "filename": record.filename,
                "sample_id": record.sample_id,
                "condition": condition_name,
                "seed": image_seed,
                "inference_seconds": elapsed,
                **metrics,
            }
        )

        if SAVE_PREVIEWS:
            lr_up = lr_image.resize(gt_image.size, Image.Resampling.BICUBIC)
            preview = Image.new("RGB", (gt_image.width * 3, gt_image.height))
            preview.paste(lr_up, (0, 0))
            preview.paste(sr_image, (gt_image.width, 0))
            preview.paste(gt_image, (2 * gt_image.width, 0))
            preview.save(preview_dir / f"{Path(record.filename).stem}_lr_sr_gt.png")

    frame = pd.DataFrame(rows)
    frame.to_csv(condition_dir / "per_image_metrics.csv", index=False)
    summary = summarize(frame)
    (condition_dir / "summary_metrics.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    pd.DataFrame(
        [{"metric": metric, **statistics} for metric, statistics in summary.items()]
    ).to_csv(condition_dir / "summary_metrics.csv", index=False)
    return frame, summary


def main() -> None:
    configure_logging()
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(OUTPUT_ROOT / "run.log", encoding="utf-8")
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    )
    logging.getLogger().addHandler(file_handler)

    required = {
        "base model": MODEL_ID,
        "data root": DATA_ROOT,
        "Cas-All checkpoint": CAS_ALL_CHECKPOINT,
        "LoRA weights": CAS_ALL_CHECKPOINT / "pytorch_lora_weights.safetensors",
        "ConditionAdapter weights": CAS_ALL_CHECKPOINT / "condition_adapter.safetensors",
    }
    directory_labels = {"base model", "data root", "Cas-All checkpoint"}
    for label, path in required.items():
        expected = path.is_dir() if label in directory_labels else path.is_file()
        if not expected:
            raise FileNotFoundError(f"Required {label} does not exist: {path}")

    from src.utils import require_diffusers_version

    require_diffusers_version()
    from diffusers import StableDiffusionUpscalePipeline

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = (
        torch.float16
        if device.type == "cuda" and MIXED_PRECISION == "fp16"
        else torch.bfloat16
        if device.type == "cuda" and MIXED_PRECISION == "bf16"
        else torch.float32
    )
    LOGGER.info("Loading base pipeline: %s", MODEL_ID)
    pipe = StableDiffusionUpscalePipeline.from_pretrained(
        str(MODEL_ID), torch_dtype=dtype, safety_checker=None
    )
    normalize_tokenizer_max_length(pipe.tokenizer, pipe.text_encoder)
    LOGGER.info("Loading Cas-All UNet LoRA from %s", CAS_ALL_CHECKPOINT)
    pipe.load_lora_weights(str(CAS_ALL_CHECKPOINT))
    LOGGER.info("Loading Cas-All ConditionAdapter from %s", CAS_ALL_CHECKPOINT)
    adapter = ConditionAdapter.from_pretrained(
        CAS_ALL_CHECKPOINT, adapter_scale=ADAPTER_SCALE, device=device
    ).to(device=device, dtype=dtype)

    pipe.to(device)
    pipe.set_progress_bar_config(disable=False)
    pipe.unet.eval()
    pipe.vae.eval()
    pipe.text_encoder.eval()
    adapter.eval()
    for module in (pipe.unet, pipe.vae, pipe.text_encoder, adapter):
        module.requires_grad_(False)

    # Deliberately no LatentPhi.from_pretrained() and no custom Cas scheduler loop.
    lpips_metric: OptionalLPIPS | None = None
    lpips_status = "disabled"
    if ENABLE_LPIPS:
        try:
            lpips_metric = OptionalLPIPS(device)
            lpips_status = "enabled"
        except Exception as error:
            lpips_status = f"unavailable: {type(error).__name__}: {error}"
            LOGGER.warning("LPIPS %s", lpips_status)

    run_started = time.perf_counter()
    comparison_rows: list[dict[str, float | str | int]] = []
    counts: dict[str, int] = {}
    for condition_name, lr_subdir in CONDITIONS.items():
        frame, summary = evaluate_condition(
            pipe, adapter, condition_name, lr_subdir, device, dtype, lpips_metric
        )
        counts[condition_name] = len(frame)
        row: dict[str, float | str | int] = {
            "condition": condition_name,
            "lr_subdir": lr_subdir,
            "sample_count": len(frame),
        }
        for metric, statistics in summary.items():
            row[f"{metric}_mean"] = statistics["mean"]
        comparison_rows.append(row)
    pd.DataFrame(comparison_rows).to_csv(OUTPUT_ROOT / "summary_comparison.csv", index=False)

    run_config = {
        "benchmark": "Cas-All trained theta/adapter with latent Cas inference disabled",
        "loads_cas_all_unet_lora": True,
        "loads_cas_all_condition_adapter": True,
        "loads_latent_phi": False,
        "uses_learned_r_t": False,
        "uses_original_pipeline_scheduler_step": True,
        "model_id": str(MODEL_ID),
        "cas_all_checkpoint": str(CAS_ALL_CHECKPOINT),
        "data_root": str(DATA_ROOT),
        "split": TEST_SPLIT,
        "gt_subdir": GT_SUBDIR,
        "conditions": CONDITIONS,
        "sample_counts": counts,
        "seed": SEED,
        "per_sample_seed_rule": "seed + sorted_test_index",
        "num_inference_steps": NUM_INFERENCE_STEPS,
        "noise_level": NOISE_LEVEL,
        "guidance_scale": GUIDANCE_SCALE,
        "adapter_scale": ADAPTER_SCALE,
        "device": str(device),
        "dtype": str(dtype),
        "lpips": lpips_status,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "elapsed_seconds": time.perf_counter() - run_started,
    }
    (OUTPUT_ROOT / "run_config.json").write_text(
        json.dumps(run_config, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    LOGGER.info("Finished Cas-All backbone-only benchmark under %s", OUTPUT_ROOT)


if __name__ == "__main__":
    main()
