#!/usr/bin/env python
"""Stage-1-only inference for the learned-r_t latent Cas-DM ablation."""

from __future__ import annotations

import argparse
import csv
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from tqdm.auto import tqdm

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.condition_adapter import ConditionAdapter
from src.dataset import PairedSatelliteDataset
from src.diffusion_prediction import (
    alpha_bar_for_timesteps,
    mix_reverse_means,
    model_output_to_x0,
    normalized_timesteps,
    timestep_range_mask,
    validate_timestep_range,
    x0_to_model_output,
)
from src.latent_phi import LatentPhi
from src.metrics import psnr, ssim
from src.utils import (
    configure_logging,
    load_yaml_config,
    normalize_tokenizer_max_length,
    resolve_project_path,
    save_json,
    tensor_to_pil,
)

LOGGER = logging.getLogger("infer_stage1_latent_phi")
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def encode_prompts(tokenizer: Any, text_encoder: Any, prompts: list[str], device: torch.device) -> torch.Tensor:
    inputs = tokenizer(
        prompts,
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    with torch.no_grad():
        return text_encoder(inputs.input_ids.to(device), attention_mask=None)[0]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/stage1_synthetic_cas_all.yaml"))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--phi_path", type=Path, default=None)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "val", "test"), default="test")
    parser.add_argument("--data_root", type=Path, default=None)
    parser.add_argument("--lr_subdir", default=None)
    parser.add_argument("--gt_subdir", default=None)
    parser.add_argument(
        "--limit",
        "--num_samples",
        dest="limit",
        type=int,
        default=None,
        help="Optional debug limit. Omit it to process every valid pair in the selected split.",
    )
    parser.add_argument("--num_inference_steps", type=int, default=None)
    parser.add_argument("--noise_level", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--phi_timestep_range", type=float, nargs=2, default=None, metavar=("MIN", "MAX"))
    parser.add_argument("--disable_phi", action="store_true")
    parser.add_argument("--skip_previews", action="store_true")
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    return parser.parse_args()


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    return torch.device(name)


def load_components(
    config: dict[str, Any],
    checkpoint: Path,
    phi_path: Path | None,
    device: torch.device,
    load_phi: bool = True,
) -> tuple[Any, ConditionAdapter, LatentPhi | None]:
    from diffusers import DDIMScheduler, StableDiffusionUpscalePipeline

    dtype = torch.float16 if device.type == "cuda" and config.get("mixed_precision") == "fp16" else torch.float32
    pipe = StableDiffusionUpscalePipeline.from_pretrained(
        str(config["model_id"]), torch_dtype=dtype, safety_checker=None
    )
    if not isinstance(pipe.scheduler, DDIMScheduler):
        raise TypeError(
            "Stage-1 latent Cas-DM inference is validated only for DDIMScheduler "
            f"with eta=0, got {type(pipe.scheduler).__name__}"
        )
    normalize_tokenizer_max_length(pipe.tokenizer, pipe.text_encoder)
    lora_file = checkpoint / "pytorch_lora_weights.safetensors"
    if not lora_file.is_file():
        raise FileNotFoundError(f"LoRA weights missing: {lora_file}")
    pipe.load_lora_weights(str(checkpoint))
    adapter = ConditionAdapter.from_pretrained(
        checkpoint, adapter_scale=float(config.get("adapter_scale", 1.0)), device=device
    ).to(device=device, dtype=dtype)
    phi = LatentPhi.from_pretrained(phi_path or checkpoint, device=device) if load_phi else None
    pipe.to(device)
    pipe.vae.to(device=device, dtype=torch.float32)
    pipe.text_encoder.to(device=device, dtype=dtype)
    modules = (pipe.unet, pipe.vae, pipe.text_encoder, adapter) + ((phi,) if phi is not None else ())
    for module in modules:
        module.eval()
        module.requires_grad_(False)
    return pipe, adapter, phi


def ddim_reverse_mean_step(
    scheduler: Any,
    base_model_output: torch.Tensor,
    timestep: torch.Tensor | int,
    sample: torch.Tensor,
    phi_model_output: torch.Tensor | None = None,
    mixing_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    """Take an exact baseline DDIM step or mix two deterministic DDIM means."""
    base_previous = scheduler.step(
        base_model_output, timestep, sample, eta=0.0, return_dict=True
    ).prev_sample
    if phi_model_output is None:
        if mixing_weight is not None:
            raise ValueError("mixing_weight was provided without phi_model_output")
        return base_previous
    if mixing_weight is None:
        raise ValueError("active Cas-DM step requires learned mixing_weight")
    phi_previous = scheduler.step(
        phi_model_output, timestep, sample, eta=0.0, return_dict=True
    ).prev_sample
    return mix_reverse_means(base_previous, phi_previous, mixing_weight)


def build_dataset(
    config: dict[str, Any],
    split: str,
    limit: int | None,
    data_root: Path | None = None,
    lr_subdir: str | None = None,
    gt_subdir: str | None = None,
) -> PairedSatelliteDataset:
    """Build a deterministic evaluation dataset; no limit means the full split."""
    return PairedSatelliteDataset(
        data_root=data_root or config["data_root"],
        split=split,
        lr_subdir=str(lr_subdir or config["val_lr_subdir"]),
        gt_subdir=str(gt_subdir or config.get("gt_subdir", "GT")),
        gt_crop_size=int(config.get("gt_crop_size", 512)),
        scale=int(config.get("scale", 4)),
        training=False,
        strict_pairs=bool(config.get("strict_pairs", False)),
        prompt_mode=str(config.get("prompt_mode", "fixed")),
        metadata_path=config.get("metadata_path"),
        prompt_dropout_probability=0.0,
        augment=False,
        validation_limit=limit,
    )


def create_output_directories(output_dir: Path) -> dict[str, Path]:
    """Create the standard layout consumed by src/evaluate.py."""
    directories = {
        name: output_dir / name for name in ("sr_raw", "gt", "lr_input", "previews")
    }
    for directory in directories.values():
        directory.mkdir(parents=True, exist_ok=True)
    return directories


@torch.no_grad()
def infer_one(
    sample: dict[str, Any],
    pipe: Any,
    adapter: ConditionAdapter,
    phi: LatentPhi | None,
    device: torch.device,
    seed: int,
    noise_level: int,
    num_inference_steps: int,
    phi_range: tuple[float, float],
    phi_enabled: bool,
) -> tuple[Image.Image, int]:
    dtype = next(pipe.unet.parameters()).dtype
    lr = sample["lr"].unsqueeze(0).to(device=device, dtype=dtype)
    adapted = adapter(lr)
    # Match the unchanged production inference boundary exactly.
    adapted_pil = tensor_to_pil(adapted[0].float().cpu())
    condition = pipe.image_processor.preprocess(adapted_pil).to(device=device, dtype=dtype)
    prompt_embeds = encode_prompts(pipe.tokenizer, pipe.text_encoder, [sample["prompt"]], device)
    pipe.scheduler.set_timesteps(num_inference_steps, device=device)
    generator = torch.Generator(device=device).manual_seed(seed)
    level = torch.tensor([noise_level], device=device, dtype=torch.long)
    condition_noise = torch.randn(condition.shape, generator=generator, device=device, dtype=dtype)
    noisy_condition = pipe.low_res_scheduler.add_noise(condition, condition_noise, level)
    latents = torch.randn(
        (1, int(pipe.vae.config.latent_channels), *noisy_condition.shape[-2:]),
        generator=generator,
        device=device,
        dtype=dtype,
    )
    latents = latents * pipe.scheduler.init_noise_sigma
    prediction_type = str(pipe.scheduler.config.prediction_type)
    total_train_timesteps = int(pipe.scheduler.config.num_train_timesteps)
    active_steps = 0

    for timestep in pipe.scheduler.timesteps:
        latent_model_input = pipe.scheduler.scale_model_input(latents, timestep)
        model_input = torch.cat((latent_model_input, noisy_condition), dim=1)
        base_model_output = pipe.unet(
            model_input,
            timestep,
            encoder_hidden_states=prompt_embeds,
            class_labels=level,
            return_dict=False,
        )[0]
        timestep_batch = timestep.reshape(1).long()
        normalized_t = normalized_timesteps(timestep_batch, total_train_timesteps)
        active = bool(timestep_range_mask(normalized_t, phi_range).item())
        if phi_enabled and active:
            if phi is None:
                raise RuntimeError("phi_enabled is true but no LatentPhi was loaded")
            alpha_bar = alpha_bar_for_timesteps(pipe.scheduler, timestep_batch)
            base_z0 = model_output_to_x0(
                base_model_output, latents, alpha_bar, prediction_type
            )
            phi_output = phi(base_z0.detach(), normalized_t)
            phi_model_output = x0_to_model_output(
                phi_output.predicted_z0,
                latents,
                alpha_bar,
                prediction_type,
            )
            latents = ddim_reverse_mean_step(
                pipe.scheduler,
                base_model_output,
                timestep,
                latents,
                phi_model_output=phi_model_output.to(dtype=base_model_output.dtype),
                mixing_weight=phi_output.mixing_weight,
            ).to(dtype=dtype)
            active_steps += 1
        else:
            # Strict baseline path: no z0 conversion and no phi/r_t construction.
            latents = ddim_reverse_mean_step(
                pipe.scheduler, base_model_output, timestep, latents
            )

    decoded = pipe.vae.decode(
        latents.float() / float(pipe.vae.config.scaling_factor), return_dict=False
    )[0]
    return tensor_to_pil(decoded[0].add(1.0).div(2.0).clamp(0.0, 1.0)), active_steps


def main() -> None:
    args = parse_args()
    configure_logging()
    if args.limit is not None and args.limit <= 0:
        raise ValueError("--limit/--num_samples must be positive when provided")
    config_path = resolve_project_path(args.config, PROJECT_ROOT)
    checkpoint = resolve_project_path(args.checkpoint, PROJECT_ROOT)
    phi_path = resolve_project_path(args.phi_path, PROJECT_ROOT) if args.phi_path else None
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_dirs = create_output_directories(output_dir)
    config = load_yaml_config(config_path)
    phi_range = validate_timestep_range(
        args.phi_timestep_range or config.get("phi_infer_timestep_range", [0.0, 1.0]),
        "phi_timestep_range",
    )
    num_steps = int(args.num_inference_steps or config.get("validation_num_inference_steps", 40))
    noise_level = int(args.noise_level if args.noise_level is not None else config.get("validation_noise_level", 10))
    seed = int(args.seed if args.seed is not None else config.get("seed", 42))
    device = resolve_device(args.device)
    pipe, adapter, phi = load_components(
        config, checkpoint, phi_path, device, load_phi=not args.disable_phi
    )
    data_root = args.data_root.expanduser().resolve() if args.data_root else None
    dataset = build_dataset(
        config,
        split=args.split,
        limit=args.limit,
        data_root=data_root,
        lr_subdir=args.lr_subdir,
        gt_subdir=args.gt_subdir,
    )
    records: list[dict[str, Any]] = []
    LOGGER.info(
        "Evaluating all %d valid pairs from split=%s, LR=%s, GT=%s (rejected=%d)",
        len(dataset),
        args.split,
        args.lr_subdir or config["val_lr_subdir"],
        args.gt_subdir or config.get("gt_subdir", "GT"),
        len(dataset.invalid_pairs),
    )

    for index in tqdm(range(len(dataset)), desc=f"Cas-DM {args.split} inference"):
        sample = dataset[index]
        output, active_steps = infer_one(
            sample, pipe, adapter, phi, device, seed + index, noise_level, num_steps,
            phi_range, not args.disable_phi,
        )
        gt = tensor_to_pil(sample["gt"])
        lr_input = tensor_to_pil(sample["lr"])
        lr_bicubic = lr_input.resize(gt.size, Image.Resampling.BICUBIC)
        if output.size != gt.size:
            raise AssertionError(f"SR={output.size} does not match GT={gt.size}")
        filename = str(sample["filename"])
        output.save(output_dirs["sr_raw"] / filename)
        gt.save(output_dirs["gt"] / filename)
        lr_input.save(output_dirs["lr_input"] / filename)
        if not args.skip_previews:
            preview = Image.new("RGB", (gt.width * 3, gt.height))
            preview.paste(lr_bicubic, (0, 0))
            preview.paste(output, (gt.width, 0))
            preview.paste(gt, (2 * gt.width, 0))
            preview.save(
                output_dirs["previews"] / f"{Path(filename).stem}_lr_sr_gt.png"
            )
        pred_array = np.asarray(output, dtype=np.float32) / 255.0
        gt_array = np.asarray(gt, dtype=np.float32) / 255.0
        records.append(
            {
                "sample_id": sample["sample_id"],
                "filename": filename,
                "psnr": psnr(pred_array, gt_array),
                "ssim": ssim(pred_array, gt_array),
                "phi_active_steps": active_steps,
            }
        )
    save_json(
        {
            "checkpoint": str(checkpoint),
            "phi_path": str(phi_path or checkpoint),
            "phi_enabled": not args.disable_phi,
            "phi_timestep_range": list(phi_range),
            "num_inference_steps": num_steps,
            "noise_level": noise_level,
            "seed": seed,
            "split": args.split,
            "data_root": str(data_root or config["data_root"]),
            "lr_subdir": str(args.lr_subdir or config["val_lr_subdir"]),
            "gt_subdir": str(args.gt_subdir or config.get("gt_subdir", "GT")),
            "requested_limit": args.limit,
            "evaluated_count": len(records),
            "invalid_pair_count": len(dataset.invalid_pairs),
            "mean_psnr": float(np.mean([record["psnr"] for record in records])),
            "mean_ssim": float(np.mean([record["ssim"] for record in records])),
            "samples": records,
        },
        output_dir / "metrics.json",
    )
    with (output_dir / "per_image_metrics.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("sample_id", "filename", "psnr", "ssim", "phi_active_steps"),
        )
        writer.writeheader()
        writer.writerows(records)
    LOGGER.info(
        "Finished full Stage-1 Cas-DM evaluation: %d samples under %s",
        len(records),
        output_dir,
    )


if __name__ == "__main__":
    main()
