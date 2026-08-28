#!/usr/bin/env python
"""Stage-1-only whole-image inference for the fixed latent-phi timestep ablation."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.condition_adapter import ConditionAdapter
from src.dataset import PairedSatelliteDataset
from src.diffusion_prediction import (
    alpha_bar_for_timesteps,
    mix_x0_and_convert_model_output,
    model_output_to_x0,
    normalized_timesteps,
    timestep_range_mask,
    validate_timestep_range,
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
    parser.add_argument("--config", type=Path, default=Path("configs/stage1_synthetic_latent_phi.yaml"))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--phi_path", type=Path, default=None)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--num_samples", type=int, default=5)
    parser.add_argument("--num_inference_steps", type=int, default=None)
    parser.add_argument("--noise_level", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--phi_timestep_range", type=float, nargs=2, default=None, metavar=("MIN", "MAX"))
    parser.add_argument("--phi_weight", type=float, default=None)
    parser.add_argument("--disable_phi", action="store_true")
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
    from diffusers import StableDiffusionUpscalePipeline

    dtype = torch.float16 if device.type == "cuda" and config.get("mixed_precision") == "fp16" else torch.float32
    pipe = StableDiffusionUpscalePipeline.from_pretrained(
        str(config["model_id"]), torch_dtype=dtype, safety_checker=None
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


def build_dataset(config: dict[str, Any], num_samples: int) -> PairedSatelliteDataset:
    return PairedSatelliteDataset(
        data_root=config["data_root"],
        split="val",
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
    phi_weight: float,
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
    extra_step_kwargs = pipe.prepare_extra_step_kwargs(generator, eta=0.0)
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
        step_model_output = base_model_output
        timestep_batch = timestep.reshape(1).long()
        normalized_t = normalized_timesteps(timestep_batch, total_train_timesteps)
        active = bool(timestep_range_mask(normalized_t, phi_range).item())
        if phi_enabled and phi_weight != 0.0 and active:
            if phi is None:
                raise RuntimeError("phi_enabled is true but no LatentPhi was loaded")
            alpha_bar = alpha_bar_for_timesteps(pipe.scheduler, timestep_batch)
            base_z0 = model_output_to_x0(
                base_model_output, latents, alpha_bar, prediction_type
            )
            phi_z0 = phi(base_z0.detach(), normalized_t)
            _, converted = mix_x0_and_convert_model_output(
                base_z0, phi_z0, phi_weight, latents, alpha_bar, prediction_type
            )
            step_model_output = converted.to(dtype=base_model_output.dtype)
            active_steps += 1
        latents = pipe.scheduler.step(
            step_model_output,
            timestep,
            latents,
            **extra_step_kwargs,
            return_dict=False,
        )[0]

    decoded = pipe.vae.decode(
        latents.float() / float(pipe.vae.config.scaling_factor), return_dict=False
    )[0]
    return tensor_to_pil(decoded[0].add(1.0).div(2.0).clamp(0.0, 1.0)), active_steps


def main() -> None:
    args = parse_args()
    configure_logging()
    if args.num_samples <= 0:
        raise ValueError("--num_samples must be positive")
    config_path = resolve_project_path(args.config, PROJECT_ROOT)
    checkpoint = resolve_project_path(args.checkpoint, PROJECT_ROOT)
    phi_path = resolve_project_path(args.phi_path, PROJECT_ROOT) if args.phi_path else None
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    config = load_yaml_config(config_path)
    phi_range = validate_timestep_range(
        args.phi_timestep_range or config.get("phi_infer_timestep_range", [0.0, 1.0]),
        "phi_timestep_range",
    )
    phi_weight = float(args.phi_weight if args.phi_weight is not None else config.get("phi_weight", 1.0))
    if not 0.0 <= phi_weight <= 1.0:
        raise ValueError("--phi_weight must be in [0,1]")
    num_steps = int(args.num_inference_steps or config.get("validation_num_inference_steps", 40))
    noise_level = int(args.noise_level if args.noise_level is not None else config.get("validation_noise_level", 10))
    seed = int(args.seed if args.seed is not None else config.get("seed", 42))
    device = resolve_device(args.device)
    pipe, adapter, phi = load_components(
        config, checkpoint, phi_path, device, load_phi=not args.disable_phi
    )
    dataset = build_dataset(config, args.num_samples)
    records: list[dict[str, Any]] = []

    for index in range(min(args.num_samples, len(dataset))):
        sample = dataset[index]
        output, active_steps = infer_one(
            sample, pipe, adapter, phi, device, seed + index, noise_level, num_steps,
            phi_range, phi_weight, not args.disable_phi,
        )
        gt = tensor_to_pil(sample["gt"])
        lr = tensor_to_pil(sample["lr"]).resize(gt.size, Image.Resampling.BICUBIC)
        if output.size != gt.size:
            raise AssertionError(f"SR={output.size} does not match GT={gt.size}")
        output.save(output_dir / f"{sample['sample_id']}_sr.png")
        preview = Image.new("RGB", (gt.width * 3, gt.height))
        preview.paste(lr, (0, 0))
        preview.paste(output, (gt.width, 0))
        preview.paste(gt, (2 * gt.width, 0))
        preview.save(output_dir / f"{sample['sample_id']}_lr_sr_gt.png")
        pred_array = np.asarray(output, dtype=np.float32) / 255.0
        gt_array = np.asarray(gt, dtype=np.float32) / 255.0
        records.append(
            {
                "sample_id": sample["sample_id"],
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
            "phi_weight": phi_weight,
            "num_inference_steps": num_steps,
            "noise_level": noise_level,
            "seed": seed,
            "samples": records,
        },
        output_dir / "metrics.json",
    )
    LOGGER.info("Finished Stage-1 latent-phi inference: %s", output_dir)


if __name__ == "__main__":
    main()
