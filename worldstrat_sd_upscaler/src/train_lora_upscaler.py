#!/usr/bin/env python
"""Two-stage LoRA training for StableDiffusionUpscalePipeline."""

from __future__ import annotations

import argparse
import logging
import math
import sys
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration, set_seed
from PIL import Image
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.condition_adapter import ConditionAdapter
from src.dataset import PairedSatelliteDataset
from src.metrics import psnr, ssim
from src.utils import (
    FIXED_PROMPT,
    atomic_torch_save,
    configure_logging,
    enforce_checkpoint_limit,
    find_latest_checkpoint,
    load_yaml_config,
    normalize_tokenizer_max_length,
    pil_to_tensor,
    require_config,
    require_diffusers_version,
    resolve_project_path,
    save_json,
    save_yaml,
    tensor_to_pil,
    worker_init_fn,
)

LOGGER = logging.getLogger("train")
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--max_train_steps", type=int, default=None, help="Override YAML for smoke tests")
    parser.add_argument("--train_batch_size", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--data_root", type=Path, default=None)
    parser.add_argument("--output_dir", type=Path, default=None)
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)
    return parser.parse_args()


def apply_cli_overrides(config: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    """Apply the small set of explicit runtime overrides."""
    result = dict(config)
    for key in (
        "max_train_steps",
        "train_batch_size",
        "num_workers",
        "data_root",
        "output_dir",
        "resume_from_checkpoint",
    ):
        value = getattr(args, key)
        if value is not None:
            result[key] = str(value) if isinstance(value, Path) else value
    return result


def compute_snr(scheduler: Any, timesteps: torch.Tensor) -> torch.Tensor:
    """Compute scheduler signal-to-noise ratios for sampled timesteps."""
    alphas = scheduler.alphas_cumprod.to(device=timesteps.device, dtype=torch.float32)
    alpha = alphas[timesteps]
    return alpha / (1.0 - alpha).clamp_min(1e-12)


def snr_weighted_mse(
    prediction: torch.Tensor,
    target: torch.Tensor,
    timesteps: torch.Tensor,
    scheduler: Any,
    snr_gamma: float | None,
) -> torch.Tensor:
    """Apply Min-SNR weighting for epsilon or velocity prediction."""
    per_sample = F.mse_loss(prediction.float(), target.float(), reduction="none")
    per_sample = per_sample.mean(dim=tuple(range(1, per_sample.ndim)))
    if snr_gamma is None or snr_gamma <= 0:
        return per_sample.mean()
    snr = compute_snr(scheduler, timesteps)
    gamma = torch.full_like(snr, float(snr_gamma))
    prediction_type = scheduler.config.prediction_type
    denominator = snr + 1.0 if prediction_type == "v_prediction" else snr
    weights = torch.minimum(snr, gamma) / denominator.clamp_min(1e-12)
    return (per_sample * weights).mean()


def get_trainable_parameters(module: nn.Module) -> list[nn.Parameter]:
    """Return trainable parameters and fail if adapter setup was ineffective."""
    parameters = [parameter for parameter in module.parameters() if parameter.requires_grad]
    if not parameters:
        raise RuntimeError(f"No trainable parameters found in {type(module).__name__}")
    return parameters


def resolve_resume_path(config: dict[str, Any], output_dir: Path) -> Path | None:
    """Resolve explicit checkpoint, `latest`, or no resume."""
    value = config.get("resume_from_checkpoint")
    if not value:
        return None
    if str(value).lower() == "latest":
        latest = find_latest_checkpoint(output_dir)
        if latest is None:
            raise FileNotFoundError(f"No checkpoint-* directories exist in {output_dir}")
        return latest
    path = resolve_project_path(str(value), PROJECT_ROOT)
    if not path.is_dir():
        raise FileNotFoundError(f"Resume checkpoint does not exist: {path}")
    return path


def configure_lora(pipe: Any, config: dict[str, Any], initial_path: Path | None) -> nn.Module:
    """Create or load the official PEFT LoRA adapter on the UNet only."""
    from peft import LoraConfig

    unet = pipe.unet
    for parameter in unet.parameters():
        parameter.requires_grad_(False)
    if initial_path is not None:
        lora_file = initial_path / "pytorch_lora_weights.safetensors" if initial_path.is_dir() else initial_path
        if not lora_file.is_file():
            raise FileNotFoundError(f"Initial LoRA safetensors not found: {lora_file}")
        pipe.load_lora_weights(
            str(initial_path if initial_path.is_dir() else initial_path.parent),
            adapter_name="default",
        )
        for name, parameter in unet.named_parameters():
            parameter.requires_grad_("lora_" in name.lower())
        LOGGER.info("Initialized UNet LoRA from %s", lora_file)
    else:
        lora_config = LoraConfig(
            r=int(config.get("lora_rank", 16)),
            lora_alpha=int(config.get("lora_alpha", 16)),
            lora_dropout=float(config.get("lora_dropout", 0.05)),
            init_lora_weights="gaussian",
            target_modules=["to_q", "to_k", "to_v", "to_out.0"],
        )
        unet.add_adapter(lora_config)
    for parameter in get_trainable_parameters(unet):
        parameter.data = parameter.data.float()
    return unet


def save_artifacts(
    accelerator: Accelerator,
    unet: nn.Module,
    adapter: nn.Module,
    output_path: Path,
    config: dict[str, Any],
    global_step: int,
    optimizer: Optimizer | None = None,
    lr_scheduler: Any | None = None,
) -> None:
    """Save reloadable Diffusers LoRA, adapter, actual config, and trainer state."""
    if not accelerator.is_main_process:
        return
    require_diffusers_version()
    from diffusers import StableDiffusionUpscalePipeline
    from diffusers.utils import convert_state_dict_to_diffusers
    from peft import get_peft_model_state_dict

    output_path.mkdir(parents=True, exist_ok=True)
    unwrapped_unet = accelerator.unwrap_model(unet)
    unwrapped_adapter = accelerator.unwrap_model(adapter)
    peft_state = get_peft_model_state_dict(unwrapped_unet)
    lora_state = convert_state_dict_to_diffusers(peft_state)
    StableDiffusionUpscalePipeline.save_lora_weights(
        save_directory=str(output_path),
        unet_lora_layers=lora_state,
        safe_serialization=True,
    )
    if not isinstance(unwrapped_adapter, ConditionAdapter):
        raise TypeError(f"Unexpected adapter type: {type(unwrapped_adapter)}")
    unwrapped_adapter.save_pretrained(output_path)
    save_yaml(config, output_path / "training_config.yaml")
    save_json(
        {
            "model_id": config["model_id"],
            "pipeline_class": "StableDiffusionUpscalePipeline",
            "diffusers_version": __import__("diffusers").__version__,
            "global_step": global_step,
            "lora_rank": int(config.get("lora_rank", 16)),
            "adapter_scale": float(config.get("adapter_scale", 1.0)),
        },
        output_path / "model_info.json",
    )
    if optimizer is not None and lr_scheduler is not None:
        atomic_torch_save(optimizer.state_dict(), output_path / "optimizer.pt")
        atomic_torch_save(lr_scheduler.state_dict(), output_path / "lr_scheduler.pt")
        atomic_torch_save(
            {"global_step": global_step, "torch_rng_state": torch.get_rng_state()},
            output_path / "trainer_state.pt",
        )
    LOGGER.info("Saved training artifacts to %s", output_path)


def encode_prompts(tokenizer: Any, text_encoder: nn.Module, prompts: list[str], device: torch.device) -> torch.Tensor:
    """Tokenize and encode prompt strings with the frozen text encoder."""
    inputs = tokenizer(
        prompts,
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    with torch.no_grad():
        return text_encoder(inputs.input_ids.to(device), attention_mask=None)[0]


def adapt_pil(adapter: nn.Module, lr_tensor: torch.Tensor, device: torch.device, dtype: torch.dtype) -> Image.Image:
    """Apply ConditionAdapter and return a PIL image suitable for the pipeline."""
    with torch.no_grad():
        adapted = adapter(lr_tensor.unsqueeze(0).to(device=device, dtype=dtype)).float().cpu()[0]
    return tensor_to_pil(adapted)


@torch.no_grad()
def run_validation(
    accelerator: Accelerator,
    pipe: Any,
    unet: nn.Module,
    adapter: nn.Module,
    dataset: PairedSatelliteDataset,
    output_dir: Path,
    global_step: int,
    config: dict[str, Any],
    weight_dtype: torch.dtype,
) -> None:
    """Generate fixed validation examples and record PSNR/SSIM plus triptychs."""
    if not accelerator.is_main_process:
        return
    unwrapped_unet = accelerator.unwrap_model(unet)
    unwrapped_adapter = accelerator.unwrap_model(adapter)
    was_training = unwrapped_unet.training
    unwrapped_unet.eval()
    unwrapped_adapter.eval()
    pipe.unet = unwrapped_unet
    pipe.to(accelerator.device)
    pipe.set_progress_bar_config(disable=True)
    validation_dir = output_dir / "validation" / f"step-{global_step:06d}"
    validation_dir.mkdir(parents=True, exist_ok=True)
    sample_count = min(int(config.get("validation_num_samples", 2)), len(dataset))
    results: list[dict[str, float | str]] = []
    for index in range(sample_count):
        sample = dataset[index]
        lr_pil = adapt_pil(unwrapped_adapter, sample["lr"], accelerator.device, weight_dtype)
        generator = torch.Generator(device=accelerator.device).manual_seed(int(config.get("seed", 42)) + index)
        output = pipe(
            prompt=sample["prompt"],
            image=lr_pil,
            noise_level=int(config.get("validation_noise_level", 10)),
            guidance_scale=float(config.get("validation_guidance_scale", 1.0)),
            num_inference_steps=int(config.get("validation_num_inference_steps", 20)),
            generator=generator,
        ).images[0]
        gt_pil = tensor_to_pil(sample["gt"])
        if output.size != gt_pil.size:
            raise AssertionError(
                f"Validation SR size mismatch for {sample['filename']}: SR={output.size}, GT={gt_pil.size}"
            )
        pred_array = np.asarray(output, dtype=np.float32) / 255.0
        gt_array = np.asarray(gt_pil, dtype=np.float32) / 255.0
        result = {"sample_id": sample["sample_id"], "psnr": psnr(pred_array, gt_array), "ssim": ssim(pred_array, gt_array)}
        results.append(result)
        preview = Image.new("RGB", (gt_pil.width * 3, gt_pil.height))
        preview.paste(lr_pil.resize(gt_pil.size, Image.Resampling.BICUBIC), (0, 0))
        preview.paste(output, (gt_pil.width, 0))
        preview.paste(gt_pil, (gt_pil.width * 2, 0))
        preview.save(validation_dir / f"{sample['sample_id']}_lr_sr_gt.png")
    save_json(
        {
            "global_step": global_step,
            "samples": results,
            "mean_psnr": float(np.mean([float(item["psnr"]) for item in results])),
            "mean_ssim": float(np.mean([float(item["ssim"]) for item in results])),
        },
        validation_dir / "metrics.json",
    )
    if was_training:
        unwrapped_unet.train()
        unwrapped_adapter.train()


def build_datasets(
    config: dict[str, Any], output_dir: Path, write_invalid_logs: bool = True
) -> tuple[PairedSatelliteDataset, PairedSatelliteDataset]:
    """Construct train/validation datasets from every relevant YAML setting."""
    common = {
        "data_root": config["data_root"],
        "gt_subdir": config.get("gt_subdir", "GT"),
        "gt_crop_size": int(config.get("gt_crop_size", 512)),
        "scale": int(config.get("scale", 4)),
        "strict_pairs": bool(config.get("strict_pairs", False)),
        "prompt_mode": str(config.get("prompt_mode", "fixed")),
        "metadata_path": config.get("metadata_path"),
    }
    train_dataset = PairedSatelliteDataset(
        **common,
        split="train",
        lr_subdir=str(config["train_lr_subdir"]),
        training=True,
        invalid_log_path=output_dir / "invalid_train_pairs.csv" if write_invalid_logs else None,
        synthetic_lr_subdir=config.get("synthetic_lr_subdir"),
        synthetic_replay_probability=float(config.get("synthetic_replay_probability", 0.0)),
        prompt_dropout_probability=float(config.get("prompt_dropout_probability", 0.1)),
        augment=bool(config.get("augment", True)),
    )
    validation_dataset = PairedSatelliteDataset(
        **common,
        split="val",
        lr_subdir=str(config["val_lr_subdir"]),
        training=False,
        invalid_log_path=output_dir / "invalid_val_pairs.csv" if write_invalid_logs else None,
        synthetic_replay_probability=0.0,
        prompt_dropout_probability=0.0,
        augment=False,
    )
    return train_dataset, validation_dataset


def main() -> None:
    args = parse_args()
    configure_logging()
    config = apply_cli_overrides(load_yaml_config(args.config), args)
    require_config(
        config,
        "model_id",
        "data_root",
        "train_lr_subdir",
        "val_lr_subdir",
        "output_dir",
        "max_train_steps",
    )
    output_dir = resolve_project_path(config["output_dir"], PROJECT_ROOT)
    output_dir.mkdir(parents=True, exist_ok=True)
    config["output_dir"] = str(output_dir)
    if int(config.get("scale", 4)) != 4:
        raise ValueError("Stable Diffusion x4 upscaler requires scale=4")
    if int(config.get("low_res_noise_level_min", 0)) > int(config.get("low_res_noise_level_max", 20)):
        raise ValueError("low_res_noise_level_min must not exceed low_res_noise_level_max")

    accelerator = Accelerator(
        gradient_accumulation_steps=int(config.get("gradient_accumulation_steps", 1)),
        mixed_precision=str(config.get("mixed_precision", "no")),
        log_with="tensorboard",
        project_config=ProjectConfiguration(project_dir=str(output_dir), logging_dir=str(output_dir / "logs")),
    )
    if accelerator.is_main_process:
        save_yaml(config, output_dir / "training_config.yaml")
    set_seed(int(config.get("seed", 42)), device_specific=True)

    from diffusers import StableDiffusionUpscalePipeline
    from diffusers.optimization import get_scheduler

    mixed_precision = accelerator.mixed_precision
    weight_dtype = torch.float16 if mixed_precision == "fp16" else torch.bfloat16 if mixed_precision == "bf16" else torch.float32
    pipe = StableDiffusionUpscalePipeline.from_pretrained(
        str(config["model_id"]),
        torch_dtype=weight_dtype,
        safety_checker=None,
    )
    tokenizer_max_length = normalize_tokenizer_max_length(pipe.tokenizer, pipe.text_encoder)
    LOGGER.info("Using tokenizer max length %d from the text encoder configuration", tokenizer_max_length)
    for frozen in (pipe.vae, pipe.text_encoder):
        frozen.requires_grad_(False)
        frozen.eval()
    resume_path = resolve_resume_path(config, output_dir)
    initial_lora = resume_path
    if initial_lora is None and config.get("init_lora_path"):
        initial_lora = resolve_project_path(config["init_lora_path"], PROJECT_ROOT)
    unet = configure_lora(pipe, config, initial_lora)
    if bool(config.get("gradient_checkpointing", True)):
        unet.enable_gradient_checkpointing()
    if bool(config.get("enable_xformers_memory_efficient_attention", False)):
        try:
            unet.enable_xformers_memory_efficient_attention()
        except (ImportError, ModuleNotFoundError) as error:
            raise RuntimeError("xFormers was requested but is unavailable; install optional dependency xformers") from error

    adapter_path: Path | None = resume_path
    if adapter_path is None and config.get("init_adapter_path"):
        adapter_path = resolve_project_path(config["init_adapter_path"], PROJECT_ROOT)
    if adapter_path is not None:
        adapter = ConditionAdapter.from_pretrained(
            adapter_path,
            adapter_scale=float(config.get("adapter_scale", 1.0)),
        )
        LOGGER.info("Initialized ConditionAdapter from %s", adapter_path)
    else:
        adapter = ConditionAdapter(adapter_scale=float(config.get("adapter_scale", 1.0)))

    with accelerator.main_process_first():
        train_dataset, validation_dataset = build_datasets(
            config, output_dir, write_invalid_logs=accelerator.is_main_process
        )
    generator = torch.Generator().manual_seed(int(config.get("seed", 42)))
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=int(config.get("train_batch_size", 1)),
        shuffle=True,
        num_workers=int(config.get("num_workers", 4)),
        pin_memory=bool(config.get("pin_memory", True)),
        worker_init_fn=worker_init_fn,
        generator=generator,
        persistent_workers=int(config.get("num_workers", 4)) > 0,
    )

    optimizer_class: type[Optimizer]
    if bool(config.get("use_8bit_adam", False)):
        try:
            import bitsandbytes as bnb  # type: ignore
        except ImportError as error:
            raise RuntimeError("8-bit Adam was requested but bitsandbytes is not installed") from error
        optimizer_class = bnb.optim.AdamW8bit
    else:
        optimizer_class = torch.optim.AdamW
    optimizer = optimizer_class(
        [
            {"params": get_trainable_parameters(unet), "lr": float(config.get("learning_rate", 1e-4))},
            {"params": list(adapter.parameters()), "lr": float(config.get("adapter_learning_rate", 1e-4))},
        ],
        betas=(float(config.get("adam_beta1", 0.9)), float(config.get("adam_beta2", 0.999))),
        weight_decay=float(config.get("adam_weight_decay", 0.01)),
        eps=float(config.get("adam_epsilon", 1e-8)),
    )
    max_steps = int(config["max_train_steps"])
    lr_scheduler = get_scheduler(
        str(config.get("lr_scheduler", "constant")),
        optimizer=optimizer,
        num_warmup_steps=int(config.get("lr_warmup_steps", 0)) * accelerator.num_processes,
        num_training_steps=max_steps * accelerator.num_processes,
    )
    unet, adapter, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        unet, adapter, optimizer, train_dataloader, lr_scheduler
    )
    vae_dtype = torch.float32
    pipe.vae.to(accelerator.device, dtype=vae_dtype)
    pipe.text_encoder.to(accelerator.device, dtype=weight_dtype)
    if accelerator.is_main_process:
        LOGGER.info("Using VAE dtype %s for numerically stable latent encoding", vae_dtype)

    global_step = 0
    if resume_path is not None:
        optimizer_path = resume_path / "optimizer.pt"
        scheduler_path = resume_path / "lr_scheduler.pt"
        trainer_path = resume_path / "trainer_state.pt"
        for required in (optimizer_path, scheduler_path, trainer_path):
            if not required.is_file():
                raise FileNotFoundError(f"Resume state file missing: {required}")
        optimizer.load_state_dict(torch.load(optimizer_path, map_location="cpu", weights_only=True))
        lr_scheduler.load_state_dict(torch.load(scheduler_path, map_location="cpu", weights_only=True))
        trainer_state = torch.load(trainer_path, map_location="cpu", weights_only=True)
        global_step = int(trainer_state["global_step"])
        torch.set_rng_state(trainer_state["torch_rng_state"])
        LOGGER.info("Resumed optimizer/scheduler at global step %d from %s", global_step, resume_path)

    accelerator.init_trackers("worldstrat_sd_upscaler", config=config)
    noise_scheduler = pipe.scheduler
    low_res_scheduler = pipe.low_res_scheduler
    num_epochs = math.ceil(max_steps * int(config.get("gradient_accumulation_steps", 1)) / max(1, len(train_dataloader)))
    progress = tqdm(range(global_step, max_steps), disable=not accelerator.is_local_main_process, desc="training")
    warned_resize = False
    unet.train()
    adapter.train()

    for _epoch in range(num_epochs):
        for batch in train_dataloader:
            if global_step >= max_steps:
                break
            with accelerator.accumulate(unet, adapter):
                gt = batch["gt"].to(accelerator.device, dtype=vae_dtype)
                lr = batch["lr"].to(accelerator.device, dtype=weight_dtype)
                with torch.no_grad():
                    latents = pipe.vae.encode(gt).latent_dist.sample()
                    latents = (latents * pipe.vae.config.scaling_factor).to(dtype=weight_dtype)
                    prompt_embeds = encode_prompts(pipe.tokenizer, pipe.text_encoder, list(batch["prompt"]), accelerator.device)
                noise = torch.randn_like(latents)
                timesteps = torch.randint(
                    0,
                    noise_scheduler.config.num_train_timesteps,
                    (latents.shape[0],),
                    device=latents.device,
                    dtype=torch.long,
                )
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
                adapted_lr = adapter(lr)
                low_levels = torch.randint(
                    int(config.get("low_res_noise_level_min", 0)),
                    int(config.get("low_res_noise_level_max", 20)) + 1,
                    (lr.shape[0],),
                    device=lr.device,
                    dtype=torch.long,
                )
                low_noise = torch.randn_like(adapted_lr)
                noisy_low = low_res_scheduler.add_noise(adapted_lr, low_noise, low_levels)
                if noisy_low.shape[-2:] != noisy_latents.shape[-2:]:
                    message = (
                        "Low-resolution condition and latent spatial sizes differ: "
                        f"condition={tuple(noisy_low.shape[-2:])}, latent={tuple(noisy_latents.shape[-2:])}. "
                        "The official x4 upscaler normally expects them to match."
                    )
                    if not bool(config.get("resize_low_res_condition_if_needed", False)):
                        raise AssertionError(message + " Set resize_low_res_condition_if_needed=true to bicubic-resize explicitly.")
                    if not warned_resize:
                        if accelerator.is_main_process:
                            LOGGER.warning("%s Applying explicit bicubic compatibility resize.", message)
                        warned_resize = True
                    noisy_low = F.interpolate(noisy_low, size=noisy_latents.shape[-2:], mode="bicubic", align_corners=False)
                assert noisy_low.shape[-2:] == noisy_latents.shape[-2:], (
                    f"Condition/latent mismatch after compatibility handling: {noisy_low.shape} vs {noisy_latents.shape}"
                )
                model_input = torch.cat([noisy_latents, noisy_low], dim=1)
                prediction = unet(
                    model_input,
                    timesteps,
                    encoder_hidden_states=prompt_embeds,
                    class_labels=low_levels,
                    return_dict=False,
                )[0]
                prediction_type = noise_scheduler.config.prediction_type
                if prediction_type == "epsilon":
                    target = noise
                elif prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(latents, noise, timesteps)
                else:
                    raise ValueError(f"Unsupported scheduler prediction_type: {prediction_type}")
                gamma_value = config.get("snr_gamma", 5.0)
                loss = snr_weighted_mse(
                    prediction,
                    target,
                    timesteps,
                    noise_scheduler,
                    None if gamma_value is None else float(gamma_value),
                )
                if not torch.isfinite(loss):
                    raise FloatingPointError(
                        "Non-finite training loss detected before backward. "
                        f"loss={loss.detach().float().item()}, "
                        f"latents_finite={torch.isfinite(latents).all().item()}, "
                        f"prediction_finite={torch.isfinite(prediction).all().item()}, "
                        f"target_finite={torch.isfinite(target).all().item()}"
                    )
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    parameters: Iterable[nn.Parameter] = list(get_trainable_parameters(unet)) + list(adapter.parameters())
                    accelerator.clip_grad_norm_(parameters, float(config.get("max_grad_norm", 1.0)))
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            if accelerator.sync_gradients:
                global_step += 1
                progress.update(1)
                loss_value = accelerator.gather(loss.detach().reshape(1)).mean().item()
                logs = {"train_loss": loss_value, "lr": lr_scheduler.get_last_lr()[0]}
                progress.set_postfix(loss=f"{loss_value:.4f}")
                accelerator.log(logs, step=global_step)

                if global_step % int(config.get("checkpointing_steps", 1000)) == 0:
                    accelerator.wait_for_everyone()
                    checkpoint = output_dir / f"checkpoint-{global_step:05d}"
                    save_artifacts(accelerator, unet, adapter, checkpoint, config, global_step, optimizer, lr_scheduler)
                    if accelerator.is_main_process:
                        enforce_checkpoint_limit(output_dir, config.get("checkpoints_total_limit", 3))
                if global_step % int(config.get("validation_steps", 1000)) == 0:
                    accelerator.wait_for_everyone()
                    run_validation(
                        accelerator,
                        pipe,
                        unet,
                        adapter,
                        validation_dataset,
                        output_dir,
                        global_step,
                        config,
                        weight_dtype,
                    )
                    accelerator.wait_for_everyone()
            if global_step >= max_steps:
                break

    accelerator.wait_for_everyone()
    save_artifacts(accelerator, unet, adapter, output_dir / "final", config, global_step, optimizer, lr_scheduler)
    accelerator.end_training()


if __name__ == "__main__":
    main()
