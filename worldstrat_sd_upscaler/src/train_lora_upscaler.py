#!/usr/bin/env python
"""Two-stage LoRA training for StableDiffusionUpscalePipeline."""

from __future__ import annotations

import argparse
import logging
import math
import random
import sys
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs, ProjectConfiguration, set_seed
from PIL import Image
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.condition_adapter import ConditionAdapter
from src.dataset import PairedSatelliteDataset
from src.diffusion_prediction import (
    alpha_bar_for_timesteps,
    model_output_to_x0,
    normalized_timesteps,
    validate_timestep_range,
)
from src.latent_phi import LatentPhi, compute_phi_training_losses
from src.metrics import psnr, ssim
from src.rgb_auxiliary_loss import compute_rgb_auxiliary_losses
from src.tri_input_bridge import (
    TriConditionedUNet,
    TriInputBridge,
    fresh_down_intrablock_residuals,
    pipeline_cross_attention_kwargs,
)
from src.tri_input_conditioner import TriInputConditioner
from src.tri_input_data import (
    RawBandStats,
    RawValueConversion,
    load_raw_band_stats,
    save_raw_band_stats,
    validate_sentinel2_l2a_band_names,
)
from src.tri_input_runtime import (
    PreparedTriCondition,
    masked_preview_l1,
    prepare_tri_condition_from_batch,
    tri_diagnostic_metrics,
)
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
    tracker_safe_config,
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


def trainable_parameters(module: nn.Module) -> list[nn.Parameter]:
    """Return trainable parameters without forcing frozen modules to be trainable."""

    return [parameter for parameter in module.parameters() if parameter.requires_grad]


def tri_input_config(config: dict[str, Any]) -> dict[str, Any]:
    """Validate and return the optional tri-input mapping.

    Keeping this validation independent from model loading makes null
    val/test paths and unconfirmed physical value conversion fail with a clear
    message rather than much later in a worker process.
    """

    value = config.get("tri_input", {})
    if value is None:
        value = {}
    if not isinstance(value, dict):
        raise TypeError("tri_input must be a YAML mapping")
    if not bool(value.get("enabled", False)):
        return value
    if bool(config.get("phi_enabled", False)):
        raise ValueError(
            "tri_input.enabled and phi_enabled are not jointly supported in V1; "
            "disable phi explicitly instead of silently ignoring either branch"
        )
    if int(config.get("scale", 4)) != 4:
        raise ValueError("Tri-input V1 requires scale=4")
    if str(value.get("training_stage", "diffusion")) != "diffusion":
        raise ValueError(
            "train_lora_upscaler.py only handles tri_input.training_stage='diffusion'; "
            "use scripts/train_tri_input_warmup.py for stages A/B"
        )
    validate_sentinel2_l2a_band_names(value.get("band_names"))
    RawValueConversion.from_config(value.get("raw_value_conversion"))
    if not value.get("raw_stats_path"):
        raise ValueError(
            "tri_input.raw_stats_path is required for diffusion training and must be "
            "computed from the train split"
        )
    stats_path = resolve_project_path(value["raw_stats_path"], PROJECT_ROOT)
    if not stats_path.is_file():
        raise FileNotFoundError(f"Tri-input train statistics do not exist: {stats_path}")
    data = value.get("data")
    if not isinstance(data, dict):
        raise ValueError("tri_input.data must explicitly configure train and val auxiliary paths")
    for split in ("train", "val"):
        split_config = data.get(split)
        if not isinstance(split_config, dict):
            raise ValueError(f"tri_input.data.{split} must be a mapping")
        for name in ("raw_ms_dir", "unmixing_dir"):
            if not split_config.get(name):
                raise ValueError(
                    f"tri_input.data.{split}.{name} is required; null means unconfirmed, "
                    "not an instruction to guess a path"
                )
            directory = resolve_project_path(split_config[name], PROJECT_ROOT)
            if not directory.is_dir():
                raise FileNotFoundError(
                    f"tri_input.data.{split}.{name} does not exist: {directory}"
                )
        manifest = split_config.get("manifest_path")
        if manifest is not None:
            manifest_path = resolve_project_path(manifest, PROJECT_ROOT)
            if not manifest_path.is_file():
                raise FileNotFoundError(
                    f"tri_input.data.{split}.manifest_path does not exist: {manifest_path}"
                )
    replay_probability = float(config.get("synthetic_replay_probability", 0.0))
    policy = str(value.get("synthetic_aux_policy", "error"))
    if policy not in {"error", "disable"}:
        raise ValueError("tri_input.synthetic_aux_policy must be 'error' or 'disable'")
    if replay_probability > 0 and policy == "error":
        raise ValueError(
            "Tri-input diffusion training cannot pair real auxiliary observations with "
            "synthetic RGB replay. Keep synthetic_replay_probability=0, or explicitly "
            "choose synthetic_aux_policy=disable."
        )
    if float(value.get("preview_loss_weight", 0.0)) < 0:
        raise ValueError("tri_input.preview_loss_weight must be non-negative")
    if float(config.get("tri_input_learning_rate", 1.0e-4)) <= 0:
        raise ValueError("tri_input_learning_rate must be positive")
    return value


def _assert_conditioner_contract(
    conditioner: TriInputConditioner,
    *,
    tri_config: dict[str, Any],
    stats: RawBandStats,
) -> None:
    layout = validate_sentinel2_l2a_band_names(tri_config.get("band_names"))
    expected_checker = dict(tri_config.get("checker") or {})
    actual = conditioner.config
    mismatches: list[str] = []
    if tuple(actual.get("surface_band_indices", ())) != layout.surface_indices:
        mismatches.append("surface_band_indices")
    if int(actual.get("components", -1)) != int(tri_config.get("components", 8)):
        mismatches.append("components")
    if int(actual.get("scale", -1)) != int(expected_checker.get("scale", 4)):
        mismatches.append("scale")
    actual_checker = dict(actual.get("checker") or {})
    # CheckerConfig adds explicit conservative defaults; compare every value
    # supplied by YAML and let the module own unspecified defaults.
    for key, expected in expected_checker.items():
        if actual_checker.get(key) != expected:
            mismatches.append(f"checker.{key}")
    expected_mean = torch.tensor(stats.mean, dtype=torch.float32)
    expected_std = torch.tensor(stats.std, dtype=torch.float32)
    if not torch.equal(conditioner.raw_mean.detach().cpu().flatten(), expected_mean):
        mismatches.append("raw_mean")
    if not torch.equal(conditioner.raw_std.detach().cpu().flatten(), expected_std):
        mismatches.append("raw_std")
    if mismatches:
        raise ValueError(
            "TriInputConditioner checkpoint/config contract mismatch: " + ", ".join(mismatches)
        )


def build_tri_input_modules(
    config: dict[str, Any],
    unet: nn.Module,
    resume_path: Path | None,
) -> tuple[TriInputConditioner, TriInputBridge, RawBandStats, dict[str, Any]]:
    """Build or strictly restore the Stage-3 modules and provenance."""

    tri_config = tri_input_config(config)
    layout = validate_sentinel2_l2a_band_names(tri_config.get("band_names"))
    conversion = RawValueConversion.from_config(tri_config.get("raw_value_conversion"))
    stats_path = resolve_project_path(tri_config["raw_stats_path"], PROJECT_ROOT)
    stats = load_raw_band_stats(
        stats_path,
        expected_band_names=layout.band_names,
        expected_conversion=conversion,
    )

    conditioner_source: Path | None = resume_path
    if conditioner_source is None and tri_config.get("init_conditioner_path"):
        conditioner_source = resolve_project_path(tri_config["init_conditioner_path"], PROJECT_ROOT)
    if conditioner_source is not None:
        conditioner = TriInputConditioner.from_pretrained(conditioner_source)
    else:
        conditioner = TriInputConditioner(
            raw_mean=stats.mean,
            raw_std=stats.std,
            surface_band_indices=layout.surface_indices,
            components=int(tri_config.get("components", 8)),
            scale=int(config.get("scale", 4)),
            checker_config=tri_config.get("checker"),
        )
    _assert_conditioner_contract(conditioner, tri_config=tri_config, stats=stats)

    bridge_source: Path | None = resume_path
    if bridge_source is None and tri_config.get("init_bridge_path"):
        bridge_source = resolve_project_path(tri_config["init_bridge_path"], PROJECT_ROOT)
    if bridge_source is not None:
        bridge = TriInputBridge.from_pretrained(bridge_source, unet=unet)
    else:
        bridge = TriInputBridge.from_unet(
            unet,
            geometry_channels=(32, 64, 96, 128),
            context_channels=64,
            hidden_channels=tri_config.get("bridge_hidden_channels", 64),
        )

    previous_dependencies = conditioner.artifact_metadata.get(
        "dependency_rgb_checkpoint", {}
    )
    if not isinstance(previous_dependencies, dict):
        previous_dependencies = {}
    dependency_paths = {
        "lora": str(
            config.get("init_lora_path") or previous_dependencies.get("lora") or ""
        ),
        "condition_adapter": str(
            config.get("init_adapter_path")
            or previous_dependencies.get("condition_adapter")
            or ""
        ),
    }
    metadata: dict[str, Any] = {
        "tri_input_enabled": True,
        "module_version": int(tri_config.get("module_version", 1)),
        "band_names": list(layout.band_names),
        "rgb_indices": list(layout.rgb_indices),
        "b8_index": layout.b8_index,
        "b1_index": layout.b1_index,
        "b9_index": layout.b9_index,
        "surface_band_indices": list(layout.surface_indices),
        "raw_value_conversion": conversion.as_dict(),
        "raw_stats_source": str(stats_path),
        "checker": conditioner.config["checker"],
        "components": int(tri_config.get("components", 8)),
        "geometry_channels": [32, 64, 96, 128],
        "context_channels": 64,
        "injection": "UNet2DConditionModel.down_intrablock_additional_residuals",
        "training_stage": "diffusion",
        "dependency_rgb_checkpoint": dependency_paths,
    }
    conditioner.artifact_metadata.update(metadata)
    return conditioner, bridge, stats, metadata


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
    phi: nn.Module | None = None,
    tri_conditioner: nn.Module | None = None,
    tri_bridge: nn.Module | None = None,
    tri_stats: RawBandStats | None = None,
    tri_metadata: dict[str, Any] | None = None,
    data_generator: torch.Generator | None = None,
) -> None:
    """Save reloadable Diffusers LoRA, adapter, actual config, and trainer state."""
    def rng_state() -> dict[str, Any]:
        numpy_state = np.random.get_state()
        result: dict[str, Any] = {
            "torch_rng_state": torch.get_rng_state(),
            "python_rng_state": random.getstate(),
            "numpy_rng_state": {
                "bit_generator": numpy_state[0],
                "state": numpy_state[1].tolist(),
                "position": int(numpy_state[2]),
                "has_gauss": int(numpy_state[3]),
                "cached_gaussian": float(numpy_state[4]),
            },
        }
        if torch.cuda.is_available():
            result["cuda_rng_state_all"] = torch.cuda.get_rng_state_all()
        if data_generator is not None:
            result["data_generator_state"] = data_generator.get_state()
        return result

    tri_checkpoint = tri_conditioner is not None
    if optimizer is not None and lr_scheduler is not None and tri_checkpoint:
        # Device-specific seeds make each rank's RNG stream distinct.  Save a
        # small per-rank state before the main-process artifact return.
        output_path.mkdir(parents=True, exist_ok=True)
        atomic_torch_save(
            rng_state(), output_path / f"rng_state_rank-{accelerator.process_index:04d}.pt"
        )
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
    if phi is not None:
        unwrapped_phi = accelerator.unwrap_model(phi)
        if not isinstance(unwrapped_phi, LatentPhi):
            raise TypeError(f"Unexpected latent phi type: {type(unwrapped_phi)}")
        unwrapped_phi.save_pretrained(output_path)
    if (tri_conditioner is None) != (tri_bridge is None):
        raise ValueError("Tri-input conditioner and bridge must be saved together")
    if tri_conditioner is not None and tri_bridge is not None:
        unwrapped_conditioner = accelerator.unwrap_model(tri_conditioner)
        unwrapped_bridge = accelerator.unwrap_model(tri_bridge)
        if not isinstance(unwrapped_conditioner, TriInputConditioner):
            raise TypeError(f"Unexpected tri-input conditioner type: {type(unwrapped_conditioner)}")
        if not isinstance(unwrapped_bridge, TriInputBridge):
            raise TypeError(f"Unexpected tri-input bridge type: {type(unwrapped_bridge)}")
        unwrapped_conditioner.save_pretrained(
            output_path, artifact_metadata=tri_metadata
        )
        unwrapped_bridge.save_pretrained(output_path)
        if tri_stats is None:
            raise ValueError("Tri-input artifacts require the train-only raw band statistics")
        save_raw_band_stats(tri_stats, output_path / "raw_band_stats.json")
    save_yaml(config, output_path / "training_config.yaml")
    model_info = {
        "model_id": config["model_id"],
        "pipeline_class": "StableDiffusionUpscalePipeline",
        "diffusers_version": __import__("diffusers").__version__,
        "global_step": global_step,
        "lora_rank": int(config.get("lora_rank", 16)),
        "adapter_scale": float(config.get("adapter_scale", 1.0)),
    }
    if tri_conditioner is not None:
        model_info["tri_input_enabled"] = True
    save_json(model_info, output_path / "model_info.json")
    if optimizer is not None and lr_scheduler is not None:
        atomic_torch_save(optimizer.state_dict(), output_path / "optimizer.pt")
        atomic_torch_save(lr_scheduler.state_dict(), output_path / "lr_scheduler.pt")
        trainer_state: dict[str, Any]
        if tri_checkpoint:
            trainer_state = {"global_step": global_step, **rng_state()}
        else:
            # Preserve the legacy RGB-only checkpoint contract exactly.
            trainer_state = {
                "global_step": global_step,
                "torch_rng_state": torch.get_rng_state(),
            }
        atomic_torch_save(trainer_state, output_path / "trainer_state.pt")
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
    tri_conditioner: nn.Module | None = None,
    tri_bridge: nn.Module | None = None,
) -> None:
    """Generate fixed validation examples and record PSNR/SSIM plus triptychs."""
    if not accelerator.is_main_process:
        return
    unwrapped_unet = accelerator.unwrap_model(unet)
    unwrapped_adapter = accelerator.unwrap_model(adapter)
    unwrapped_conditioner = (
        accelerator.unwrap_model(tri_conditioner) if tri_conditioner is not None else None
    )
    unwrapped_bridge = (
        accelerator.unwrap_model(tri_bridge) if tri_bridge is not None else None
    )
    if (unwrapped_conditioner is None) != (unwrapped_bridge is None):
        raise ValueError("Validation requires tri-input conditioner and bridge together")
    was_training = unwrapped_unet.training
    adapter_was_training = unwrapped_adapter.training
    conditioner_was_training = (
        unwrapped_conditioner.training if unwrapped_conditioner is not None else False
    )
    bridge_was_training = unwrapped_bridge.training if unwrapped_bridge is not None else False
    unwrapped_unet.eval()
    unwrapped_adapter.eval()
    if unwrapped_conditioner is not None and unwrapped_bridge is not None:
        unwrapped_conditioner.eval()
        unwrapped_bridge.eval()
        pipeline_unet: nn.Module = TriConditionedUNet(unwrapped_unet, unwrapped_bridge)
    else:
        pipeline_unet = unwrapped_unet
    pipe.unet = pipeline_unet
    pipe.to(accelerator.device)
    pipe.set_progress_bar_config(disable=True)
    validation_dir = output_dir / "validation" / f"step-{global_step:06d}"
    validation_dir.mkdir(parents=True, exist_ok=True)
    sample_count = min(int(config.get("validation_num_samples", 2)), len(dataset))
    results: list[dict[str, float | str]] = []
    try:
        for index in range(sample_count):
            sample = dataset[index]
            lr_pil = adapt_pil(unwrapped_adapter, sample["lr"], accelerator.device, weight_dtype)
            generator = torch.Generator(device=accelerator.device).manual_seed(
                int(config.get("seed", 42)) + index
            )
            guidance_scale = float(config.get("validation_guidance_scale", 1.0))
            call: dict[str, Any] = {
                "prompt": sample["prompt"],
                "image": lr_pil,
                "noise_level": int(config.get("validation_noise_level", 10)),
                "guidance_scale": guidance_scale,
                "num_inference_steps": int(
                    config.get("validation_num_inference_steps", 20)
                ),
                "generator": generator,
            }
            if unwrapped_conditioner is not None and unwrapped_bridge is not None:
                lr_batch = sample["lr"].unsqueeze(0).to(
                    accelerator.device, dtype=torch.float32
                )
                validation_batch = {
                    name: sample[name].unsqueeze(0)
                    for name in ("raw_ms", "unmixing", "raw_valid", "aux_present")
                }
                dummy_sample = torch.empty(
                    (
                        1,
                        int(unwrapped_unet.config.in_channels),
                        lr_batch.shape[-2],
                        lr_batch.shape[-1],
                    ),
                    device=accelerator.device,
                    dtype=weight_dtype,
                )
                prepared = prepare_tri_condition_from_batch(
                    unwrapped_conditioner,
                    unwrapped_bridge,
                    validation_batch,
                    rgb_minus_one_one=lr_batch,
                    unet_sample=dummy_sample,
                    device=accelerator.device,
                    do_classifier_free_guidance=guidance_scale > 1.0,
                )
                if prepared is not None:
                    call["cross_attention_kwargs"] = pipeline_cross_attention_kwargs(
                        prepared.residuals
                    )
            output = pipe(**call).images[0]
            gt_pil = tensor_to_pil(sample["gt"])
            if output.size != gt_pil.size:
                raise AssertionError(
                    f"Validation SR size mismatch for {sample['filename']}: SR={output.size}, GT={gt_pil.size}"
                )
            pred_array = np.asarray(output, dtype=np.float32) / 255.0
            gt_array = np.asarray(gt_pil, dtype=np.float32) / 255.0
            result = {
                "sample_id": sample["sample_id"],
                "psnr": psnr(pred_array, gt_array),
                "ssim": ssim(pred_array, gt_array),
            }
            results.append(result)
            preview = Image.new("RGB", (gt_pil.width * 3, gt_pil.height))
            preview.paste(lr_pil.resize(gt_pil.size, Image.Resampling.BICUBIC), (0, 0))
            preview.paste(output, (gt_pil.width, 0))
            preview.paste(gt_pil, (gt_pil.width * 2, 0))
            preview.save(validation_dir / f"{sample['sample_id']}_lr_sr_gt.png")
    finally:
        # The pipeline wrapper and eval states are request-scoped.  Restore
        # them even when sampling or metric serialization raises, otherwise a
        # failed validation can silently leave the training loop in eval mode.
        pipe.unet = unwrapped_unet
        unwrapped_unet.train(was_training)
        unwrapped_adapter.train(adapter_was_training)
        if unwrapped_conditioner is not None and unwrapped_bridge is not None:
            unwrapped_conditioner.train(conditioner_was_training)
            unwrapped_bridge.train(bridge_was_training)
    save_json(
        {
            "global_step": global_step,
            "samples": results,
            "mean_psnr": float(np.mean([float(item["psnr"]) for item in results])),
            "mean_ssim": float(np.mean([float(item["ssim"]) for item in results])),
        },
        validation_dir / "metrics.json",
    )
def build_datasets(
    config: dict[str, Any], output_dir: Path, write_invalid_logs: bool = True
) -> tuple[PairedSatelliteDataset, PairedSatelliteDataset]:
    """Construct train/validation datasets from every relevant YAML setting."""
    tri_config = config.get("tri_input", {}) or {}
    tri_enabled = bool(tri_config.get("enabled", False))
    tri_data = tri_config.get("data", {}) if tri_enabled else {}

    def split_auxiliary(split: str) -> dict[str, Any]:
        if not tri_enabled:
            return {}
        value = tri_data.get(split)
        if not isinstance(value, dict):
            raise ValueError(f"tri_input.data.{split} must be an explicit mapping")
        return {
            "tri_input_enabled": True,
            "raw_ms_dir": value.get("raw_ms_dir"),
            "unmixing_dir": value.get("unmixing_dir"),
            "raw_band_names": tri_config.get("band_names"),
            "raw_value_conversion": tri_config.get("raw_value_conversion"),
            "aux_manifest_path": value.get("manifest_path"),
            "aux_recursive": bool(value.get("recursive", False)),
            "synthetic_aux_policy": str(tri_config.get("synthetic_aux_policy", "error")),
        }

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
        **split_auxiliary("train"),
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
        **split_auxiliary("val"),
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
    phi_enabled = bool(config.get("phi_enabled", False))
    rgb_aux_loss_enabled = bool(config.get("rgb_aux_loss_enabled", False))
    tri_config = tri_input_config(config)
    tri_enabled = bool(tri_config.get("enabled", False))
    if tri_enabled and not config.get("resume_from_checkpoint"):
        missing_initial = [
            name
            for name in ("init_lora_path", "init_adapter_path")
            if not config.get(name)
        ]
        if missing_initial:
            raise ValueError(
                "Tri-input diffusion adaptation must start from an explicit trained RGB artifact; "
                f"missing {missing_initial}. Use resume_from_checkpoint only for a same-structure "
                "tri-input checkpoint."
            )
    if rgb_aux_loss_enabled and any(
        float(config.get(name, default)) < 0
        for name, default in (("lambda_l1", 0.1), ("lambda_lpips_rgb", 0.1))
    ):
        raise ValueError("lambda_l1 and lambda_lpips_rgb must be non-negative")
    if phi_enabled:
        validate_timestep_range(
            config.get("phi_train_timestep_range", [0.0, 1.0]),
            "phi_train_timestep_range",
        )
        validate_timestep_range(
            config.get("phi_infer_timestep_range", [0.0, 1.0]),
            "phi_infer_timestep_range",
        )
        if any(
            float(config.get(name, default)) < 0
            for name, default in (
                ("lambda_z0", 1.0),
                ("lambda_mu", 1.0),
                ("lambda_lpips", 0.1),
            )
        ):
            raise ValueError("lambda_z0, lambda_mu, and lambda_lpips must be non-negative")

    accelerator_options: dict[str, Any] = {}
    if phi_enabled or tri_enabled:
        accelerator_options["kwargs_handlers"] = [
            DistributedDataParallelKwargs(find_unused_parameters=True)
        ]
    accelerator = Accelerator(
        gradient_accumulation_steps=int(config.get("gradient_accumulation_steps", 1)),
        mixed_precision=str(config.get("mixed_precision", "no")),
        log_with="tensorboard",
        project_config=ProjectConfiguration(project_dir=str(output_dir), logging_dir=str(output_dir / "logs")),
        **accelerator_options,
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

    tri_conditioner: TriInputConditioner | None = None
    tri_bridge: TriInputBridge | None = None
    tri_stats: RawBandStats | None = None
    tri_metadata: dict[str, Any] | None = None
    train_existing_lora = True
    train_existing_adapter = True
    if tri_enabled:
        train_existing_lora = bool(tri_config.get("train_existing_lora", False))
        train_existing_adapter = bool(
            tri_config.get("train_existing_condition_adapter", False)
        )
        if not train_existing_lora:
            unet.requires_grad_(False)
        if not train_existing_adapter:
            adapter.requires_grad_(False)
        tri_conditioner, tri_bridge, tri_stats, tri_metadata = build_tri_input_modules(
            config, unet, resume_path
        )
        LOGGER.info(
            "Tri-input enabled: checker=%s, train_existing_lora=%s, "
            "train_existing_condition_adapter=%s, VAE scale factor=%s",
            bool((tri_config.get("checker") or {}).get("enabled", True)),
            train_existing_lora,
            train_existing_adapter,
            getattr(pipe, "vae_scale_factor", "unknown"),
        )

    phi: LatentPhi | None = None
    differentiable_lpips: nn.Module | None = None
    if phi_enabled:
        if resume_path is not None:
            phi = LatentPhi.from_pretrained(resume_path)
            if int(phi.config["latent_channels"]) != int(pipe.vae.config.latent_channels):
                raise ValueError(
                    "LatentPhi checkpoint latent_channels does not match the loaded VAE: "
                    f"{phi.config['latent_channels']} vs {pipe.vae.config.latent_channels}"
                )
            LOGGER.info("Initialized LatentPhi from %s", resume_path)
        else:
            phi = LatentPhi(
                latent_channels=int(pipe.vae.config.latent_channels),
                hidden_channels=int(config.get("phi_hidden_channels", 64)),
                time_embed_dim=int(config.get("phi_time_embed_dim", 128)),
                num_blocks=int(config.get("phi_num_blocks", 3)),
            )
    lpips_required = (
        phi_enabled and float(config.get("lambda_lpips", 0.1)) > 0
    ) or (
        rgb_aux_loss_enabled and float(config.get("lambda_lpips_rgb", 0.1)) > 0
    )
    if lpips_required:
        try:
            import lpips  # type: ignore
        except ImportError as error:
            raise RuntimeError(
                "Enabled differentiable LPIPS loss requires the `lpips` package"
            ) from error
        differentiable_lpips = lpips.LPIPS(net="alex").eval()
        differentiable_lpips.requires_grad_(False)

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
    parameter_groups: list[dict[str, Any]] = []
    if not tri_enabled or train_existing_lora:
        parameter_groups.append(
            {
                "params": get_trainable_parameters(unet),
                "lr": float(config.get("learning_rate", 1e-4)),
            }
        )
    if not tri_enabled or train_existing_adapter:
        adapter_parameters = trainable_parameters(adapter)
        if not adapter_parameters:
            raise RuntimeError("ConditionAdapter was requested for training but has no trainable parameters")
        parameter_groups.append(
            {
                "params": adapter_parameters,
                "lr": float(config.get("adapter_learning_rate", 1e-4)),
            }
        )
    if tri_conditioner is not None and tri_bridge is not None:
        tri_lr = float(config.get("tri_input_learning_rate", 1e-4))
        parameter_groups.extend(
            (
                {
                    "params": get_trainable_parameters(tri_conditioner),
                    "lr": tri_lr,
                },
                {
                    "params": get_trainable_parameters(tri_bridge),
                    "lr": tri_lr,
                },
            )
        )
    if phi is not None:
        parameter_groups.append(
            {
                "params": list(phi.parameters()),
                "lr": float(config.get("phi_learning_rate", 1e-4)),
            }
        )
    if not parameter_groups:
        raise RuntimeError("Training configuration selected no trainable parameters")
    optimizer = optimizer_class(
        parameter_groups,
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
    if tri_conditioner is not None and tri_bridge is not None:
        # DDP rejects modules with no trainable parameters on some PyTorch
        # versions.  Frozen RGB modules are therefore moved explicitly while
        # every trainable module remains registered/prepared normally.
        trainable_models: list[tuple[str, nn.Module]] = []
        if trainable_parameters(unet):
            trainable_models.append(("unet", unet))
        else:
            unet.to(accelerator.device)
        if trainable_parameters(adapter):
            trainable_models.append(("adapter", adapter))
        else:
            adapter.to(accelerator.device, dtype=weight_dtype)
        trainable_models.extend(
            (("tri_conditioner", tri_conditioner), ("tri_bridge", tri_bridge))
        )
        prepared_values = accelerator.prepare(
            *(module for _, module in trainable_models),
            optimizer,
            train_dataloader,
            lr_scheduler,
        )
        prepared_models = prepared_values[: len(trainable_models)]
        for (name, _), prepared_model in zip(
            trainable_models, prepared_models, strict=True
        ):
            if name == "unet":
                unet = prepared_model
            elif name == "adapter":
                adapter = prepared_model
            elif name == "tri_conditioner":
                tri_conditioner = prepared_model
            elif name == "tri_bridge":
                tri_bridge = prepared_model
        optimizer, train_dataloader, lr_scheduler = prepared_values[-3:]
    elif phi is not None:
        unet, adapter, phi, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
            unet, adapter, phi, optimizer, train_dataloader, lr_scheduler
        )
    else:
        unet, adapter, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
            unet, adapter, optimizer, train_dataloader, lr_scheduler
        )
    vae_dtype = torch.float32
    pipe.vae.to(accelerator.device, dtype=vae_dtype)
    pipe.text_encoder.to(accelerator.device, dtype=weight_dtype)
    if differentiable_lpips is not None:
        differentiable_lpips.to(accelerator.device)
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
        rank_rng_path = resume_path / f"rng_state_rank-{accelerator.process_index:04d}.pt"
        rng_state = (
            torch.load(rank_rng_path, map_location="cpu", weights_only=True)
            if rank_rng_path.is_file()
            else trainer_state
        )
        torch.set_rng_state(rng_state["torch_rng_state"])
        if "python_rng_state" in rng_state:
            random.setstate(rng_state["python_rng_state"])
        if "numpy_rng_state" in rng_state:
            numpy_state = rng_state["numpy_rng_state"]
            np.random.set_state(
                (
                    str(numpy_state["bit_generator"]),
                    np.asarray(numpy_state["state"], dtype=np.uint32),
                    int(numpy_state["position"]),
                    int(numpy_state["has_gauss"]),
                    float(numpy_state["cached_gaussian"]),
                )
            )
        if "cuda_rng_state_all" in rng_state and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(rng_state["cuda_rng_state_all"])
        if "data_generator_state" in rng_state:
            generator.set_state(rng_state["data_generator_state"])
        LOGGER.info("Resumed optimizer/scheduler at global step %d from %s", global_step, resume_path)

    accelerator.init_trackers(
        "worldstrat_sd_upscaler", config=tracker_safe_config(config)
    )
    noise_scheduler = pipe.scheduler
    low_res_scheduler = pipe.low_res_scheduler
    num_epochs = math.ceil(max_steps * int(config.get("gradient_accumulation_steps", 1)) / max(1, len(train_dataloader)))
    progress = tqdm(range(global_step, max_steps), disable=not accelerator.is_local_main_process, desc="training")
    warned_resize = False
    accumulation_active_count = 0.0
    accumulation_sample_count = 0.0
    accumulation_phi_z0_sum = 0.0
    accumulation_phi_mu_sum = 0.0
    accumulation_phi_lpips_sum = 0.0
    accumulation_phi_total_sum = 0.0
    accumulation_r_mean_sum = 0.0
    accumulation_r_std_sum = 0.0
    accumulation_r_min = float("inf")
    accumulation_r_max = float("-inf")
    accumulation_mu_base_sum = 0.0
    accumulation_mu_phi_sum = 0.0
    accumulation_mu_target_sum = 0.0
    accumulation_mu_mixed_sum = 0.0
    accumulation_rgb_sample_count = 0.0
    accumulation_rgb_l1_sum = 0.0
    accumulation_rgb_lpips_sum = 0.0
    accumulation_rgb_total_sum = 0.0
    accumulation_tri_sample_count = 0.0
    accumulation_tri_aux_count = 0.0
    accumulation_tri_preview_sum = 0.0
    accumulation_tri_diagnostic_sums = {
        "checker_valid_windows": 0.0,
        "checker_accepted_fraction": 0.0,
        "checker_mean_shift_hr": 0.0,
        "checker_internal_gain": 0.0,
    }
    accumulation_tri_diagnostic_batches = 0.0
    seen_checker_reasons: set[str] = set()
    if tri_enabled:
        unet.train(mode=train_existing_lora)
        adapter.train(mode=train_existing_adapter)
        assert tri_conditioner is not None and tri_bridge is not None
        tri_conditioner.train()
        tri_bridge.train()
    else:
        unet.train()
        adapter.train()
    if phi is not None:
        phi.train()

    for _epoch in range(num_epochs):
        for batch in train_dataloader:
            if global_step >= max_steps:
                break
            if tri_conditioner is not None and tri_bridge is not None:
                accumulation_modules = (unet, adapter, tri_conditioner, tri_bridge)
            else:
                accumulation_modules = (unet, adapter, phi) if phi is not None else (unet, adapter)
            with accelerator.accumulate(*accumulation_modules):
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
                # The tri-input condition is computed from the original
                # preprocessed RGB/raw/prior tensors exactly once, before the
                # low-resolution condition receives random scheduler noise.
                tri_prepared: PreparedTriCondition | None = None
                if tri_conditioner is not None and tri_bridge is not None:
                    residual_shape_sample = torch.empty(
                        (
                            latents.shape[0],
                            latents.shape[1] + adapted_lr.shape[1],
                            latents.shape[2],
                            latents.shape[3],
                        ),
                        device=latents.device,
                        dtype=weight_dtype,
                    )
                    tri_prepared = prepare_tri_condition_from_batch(
                        tri_conditioner,
                        tri_bridge,
                        batch,
                        rgb_minus_one_one=lr,
                        unet_sample=residual_shape_sample,
                        device=accelerator.device,
                    )
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
                if tri_enabled:
                    # A prepared trainable adapter may return fp32 while the
                    # frozen UNet remains fp16/bf16. Keep this boundary explicit;
                    # .to() preserves gradients to any unfrozen adapter.
                    model_input = model_input.to(dtype=weight_dtype)
                tri_unet_kwargs: dict[str, Any] = {}
                if tri_prepared is not None:
                    tri_unet_kwargs["down_intrablock_additional_residuals"] = (
                        fresh_down_intrablock_residuals(tri_prepared.residuals, sample=model_input)
                    )
                model_output = unet(
                    model_input,
                    timesteps,
                    encoder_hidden_states=prompt_embeds,
                    class_labels=low_levels,
                    return_dict=False,
                    **tri_unet_kwargs,
                )[0]
                prediction_type = noise_scheduler.config.prediction_type
                if prediction_type == "epsilon":
                    target = noise
                elif prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(latents, noise, timesteps)
                elif prediction_type in {"sample", "x0", "x_start"}:
                    target = latents
                else:
                    raise ValueError(f"Unsupported scheduler prediction_type: {prediction_type}")
                gamma_value = config.get("snr_gamma", 5.0)
                diffusion_loss = snr_weighted_mse(
                    model_output,
                    target,
                    timesteps,
                    noise_scheduler,
                    None if gamma_value is None else float(gamma_value),
                )
                zero = diffusion_loss.new_zeros(())
                phi_results = {
                    "loss": zero,
                    "z0_loss": zero,
                    "mu_loss": zero,
                    "lpips_loss": zero,
                    "active_count": zero,
                    "active_fraction": zero,
                    "r_mean": zero,
                    "r_std": zero,
                    "r_min": zero,
                    "r_max": zero,
                    "mu_base_abs_mean": zero,
                    "mu_phi_abs_mean": zero,
                    "mu_target_abs_mean": zero,
                    "mu_mixed_abs_mean": zero,
                }
                base_z0: torch.Tensor | None = None
                if phi is not None:
                    alpha_bar = alpha_bar_for_timesteps(noise_scheduler, timesteps)
                    base_z0 = model_output_to_x0(
                        model_output, noisy_latents, alpha_bar, str(prediction_type)
                    )
                    normalized_t = normalized_timesteps(
                        timesteps, int(noise_scheduler.config.num_train_timesteps)
                    )
                    phi_results = compute_phi_training_losses(
                        phi=phi,
                        predicted_z0=base_z0,
                        target_z0=latents,
                        normalized_t=normalized_t,
                        target_rgb_minus_one_one=gt,
                        sample_zt=noisy_latents,
                        timesteps=timesteps,
                        scheduler=noise_scheduler,
                        vae=pipe.vae,
                        lpips_model=differentiable_lpips,
                        timestep_range=config.get("phi_train_timestep_range", [0.0, 1.0]),
                        lambda_z0=float(config.get("lambda_z0", 1.0)),
                        lambda_mu=float(config.get("lambda_mu", 1.0)),
                        lambda_lpips=float(config.get("lambda_lpips", 0.1)),
                        scaling_factor=float(pipe.vae.config.scaling_factor),
                    )
                predicted_z0_for_rgb: torch.Tensor | None = None
                if rgb_aux_loss_enabled:
                    if base_z0 is not None:
                        predicted_z0_for_rgb = base_z0
                    else:
                        alpha_bar_rgb = alpha_bar_for_timesteps(
                            noise_scheduler, timesteps
                        )
                        predicted_z0_for_rgb = model_output_to_x0(
                            model_output,
                            noisy_latents,
                            alpha_bar_rgb,
                            str(prediction_type),
                        )
                rgb_results = compute_rgb_auxiliary_losses(
                    predicted_z0=predicted_z0_for_rgb,
                    target_rgb_minus_one_one=gt,
                    vae=pipe.vae,
                    lpips_model=differentiable_lpips,
                    enabled=rgb_aux_loss_enabled,
                    lambda_l1=float(config.get("lambda_l1", 0.1)),
                    lambda_lpips_rgb=float(config.get("lambda_lpips_rgb", 0.1)),
                    scaling_factor=float(pipe.vae.config.scaling_factor),
                )
                tri_preview_loss = zero
                if tri_prepared is not None:
                    preview_weight = float(tri_config.get("preview_loss_weight", 0.0))
                    if preview_weight < 0:
                        raise ValueError("tri_input.preview_loss_weight must be non-negative")
                    if preview_weight > 0:
                        tri_preview_loss = masked_preview_l1(
                            tri_prepared.output.preview,
                            gt,
                            tri_prepared.output.active_mask,
                        ) * preview_weight
                    micro_batch = float(latents.shape[0])
                    micro_aux = float(tri_prepared.output.active_mask.float().sum().detach().item())
                    accumulation_tri_sample_count += micro_batch
                    accumulation_tri_aux_count += micro_aux
                    accumulation_tri_preview_sum += (
                        float(tri_preview_loss.detach().item()) * micro_batch
                    )
                    tri_metrics = tri_diagnostic_metrics(tri_prepared)
                    for name in accumulation_tri_diagnostic_sums:
                        accumulation_tri_diagnostic_sums[name] += float(
                            tri_metrics[name].detach().item()
                        )
                    accumulation_tri_diagnostic_batches += 1.0
                    for reason in tri_prepared.output.checker_diagnostics.fallback_reasons:
                        if reason not in seen_checker_reasons:
                            LOGGER.info(
                                "Tri-input checker diagnostic reason observed: %s "
                                "(internal heuristic, not a truth probability)",
                                reason,
                            )
                            seen_checker_reasons.add(reason)
                elif tri_enabled:
                    # Explicit synthetic_aux_policy=disable: no conditioner,
                    # checker, bridge, or preview supervision is evaluated.
                    accumulation_tri_sample_count += float(latents.shape[0])
                loss = (
                    diffusion_loss
                    + phi_results["loss"]
                    + rgb_results["loss"]
                    + tri_preview_loss
                )
                if rgb_aux_loss_enabled:
                    micro_rgb_batch = float(latents.shape[0])
                    accumulation_rgb_sample_count += micro_rgb_batch
                    accumulation_rgb_l1_sum += (
                        float(rgb_results["l1_loss"].detach().item()) * micro_rgb_batch
                    )
                    accumulation_rgb_lpips_sum += (
                        float(rgb_results["lpips_loss"].detach().item()) * micro_rgb_batch
                    )
                    accumulation_rgb_total_sum += (
                        float(rgb_results["loss"].detach().item()) * micro_rgb_batch
                    )
                if phi is not None:
                    micro_active = float(phi_results["active_count"].detach().item())
                    micro_batch = float(latents.shape[0])
                    accumulation_active_count += micro_active
                    accumulation_sample_count += micro_batch
                    accumulation_phi_z0_sum += float(phi_results["z0_loss"].detach().item()) * micro_active
                    accumulation_phi_mu_sum += float(phi_results["mu_loss"].detach().item()) * micro_active
                    accumulation_phi_lpips_sum += float(phi_results["lpips_loss"].detach().item()) * micro_active
                    accumulation_phi_total_sum += float(phi_results["loss"].detach().item()) * micro_batch
                    if micro_active > 0:
                        accumulation_r_mean_sum += float(phi_results["r_mean"].detach().item()) * micro_active
                        accumulation_r_std_sum += float(phi_results["r_std"].detach().item()) * micro_active
                        accumulation_r_min = min(accumulation_r_min, float(phi_results["r_min"].detach().item()))
                        accumulation_r_max = max(accumulation_r_max, float(phi_results["r_max"].detach().item()))
                        accumulation_mu_base_sum += float(phi_results["mu_base_abs_mean"].detach().item()) * micro_active
                        accumulation_mu_phi_sum += float(phi_results["mu_phi_abs_mean"].detach().item()) * micro_active
                        accumulation_mu_target_sum += float(phi_results["mu_target_abs_mean"].detach().item()) * micro_active
                        accumulation_mu_mixed_sum += float(phi_results["mu_mixed_abs_mean"].detach().item()) * micro_active
                if not torch.isfinite(loss):
                    raise FloatingPointError(
                        "Non-finite training loss detected before backward. "
                        f"loss={loss.detach().float().item()}, "
                        f"latents_finite={torch.isfinite(latents).all().item()}, "
                        f"prediction_finite={torch.isfinite(model_output).all().item()}, "
                        f"target_finite={torch.isfinite(target).all().item()}"
                    )
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    modules_for_clipping: list[nn.Module] = [unet, adapter]
                    if phi is not None:
                        modules_for_clipping.append(phi)
                    if tri_conditioner is not None and tri_bridge is not None:
                        modules_for_clipping.extend((tri_conditioner, tri_bridge))
                    parameters: Iterable[nn.Parameter] = [
                        parameter
                        for module in modules_for_clipping
                        for parameter in module.parameters()
                        if parameter.requires_grad
                    ]
                    accelerator.clip_grad_norm_(parameters, float(config.get("max_grad_norm", 1.0)))
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            if accelerator.sync_gradients:
                global_step += 1
                progress.update(1)
                loss_value = accelerator.gather(loss.detach().reshape(1)).mean().item()
                logs = {
                    "train_loss": loss_value,
                    "diffusion_loss": accelerator.gather(
                        diffusion_loss.detach().reshape(1)
                    ).mean().item(),
                    "lr": lr_scheduler.get_last_lr()[0],
                }
                if rgb_aux_loss_enabled:
                    gathered_rgb_count = accelerator.gather(
                        torch.tensor(
                            [accumulation_rgb_sample_count], device=latents.device
                        )
                    ).sum()

                    def gathered_rgb_sum(value: float) -> torch.Tensor:
                        return accelerator.gather(
                            torch.tensor([value], device=latents.device)
                        ).sum()

                    logs.update(
                        {
                            "rgb_l1_loss": float(
                                (
                                    gathered_rgb_sum(accumulation_rgb_l1_sum)
                                    / gathered_rgb_count.clamp_min(1)
                                ).item()
                            ),
                            "rgb_lpips_loss": float(
                                (
                                    gathered_rgb_sum(accumulation_rgb_lpips_sum)
                                    / gathered_rgb_count.clamp_min(1)
                                ).item()
                            ),
                            "rgb_aux_loss": float(
                                (
                                    gathered_rgb_sum(accumulation_rgb_total_sum)
                                    / gathered_rgb_count.clamp_min(1)
                                ).item()
                            ),
                        }
                    )
                    accumulation_rgb_sample_count = 0.0
                    accumulation_rgb_l1_sum = 0.0
                    accumulation_rgb_lpips_sum = 0.0
                    accumulation_rgb_total_sum = 0.0
                if tri_enabled:
                    gathered_tri_samples = accelerator.gather(
                        torch.tensor([accumulation_tri_sample_count], device=latents.device)
                    ).sum()
                    gathered_tri_aux = accelerator.gather(
                        torch.tensor([accumulation_tri_aux_count], device=latents.device)
                    ).sum()
                    gathered_preview_sum = accelerator.gather(
                        torch.tensor([accumulation_tri_preview_sum], device=latents.device)
                    ).sum()
                    gathered_diagnostic_batches = accelerator.gather(
                        torch.tensor(
                            [accumulation_tri_diagnostic_batches], device=latents.device
                        )
                    ).sum()

                    def gathered_tri_sum(value: float) -> torch.Tensor:
                        return accelerator.gather(
                            torch.tensor([value], device=latents.device)
                        ).sum()

                    logs.update(
                        {
                            "tri_preview_loss": float(
                                (
                                    gathered_preview_sum
                                    / gathered_tri_samples.clamp_min(1)
                                ).item()
                            ),
                            "tri_aux_present_fraction": float(
                                (
                                    gathered_tri_aux
                                    / gathered_tri_samples.clamp_min(1)
                                ).item()
                            ),
                            **{
                                name: float(
                                    (
                                        gathered_tri_sum(value)
                                        / gathered_diagnostic_batches.clamp_min(1)
                                    ).item()
                                )
                                for name, value in accumulation_tri_diagnostic_sums.items()
                            },
                        }
                    )
                    accumulation_tri_sample_count = 0.0
                    accumulation_tri_aux_count = 0.0
                    accumulation_tri_preview_sum = 0.0
                    accumulation_tri_diagnostic_batches = 0.0
                    for name in accumulation_tri_diagnostic_sums:
                        accumulation_tri_diagnostic_sums[name] = 0.0
                if phi is not None:
                    gathered_count = accelerator.gather(
                        torch.tensor([accumulation_active_count], device=latents.device)
                    ).sum()
                    gathered_batch = accelerator.gather(
                        torch.tensor([accumulation_sample_count], device=latents.device)
                    ).sum()
                    active_fraction = float((gathered_count / gathered_batch.clamp_min(1)).item())
                    def gathered_sum(value: float) -> torch.Tensor:
                        return accelerator.gather(
                            torch.tensor([value], device=latents.device)
                        ).sum()

                    has_active = bool(gathered_count.item() > 0)
                    local_r_min = accumulation_r_min if accumulation_active_count > 0 else float("inf")
                    local_r_max = accumulation_r_max if accumulation_active_count > 0 else float("-inf")
                    gathered_r_min = accelerator.gather(torch.tensor([local_r_min], device=latents.device)).min()
                    gathered_r_max = accelerator.gather(torch.tensor([local_r_max], device=latents.device)).max()

                    logs.update(
                        {
                            "phi_z0_loss": float((gathered_sum(accumulation_phi_z0_sum) / gathered_count.clamp_min(1)).item()),
                            "phi_mu_loss": float((gathered_sum(accumulation_phi_mu_sum) / gathered_count.clamp_min(1)).item()),
                            "phi_lpips_loss": float((gathered_sum(accumulation_phi_lpips_sum) / gathered_count.clamp_min(1)).item()),
                            "phi_total_loss": float((gathered_sum(accumulation_phi_total_sum) / gathered_batch.clamp_min(1)).item()),
                            "phi_active_fraction": active_fraction,
                            "phi_active_count": float(gathered_count.item()),
                            "r_t_mean": float((gathered_sum(accumulation_r_mean_sum) / gathered_count.clamp_min(1)).item()),
                            "r_t_std": float((gathered_sum(accumulation_r_std_sum) / gathered_count.clamp_min(1)).item()),
                            "r_t_min": float(gathered_r_min.item()) if has_active else 0.0,
                            "r_t_max": float(gathered_r_max.item()) if has_active else 0.0,
                            "mu_base_abs_mean": float((gathered_sum(accumulation_mu_base_sum) / gathered_count.clamp_min(1)).item()),
                            "mu_phi_abs_mean": float((gathered_sum(accumulation_mu_phi_sum) / gathered_count.clamp_min(1)).item()),
                            "mu_target_abs_mean": float((gathered_sum(accumulation_mu_target_sum) / gathered_count.clamp_min(1)).item()),
                            "mu_mixed_abs_mean": float((gathered_sum(accumulation_mu_mixed_sum) / gathered_count.clamp_min(1)).item()),
                        }
                    )
                    accumulation_active_count = 0.0
                    accumulation_sample_count = 0.0
                    accumulation_phi_z0_sum = 0.0
                    accumulation_phi_mu_sum = 0.0
                    accumulation_phi_lpips_sum = 0.0
                    accumulation_phi_total_sum = 0.0
                    accumulation_r_mean_sum = 0.0
                    accumulation_r_std_sum = 0.0
                    accumulation_r_min = float("inf")
                    accumulation_r_max = float("-inf")
                    accumulation_mu_base_sum = 0.0
                    accumulation_mu_phi_sum = 0.0
                    accumulation_mu_target_sum = 0.0
                    accumulation_mu_mixed_sum = 0.0
                progress.set_postfix(loss=f"{loss_value:.4f}")
                accelerator.log(logs, step=global_step)

                if global_step % int(config.get("checkpointing_steps", 1000)) == 0:
                    accelerator.wait_for_everyone()
                    checkpoint = output_dir / f"checkpoint-{global_step:05d}"
                    save_artifacts(
                        accelerator, unet, adapter, checkpoint, config, global_step,
                        optimizer, lr_scheduler, phi=phi,
                        tri_conditioner=tri_conditioner,
                        tri_bridge=tri_bridge,
                        tri_stats=tri_stats,
                        tri_metadata=tri_metadata,
                        data_generator=generator,
                    )
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
                        tri_conditioner=tri_conditioner,
                        tri_bridge=tri_bridge,
                    )
                    accelerator.wait_for_everyone()
            if global_step >= max_steps:
                break

    accelerator.wait_for_everyone()
    save_artifacts(
        accelerator, unet, adapter, output_dir / "final", config, global_step,
        optimizer, lr_scheduler, phi=phi,
        tri_conditioner=tri_conditioner,
        tri_bridge=tri_bridge,
        tri_stats=tri_stats,
        tri_metadata=tri_metadata,
        data_generator=generator,
    )
    accelerator.end_training()


if __name__ == "__main__":
    main()
