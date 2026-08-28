"""Float32 conversions between diffusion model parameterizations and clean latents."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch


SUPPORTED_PREDICTION_TYPES = {"epsilon", "v_prediction", "sample", "x0", "x_start"}


def _prediction_type(value: Any) -> str:
    prediction_type = str(value)
    if prediction_type not in SUPPORTED_PREDICTION_TYPES:
        raise ValueError(f"Unsupported scheduler prediction_type={prediction_type!r}")
    return prediction_type


def alpha_bar_for_timesteps(scheduler: Any, timesteps: torch.Tensor) -> torch.Tensor:
    """Return broadcastable alpha_bar values in float32 for a timestep batch."""
    if timesteps.ndim == 0:
        timesteps = timesteps.reshape(1)
    alphas = scheduler.alphas_cumprod.to(device=timesteps.device, dtype=torch.float32)
    selected = alphas[timesteps.long()]
    return selected.reshape((-1,) + (1,) * 3)


def model_output_to_x0(
    model_output: torch.Tensor,
    sample: torch.Tensor,
    alpha_bar: torch.Tensor,
    prediction_type: str,
) -> torch.Tensor:
    """Convert a UNet output to predicted clean latent, always in float32."""
    kind = _prediction_type(prediction_type)
    output = model_output.float()
    z_t = sample.float()
    alpha = alpha_bar.to(device=z_t.device, dtype=torch.float32)
    sqrt_alpha = alpha.sqrt()
    sqrt_beta = (1.0 - alpha).clamp_min(0.0).sqrt()
    if kind == "epsilon":
        return (z_t - sqrt_beta * output) / sqrt_alpha.clamp_min(1e-12)
    if kind == "v_prediction":
        return sqrt_alpha * z_t - sqrt_beta * output
    return output


def x0_to_model_output(
    predicted_x0: torch.Tensor,
    sample: torch.Tensor,
    alpha_bar: torch.Tensor,
    prediction_type: str,
) -> torch.Tensor:
    """Convert a clean latent to the scheduler's model-output parameterization."""
    kind = _prediction_type(prediction_type)
    x0 = predicted_x0.float()
    z_t = sample.float()
    alpha = alpha_bar.to(device=z_t.device, dtype=torch.float32)
    sqrt_alpha = alpha.sqrt()
    sqrt_beta = (1.0 - alpha).clamp_min(0.0).sqrt()
    if kind == "epsilon":
        return (z_t - sqrt_alpha * x0) / sqrt_beta.clamp_min(1e-12)
    if kind == "v_prediction":
        return (sqrt_alpha * z_t - x0) / sqrt_beta.clamp_min(1e-12)
    return x0


def normalized_timesteps(timesteps: torch.Tensor, num_train_timesteps: int) -> torch.Tensor:
    """Map t=0..T-1 to an inclusive float32 [0,1] coordinate."""
    if num_train_timesteps <= 1:
        raise ValueError(f"num_train_timesteps must exceed 1, got {num_train_timesteps}")
    return timesteps.float() / float(num_train_timesteps - 1)


def validate_timestep_range(value: Sequence[float], name: str = "timestep_range") -> tuple[float, float]:
    if len(value) != 2:
        raise ValueError(f"{name} must contain exactly [minimum, maximum], got {value!r}")
    lower, upper = float(value[0]), float(value[1])
    if not 0.0 <= lower <= upper <= 1.0:
        raise ValueError(f"{name} must satisfy 0 <= min <= max <= 1, got [{lower}, {upper}]")
    return lower, upper


def timestep_range_mask(normalized_t: torch.Tensor, value: Sequence[float]) -> torch.Tensor:
    """Inclusive hard range mask; no learned or adaptive gating."""
    lower, upper = validate_timestep_range(value)
    values = normalized_t.float()
    return (values >= lower) & (values <= upper)


def mix_x0_and_convert_model_output(
    base_x0: torch.Tensor,
    phi_x0: torch.Tensor,
    weight: float | torch.Tensor,
    sample: torch.Tensor,
    alpha_bar: torch.Tensor,
    prediction_type: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply the fixed Cas-DM-style mixture and return scheduler model output."""
    weight_tensor = torch.as_tensor(weight, device=base_x0.device, dtype=torch.float32)
    while weight_tensor.ndim < base_x0.ndim:
        weight_tensor = weight_tensor.unsqueeze(-1)
    mixed_x0 = (1.0 - weight_tensor) * base_x0.float() + weight_tensor * phi_x0.float()
    converted = x0_to_model_output(mixed_x0, sample, alpha_bar, prediction_type)
    return mixed_x0, converted
