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


def posterior_mean_from_x0(
    sample_zt: torch.Tensor,
    predicted_z0: torch.Tensor,
    timesteps: torch.Tensor,
    scheduler: Any,
) -> torch.Tensor:
    """Compute q(z_{t-1} | z_t, predicted_z0)'s mean in float32.

    This is the adjacent training posterior used by L_mu. It is intentionally
    separate from a DDIM inference step, which may jump over timesteps.
    """
    if sample_zt.shape != predicted_z0.shape:
        raise ValueError(
            "sample_zt and predicted_z0 must have identical shapes, got "
            f"{tuple(sample_zt.shape)} and {tuple(predicted_z0.shape)}"
        )
    timestep_batch = timesteps.long().reshape(-1)
    if timestep_batch.numel() != sample_zt.shape[0]:
        raise ValueError("timesteps must contain one value per sample")

    alphas_cumprod = scheduler.alphas_cumprod.to(
        device=sample_zt.device, dtype=torch.float32
    )
    if bool((timestep_batch < 0).any()) or bool(
        (timestep_batch >= alphas_cumprod.numel()).any()
    ):
        raise IndexError("timesteps contain an index outside scheduler.alphas_cumprod")

    alpha_bar_t = alphas_cumprod[timestep_batch]
    previous_indices = (timestep_batch - 1).clamp_min(0)
    alpha_bar_previous = alphas_cumprod[previous_indices]
    alpha_bar_previous = torch.where(
        timestep_batch > 0,
        alpha_bar_previous,
        torch.ones_like(alpha_bar_previous),
    )
    alpha_t = alpha_bar_t / alpha_bar_previous
    beta_t = 1.0 - alpha_t
    denominator = (1.0 - alpha_bar_t).clamp_min(torch.finfo(torch.float32).eps)
    coefficient_z0 = alpha_bar_previous.sqrt() * beta_t / denominator
    coefficient_zt = (
        alpha_t.clamp_min(0.0).sqrt()
        * (1.0 - alpha_bar_previous)
        / denominator
    )
    broadcast_shape = (-1,) + (1,) * (sample_zt.ndim - 1)
    return (
        coefficient_z0.reshape(broadcast_shape) * predicted_z0.float()
        + coefficient_zt.reshape(broadcast_shape) * sample_zt.float()
    )


def mix_reverse_means(
    base_mean: torch.Tensor,
    phi_mean: torch.Tensor,
    mixing_weight: torch.Tensor,
) -> torch.Tensor:
    """Mix two reverse means using a learned single-channel spatial r_t."""
    if base_mean.shape != phi_mean.shape:
        raise ValueError("base_mean and phi_mean must have identical shapes")
    if mixing_weight.ndim != base_mean.ndim:
        raise ValueError("mixing_weight must have the same rank as the latent tensors")
    if mixing_weight.shape[0] != base_mean.shape[0] or mixing_weight.shape[1] != 1:
        raise ValueError("mixing_weight must have shape [B, 1, ...]")
    if mixing_weight.shape[2:] != base_mean.shape[2:]:
        raise ValueError("mixing_weight spatial dimensions must match the latent tensors")
    r_t = mixing_weight.float()
    return r_t * phi_mean.float() + (1.0 - r_t) * base_mean.float()
