"""Direct RGB supervision for a diffusion model's predicted clean latent."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


def compute_rgb_auxiliary_losses(
    predicted_z0: torch.Tensor | None,
    target_rgb_minus_one_one: torch.Tensor,
    vae: nn.Module,
    lpips_model: nn.Module | None,
    enabled: bool,
    lambda_l1: float,
    lambda_lpips_rgb: float,
    scaling_factor: float,
) -> dict[str, torch.Tensor]:
    """Decode predicted z0 and compute optional direct RGB supervision losses."""
    zero = target_rgb_minus_one_one.new_zeros((), dtype=torch.float32)
    result = {"loss": zero, "l1_loss": zero, "lpips_loss": zero}
    if not enabled:
        return result
    if predicted_z0 is None:
        raise ValueError("predicted_z0 is required when RGB auxiliary loss is enabled")

    # The VAE is frozen, but this decode must remain autograd-enabled so RGB
    # supervision reaches predicted_z0 and the trainable theta/adapter path.
    predicted_rgb = vae.decode(
        predicted_z0.float() / float(scaling_factor), return_dict=False
    )[0]
    l1_loss = F.l1_loss(
        predicted_rgb.float(), target_rgb_minus_one_one.float()
    )
    if float(lambda_lpips_rgb) != 0.0:
        if lpips_model is None:
            raise RuntimeError(
                "lambda_lpips_rgb is nonzero but no differentiable LPIPS model was provided"
            )
        lpips_loss = lpips_model(
            predicted_rgb.float().clamp(-1.0, 1.0),
            target_rgb_minus_one_one.float().clamp(-1.0, 1.0),
        ).mean()
    else:
        lpips_loss = zero
    auxiliary_loss = (
        float(lambda_l1) * l1_loss
        + float(lambda_lpips_rgb) * lpips_loss
    )
    result.update(
        {"loss": auxiliary_loss, "l1_loss": l1_loss, "lpips_loss": lpips_loss}
    )
    return result
