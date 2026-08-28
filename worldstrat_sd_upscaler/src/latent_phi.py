"""Lightweight fixed-resolution timestep-conditioned clean-latent CNN."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from src.diffusion_prediction import timestep_range_mask


PHI_WEIGHTS_NAME = "latent_phi.safetensors"
PHI_CONFIG_NAME = "latent_phi_config.json"


def sinusoidal_timestep_embedding(normalized_t: torch.Tensor, dimension: int) -> torch.Tensor:
    """Embed normalized timestep values without any learned gate or threshold."""
    if dimension < 2:
        raise ValueError(f"dimension must be at least 2, got {dimension}")
    values = normalized_t.float().reshape(-1)
    half = dimension // 2
    frequencies = torch.exp(
        -math.log(10000.0)
        * torch.arange(half, device=values.device, dtype=torch.float32)
        / max(half - 1, 1)
    )
    angles = values[:, None] * frequencies[None, :] * 1000.0
    embedding = torch.cat((angles.sin(), angles.cos()), dim=1)
    if embedding.shape[1] < dimension:
        embedding = torch.nn.functional.pad(embedding, (0, dimension - embedding.shape[1]))
    return embedding


class PhiBlock(nn.Module):
    def __init__(self, channels: int, time_embed_dim: int) -> None:
        super().__init__()
        self.norm1 = nn.GroupNorm(min(8, channels), channels)
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.time_projection = nn.Linear(time_embed_dim, channels)
        self.norm2 = nn.GroupNorm(min(8, channels), channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.activation = nn.SiLU()

    def forward(self, features: torch.Tensor, time_embedding: torch.Tensor) -> torch.Tensor:
        residual = features
        features = self.conv1(self.activation(self.norm1(features)))
        features = features + self.time_projection(time_embedding)[:, :, None, None]
        features = self.conv2(self.activation(self.norm2(features)))
        return residual + features


class LatentPhi(nn.Module):
    """Small same-resolution CNN mapping predicted z0 to supervised z0."""

    def __init__(
        self,
        latent_channels: int,
        hidden_channels: int = 64,
        time_embed_dim: int = 128,
        num_blocks: int = 3,
    ) -> None:
        super().__init__()
        if min(latent_channels, hidden_channels, time_embed_dim, num_blocks) <= 0:
            raise ValueError("All LatentPhi dimensions must be positive")
        self.config = {
            "latent_channels": int(latent_channels),
            "hidden_channels": int(hidden_channels),
            "time_embed_dim": int(time_embed_dim),
            "num_blocks": int(num_blocks),
        }
        self.time_mlp = nn.Sequential(
            nn.Linear(time_embed_dim, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )
        self.input_conv = nn.Conv2d(latent_channels, hidden_channels, 3, padding=1)
        self.blocks = nn.ModuleList(
            PhiBlock(hidden_channels, time_embed_dim) for _ in range(num_blocks)
        )
        self.output_norm = nn.GroupNorm(min(8, hidden_channels), hidden_channels)
        self.output_conv = nn.Conv2d(hidden_channels, latent_channels, 3, padding=1)
        self.activation = nn.SiLU()

    def forward(self, predicted_z0: torch.Tensor, normalized_t: torch.Tensor) -> torch.Tensor:
        if predicted_z0.ndim != 4 or predicted_z0.shape[1] != self.config["latent_channels"]:
            raise ValueError(
                f"Expected BCHW with {self.config['latent_channels']} latent channels, "
                f"got {tuple(predicted_z0.shape)}"
            )
        if normalized_t.numel() != predicted_z0.shape[0]:
            raise ValueError("normalized_t must contain one value per latent sample")
        time_embedding = sinusoidal_timestep_embedding(
            normalized_t, self.config["time_embed_dim"]
        )
        time_embedding = self.time_mlp(time_embedding)
        features = self.input_conv(predicted_z0.float())
        for block in self.blocks:
            features = block(features, time_embedding)
        return self.output_conv(self.activation(self.output_norm(features)))

    def save_pretrained(self, directory: str | Path) -> tuple[Path, Path]:
        from safetensors.torch import save_file

        output_dir = Path(directory)
        output_dir.mkdir(parents=True, exist_ok=True)
        weights_path = output_dir / PHI_WEIGHTS_NAME
        config_path = output_dir / PHI_CONFIG_NAME
        state = {name: value.detach().cpu().contiguous() for name, value in self.state_dict().items()}
        save_file(state, str(weights_path))
        config_path.write_text(json.dumps(self.config, indent=2) + "\n", encoding="utf-8")
        return weights_path, config_path

    @classmethod
    def from_pretrained(
        cls, path_or_directory: str | Path, device: Any = "cpu"
    ) -> "LatentPhi":
        from safetensors.torch import load_file

        path = Path(path_or_directory)
        directory = path if path.is_dir() else path.parent
        weights_path = path if path.is_file() and path.name == PHI_WEIGHTS_NAME else directory / PHI_WEIGHTS_NAME
        config_path = directory / PHI_CONFIG_NAME
        if not weights_path.is_file() or not config_path.is_file():
            raise FileNotFoundError(
                f"LatentPhi requires {weights_path} and {config_path}"
            )
        config = json.loads(config_path.read_text(encoding="utf-8"))
        model = cls(**config)
        model.load_state_dict(load_file(str(weights_path), device=str(device)), strict=True)
        return model.to(device)


def compute_phi_training_losses(
    phi: nn.Module,
    predicted_z0: torch.Tensor,
    target_z0: torch.Tensor,
    normalized_t: torch.Tensor,
    target_rgb_minus_one_one: torch.Tensor,
    vae: nn.Module,
    lpips_model: nn.Module | None,
    timestep_range: list[float] | tuple[float, float],
    phi_weight: float,
    lambda_z0: float,
    lambda_lpips: float,
    scaling_factor: float,
) -> dict[str, torch.Tensor]:
    """Compute only the requested hard-range φ losses with θ detached."""
    batch_size = predicted_z0.shape[0]
    active = timestep_range_mask(normalized_t, timestep_range)
    if float(phi_weight) == 0.0:
        active = torch.zeros_like(active)
    active_count = active.sum()
    zero = predicted_z0.new_zeros((), dtype=torch.float32)
    result = {
        "loss": zero,
        "z0_loss": zero,
        "lpips_loss": zero,
        "active_count": active_count.float(),
        "active_fraction": active.float().mean(),
        "weight_mean": active.float().mean() * float(phi_weight),
        "base_z0_abs_mean": predicted_z0.detach().float().abs().mean(),
        "phi_z0_abs_mean": zero,
    }
    if not bool(active.any().item()):
        return result

    base_active = predicted_z0.detach().float()[active]
    target_active = target_z0.detach().float()[active]
    normalized_active = normalized_t.float()[active]
    phi_z0 = phi(base_active, normalized_active)
    z0_per_sample = F.mse_loss(phi_z0.float(), target_active, reduction="none").mean(
        dim=tuple(range(1, phi_z0.ndim))
    )

    if lambda_lpips != 0.0:
        if lpips_model is None:
            raise RuntimeError("lambda_lpips is nonzero but no differentiable LPIPS model was provided")
        decoded = vae.decode(phi_z0.float() / float(scaling_factor), return_dict=False)[0]
        lpips_values = lpips_model(
            decoded.clamp(-1.0, 1.0),
            target_rgb_minus_one_one.float()[active].clamp(-1.0, 1.0),
        ).reshape(-1)
    else:
        lpips_values = torch.zeros_like(z0_per_sample)

    # Inactive samples are mathematically zero-weighted. Dividing by the full
    # batch preserves the requested per-sample timestep_weight definition.
    auxiliary = float(phi_weight) * (
        float(lambda_z0) * z0_per_sample + float(lambda_lpips) * lpips_values
    )
    result.update(
        {
            "loss": auxiliary.sum() / batch_size,
            "z0_loss": z0_per_sample.mean(),
            "lpips_loss": lpips_values.mean(),
            "phi_z0_abs_mean": phi_z0.float().abs().mean(),
        }
    )
    return result
