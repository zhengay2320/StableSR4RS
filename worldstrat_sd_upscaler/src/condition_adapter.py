"""Lightweight residual domain adapter for low-resolution conditions."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file
from torch import nn


class ConditionAdapter(nn.Module):
    """Adapt Sentinel-2 RGB conditions while starting as an identity mapping."""

    def __init__(self, adapter_scale: float = 1.0) -> None:
        super().__init__()
        self.adapter_scale = float(adapter_scale)
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(32, 3, kernel_size=3, padding=1),
        )
        final = self.net[-1]
        if not isinstance(final, nn.Conv2d):
            raise TypeError("ConditionAdapter final layer must be Conv2d")
        nn.init.zeros_(final.weight)
        nn.init.zeros_(final.bias)

    def forward(self, lr_image: torch.Tensor) -> torch.Tensor:
        """Return the clamped residual adaptation in [-1, 1]."""
        if lr_image.ndim != 4 or lr_image.shape[1] != 3:
            raise ValueError(f"Expected BCHW RGB input, got {tuple(lr_image.shape)}")
        return (lr_image + self.adapter_scale * self.net(lr_image)).clamp(-1.0, 1.0)

    def save_pretrained(self, directory: str | Path) -> Path:
        """Save weights and scale to ``condition_adapter.safetensors``."""
        output_dir = Path(directory)
        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / "condition_adapter.safetensors"
        state = {name: value.detach().cpu().contiguous() for name, value in self.state_dict().items()}
        save_file(state, str(path), metadata={"adapter_scale": str(self.adapter_scale)})
        return path

    @classmethod
    def from_pretrained(
        cls, path_or_directory: str | Path, adapter_scale: float | None = None, device: Any = "cpu"
    ) -> "ConditionAdapter":
        """Load adapter weights from a file or artifact directory."""
        path = Path(path_or_directory)
        if path.is_dir():
            path = path / "condition_adapter.safetensors"
        if not path.is_file():
            raise FileNotFoundError(f"ConditionAdapter weights not found: {path}")
        if adapter_scale is None:
            with safe_open(str(path), framework="pt", device="cpu") as handle:
                metadata = handle.metadata() or {}
            scale = float(metadata.get("adapter_scale", 1.0))
        else:
            scale = float(adapter_scale)
        adapter = cls(adapter_scale=scale)
        state = load_file(str(path), device=str(device))
        adapter.load_state_dict(state, strict=True)
        return adapter.to(device)
