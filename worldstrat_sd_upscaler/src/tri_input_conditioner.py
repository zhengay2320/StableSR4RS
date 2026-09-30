"""Three-input conditioner for RGB, raw Sentinel-2, and offline unmixing priors.

The layout responses and checker scores produced here are internal conditioning
features and heuristic diagnostics.  They are not semantic probabilities,
calibrated confidence, or estimates of unknown high-resolution truth.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import torch
from torch import nn

from src.local_spectral_checker import CheckerConfig, LocalSpectralChecker


TRI_INPUT_WEIGHTS_NAME = "tri_input_conditioner.safetensors"
TRI_INPUT_CONFIG_NAME = "tri_input_config.json"
TRI_INPUT_MODULE_VERSION = 1


@dataclass(frozen=True)
class CheckerDiagnostics:
    """Per-sample diagnostics from the internal finite-candidate checker."""

    valid_window_count: torch.Tensor
    accepted_window_count: torch.Tensor
    internal_error_before: torch.Tensor
    internal_error_after: torch.Tensor
    fallback_reasons: tuple[str, ...]


@dataclass(frozen=True)
class TriInputConditionOutput:
    q0: torch.Tensor
    q_star: torch.Tensor
    flow_hr: torch.Tensor
    support: torch.Tensor
    scores: torch.Tensor
    geometry: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
    context_lr: torch.Tensor
    preview: torch.Tensor
    active_mask: torch.Tensor
    checker_diagnostics: CheckerDiagnostics


def _encoder(in_channels: int, out_channels: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.GroupNorm(min(8, out_channels), out_channels),
        nn.SiLU(),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
        nn.GroupNorm(min(8, out_channels), out_channels),
        nn.SiLU(),
    )


def _pixel_shuffle_stage(channels: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(channels, channels * 4, kernel_size=3, padding=1),
        nn.PixelShuffle(2),
        nn.GroupNorm(min(8, channels), channels),
        nn.SiLU(),
    )


class TriInputConditioner(nn.Module):
    """Build multiscale diffusion conditions from three aligned LR inputs.

    ``surface_band_indices`` must explicitly identify the ten raw Sentinel-2
    surface bands used by the checker, in the exact order documented by the
    data manifest.  The two remaining raw bands still participate in the
    context encoder.
    """

    def __init__(
        self,
        raw_mean: Sequence[float] | torch.Tensor,
        raw_std: Sequence[float] | torch.Tensor,
        surface_band_indices: Sequence[int],
        components: int = 8,
        scale: int = 4,
        checker_config: CheckerConfig | dict[str, Any] | None = None,
    ) -> None:
        super().__init__()
        if int(components) != 8:
            raise ValueError(f"The V1 conditioner requires components=8, got {components}")
        if int(scale) != 4:
            raise ValueError(f"The two PixelShuffle stages require scale=4, got {scale}")

        indices = tuple(int(index) for index in surface_band_indices)
        if len(indices) != 10 or len(set(indices)) != 10:
            raise ValueError("surface_band_indices must contain exactly ten unique indices")
        if min(indices) < 0 or max(indices) >= 12:
            raise ValueError("surface_band_indices must lie in the inclusive range [0, 11]")

        mean = torch.as_tensor(raw_mean, dtype=torch.float32).reshape(-1)
        std = torch.as_tensor(raw_std, dtype=torch.float32).reshape(-1)
        if mean.numel() != 12 or std.numel() != 12:
            raise ValueError("raw_mean and raw_std must each contain exactly 12 values")
        if not bool(torch.isfinite(mean).all()) or not bool(torch.isfinite(std).all()):
            raise ValueError("raw_mean and raw_std must be finite")
        if bool((std <= 0).any()):
            raise ValueError("raw_std must be strictly positive")

        checker = (
            checker_config
            if isinstance(checker_config, CheckerConfig)
            else CheckerConfig.from_mapping(checker_config)
        )
        if checker.scale != int(scale):
            raise ValueError(
                f"checker scale={checker.scale} does not match conditioner scale={scale}"
            )

        self.config: dict[str, Any] = {
            "module_version": TRI_INPUT_MODULE_VERSION,
            "components": int(components),
            "scale": int(scale),
            "surface_band_indices": list(indices),
            "checker": checker.to_dict(),
        }
        # Additional artifact provenance (band order, value conversion,
        # statistics source, injection contract, and parent RGB checkpoint) is
        # populated by the training entry point.  Unknown metadata is retained
        # across strict load/save cycles but never changes the architecture.
        self.artifact_metadata: dict[str, Any] = {}
        self.surface_band_indices = indices
        self.register_buffer("raw_mean", mean.reshape(1, 12, 1, 1), persistent=True)
        self.register_buffer("raw_std", std.reshape(1, 12, 1, 1), persistent=True)

        self.rgb_encoder = _encoder(3, 32)
        self.raw_encoder = _encoder(12, 48)
        self.prior_encoder = _encoder(15, 32)
        self.fusion = nn.Sequential(
            nn.Conv2d(112, 64, kernel_size=3, padding=1),
            nn.GroupNorm(8, 64),
            nn.SiLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.GroupNorm(8, 64),
            nn.SiLU(),
        )
        self.upsample_1 = _pixel_shuffle_stage(64)
        self.upsample_2 = _pixel_shuffle_stage(64)
        self.layout_head = nn.Conv2d(64, 8, kernel_size=3, padding=1)
        self.preview_head = nn.Conv2d(64, 3, kernel_size=3, padding=1)

        # Q*, Q*-Q0, flow, and support contain 8+8+2+1=19 channels.
        self.geometry_hr = _encoder(19, 32)
        self.geometry_half = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, 64),
            nn.SiLU(),
        )
        self.geometry_quarter = nn.Sequential(
            nn.Conv2d(64, 96, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, 96),
            nn.SiLU(),
        )
        self.geometry_eighth = nn.Sequential(
            nn.Conv2d(96, 128, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, 128),
            nn.SiLU(),
        )
        self.checker = LocalSpectralChecker(checker)

    @staticmethod
    def _active_mask(aux_present: torch.Tensor | bool, batch_size: int, device: torch.device) -> torch.Tensor:
        value = torch.as_tensor(aux_present, device=device)
        if value.ndim == 0:
            value = value.expand(batch_size)
        else:
            value = value.reshape(batch_size, -1)
            if value.shape[1] != 1:
                raise ValueError("aux_present must provide exactly one value per sample")
            value = value[:, 0]
        if value.numel() != batch_size:
            raise ValueError("aux_present must provide exactly one value per sample")
        return value.bool()

    @staticmethod
    def _validate_inputs(
        rgb: torch.Tensor,
        raw_ms: torch.Tensor,
        unmixing: torch.Tensor,
        raw_valid: torch.Tensor,
    ) -> None:
        if rgb.ndim != 4 or rgb.shape[1] != 3:
            raise ValueError(f"rgb must have shape [B,3,H,W], got {tuple(rgb.shape)}")
        expected_raw = (rgb.shape[0], 12, rgb.shape[2], rgb.shape[3])
        expected_unmixing = (rgb.shape[0], 10, rgb.shape[2], rgb.shape[3])
        expected_valid = (rgb.shape[0], 1, rgb.shape[2], rgb.shape[3])
        if tuple(raw_ms.shape) != expected_raw:
            raise ValueError(f"raw_ms must have shape {expected_raw}, got {tuple(raw_ms.shape)}")
        if tuple(unmixing.shape) != expected_unmixing:
            raise ValueError(
                f"unmixing must have shape {expected_unmixing}, got {tuple(unmixing.shape)}"
            )
        if tuple(raw_valid.shape) != expected_valid:
            raise ValueError(
                f"raw_valid must have shape {expected_valid}, got {tuple(raw_valid.shape)}"
            )
        if unmixing.dtype != torch.float32:
            raise TypeError(f"unmixing must be float32, got {unmixing.dtype}")
        for name, tensor in (("rgb", rgb), ("raw_ms", raw_ms), ("unmixing", unmixing)):
            if not bool(torch.isfinite(tensor).all()):
                raise ValueError(f"{name} contains NaN or Inf")
        if bool((unmixing < 0).any()) or bool((unmixing > 1).any()):
            raise ValueError("unmixing values must lie in [0,1]")

    def forward(
        self,
        rgb: torch.Tensor,
        raw_ms: torch.Tensor,
        unmixing: torch.Tensor,
        raw_valid: torch.Tensor,
        aux_present: torch.Tensor | bool = True,
    ) -> TriInputConditionOutput:
        self._validate_inputs(rgb, raw_ms, unmixing, raw_valid)
        batch_size = rgb.shape[0]
        active_mask = self._active_mask(aux_present, batch_size, rgb.device)
        active_spatial = active_mask[:, None, None, None].to(dtype=torch.float32)
        valid = raw_valid.bool()
        valid_float = valid.to(dtype=torch.float32)

        rgb_features = self.rgb_encoder(rgb.float())
        normalized_raw = (raw_ms.float() - self.raw_mean) / self.raw_std
        raw_features = self.raw_encoder(normalized_raw * valid_float)
        raw_features = raw_features * valid_float * active_spatial

        # Offline priors are immutable conditioning data.  F is weakly weighted
        # by (1-U); U is not interpreted as variance or calibrated confidence.
        detached = unmixing.detach()
        fractions, uncertainty = detached[:, :5], detached[:, 5:]
        confidence = (1.0 - uncertainty).clamp(0.0, 1.0)
        use_marker = (uncertainty < 1.0 - 1.0e-6).to(dtype=torch.float32)
        prior_valid = valid_float * active_spatial
        use_marker = use_marker * prior_valid
        prior_input = torch.cat(
            (
                fractions * confidence * prior_valid,
                uncertainty * prior_valid,
                use_marker,
            ),
            dim=1,
        )
        prior_features = self.prior_encoder(prior_input)
        # F=0,U=1 means unknown, not zero cover. Suppress only the prior path;
        # raw context remains active and independent of this gate.
        prior_features = prior_features * use_marker.amax(dim=1, keepdim=True)

        context_lr = self.fusion(torch.cat((rgb_features, raw_features, prior_features), dim=1))
        hr_features = self.upsample_2(self.upsample_1(context_lr))
        q0 = self.layout_head(hr_features).softmax(dim=1)

        surface_observation = raw_ms[:, self.surface_band_indices].float()
        checker_valid = valid & active_mask[:, None, None, None]
        checked = self.checker(q0, surface_observation, checker_valid)
        q_star = checked.q_star

        geometry_input = torch.cat(
            (q_star, q_star - q0, checked.flow_hr, checked.support), dim=1
        )
        geometry_0 = self.geometry_hr(geometry_input)
        geometry_1 = self.geometry_half(geometry_0)
        geometry_2 = self.geometry_quarter(geometry_1)
        geometry_3 = self.geometry_eighth(geometry_2)
        preview = self.preview_head(hr_features).tanh()

        return TriInputConditionOutput(
            q0=q0,
            q_star=q_star,
            flow_hr=checked.flow_hr,
            support=checked.support,
            scores=checked.scores,
            geometry=(geometry_0, geometry_1, geometry_2, geometry_3),
            context_lr=context_lr,
            preview=preview,
            active_mask=active_mask,
            checker_diagnostics=CheckerDiagnostics(
                valid_window_count=checked.valid_window_count,
                accepted_window_count=checked.accepted_window_count,
                internal_error_before=checked.internal_error_before,
                internal_error_after=checked.internal_error_after,
                fallback_reasons=checked.fallback_reasons,
            ),
        )

    def save_pretrained(
        self,
        directory: str | Path,
        *,
        artifact_metadata: dict[str, Any] | None = None,
    ) -> tuple[Path, Path]:
        from safetensors.torch import save_file

        output_dir = Path(directory)
        output_dir.mkdir(parents=True, exist_ok=True)
        weights_path = output_dir / TRI_INPUT_WEIGHTS_NAME
        config_path = output_dir / TRI_INPUT_CONFIG_NAME
        state = {
            name: value.detach().cpu().contiguous()
            for name, value in self.state_dict().items()
        }
        save_file(state, str(weights_path))
        config = dict(self.config)
        config["raw_mean"] = self.raw_mean.flatten().detach().cpu().tolist()
        config["raw_std"] = self.raw_std.flatten().detach().cpu().tolist()
        merged_metadata = dict(self.artifact_metadata)
        if artifact_metadata:
            merged_metadata.update(artifact_metadata)
        if merged_metadata:
            config["artifact_metadata"] = merged_metadata
        config_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
        return weights_path, config_path

    @classmethod
    def from_pretrained(
        cls, path_or_directory: str | Path, device: Any = "cpu"
    ) -> "TriInputConditioner":
        from safetensors.torch import load_file

        path = Path(path_or_directory)
        directory = path if path.is_dir() else path.parent
        weights_path = (
            path
            if path.is_file() and path.name == TRI_INPUT_WEIGHTS_NAME
            else directory / TRI_INPUT_WEIGHTS_NAME
        )
        config_path = directory / TRI_INPUT_CONFIG_NAME
        if not weights_path.is_file() or not config_path.is_file():
            raise FileNotFoundError(
                f"TriInputConditioner requires {weights_path} and {config_path}"
            )
        config = json.loads(config_path.read_text(encoding="utf-8"))
        if config.get("module_version") != TRI_INPUT_MODULE_VERSION:
            raise ValueError(
                f"Unsupported tri-input module version {config.get('module_version')!r}; "
                f"expected {TRI_INPUT_MODULE_VERSION}"
            )
        required = {
            "raw_mean",
            "raw_std",
            "surface_band_indices",
            "components",
            "scale",
            "checker",
        }
        missing = sorted(required.difference(config))
        if missing:
            raise ValueError(f"{config_path} is missing required fields: {missing}")
        model = cls(
            raw_mean=config["raw_mean"],
            raw_std=config["raw_std"],
            surface_band_indices=config["surface_band_indices"],
            components=int(config["components"]),
            scale=int(config["scale"]),
            checker_config=config["checker"],
        )
        try:
            state = load_file(str(weights_path), device=str(device))
            model.load_state_dict(state, strict=True)
        except RuntimeError as error:
            raise RuntimeError(
                f"TriInputConditioner checkpoint is structurally incompatible: {error}"
            ) from error
        metadata = config.get("artifact_metadata", {})
        if metadata is not None and not isinstance(metadata, dict):
            raise ValueError(f"{config_path} artifact_metadata must be an object")
        model.artifact_metadata = dict(metadata or {})
        return model.to(device)
