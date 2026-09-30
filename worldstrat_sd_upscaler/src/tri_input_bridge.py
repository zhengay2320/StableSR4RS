"""Zero-initialized spatial residual bridge for tri-input conditioning.

This module deliberately uses the public ``UNet2DConditionModel``
``down_intrablock_additional_residuals`` interface used by Diffusers'
T2I-Adapter pipelines.  It does not install hooks and it does not retain a
condition between calls.  A condition is prepared once per image/tile as an
immutable tuple; :class:`TriConditionedUNet` creates a fresh, disposable list
for every UNet invocation because the pinned Diffusers implementation consumes
that list with ``pop(0)``.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence, runtime_checkable

import torch
import torch.nn.functional as F
from safetensors.torch import load_file, save_file
from torch import nn


TRI_RESIDUALS_CROSS_ATTENTION_KEY = "_tri_input_down_intrablock_residuals"
BRIDGE_WEIGHTS_NAME = "tri_input_bridges.safetensors"
BRIDGE_CONFIG_NAME = "tri_input_bridge_config.json"
BRIDGE_FORMAT_VERSION = 1
_SUPPORTED_DOWN_BLOCKS = {"CrossAttnDownBlock2D", "DownBlock2D"}


@runtime_checkable
class TriConditioningProtocol(Protocol):
    """Minimal structural contract expected from a tri-input conditioner."""

    geometry: Sequence[torch.Tensor]
    context_lr: torch.Tensor


@dataclass(frozen=True)
class DownsampleShapeSpec:
    """Serializable spatial part of one standard Diffusers downsampler."""

    kind: str
    kernel_size: tuple[int, int]
    stride: tuple[int, int]
    padding: tuple[int, int]
    dilation: tuple[int, int]
    ceil_mode: bool = False
    prepad_right_bottom: int = 0

    def output_size(self, height: int, width: int) -> tuple[int, int]:
        if height <= 0 or width <= 0:
            raise ValueError(f"Spatial dimensions must be positive, got {(height, width)}")

        def one(value: int, axis: int) -> int:
            value += self.prepad_right_bottom
            numerator = (
                value
                + 2 * self.padding[axis]
                - self.dilation[axis] * (self.kernel_size[axis] - 1)
                - 1
            )
            raw = numerator / self.stride[axis] + 1
            result = math.ceil(raw) if self.ceil_mode else math.floor(raw)
            if result <= 0:
                raise ValueError(
                    "Downsampler produced a non-positive spatial size: "
                    f"input={(height, width)}, spec={self}"
                )
            return result

        return one(height, 0), one(width, 1)


@dataclass(frozen=True)
class DownBlockLayout:
    """The residual target contract of one supported UNet down block."""

    block_type: str
    out_channels: int
    has_cross_attention: bool
    downsample: DownsampleShapeSpec | None


@dataclass(frozen=True)
class PreparedTriResiduals:
    """Immutable residual cache safe to reuse across denoising timesteps."""

    residuals: tuple[torch.Tensor, ...]
    base_batch_size: int
    num_images_per_prompt: int
    do_classifier_free_guidance: bool

    @property
    def expanded_batch_size(self) -> int:
        multiplier = 2 if self.do_classifier_free_guidance else 1
        return self.base_batch_size * self.num_images_per_prompt * multiplier


def _pair(value: int | Sequence[int]) -> tuple[int, int]:
    if isinstance(value, Sequence):
        values = tuple(int(item) for item in value)
        if len(values) != 2:
            raise ValueError(f"Expected a scalar or pair, got {value!r}")
        return values
    return int(value), int(value)


def _downsample_shape_spec(module: nn.Module) -> DownsampleShapeSpec:
    """Inspect a standard Diffusers downsampler without executing it."""

    operator: nn.Module = getattr(module, "conv", module)
    prepad = 0
    # Diffusers Downsample2D explicitly pads right/bottom when convolutional
    # padding is zero (see its forward implementation).
    if type(module).__name__ == "Downsample2D" and bool(getattr(module, "use_conv", False)):
        if int(getattr(module, "padding", 1)) == 0:
            prepad = 1

    if isinstance(operator, nn.Conv2d):
        return DownsampleShapeSpec(
            kind="conv2d",
            kernel_size=_pair(operator.kernel_size),
            stride=_pair(operator.stride),
            padding=_pair(operator.padding),
            dilation=_pair(operator.dilation),
            prepad_right_bottom=prepad,
        )
    if isinstance(operator, (nn.AvgPool2d, nn.MaxPool2d)):
        stride = operator.stride if operator.stride is not None else operator.kernel_size
        return DownsampleShapeSpec(
            kind=type(operator).__name__.lower(),
            kernel_size=_pair(operator.kernel_size),
            stride=_pair(stride),
            padding=_pair(operator.padding),
            dilation=_pair(getattr(operator, "dilation", 1)),
            ceil_mode=bool(getattr(operator, "ceil_mode", False)),
        )
    raise TypeError(
        "Unsupported UNet downsampler for tri-input residual shape inference: "
        f"{type(module).__name__} (operator={type(operator).__name__}). "
        "Only standard Conv2d/AvgPool2d/MaxPool2d downsamplers are supported."
    )


def inspect_unet_down_blocks(unet: nn.Module) -> tuple[DownBlockLayout, ...]:
    """Build and validate the four-scale residual layout of a runtime UNet."""

    down_blocks = getattr(unet, "down_blocks", None)
    config = getattr(unet, "config", None)
    channels = getattr(config, "block_out_channels", None)
    if down_blocks is None or channels is None:
        raise TypeError("UNet must expose down_blocks and config.block_out_channels")
    if len(down_blocks) != 4 or len(channels) != 4:
        raise ValueError(
            "TriInputBridge V1 requires exactly four UNet down blocks/scales; "
            f"got down_blocks={len(down_blocks)}, block_out_channels={len(channels)}"
        )

    layouts: list[DownBlockLayout] = []
    for index, (block, out_channels) in enumerate(zip(down_blocks, channels, strict=True)):
        block_type = type(block).__name__
        if block_type not in _SUPPORTED_DOWN_BLOCKS:
            raise TypeError(
                f"Unsupported UNet down block at index {index}: {block_type}; "
                f"supported={sorted(_SUPPORTED_DOWN_BLOCKS)}"
            )
        has_cross_attention = bool(getattr(block, "has_cross_attention", False))
        expected_cross_attention = block_type == "CrossAttnDownBlock2D"
        if has_cross_attention != expected_cross_attention:
            raise ValueError(
                f"UNet down block {index} has inconsistent type/flag: "
                f"type={block_type}, has_cross_attention={has_cross_attention}"
            )
        downsamplers = getattr(block, "downsamplers", None)
        downsample: DownsampleShapeSpec | None = None
        if downsamplers is not None:
            if len(downsamplers) != 1:
                raise ValueError(
                    f"UNet down block {index} must have zero or one downsampler, got {len(downsamplers)}"
                )
            downsample = _downsample_shape_spec(downsamplers[0])
        layouts.append(
            DownBlockLayout(
                block_type=block_type,
                out_channels=int(out_channels),
                has_cross_attention=has_cross_attention,
                downsample=downsample,
            )
        )
    return tuple(layouts)


def residual_target_sizes(
    layouts: Sequence[DownBlockLayout], sample_height: int, sample_width: int
) -> tuple[tuple[int, int], ...]:
    """Return the exact target size at which each V1 residual is consumed.

    In pinned Diffusers, ``CrossAttnDownBlock2D`` consumes its additional
    residual before its downsampler.  ``DownBlock2D`` receives no intrablock
    argument, so UNet2DConditionModel adds the residual to the returned sample,
    after that block's downsampler when present.
    """

    current = (int(sample_height), int(sample_width))
    targets: list[tuple[int, int]] = []
    for layout in layouts:
        downsampled = layout.downsample.output_size(*current) if layout.downsample is not None else current
        targets.append(current if layout.has_cross_attention else downsampled)
        current = downsampled
    return tuple(targets)


def _field(conditioning: TriConditioningProtocol | Mapping[str, Any], name: str) -> Any:
    if isinstance(conditioning, Mapping):
        if name not in conditioning:
            raise KeyError(f"Tri-input conditioner output is missing {name!r}")
        return conditioning[name]
    if not hasattr(conditioning, name):
        raise AttributeError(f"Tri-input conditioner output is missing {name!r}")
    return getattr(conditioning, name)


def _optional_field(conditioning: TriConditioningProtocol | Mapping[str, Any], names: Sequence[str]) -> Any:
    for name in names:
        if isinstance(conditioning, Mapping):
            if name in conditioning:
                return conditioning[name]
        elif hasattr(conditioning, name):
            return getattr(conditioning, name)
    return None


def _mask_for_target(
    mask: torch.Tensor | None,
    *,
    batch_size: int,
    target_size: tuple[int, int],
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    if mask is None:
        return torch.ones((batch_size, 1, *target_size), device=device, dtype=dtype)
    if not isinstance(mask, torch.Tensor):
        mask = torch.as_tensor(mask)
    if mask.ndim == 1:
        mask = mask[:, None, None, None]
    elif mask.ndim == 2 and mask.shape[1] == 1:
        mask = mask[:, :, None, None]
    elif mask.ndim == 3:
        mask = mask[:, None]
    elif mask.ndim != 4 or mask.shape[1] != 1:
        raise ValueError(
            "active_mask/aux_present must have shape [B], [B,1], [B,H,W], or [B,1,H,W], "
            f"got {tuple(mask.shape)}"
        )
    if mask.shape[0] != batch_size:
        raise ValueError(f"Active mask batch {mask.shape[0]} does not match condition batch {batch_size}")
    mask = mask.to(device=device, dtype=dtype)
    if mask.shape[-2:] != target_size:
        mask = F.interpolate(mask, size=target_size, mode="nearest")
    return mask.clamp(0, 1)


class _ZeroResidualProjection(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1),
            nn.SiLU(),
        )
        self.zero_conv = nn.Conv2d(hidden_channels, out_channels, kernel_size=1)
        nn.init.zeros_(self.zero_conv.weight)
        nn.init.zeros_(self.zero_conv.bias)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.zero_conv(self.features(value))


class TriInputBridge(nn.Module):
    """Project multi-scale geometry and LR spectral context into UNet residuals."""

    def __init__(
        self,
        layouts: Sequence[DownBlockLayout],
        geometry_channels: Sequence[int],
        context_channels: int,
        hidden_channels: int | Sequence[int] = 64,
    ) -> None:
        super().__init__()
        self.layouts = tuple(layouts)
        self.geometry_channels = tuple(int(value) for value in geometry_channels)
        self.context_channels = int(context_channels)
        if len(self.layouts) != 4 or len(self.geometry_channels) != 4:
            raise ValueError(
                "TriInputBridge V1 is four-scale: layouts and geometry_channels must both have length 4"
            )
        if self.context_channels <= 0 or any(value <= 0 for value in self.geometry_channels):
            raise ValueError("All geometry/context channel counts must be positive")
        if isinstance(hidden_channels, int):
            hidden = (int(hidden_channels),) * 4
        else:
            hidden = tuple(int(value) for value in hidden_channels)
        if len(hidden) != 4 or any(value <= 0 for value in hidden):
            raise ValueError("hidden_channels must be a positive scalar or a four-item sequence")
        self.hidden_channels = hidden
        self.projections = nn.ModuleList(
            _ZeroResidualProjection(
                geometry_channels[index] + self.context_channels,
                hidden[index],
                layout.out_channels,
            )
            for index, layout in enumerate(self.layouts)
        )

    @classmethod
    def from_unet(
        cls,
        unet: nn.Module,
        geometry_channels: Sequence[int],
        context_channels: int,
        hidden_channels: int | Sequence[int] = 64,
    ) -> "TriInputBridge":
        return cls(
            inspect_unet_down_blocks(unet),
            geometry_channels=geometry_channels,
            context_channels=context_channels,
            hidden_channels=hidden_channels,
        )

    def validate_unet(self, unet: nn.Module) -> None:
        runtime = inspect_unet_down_blocks(unet)
        if runtime != self.layouts:
            raise ValueError(
                "TriInputBridge layout does not match the runtime UNet. "
                f"checkpoint={self.layouts}, runtime={runtime}"
            )

    def forward(
        self,
        conditioning: TriConditioningProtocol | Mapping[str, Any],
        sample: torch.Tensor,
        active_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, ...]:
        if sample.ndim != 4:
            raise ValueError(f"UNet sample must be BCHW, got {tuple(sample.shape)}")
        geometry_value = _field(conditioning, "geometry")
        context = _field(conditioning, "context_lr")
        if isinstance(geometry_value, torch.Tensor):
            geometry = (geometry_value,) * 4
        else:
            geometry = tuple(geometry_value)
        if len(geometry) != 4:
            raise ValueError(f"Expected four geometry scales, got {len(geometry)}")
        if not isinstance(context, torch.Tensor) or context.ndim != 4:
            raise ValueError("context_lr must be a BCHW tensor")
        batch_size = context.shape[0]
        if context.shape[1] != self.context_channels:
            raise ValueError(
                f"context_lr has {context.shape[1]} channels; expected {self.context_channels}"
            )
        if active_mask is None:
            active_mask = _optional_field(conditioning, ("active_mask", "aux_present"))
        targets = residual_target_sizes(self.layouts, sample.shape[-2], sample.shape[-1])

        residuals: list[torch.Tensor] = []
        for index, (geometry_stage, projection, target_size) in enumerate(
            zip(geometry, self.projections, targets, strict=True)
        ):
            if not isinstance(geometry_stage, torch.Tensor) or geometry_stage.ndim != 4:
                raise ValueError(f"geometry[{index}] must be a BCHW tensor")
            if geometry_stage.shape[:2] != (batch_size, self.geometry_channels[index]):
                raise ValueError(
                    f"geometry[{index}] has shape {tuple(geometry_stage.shape)}; expected batch/channels "
                    f"({batch_size}, {self.geometry_channels[index]})"
                )
            if geometry_stage.device != context.device:
                raise ValueError(
                    f"geometry[{index}] device {geometry_stage.device} differs from context device {context.device}"
                )
            geometry_scaled = F.interpolate(
                geometry_stage, size=target_size, mode="bilinear", align_corners=False
            )
            context_scaled = F.interpolate(context, size=target_size, mode="bilinear", align_corners=False)
            context_scaled = context_scaled.to(dtype=geometry_scaled.dtype)
            fused = torch.cat([geometry_scaled, context_scaled], dim=1)
            residual = projection(fused).to(device=sample.device, dtype=sample.dtype)
            mask = _mask_for_target(
                active_mask,
                batch_size=batch_size,
                target_size=target_size,
                device=residual.device,
                dtype=residual.dtype,
            )
            residuals.append(residual * mask)
        return tuple(residuals)

    def _config_dict(self) -> dict[str, Any]:
        return {
            "format_version": BRIDGE_FORMAT_VERSION,
            "geometry_channels": list(self.geometry_channels),
            "context_channels": self.context_channels,
            "hidden_channels": list(self.hidden_channels),
            "layouts": [asdict(layout) for layout in self.layouts],
        }

    def save_pretrained(self, directory: str | Path) -> tuple[Path, Path]:
        output_dir = Path(directory)
        output_dir.mkdir(parents=True, exist_ok=True)
        weights_path = output_dir / BRIDGE_WEIGHTS_NAME
        config_path = output_dir / BRIDGE_CONFIG_NAME
        state = {name: tensor.detach().cpu().contiguous() for name, tensor in self.state_dict().items()}
        save_file(state, str(weights_path), metadata={"format_version": str(BRIDGE_FORMAT_VERSION)})
        config_path.write_text(
            json.dumps(self._config_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        return weights_path, config_path

    @classmethod
    def from_pretrained(
        cls,
        path_or_directory: str | Path,
        *,
        unet: nn.Module | None = None,
        device: str | torch.device = "cpu",
    ) -> "TriInputBridge":
        path = Path(path_or_directory)
        if path.is_dir():
            weights_path = path / BRIDGE_WEIGHTS_NAME
            config_path = path / BRIDGE_CONFIG_NAME
        else:
            weights_path = path
            config_path = path.with_name(BRIDGE_CONFIG_NAME)
        if not weights_path.is_file():
            raise FileNotFoundError(f"Tri-input bridge weights not found: {weights_path}")
        if not config_path.is_file():
            raise FileNotFoundError(f"Tri-input bridge config not found: {config_path}")
        config = json.loads(config_path.read_text(encoding="utf-8"))
        if int(config.get("format_version", -1)) != BRIDGE_FORMAT_VERSION:
            raise ValueError(
                f"Unsupported tri-input bridge format_version={config.get('format_version')!r}; "
                f"expected {BRIDGE_FORMAT_VERSION}"
            )
        layouts = []
        for raw_layout in config["layouts"]:
            raw_downsample = raw_layout.get("downsample")
            downsample = (
                DownsampleShapeSpec(
                    kind=str(raw_downsample["kind"]),
                    kernel_size=_pair(raw_downsample["kernel_size"]),
                    stride=_pair(raw_downsample["stride"]),
                    padding=_pair(raw_downsample["padding"]),
                    dilation=_pair(raw_downsample["dilation"]),
                    ceil_mode=bool(raw_downsample.get("ceil_mode", False)),
                    prepad_right_bottom=int(raw_downsample.get("prepad_right_bottom", 0)),
                )
                if raw_downsample is not None
                else None
            )
            layouts.append(
                DownBlockLayout(
                    block_type=str(raw_layout["block_type"]),
                    out_channels=int(raw_layout["out_channels"]),
                    has_cross_attention=bool(raw_layout["has_cross_attention"]),
                    downsample=downsample,
                )
            )
        bridge = cls(
            layouts=layouts,
            geometry_channels=config["geometry_channels"],
            context_channels=int(config["context_channels"]),
            hidden_channels=config["hidden_channels"],
        )
        state = load_file(str(weights_path), device=str(device))
        expected, actual = set(bridge.state_dict()), set(state)
        if expected != actual:
            missing = sorted(expected - actual)
            unexpected = sorted(actual - expected)
            raise RuntimeError(
                "Tri-input bridge checkpoint keys do not match strictly: "
                f"missing={missing}, unexpected={unexpected}"
            )
        bridge.load_state_dict(state, strict=True)
        bridge.to(device)
        if unet is not None:
            bridge.validate_unet(unet)
        return bridge


def expand_tri_residuals_for_generation(
    residuals: Sequence[torch.Tensor],
    *,
    num_images_per_prompt: int = 1,
    do_classifier_free_guidance: bool = False,
) -> PreparedTriResiduals:
    """Match the x4 pipeline's image-condition batch replication order."""

    if not residuals:
        raise ValueError("At least one tri-input residual is required")
    if num_images_per_prompt <= 0:
        raise ValueError("num_images_per_prompt must be positive")
    base_batch_size = int(residuals[0].shape[0])
    expanded: list[torch.Tensor] = []
    for index, residual in enumerate(residuals):
        if residual.ndim != 4 or residual.shape[0] != base_batch_size:
            raise ValueError(
                f"Residual {index} must be BCHW with batch {base_batch_size}, got {tuple(residual.shape)}"
            )
        value = residual
        if num_images_per_prompt > 1:
            # This intentionally repeats the complete batch, matching
            # StableDiffusionUpscalePipeline's `torch.cat([image] * N)`.
            value = value.repeat(num_images_per_prompt, 1, 1, 1)
        if do_classifier_free_guidance:
            # Both the unconditional and conditional text branches see the
            # same image/tri-input observation, just like the LR image.
            value = torch.cat([value, value], dim=0)
        expanded.append(value)
    return PreparedTriResiduals(
        residuals=tuple(expanded),
        base_batch_size=base_batch_size,
        num_images_per_prompt=int(num_images_per_prompt),
        do_classifier_free_guidance=bool(do_classifier_free_guidance),
    )


def prepare_tri_residuals(
    bridge: TriInputBridge,
    conditioning: TriConditioningProtocol | Mapping[str, Any],
    sample: torch.Tensor,
    *,
    active_mask: torch.Tensor | None = None,
    num_images_per_prompt: int = 1,
    do_classifier_free_guidance: bool = False,
) -> PreparedTriResiduals:
    """Compute a condition once, then expand it for a complete sampler call."""

    base = bridge(conditioning, sample, active_mask=active_mask)
    return expand_tri_residuals_for_generation(
        base,
        num_images_per_prompt=num_images_per_prompt,
        do_classifier_free_guidance=do_classifier_free_guidance,
    )


def fresh_down_intrablock_residuals(
    prepared: PreparedTriResiduals, *, sample: torch.Tensor | None = None
) -> list[torch.Tensor]:
    """Make a disposable list, matching precision at the UNet call boundary.

    Accelerate can convert a prepared bridge's outputs back to float32 even
    though the bridge itself casts to the UNet sample dtype. A frozen fp16
    UNet is not necessarily wrapped in autocast: adding those float32
    residuals promotes its hidden states and breaks the next half convolution.
    Reconcile dtype/device *after* the prepared bridge returns, without
    detaching the gradient or mutating the reusable cache. Diffusers consumes
    the new list with ``pop``. Omitting sample preserves the legacy list API.
    """

    if sample is None:
        return list(prepared.residuals)
    return [value.to(device=sample.device, dtype=sample.dtype) for value in prepared.residuals]


class TriConditionedUNet(nn.Module):
    """Stateless UNet wrapper routing prepared residuals through its public API.

    Prepared residuals may be passed explicitly as ``tri_residuals`` during
    training, or embedded under :data:`TRI_RESIDUALS_CROSS_ATTENTION_KEY` in a
    copied ``cross_attention_kwargs`` mapping for the existing upscaler
    pipeline.  The private entry is removed before forwarding kwargs to the
    underlying attention processors.
    """

    def __init__(self, base_unet: nn.Module, bridge: TriInputBridge) -> None:
        super().__init__()
        bridge.validate_unet(base_unet)
        self.base_unet = base_unet
        self.bridge = bridge

    @property
    def config(self) -> Any:
        return self.base_unet.config

    @property
    def dtype(self) -> torch.dtype:
        parameter = next(self.base_unet.parameters(), None)
        return parameter.dtype if parameter is not None else torch.float32

    def enable_gradient_checkpointing(self, *args: Any, **kwargs: Any) -> Any:
        return self.base_unet.enable_gradient_checkpointing(*args, **kwargs)

    def forward(
        self,
        sample: torch.Tensor,
        timestep: torch.Tensor | float | int,
        encoder_hidden_states: torch.Tensor | None = None,
        *,
        tri_residuals: PreparedTriResiduals | None = None,
        cross_attention_kwargs: Mapping[str, Any] | None = None,
        **kwargs: Any,
    ) -> Any:
        clean_cross_attention_kwargs: dict[str, Any] | None = None
        payload_from_cross_attention: Any = None
        if cross_attention_kwargs is not None:
            clean_cross_attention_kwargs = dict(cross_attention_kwargs)
            payload_from_cross_attention = clean_cross_attention_kwargs.pop(
                TRI_RESIDUALS_CROSS_ATTENTION_KEY, None
            )
            if not clean_cross_attention_kwargs:
                clean_cross_attention_kwargs = None
        if tri_residuals is not None and payload_from_cross_attention is not None:
            raise ValueError("Tri residuals were provided both explicitly and through cross_attention_kwargs")
        prepared = tri_residuals if tri_residuals is not None else payload_from_cross_attention
        if prepared is not None and not isinstance(prepared, PreparedTriResiduals):
            raise TypeError(
                f"Tri residual payload must be PreparedTriResiduals, got {type(prepared).__name__}"
            )
        if prepared is not None and "down_intrablock_additional_residuals" in kwargs:
            raise ValueError("Cannot combine tri residuals with another down_intrablock residual source")

        call_kwargs = dict(kwargs)
        if encoder_hidden_states is not None:
            call_kwargs["encoder_hidden_states"] = encoder_hidden_states
        if clean_cross_attention_kwargs is not None:
            call_kwargs["cross_attention_kwargs"] = clean_cross_attention_kwargs
        if prepared is not None:
            if sample.shape[0] != prepared.expanded_batch_size:
                raise ValueError(
                    f"Prepared tri residual batch={prepared.expanded_batch_size} does not match "
                    f"UNet sample batch={sample.shape[0]}"
                )
            expected_sizes = residual_target_sizes(self.bridge.layouts, sample.shape[-2], sample.shape[-1])
            if len(prepared.residuals) != len(self.bridge.layouts):
                raise ValueError(
                    f"Prepared residual count={len(prepared.residuals)}; expected {len(self.bridge.layouts)}"
                )
            for index, (residual, layout, expected_size) in enumerate(
                zip(prepared.residuals, self.bridge.layouts, expected_sizes, strict=True)
            ):
                expected_shape = (sample.shape[0], layout.out_channels, *expected_size)
                if tuple(residual.shape) != expected_shape:
                    raise ValueError(
                        f"Tri residual {index} shape={tuple(residual.shape)}; expected={expected_shape}"
                    )
            call_kwargs["down_intrablock_additional_residuals"] = fresh_down_intrablock_residuals(
                prepared, sample=sample
            )
        return self.base_unet(sample, timestep, **call_kwargs)


def pipeline_cross_attention_kwargs(
    prepared: PreparedTriResiduals,
    existing: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Create request-scoped pipeline kwargs without mutating the caller's map."""

    result = dict(existing or {})
    if TRI_RESIDUALS_CROSS_ATTENTION_KEY in result:
        raise ValueError(f"Existing kwargs already contain {TRI_RESIDUALS_CROSS_ATTENTION_KEY!r}")
    result[TRI_RESIDUALS_CROSS_ATTENTION_KEY] = prepared
    return result
