from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import torch
from PIL import Image
from torch import nn

from src.condition_adapter import ConditionAdapter
from src.infer_upscaler import (
    InferenceAuxiliary,
    TriInferenceRuntime,
    _crop_auxiliary,
    _load_tri_runtime,
    _pad_auxiliary,
    _run_pipeline,
    tiled_inference,
)
from src.local_spectral_checker import CheckerConfig
from src.tri_input_bridge import TriConditionedUNet, TriInputBridge
from src.tri_input_conditioner import TriInputConditioner
from src.tri_input_data import RawBandStats, RawValueConversion, SENTINEL2_L2A_BANDS


class Downsample2D(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.use_conv = True
        self.padding = 1
        self.conv = nn.Conv2d(channels, channels, 3, stride=2, padding=1)


class CrossAttnDownBlock2D(nn.Module):
    has_cross_attention = True

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.downsamplers = nn.ModuleList([Downsample2D(channels)])


class DownBlock2D(nn.Module):
    has_cross_attention = False
    downsamplers = None


class _BaseUNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = SimpleNamespace(block_out_channels=(4, 8, 16, 16), in_channels=7)
        self.down_blocks = nn.ModuleList(
            [CrossAttnDownBlock2D(4), CrossAttnDownBlock2D(8), CrossAttnDownBlock2D(16), DownBlock2D()]
        )
        self.anchor = nn.Parameter(torch.ones(()))
        self.list_ids: list[int] = []
        # Keep only the mock's consumed lists alive for meaningful id comparisons.
        self.residual_lists: list[list[torch.Tensor]] = []
        self.residual_signal: list[float] = []

    def forward(self, sample, timestep, down_intrablock_additional_residuals=None, **kwargs):
        del timestep, kwargs
        output = sample[:, :4] * self.anchor
        if down_intrablock_additional_residuals is not None:
            self.list_ids.append(id(down_intrablock_additional_residuals))
            self.residual_lists.append(down_intrablock_additional_residuals)
            signal = 0.0
            while down_intrablock_additional_residuals:
                signal = signal + down_intrablock_additional_residuals.pop(0).mean()
            output = output + signal
            self.residual_signal.append(float(signal.detach()))
        return (output,)


class _CountingConditioner(TriInputConditioner):
    def __init__(self) -> None:
        super().__init__(
            raw_mean=[0.0] * 12,
            raw_std=[1.0] * 12,
            surface_band_indices=range(10),
            checker_config=CheckerConfig(enabled=False),
        )
        self.calls = 0
        self.seen_raw_origins: list[float] = []

    def forward(self, *args, **kwargs):
        self.calls += 1
        raw = args[1]
        self.seen_raw_origins.append(float(raw[0, 0, 0, 0]))
        return super().forward(*args, **kwargs)


class _FakePipeline:
    vae_scale_factor = 4

    def __init__(self, unet: nn.Module) -> None:
        self.unet = unet
        self.calls: list[dict] = []

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        image = kwargs["image"]
        batch = 2 if kwargs["guidance_scale"] > 1 else 1
        sample = torch.zeros(batch, 7, image.height, image.width)
        for timestep in (3, 2, 1):
            self.unet(
                sample,
                timestep,
                cross_attention_kwargs=kwargs.get("cross_attention_kwargs"),
                return_dict=False,
            )
        return SimpleNamespace(
            images=[image.resize((image.width * 4, image.height * 4), Image.Resampling.NEAREST)]
        )


def _aux(height: int = 5, width: int = 7) -> InferenceAuxiliary:
    prior = torch.rand(10, height, width, dtype=torch.float32)
    return InferenceAuxiliary(
        raw_ms=torch.ones(12, height, width),
        unmixing=prior,
        raw_valid=torch.ones(1, height, width, dtype=torch.bool),
        raw_ms_path=Path("raw.tif"),
        unmixing_path=Path("prior.npy"),
    )


def _args(**overrides):
    values = dict(
        noise_level=10,
        guidance_scale=1.0,
        num_inference_steps=3,
        tile_size=8,
        tile_overlap=2,
    )
    values.update(overrides)
    return SimpleNamespace(**values)


def test_auxiliary_padding_uses_invalid_raw_and_unknown_prior() -> None:
    auxiliary = _aux(2, 3)
    padded = _pad_auxiliary(auxiliary, 4, 5)
    assert torch.equal(padded.raw_ms[:, :2, :3], auxiliary.raw_ms)
    assert torch.count_nonzero(padded.raw_ms[:, 2:, :]).item() == 0
    assert not padded.raw_valid[:, 2:, :].any()
    assert torch.count_nonzero(padded.unmixing[:5, 2:, :]).item() == 0
    assert torch.all(padded.unmixing[5:, 2:, :] == 1)
    cropped = _crop_auxiliary(padded, 1, 1, 3, 2)
    assert cropped.raw_ms.shape == (12, 2, 3)
    assert cropped.unmixing.shape == (10, 2, 3)


def test_whole_image_condition_is_computed_once_and_reused_each_denoising_step() -> None:
    base = _BaseUNet()
    conditioner = _CountingConditioner()
    bridge = TriInputBridge.from_unet(
        base, geometry_channels=(32, 64, 96, 128), context_channels=64, hidden_channels=8
    )
    with torch.no_grad():
        for projection in bridge.projections:
            projection.zero_conv.weight.fill_(0.01)
    pipe = _FakePipeline(TriConditionedUNet(base, bridge))
    stats = RawBandStats(
        format_version=1,
        split="train",
        band_names=SENTINEL2_L2A_BANDS,
        raw_value_conversion=RawValueConversion.from_config({"mode": "identity"}),
        mean=(0.0,) * 12,
        std=(1.0,) * 12,
        valid_pixel_count=(1,) * 12,
        sample_count=1,
        data_source="fixture",
        created_utc="fixture",
        software_versions={},
    )
    runtime = TriInferenceRuntime(
        conditioner, bridge, SENTINEL2_L2A_BANDS, stats.raw_value_conversion, stats
    )
    image = Image.new("RGB", (7, 5), color=(20, 30, 40))
    result = _run_pipeline(
        pipe,
        ConditionAdapter().eval(),
        image,
        "prompt",
        _args(),
        torch.Generator().manual_seed(1),
        torch.device("cpu"),
        torch.float32,
        runtime,
        _aux(),
    )
    assert result.size == (28, 20)
    assert conditioner.calls == 1
    assert len(base.list_ids) == 3
    assert len(set(base.list_ids)) == 3
    assert all(value != 0 for value in base.residual_signal)
    assert "cross_attention_kwargs" in pipe.calls[0]


def test_tiled_small_image_explicitly_falls_back_to_whole_without_resize() -> None:
    pipe = _FakePipeline(_BaseUNet())
    image = Image.new("RGB", (5, 6))
    result = tiled_inference(
        pipe,
        ConditionAdapter().eval(),
        image,
        "prompt",
        _args(tile_size=8),
        torch.device("cpu"),
        torch.float32,
        9,
    )
    assert result.size == (20, 24)
    assert len(pipe.calls) == 1
    assert "cross_attention_kwargs" not in pipe.calls[0]


def test_tiled_tri_input_uses_identical_lr_coordinates_and_no_cross_tile_cache() -> None:
    base = _BaseUNet()
    conditioner = _CountingConditioner()
    bridge = TriInputBridge.from_unet(
        base, geometry_channels=(32, 64, 96, 128), context_channels=64, hidden_channels=8
    )
    pipe = _FakePipeline(TriConditionedUNet(base, bridge))
    conversion = RawValueConversion.from_config({"mode": "identity"})
    stats = RawBandStats(
        1,
        "train",
        SENTINEL2_L2A_BANDS,
        conversion,
        (0.0,) * 12,
        (1.0,) * 12,
        (1,) * 12,
        1,
        "fixture",
        "fixture",
        {},
    )
    runtime = TriInferenceRuntime(conditioner, bridge, SENTINEL2_L2A_BANDS, conversion, stats)
    auxiliary = _aux(12, 12)
    yy, xx = torch.meshgrid(torch.arange(12), torch.arange(12), indexing="ij")
    raw = auxiliary.raw_ms.clone()
    raw[0] = yy * 10 + xx
    auxiliary = InferenceAuxiliary(
        raw, auxiliary.unmixing, auxiliary.raw_valid, auxiliary.raw_ms_path, auxiliary.unmixing_path
    )
    result = tiled_inference(
        pipe,
        ConditionAdapter().eval(),
        Image.new("RGB", (12, 12)),
        "prompt",
        _args(tile_size=8, tile_overlap=4),
        torch.device("cpu"),
        torch.float32,
        4,
        runtime,
        auxiliary,
    )
    assert result.size == (48, 48)
    assert conditioner.calls == 4
    assert conditioner.seen_raw_origins == [0.0, 4.0, 40.0, 44.0]
    assert len(pipe.calls) == 4


def test_declared_tri_artifact_missing_weights_fails_fast(tmp_path) -> None:
    pipe = SimpleNamespace(unet=_BaseUNet())
    try:
        _load_tri_runtime(
            tmp_path,
            pipe,
            raw_stats_override=None,
            cli_config={},
            artifact_config={},
            device=torch.device("cpu"),
        )
    except FileNotFoundError as error:
        assert "required weights/configs are missing" in str(error)
    else:  # pragma: no cover
        raise AssertionError("missing tri-input artifact must fail")
