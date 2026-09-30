"""CPU regression for the prepared bridge -> frozen UNet precision boundary.

The tuple-to-float32 conversion below models Accelerate's output conversion;
it deliberately does not claim to exercise a complete Accelerate/GPU runtime.
"""

from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from src.tri_input_bridge import (
    TRI_RESIDUALS_CROSS_ATTENTION_KEY,
    TriConditionedUNet,
    TriInputBridge,
    expand_tri_residuals_for_generation,
    fresh_down_intrablock_residuals,
    pipeline_cross_attention_kwargs,
)


class CrossAttnDownBlock2D(nn.Module):
    """Match the important order: add intrablock residual, then downsample."""

    has_cross_attention = True

    def __init__(self) -> None:
        super().__init__()
        self.downsamplers = nn.ModuleList([nn.Conv2d(4, 4, 3, stride=2, padding=1)])


class DownBlock2D(nn.Module):
    has_cross_attention = False
    downsamplers = None


class FrozenConvUNet(nn.Module):
    def __init__(self, dtype: torch.dtype) -> None:
        super().__init__()
        self.config = SimpleNamespace(block_out_channels=(4, 4, 4, 4), in_channels=7)
        self.down_blocks = nn.ModuleList(
            [CrossAttnDownBlock2D(), CrossAttnDownBlock2D(), CrossAttnDownBlock2D(), DownBlock2D()]
        )
        self.conv_out = nn.Conv2d(4, 4, 1)
        self.last_cross_attention_kwargs = None
        self.last_residual_list = None
        self.to(dtype=dtype).requires_grad_(False)

    def forward(
        self,
        sample,
        timestep,
        *,
        down_intrablock_additional_residuals=None,
        cross_attention_kwargs=None,
        **kwargs,
    ):
        del timestep, kwargs
        self.last_cross_attention_kwargs = cross_attention_kwargs
        self.last_residual_list = down_intrablock_additional_residuals
        value = sample[:, :4]
        for block in self.down_blocks:
            if down_intrablock_additional_residuals is not None:
                value = value + down_intrablock_additional_residuals.pop(0)
            if block.downsamplers is not None:
                value = block.downsamplers[0](value)
        return (self.conv_out(value),)


def make_fp32_prepared(dtype: torch.dtype):
    """Simulate wrapped bridge outputs promoted back to fp32 after forward."""

    torch.manual_seed(120)
    unet = FrozenConvUNet(dtype)
    bridge = TriInputBridge.from_unet(
        unet, geometry_channels=(3, 3, 3, 3), context_channels=2, hidden_channels=4
    )
    # Nonzero bridges test the actual gradient path, not only zero equivalence.
    with torch.no_grad():
        for projection in bridge.projections:
            projection.zero_conv.weight.fill_(0.05)
            projection.zero_conv.bias.fill_(0.01)
    sample = torch.randn(2, 7, 8, 10).to(dtype=dtype)
    geometry = tuple(torch.randn(2, 3, 8, 10, requires_grad=True) for _ in range(4))
    context = torch.randn(2, 2, 8, 10, requires_grad=True)
    raw_outputs = bridge({"geometry": geometry, "context_lr": context}, sample)
    assert all(value.dtype == dtype for value in raw_outputs)
    converted = tuple(value.float() for value in raw_outputs)
    for value in converted:
        value.retain_grad()
    prepared = expand_tri_residuals_for_generation(converted)
    return unet, bridge, sample, prepared, geometry, context


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_unconverted_fp32_residual_reproduces_frozen_conv_dtype_failure(dtype) -> None:
    # Disable oneDNN only within this test: some CPUs cannot backprop f16/bf16
    # convolutions there, while the native CPU implementation supports them.
    with torch.backends.mkldnn.flags(enabled=False):
        unet, _, sample, prepared, _, _ = make_fp32_prepared(dtype)
        with pytest.raises(RuntimeError, match="same|type"):
            unet(sample, 1, down_intrablock_additional_residuals=list(prepared.residuals))


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("pipeline_route", [False, True], ids=["train-boundary", "pipeline-wrapper"])
def test_fp32_cache_is_cast_at_unet_boundary_without_breaking_gradients(dtype, pipeline_route) -> None:
    with torch.backends.mkldnn.flags(enabled=False):
        unet, bridge, sample, prepared, geometry, context = make_fp32_prepared(dtype)
        cached = prepared.residuals
        snapshots = tuple(value.detach().clone() for value in cached)
        if pipeline_route:
            wrapper = TriConditionedUNet(unet, bridge)
            kwargs = pipeline_cross_attention_kwargs(prepared, {"scale": 0.75})
            result = wrapper(sample, 1, cross_attention_kwargs=kwargs)[0]
            assert kwargs[TRI_RESIDUALS_CROSS_ATTENTION_KEY] is prepared
            assert unet.last_cross_attention_kwargs == {"scale": 0.75}
        else:
            residuals = fresh_down_intrablock_residuals(prepared, sample=sample)
            assert all(value.dtype == sample.dtype and value.device == sample.device for value in residuals)
            result = unet(sample, 1, down_intrablock_additional_residuals=residuals)[0]

        assert result.dtype == dtype
        assert torch.isfinite(result).all()
        assert unet.last_residual_list == []  # Diffusers consumes a disposable list.
        result.float().square().mean().backward()
        assert prepared.residuals is cached
        for value, snapshot in zip(cached, snapshots, strict=True):
            assert value.dtype == torch.float32
            assert torch.equal(value, snapshot)
            assert value.grad is not None and torch.isfinite(value.grad).all()
            assert value.grad.abs().sum() > 0
        assert all(parameter.grad is None for parameter in unet.parameters())
        for projection in bridge.projections:
            assert projection.zero_conv.weight.grad is not None
            assert torch.isfinite(projection.zero_conv.weight.grad).all()
            assert projection.zero_conv.weight.grad.abs().sum() > 0
        assert context.grad is not None and context.grad.abs().sum() > 0
        assert all(value.grad is not None and value.grad.abs().sum() > 0 for value in geometry)


def test_fresh_lists_preserve_cache_and_legacy_no_sample_api() -> None:
    with torch.backends.mkldnn.flags(enabled=False):
        _, _, sample, prepared, _, _ = make_fp32_prepared(torch.float16)
        first = fresh_down_intrablock_residuals(prepared, sample=sample)
        second = fresh_down_intrablock_residuals(prepared, sample=sample)
        assert first is not second
        first.pop(0)
        assert len(second) == len(prepared.residuals) == 4
        assert all(value.dtype == torch.float32 for value in prepared.residuals)
        legacy = fresh_down_intrablock_residuals(prepared)
        assert all(value is cached for value, cached in zip(legacy, prepared.residuals, strict=True))


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_disabled_wrapper_is_exact_frozen_baseline(dtype) -> None:
    with torch.backends.mkldnn.flags(enabled=False):
        unet, bridge, sample, _, _, _ = make_fp32_prepared(dtype)
        wrapper = TriConditionedUNet(unet, bridge)
        baseline = unet(sample, 1)[0]
        actual = wrapper(sample, 1)[0]
        assert torch.equal(actual, baseline)
        assert unet.last_residual_list is None


def test_training_call_matches_residuals_to_actual_model_input() -> None:
    # Protect the real training call without importing/downloading its large
    # optional model dependencies. Numerical correctness is tested above.
    training = Path(__file__).resolve().parents[1] / "src" / "train_lora_upscaler.py"
    tree = ast.parse(training.read_text(encoding="utf-8"))
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "fresh_down_intrablock_residuals"
    ]
    assert calls, "Training must prepare disposable intrablock residuals at the UNet boundary"
    for call in calls:
        sample_arguments = [keyword.value for keyword in call.keywords if keyword.arg == "sample"]
        assert len(sample_arguments) == 1
        assert isinstance(sample_arguments[0], ast.Name) and sample_arguments[0].id == "model_input"
