from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from safetensors.torch import load_file, save_file
from torch import nn

from src.tri_input_bridge import (
    BRIDGE_WEIGHTS_NAME,
    TRI_RESIDUALS_CROSS_ATTENTION_KEY,
    TriConditionedUNet,
    TriInputBridge,
    expand_tri_residuals_for_generation,
    fresh_down_intrablock_residuals,
    inspect_unet_down_blocks,
    pipeline_cross_attention_kwargs,
    prepare_tri_residuals,
    residual_target_sizes,
)


class Downsample2D(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.use_conv = True
        self.padding = 1
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, stride=2, padding=1)


class CrossAttnDownBlock2D(nn.Module):
    has_cross_attention = True

    def __init__(self, channels: int, downsample: bool = True) -> None:
        super().__init__()
        self.downsamplers = nn.ModuleList([Downsample2D(channels)]) if downsample else None


class DownBlock2D(nn.Module):
    has_cross_attention = False

    def __init__(self, channels: int, downsample: bool = False) -> None:
        super().__init__()
        self.downsamplers = nn.ModuleList([Downsample2D(channels)]) if downsample else None


class MockUNet(nn.Module):
    def __init__(self, channels: tuple[int, int, int, int] = (4, 8, 16, 16)) -> None:
        super().__init__()
        self.config = SimpleNamespace(block_out_channels=channels, in_channels=7)
        self.down_blocks = nn.ModuleList(
            [
                CrossAttnDownBlock2D(channels[0]),
                CrossAttnDownBlock2D(channels[1]),
                CrossAttnDownBlock2D(channels[2]),
                DownBlock2D(channels[3]),
            ]
        )
        self.scale = nn.Parameter(torch.tensor(1.0))
        self.residual_list_ids: list[int] = []
        # Retain consumed lists so Python cannot recycle their ids between calls.
        self.residual_lists: list[list[torch.Tensor]] = []
        self.last_cross_attention_kwargs: dict | None = None

    def forward(
        self,
        sample: torch.Tensor,
        timestep: torch.Tensor | int,
        encoder_hidden_states: torch.Tensor | None = None,
        cross_attention_kwargs: dict | None = None,
        down_intrablock_additional_residuals: list[torch.Tensor] | None = None,
        return_dict: bool = False,
        **_kwargs,
    ):
        del timestep, encoder_hidden_states, return_dict
        self.last_cross_attention_kwargs = cross_attention_kwargs
        result = sample[:, :4] * self.scale
        if down_intrablock_additional_residuals is not None:
            self.residual_list_ids.append(id(down_intrablock_additional_residuals))
            self.residual_lists.append(down_intrablock_additional_residuals)
            # Match pinned Diffusers' destructive list consumption.
            while down_intrablock_additional_residuals:
                residual = down_intrablock_additional_residuals.pop(0)
                value = F.interpolate(
                    residual.mean(dim=1, keepdim=True),
                    size=result.shape[-2:],
                    mode="nearest",
                )
                result = result + value.expand(-1, result.shape[1], -1, -1)
        return (result,)


def make_bridge(unet: MockUNet) -> TriInputBridge:
    return TriInputBridge.from_unet(
        unet,
        geometry_channels=(3, 5, 7, 9),
        context_channels=6,
        hidden_channels=(4, 6, 8, 8),
    )


def make_condition(batch: int = 2, value: float = 1.0, active_mask=None) -> dict:
    condition = {
        "geometry": (
            torch.full((batch, 3, 36, 52), value),
            torch.full((batch, 5, 18, 26), value),
            torch.full((batch, 7, 9, 13), value),
            torch.full((batch, 9, 5, 7), value),
        ),
        "context_lr": torch.full((batch, 6, 9, 13), value),
    }
    if active_mask is not None:
        condition["aux_present"] = torch.as_tensor(active_mask)
    return condition


def make_sample(batch: int = 2) -> torch.Tensor:
    return torch.randn(batch, 7, 9, 13)


def make_bridge_nonzero(bridge: TriInputBridge) -> None:
    with torch.no_grad():
        for projection in bridge.projections:
            feature_conv = projection.features[0]
            feature_conv.weight.fill_(0.01)
            feature_conv.bias.zero_()
            projection.zero_conv.weight.fill_(0.02)
            projection.zero_conv.bias.zero_()


def test_runtime_block_layout_uses_true_pre_and_post_downsample_sizes() -> None:
    unet = MockUNet()
    layouts = inspect_unet_down_blocks(unet)
    # Cross-attention blocks consume residuals before downsampling. The final
    # ordinary DownBlock2D receives its residual after returning.
    assert residual_target_sizes(layouts, 9, 13) == ((9, 13), (5, 7), (3, 4), (2, 2))


def test_zero_initialized_bridge_is_exact_baseline_and_masks_inactive_samples() -> None:
    unet = MockUNet()
    bridge = make_bridge(unet)
    wrapper = TriConditionedUNet(unet, bridge)
    sample = make_sample()
    condition = make_condition(active_mask=[1, 0])
    prepared = prepare_tri_residuals(bridge, condition, sample)

    baseline = unet(sample, 5, return_dict=False)[0]
    conditioned = wrapper(sample, 5, tri_residuals=prepared, return_dict=False)[0]
    assert torch.equal(conditioned, baseline)
    assert [tuple(value.shape) for value in prepared.residuals] == [
        (2, 4, 9, 13),
        (2, 8, 5, 7),
        (2, 16, 3, 4),
        (2, 16, 2, 2),
    ]
    assert all(torch.count_nonzero(value[1]).item() == 0 for value in prepared.residuals)


def test_nonzero_bridge_changes_prediction_and_fresh_lists_survive_reuse() -> None:
    unet = MockUNet()
    bridge = make_bridge(unet)
    make_bridge_nonzero(bridge)
    wrapper = TriConditionedUNet(unet, bridge)
    sample = make_sample()
    prepared = prepare_tri_residuals(bridge, make_condition(), sample)
    first_list = fresh_down_intrablock_residuals(prepared)
    second_list = fresh_down_intrablock_residuals(prepared)
    assert first_list is not second_list
    assert first_list[0] is second_list[0]

    baseline = unet(sample, 4, return_dict=False)[0]
    first = wrapper(sample, 4, tri_residuals=prepared, return_dict=False)[0]
    second = wrapper(sample, 4, tri_residuals=prepared, return_dict=False)[0]
    assert not torch.equal(first, baseline)
    assert torch.equal(first, second)
    assert len(prepared.residuals) == 4
    assert len(unet.residual_list_ids) == 2
    assert unet.residual_list_ids[0] != unet.residual_list_ids[1]


def test_cfg_and_num_images_repeat_the_complete_batch_in_pipeline_order() -> None:
    base = torch.tensor([1.0, 2.0]).reshape(2, 1, 1, 1)
    prepared = expand_tri_residuals_for_generation(
        (base, base + 10),
        num_images_per_prompt=3,
        do_classifier_free_guidance=True,
    )
    expected = [1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2]
    assert prepared.expanded_batch_size == 12
    assert prepared.residuals[0][:, 0, 0, 0].tolist() == expected
    assert prepared.residuals[1][:, 0, 0, 0].tolist() == [value + 10 for value in expected]


def test_wrapper_strips_private_pipeline_key_and_does_not_leak_conditions() -> None:
    unet = MockUNet()
    bridge = make_bridge(unet)
    make_bridge_nonzero(bridge)
    wrapper = TriConditionedUNet(unet, bridge)
    sample = make_sample()
    condition_a = make_condition(value=0.0)
    condition_b = make_condition(value=1.0)
    prepared_a = prepare_tri_residuals(bridge, condition_a, sample)
    prepared_b = prepare_tri_residuals(bridge, condition_b, sample)

    kwargs_a = pipeline_cross_attention_kwargs(prepared_a, {"scale": 0.75})
    result_a = wrapper(sample, 3, cross_attention_kwargs=kwargs_a, return_dict=False)[0]
    assert unet.last_cross_attention_kwargs == {"scale": 0.75}
    assert TRI_RESIDUALS_CROSS_ATTENTION_KEY not in unet.last_cross_attention_kwargs
    result_b = wrapper(
        sample,
        3,
        cross_attention_kwargs=pipeline_cross_attention_kwargs(prepared_b),
        return_dict=False,
    )[0]
    result_a_again = wrapper(
        sample,
        3,
        cross_attention_kwargs=pipeline_cross_attention_kwargs(prepared_a),
        return_dict=False,
    )[0]
    baseline_after = wrapper(sample, 3, return_dict=False)[0]
    baseline_direct = unet(sample, 3, return_dict=False)[0]

    assert not torch.equal(result_a, result_b)
    assert torch.equal(result_a, result_a_again)
    assert torch.equal(baseline_after, baseline_direct)


def test_wrapper_validates_expanded_batch_and_spatial_contract() -> None:
    unet = MockUNet()
    bridge = make_bridge(unet)
    wrapper = TriConditionedUNet(unet, bridge)
    sample = make_sample()
    prepared = prepare_tri_residuals(
        bridge,
        make_condition(),
        sample,
        num_images_per_prompt=2,
        do_classifier_free_guidance=True,
    )
    expanded_sample = sample.repeat(4, 1, 1, 1)
    wrapper(expanded_sample, 2, tri_residuals=prepared, return_dict=False)
    with pytest.raises(ValueError, match="does not match UNet sample batch"):
        wrapper(sample, 2, tri_residuals=prepared, return_dict=False)


def test_bridge_save_load_round_trip_is_strict(tmp_path) -> None:
    unet = MockUNet()
    bridge = make_bridge(unet)
    make_bridge_nonzero(bridge)
    sample = make_sample()
    condition = make_condition()
    expected = bridge(condition, sample)
    bridge.save_pretrained(tmp_path)

    restored = TriInputBridge.from_pretrained(tmp_path, unet=unet)
    actual = restored(condition, sample)
    assert restored.layouts == bridge.layouts
    for expected_tensor, actual_tensor in zip(expected, actual, strict=True):
        assert torch.equal(expected_tensor, actual_tensor)

    weights_path = tmp_path / BRIDGE_WEIGHTS_NAME
    incomplete = load_file(str(weights_path))
    incomplete.pop(next(iter(incomplete)))
    save_file(incomplete, str(weights_path))
    with pytest.raises(RuntimeError, match="checkpoint keys do not match strictly"):
        TriInputBridge.from_pretrained(tmp_path)


def test_unknown_down_block_fails_fast() -> None:
    class UnknownDownBlock(nn.Module):
        has_cross_attention = True
        downsamplers = None

    unet = MockUNet()
    unet.down_blocks[0] = UnknownDownBlock()
    with pytest.raises(TypeError, match="Unsupported UNet down block"):
        inspect_unet_down_blocks(unet)


def test_checkpoint_recomputation_uses_request_scoped_residual_tensor() -> None:
    from torch.utils.checkpoint import checkpoint

    class CheckpointUNet(MockUNet):
        def forward(
            self,
            sample,
            timestep,
            down_intrablock_additional_residuals=None,
            return_dict=False,
            **kwargs,
        ):
            del timestep, return_dict, kwargs
            output = sample[:, :4]
            assert down_intrablock_additional_residuals is not None
            residual = down_intrablock_additional_residuals.pop(0)

            def recomputed(value):
                return value.square() + 0.5 * value

            value = checkpoint(recomputed, residual, use_reentrant=False)
            value = F.interpolate(value.mean(1, keepdim=True), output.shape[-2:])
            return (output + value,)

    unet = CheckpointUNet()
    bridge = make_bridge(unet)
    make_bridge_nonzero(bridge)
    wrapper = TriConditionedUNet(unet, bridge)
    sample = make_sample()
    prepared = prepare_tri_residuals(bridge, make_condition(), sample)
    loss = wrapper(sample, 1, tri_residuals=prepared, return_dict=False)[0].mean()
    loss.backward()
    assert any(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in bridge.parameters()
    )
    assert len(prepared.residuals) == 4
