from __future__ import annotations

import torch
from torch import nn

from src.local_spectral_checker import CheckerConfig
from src.tri_input_bridge import DownBlockLayout, TriInputBridge
from src.tri_input_conditioner import TriInputConditioner
from src.tri_input_runtime import masked_preview_l1, prepare_tri_condition


class _MustNotRun(nn.Module):
    def forward(self, *_args, **_kwargs):  # pragma: no cover - failure path
        raise AssertionError("inactive tri-input batch must bypass the module")


def _conditioner() -> TriInputConditioner:
    return TriInputConditioner(
        raw_mean=[0.0] * 12,
        raw_std=[1.0] * 12,
        surface_band_indices=range(10),
        checker_config=CheckerConfig(enabled=False),
    )


def _bridge() -> TriInputBridge:
    layouts = tuple(
        DownBlockLayout(
            block_type="CrossAttnDownBlock2D",
            out_channels=4,
            has_cross_attention=True,
            downsample=None,
        )
        for _ in range(4)
    )
    return TriInputBridge(
        layouts=layouts,
        geometry_channels=(32, 64, 96, 128),
        context_channels=64,
        hidden_channels=8,
    )


def _inputs(batch: int = 2):
    rgb = torch.randn(batch, 3, 4, 5)
    raw = torch.randn(batch, 12, 4, 5)
    prior = torch.rand(batch, 10, 4, 5, dtype=torch.float32)
    valid = torch.ones(batch, 1, 4, 5, dtype=torch.bool)
    sample = torch.randn(batch, 7, 4, 5)
    return rgb, raw, prior, valid, sample


def test_all_inactive_batch_bypasses_conditioner_and_bridge() -> None:
    rgb, raw, prior, valid, sample = _inputs()
    assert (
        prepare_tri_condition(
            _MustNotRun(),
            _MustNotRun(),
            rgb_minus_one_one=rgb,
            raw_ms=raw,
            unmixing=prior,
            raw_valid=valid,
            aux_present=torch.tensor([False, False]),
            unet_sample=sample,
        )
        is None
    )


def test_partial_active_batch_masks_bridge_and_preview_without_reweighting() -> None:
    conditioner, bridge = _conditioner(), _bridge()
    # Make the final bridge layers non-zero so masking can be observed.
    with torch.no_grad():
        for projection in bridge.projections:
            projection.zero_conv.weight.fill_(0.01)
    rgb, raw, prior, valid, sample = _inputs()
    prepared = prepare_tri_condition(
        conditioner,
        bridge,
        rgb_minus_one_one=rgb,
        raw_ms=raw,
        unmixing=prior,
        raw_valid=valid,
        aux_present=torch.tensor([True, False]),
        unet_sample=sample,
    )
    assert prepared is not None
    assert all(torch.count_nonzero(value[1]).item() == 0 for value in prepared.residuals.residuals)

    preview = torch.zeros(2, 3, 16, 20)
    target = torch.ones_like(preview)
    loss = masked_preview_l1(preview, target, torch.tensor([True, False]))
    # Active-only mean would be 1.0; full-batch zero weighting must remain 0.5.
    assert torch.equal(loss, torch.tensor(0.5))


def test_zero_bridge_then_second_update_reaches_conditioner_while_old_model_stays_frozen() -> None:
    torch.manual_seed(3)
    conditioner, bridge = _conditioner(), _bridge()
    old_backbone = nn.Conv2d(4, 1, 1)
    old_backbone.requires_grad_(False)
    old_before = {name: value.detach().clone() for name, value in old_backbone.state_dict().items()}
    optimizer = torch.optim.SGD(
        list(conditioner.parameters()) + list(bridge.parameters()), lr=0.05
    )
    rgb, raw, prior, valid, sample = _inputs(batch=1)

    first_conditioner_grad = None
    for update in range(2):
        optimizer.zero_grad(set_to_none=True)
        prepared = prepare_tri_condition(
            conditioner,
            bridge,
            rgb_minus_one_one=rgb,
            raw_ms=raw,
            unmixing=prior,
            raw_valid=valid,
            aux_present=torch.tensor([True]),
            unet_sample=sample,
        )
        assert prepared is not None
        prediction = sum(
            old_backbone(residual).mean() for residual in prepared.residuals.residuals
        )
        prediction.backward()
        conditioner_grad = sum(
            float(parameter.grad.abs().sum())
            for parameter in conditioner.parameters()
            if parameter.grad is not None
        )
        if update == 0:
            first_conditioner_grad = conditioner_grad
        else:
            assert conditioner_grad > 0
        optimizer.step()

    assert first_conditioner_grad == 0.0
    assert all(parameter.grad is None for parameter in old_backbone.parameters())
    for name, value in old_backbone.state_dict().items():
        assert torch.equal(value, old_before[name])
