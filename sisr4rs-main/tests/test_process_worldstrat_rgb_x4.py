from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from scripts.process_worldstrat_rgb_x4 import correct_pair


class ConstantFlowRegistration(torch.nn.Module):
    """Small deterministic stand-in; real checkpoint access is not needed."""

    def __init__(self, x: float = 0.0, y: float = 0.0) -> None:
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()), requires_grad=False)
        self.x = x
        self.y = y

    def forward(self, source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        del target
        flow = torch.empty(
            (source.shape[0], 2, source.shape[-2], source.shape[-1]),
            device=source.device,
            dtype=source.dtype,
        )
        flow[:, 0].fill_(self.x)
        flow[:, 1].fill_(self.y)
        return flow


class GridCheckingRegistration(ConstantFlowRegistration):
    """Mock a depth-4 U-Net and reject inputs that are not divisible by 8."""

    def __init__(self) -> None:
        super().__init__(x=0.25, y=-0.5)
        self.unet = torch.nn.Identity()
        self.unet.depth = 4
        self.received_size: tuple[int, int] | None = None

    def forward(self, source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        self.received_size = tuple(source.shape[-2:])
        assert source.shape[-2] % 8 == 0
        assert source.shape[-1] % 8 == 0
        return super().forward(source, target)


def test_zero_flow_geometry_and_skip_radiometric() -> None:
    torch.manual_seed(1)
    lr = torch.rand(1, 3, 8, 9)
    gt = torch.rand(1, 3, 32, 36)

    result = correct_pair(
        lr,
        gt,
        ConstantFlowRegistration(),
        scale=4,
        registration_channel=0,
        mtf=0.4,
        compute_radiometric=False,
    )

    assert result["gt_geo"].shape == gt.shape
    assert result["pseudo_lr"].shape == lr.shape
    assert result["flow_lr"].shape == (1, 2, 8, 9)
    assert torch.equal(result["flow_gt"], torch.zeros_like(result["flow_gt"]))
    assert torch.equal(result["gt_geo"], gt)
    assert torch.equal(result["gt_geo_rad"], result["gt_geo"])
    assert torch.count_nonzero(result["residual_gt"]) == 0
    assert not bool(result["radiometric_visual_enabled"].item())


def test_flow_upsampling_multiplies_displacement_by_scale() -> None:
    torch.manual_seed(2)
    lr = torch.rand(1, 3, 10, 11)
    gt = torch.rand(1, 3, 40, 44)

    result = correct_pair(
        lr,
        gt,
        ConstantFlowRegistration(x=1.0, y=-0.5),
        scale=4,
        registration_channel=0,
        mtf=0.4,
        compute_radiometric=False,
    )

    expected = 4.0 * F.interpolate(
        result["flow_lr"], scale_factor=4.0, mode="bicubic", align_corners=False
    )
    assert result["flow_gt"].shape == (1, 2, 40, 44)
    assert torch.allclose(result["flow_gt"], expected)


def test_odd_lr_size_is_padded_for_unet_then_flow_is_cropped() -> None:
    torch.manual_seed(3)
    lr = torch.rand(1, 3, 13, 15)
    gt = torch.rand(1, 3, 52, 60)
    model = GridCheckingRegistration()

    result = correct_pair(
        lr,
        gt,
        model,
        scale=4,
        registration_channel=0,
        mtf=0.4,
        compute_radiometric=False,
    )

    assert model.received_size == (16, 16)
    assert result["flow_lr"].shape == (1, 2, 13, 15)
    assert result["flow_gt"].shape == (1, 2, 52, 60)
    assert result["gt_geo"].shape == gt.shape


def test_outputs_are_safely_clipped_to_display_rgb_range() -> None:
    lr = torch.rand(1, 3, 8, 8)
    gt = 1.5 * torch.ones(1, 3, 32, 32)

    result = correct_pair(
        lr,
        gt,
        ConstantFlowRegistration(),
        scale=4,
        registration_channel=0,
        mtf=0.4,
        compute_radiometric=True,
    )

    for key in ("gt_geo", "gt_geo_rad", "pseudo_lr", "pseudo_lr_aligned"):
        assert torch.isfinite(result[key]).all()
        assert result[key].min() >= 0.0
        assert result[key].max() <= 1.0


def test_non_x4_geometry_fails_without_resizing() -> None:
    lr = torch.rand(1, 3, 8, 8)
    gt = torch.rand(1, 3, 31, 32)

    with pytest.raises(ValueError, match="does not equal 4x LR"):
        correct_pair(
            lr,
            gt,
            ConstantFlowRegistration(),
            scale=4,
            registration_channel=0,
            mtf=0.4,
            compute_radiometric=False,
        )
