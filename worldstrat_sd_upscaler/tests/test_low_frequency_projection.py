from __future__ import annotations

import torch
import torch.nn.functional as F

from src.utils import low_frequency_projection


def test_alpha_zero_preserves_sr() -> None:
    torch.manual_seed(1)
    lr = torch.rand(3, 8, 10)
    sr = torch.rand(3, 32, 40)
    assert torch.equal(low_frequency_projection(sr, lr, alpha=0.0), sr)


def test_projection_improves_low_frequency_consistency() -> None:
    torch.manual_seed(2)
    lr = torch.rand(1, 3, 8, 8)
    sr = torch.rand(1, 3, 32, 32)
    before = F.interpolate(sr, size=lr.shape[-2:], mode="bicubic", align_corners=False)
    projected = low_frequency_projection(sr, lr, alpha=1.0)
    after = F.interpolate(projected, size=lr.shape[-2:], mode="bicubic", align_corners=False)
    assert torch.mean(torch.abs(after - lr)) < torch.mean(torch.abs(before - lr))
    assert projected.min() >= 0 and projected.max() <= 1


def test_projection_rejects_wrong_scale() -> None:
    try:
        low_frequency_projection(torch.rand(3, 31, 32), torch.rand(3, 8, 8), alpha=0.5)
    except ValueError as error:
        assert "not LR size" in str(error)
    else:
        raise AssertionError("wrong-scale input was accepted")

