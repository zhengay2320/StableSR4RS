from __future__ import annotations

import torch

from src.local_spectral_checker import (
    CheckerConfig,
    LocalSpectralChecker,
    constant_shift_flow,
    warp_internal_structure,
)


def test_checker_disabled_is_exact_identity_for_odd_non_square_input() -> None:
    q0 = torch.softmax(torch.randn(2, 8, 13, 17), dim=1).requires_grad_(True)
    raw = torch.randn(2, 10, 4, 5)
    valid = torch.ones(2, 1, 4, 5, dtype=torch.bool)
    checker = LocalSpectralChecker(CheckerConfig(enabled=False))

    result = checker(q0, raw, valid)

    assert result.q_star is q0
    assert torch.count_nonzero(result.flow_hr) == 0
    assert result.fallback_reasons == ("checker_disabled", "checker_disabled")
    result.q_star.sum().backward()
    assert q0.grad is not None


def test_target_grid_positive_x_shift_moves_content_right_by_one_pixel() -> None:
    q = torch.zeros(1, 1, 5, 7)
    q[0, 0, 2, 2] = 1.0
    shifted = warp_internal_structure(q, constant_shift_flow(q, dx=1.0, dy=0.0))
    maximum = torch.nonzero(shifted[0, 0] == shifted.max(), as_tuple=False)
    assert maximum.tolist() == [[2, 3]]


def test_checker_keeps_final_warp_differentiable_but_flow_detached() -> None:
    torch.manual_seed(9)
    logits = torch.randn(1, 8, 12, 16, requires_grad=True)
    q0 = torch.softmax(logits, dim=1)
    raw = torch.randn(1, 10, 3, 4)
    valid = torch.zeros(1, 1, 3, 4, dtype=torch.bool)
    checker = LocalSpectralChecker(
        CheckerConfig(enabled=True, scale=4, common_factor=1, window=4, min_fit_samples=2)
    )

    result = checker(q0, raw, valid)

    assert not result.flow_hr.requires_grad
    assert result.q_star.requires_grad
    assert torch.isfinite(result.q_star).all()
    assert result.fallback_reasons == ("insufficient_valid_observations_or_ill_conditioned",)
    result.q_star.square().mean().backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()


def test_checker_detects_a_controlled_right_shift_on_a_common_grid() -> None:
    torch.manual_seed(21)
    q0 = torch.softmax(torch.randn(1, 8, 16, 18), dim=1)
    shifted = warp_internal_structure(q0, constant_shift_flow(q0, dx=1.0, dy=0.0))
    mixing = torch.randn(8, 10)
    raw = torch.einsum("bkhw,kc->bchw", shifted, mixing)
    valid = torch.ones(1, 1, 16, 18, dtype=torch.bool)
    checker = LocalSpectralChecker(
        CheckerConfig(
            enabled=True,
            scale=1,
            common_factor=1,
            window=8,
            stride=4,
            min_relative_gain=0.0,
            min_relative_gap=0.0,
            movement_penalty=0.0,
            absolute_gain_floor=1.0e-10,
        )
    )

    result = checker(q0, raw, valid)

    assert result.valid_window_count.item() > 0
    assert result.accepted_window_count.item() > 0
    assert result.flow_hr[:, 0].mean().item() > 0
    assert result.internal_error_after.item() < result.internal_error_before.item()


def test_uninformative_tie_conservatively_stays_stationary() -> None:
    q0 = torch.full((1, 8, 16, 20), 1.0 / 8.0)
    raw = torch.zeros(1, 10, 4, 5)
    valid = torch.ones(1, 1, 4, 5, dtype=torch.bool)
    checker = LocalSpectralChecker(
        CheckerConfig(enabled=True, scale=4, common_factor=1, window=5, min_fit_samples=2)
    )

    result = checker(q0, raw, valid)

    assert torch.count_nonzero(result.flow_hr) == 0
    assert torch.equal(result.q_star, q0)
    assert torch.isfinite(result.scores).all()


def test_pure_shared_spectral_change_does_not_invent_geometry() -> None:
    torch.manual_seed(22)
    q0 = torch.softmax(torch.randn(1, 8, 15, 19), dim=1)
    # An arbitrary shared spectral response is exactly the nuisance matrix the
    # checker refits per candidate; it must not be confused with displacement.
    coefficients = torch.randn(8, 10) * torch.linspace(0.5, 2.0, 10)
    raw = torch.einsum("bkhw,kc->bchw", q0, coefficients)
    checker = LocalSpectralChecker(
        CheckerConfig(
            enabled=True,
            scale=1,
            common_factor=1,
            window=7,
            stride=4,
            min_relative_gain=0.0,
            min_relative_gap=0.0,
            movement_penalty=1.0e-5,
            absolute_gain_floor=1.0e-9,
        )
    )
    result = checker(q0, raw, torch.ones(1, 1, 15, 19, dtype=torch.bool))
    assert result.valid_window_count.item() > 0
    assert torch.count_nonzero(result.flow_hr).item() == 0
    assert torch.allclose(result.q_star, q0)


def test_enabled_checker_covers_odd_non_square_grid_and_invalid_edges() -> None:
    torch.manual_seed(23)
    q0 = torch.softmax(torch.randn(1, 8, 20, 28), dim=1).requires_grad_(True)
    raw = torch.randn(1, 10, 5, 7)
    valid = torch.ones(1, 1, 5, 7, dtype=torch.bool)
    valid[:, :, -1, :] = False
    valid[:, :, :, -1] = False
    checker = LocalSpectralChecker(
        CheckerConfig(enabled=True, scale=4, common_factor=2, window=8, stride=4)
    )
    result = checker(q0, raw, valid)
    assert result.q_star.shape == q0.shape
    assert result.flow_hr.shape == (1, 2, 20, 28)
    assert result.support.shape == (1, 1, 20, 28)
    assert torch.isfinite(result.q_star).all()
    result.q_star.mean().backward()
    assert q0.grad is not None
