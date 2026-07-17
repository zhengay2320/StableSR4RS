from __future__ import annotations

import torch
import pytest

from scripts.analyze_z0_reliability import (
    decode_latent,
    decode_latent_raw,
    evaluated_timesteps,
    fixed_error_heatmap,
    prediction_to_epsilon,
    prediction_to_x0,
    representative_inference_steps,
    representative_timesteps,
)


class _IdentityVAE:
    def decode(self, latent: torch.Tensor, return_dict: bool = False):
        assert return_dict is False
        return (latent,)


def test_epsilon_oracle_recovers_clean_latent() -> None:
    torch.manual_seed(1)
    z0 = torch.randn(1, 4, 8, 8)
    epsilon = torch.randn_like(z0)
    alpha = torch.tensor(0.037).reshape(1, 1, 1, 1)
    zt = alpha.sqrt() * z0 + (1.0 - alpha).sqrt() * epsilon

    recovered = prediction_to_x0(epsilon, zt, alpha, "epsilon")

    assert torch.allclose(recovered, z0, atol=2e-6, rtol=2e-6)


def test_epsilon_prediction_uses_model_output_not_true_noise() -> None:
    """Guard the diagnostic against accidentally turning the oracle into the prediction."""
    torch.manual_seed(7)
    z0 = torch.randn(1, 4, 4, 4)
    true_noise = torch.randn_like(z0)
    deliberately_wrong_model_output = torch.zeros_like(z0)
    alpha = torch.tensor(0.2).reshape(1, 1, 1, 1)
    zt = alpha.sqrt() * z0 + (1.0 - alpha).sqrt() * true_noise

    predicted = prediction_to_x0(deliberately_wrong_model_output, zt, alpha, "epsilon")
    oracle = prediction_to_x0(true_noise, zt, alpha, "epsilon")

    assert torch.allclose(oracle, z0, atol=2e-6, rtol=2e-6)
    assert not torch.allclose(predicted, z0, atol=1e-3, rtol=1e-3)


def test_v_prediction_conversions_recover_x0_and_epsilon() -> None:
    torch.manual_seed(2)
    z0 = torch.randn(1, 4, 5, 6)
    epsilon = torch.randn_like(z0)
    alpha = torch.tensor(0.41).reshape(1, 1, 1, 1)
    zt = alpha.sqrt() * z0 + (1.0 - alpha).sqrt() * epsilon
    velocity = alpha.sqrt() * epsilon - (1.0 - alpha).sqrt() * z0

    assert torch.allclose(prediction_to_x0(velocity, zt, alpha, "v_prediction"), z0, atol=2e-6)
    assert torch.allclose(
        prediction_to_epsilon(velocity, zt, alpha, "v_prediction"), epsilon, atol=2e-6
    )


def test_sample_prediction_is_clean_latent() -> None:
    z0 = torch.randn(1, 4, 4, 4)
    zt = torch.randn_like(z0)
    alpha = torch.tensor(0.5).reshape(1, 1, 1, 1)
    assert torch.equal(prediction_to_x0(z0, zt, alpha, "sample"), z0)


def test_timestep_selection_is_schedule_length_independent() -> None:
    evaluated = evaluated_timesteps(total=37, stride=5)
    selected = representative_timesteps(total=37, available=evaluated)

    assert evaluated[0] == 0
    assert evaluated[-1] == 36
    assert selected[0] == 0
    assert selected[-1] == 36
    assert set(selected).issubset(evaluated)


def test_reverse_trajectory_selection_includes_first_and_last_step() -> None:
    selected = representative_inference_steps(40)

    assert 0 in selected
    assert 39 in selected
    assert all(0 <= step < 40 for step in selected)


def test_decode_diagnostics_preserve_values_before_display_clipping() -> None:
    latent = torch.tensor([[[[-3.0, 0.0, 3.0]]]])
    raw = decode_latent_raw(_IdentityVAE(), latent, scaling_factor=1.0)
    clipped = decode_latent(_IdentityVAE(), latent, scaling_factor=1.0)

    assert raw.min().item() == -1.0
    assert raw.max().item() == 2.0
    assert clipped.min().item() == 0.0
    assert clipped.max().item() == 1.0


def test_error_heatmap_uses_explicit_shared_scale() -> None:
    target = torch.zeros(1, 3, 1, 1)
    prediction = torch.full_like(target, 0.125)
    pixel = fixed_error_heatmap(prediction, target, display_max=0.25).getpixel((0, 0))

    assert pixel == (255, 0, 0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_error_heatmap_accepts_cuda_tensors() -> None:
    target = torch.zeros(1, 3, 2, 2, device="cuda")
    prediction = torch.full_like(target, 0.125)

    image = fixed_error_heatmap(prediction, target, display_max=0.25)

    assert image.size == (2, 2)
    assert image.getpixel((0, 0)) == (255, 0, 0)
