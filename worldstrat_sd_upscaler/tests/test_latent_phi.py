from __future__ import annotations

import torch
import pytest
from torch import nn

from src.diffusion_prediction import (
    model_output_to_x0,
    normalized_timesteps,
    timestep_range_mask,
    x0_to_model_output,
)
from src.latent_phi import LatentPhi, compute_phi_training_losses


def test_tracker_config_serializes_timestep_ranges() -> None:
    from src.utils import tracker_safe_config

    original = {
        "phi_enabled": True,
        "phi_train_timestep_range": [0.0, 0.2],
        "nested": {"range": [0.0, 1.0]},
        "none_value": None,
    }
    converted = tracker_safe_config(original)
    assert converted["phi_enabled"] is True
    assert converted["phi_train_timestep_range"] == "[0.0, 0.2]"
    assert converted["nested"] == '{"range": [0.0, 1.0]}'
    assert converted["none_value"] == "null"
    assert original["phi_train_timestep_range"] == [0.0, 0.2]


def _diffusion_values(alpha_value: float = 0.37):
    torch.manual_seed(11)
    z0 = torch.randn(2, 4, 5, 6)
    noise = torch.randn_like(z0)
    alpha = torch.full((2, 1, 1, 1), alpha_value)
    zt = alpha.sqrt() * z0 + (1.0 - alpha).sqrt() * noise
    velocity = alpha.sqrt() * noise - (1.0 - alpha).sqrt() * z0
    return z0, noise, alpha, zt, velocity


def test_epsilon_model_output_to_x0() -> None:
    z0, noise, alpha, zt, _ = _diffusion_values()
    assert torch.allclose(model_output_to_x0(noise, zt, alpha, "epsilon"), z0, atol=2e-6)


def test_v_prediction_model_output_to_x0() -> None:
    z0, _, alpha, zt, velocity = _diffusion_values()
    assert torch.allclose(
        model_output_to_x0(velocity, zt, alpha, "v_prediction"), z0, atol=2e-6
    )


def test_x0_to_model_output_round_trips_all_parameterizations() -> None:
    z0, noise, alpha, zt, velocity = _diffusion_values()
    expected = {"epsilon": noise, "v_prediction": velocity, "sample": z0}
    for prediction_type, target in expected.items():
        output = x0_to_model_output(z0, zt, alpha, prediction_type)
        recovered = model_output_to_x0(output, zt, alpha, prediction_type)
        assert torch.allclose(output, target, atol=2e-6, rtol=2e-6)
        assert torch.allclose(recovered, z0, atol=2e-6, rtol=2e-6)


def test_high_timestep_conversion_is_finite() -> None:
    z0, noise, _, _, _ = _diffusion_values(alpha_value=1e-7)
    alpha = torch.full((2, 1, 1, 1), 1e-7)
    zt = alpha.sqrt() * z0 + (1.0 - alpha).sqrt() * noise
    recovered = model_output_to_x0(noise, zt, alpha, "epsilon")
    assert torch.isfinite(recovered).all()


def test_normalized_timestep_hard_ranges() -> None:
    normalized = normalized_timesteps(torch.tensor([0, 199, 200, 999]), 1000)
    late = timestep_range_mask(normalized, [0.0, 0.2])
    all_steps = timestep_range_mask(normalized, [0.0, 1.0])
    assert late.tolist() == [True, True, False, False]
    assert all_steps.tolist() == [True, True, True, True]


class _CountingVAE(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(1.0), requires_grad=False)
        self.decode_calls = 0

    def decode(self, latent: torch.Tensor, return_dict: bool = False):
        self.decode_calls += 1
        return (latent[:, :3] * self.scale,)


class _DifferentiableLPIPS(nn.Module):
    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return (prediction - target).square().mean(dim=(1, 2, 3), keepdim=True)


class _CountingPhi(LatentPhi):
    def __init__(self) -> None:
        super().__init__(latent_channels=4, hidden_channels=8, time_embed_dim=8, num_blocks=1)
        self.calls = 0

    def forward(self, predicted_z0: torch.Tensor, normalized_t: torch.Tensor):
        self.calls += 1
        return super().forward(predicted_z0, normalized_t)


class _Scheduler:
    alphas_cumprod = torch.tensor([0.99, 0.92, 0.81, 0.65, 0.42])


def _loss_inputs():
    torch.manual_seed(19)
    return {
        "target_z0": torch.randn(2, 4, 4, 4),
        "sample_zt": torch.randn(2, 4, 4, 4),
        "timesteps": torch.tensor([1, 3]),
        "normalized_t": torch.tensor([0.1, 0.8]),
        "target_rgb_minus_one_one": torch.rand(2, 3, 4, 4) * 2 - 1,
    }


def test_phi_loss_does_not_backpropagate_to_upstream_theta_or_adapter() -> None:
    inputs = _loss_inputs()
    theta = nn.Parameter(torch.tensor(0.7))
    adapter = nn.Parameter(torch.tensor(0.4))
    source = torch.randn(2, 4, 4, 4)
    predicted = theta * source + adapter * source.square()
    phi = _CountingPhi()
    result = compute_phi_training_losses(
        phi=phi, predicted_z0=predicted, target_z0=inputs["target_z0"],
        normalized_t=inputs["normalized_t"],
        target_rgb_minus_one_one=inputs["target_rgb_minus_one_one"],
        sample_zt=inputs["sample_zt"], timesteps=inputs["timesteps"],
        scheduler=_Scheduler(), vae=_CountingVAE(), lpips_model=None,
        timestep_range=[0.0, 1.0], lambda_z0=1.0, lambda_mu=1.0,
        lambda_lpips=0.0, scaling_factor=1.0,
    )
    result["loss"].backward()
    assert theta.grad is None
    assert adapter.grad is None
    assert any(parameter.grad is not None for parameter in phi.parameters())


def test_decoded_lpips_loss_backpropagates_only_to_phi_and_not_vae() -> None:
    inputs = _loss_inputs()
    phi = _CountingPhi()
    vae = _CountingVAE()
    predicted = torch.randn(2, 4, 4, 4, requires_grad=True)
    result = compute_phi_training_losses(
        phi=phi, predicted_z0=predicted, target_z0=inputs["target_z0"],
        normalized_t=inputs["normalized_t"],
        target_rgb_minus_one_one=inputs["target_rgb_minus_one_one"],
        sample_zt=inputs["sample_zt"], timesteps=inputs["timesteps"],
        scheduler=_Scheduler(), vae=vae, lpips_model=_DifferentiableLPIPS(),
        timestep_range=[0.0, 1.0], lambda_z0=0.0, lambda_mu=0.0,
        lambda_lpips=1.0, scaling_factor=1.0,
    )
    result["loss"].backward()
    assert vae.decode_calls == 1
    assert any(parameter.grad is not None for parameter in phi.parameters())
    assert predicted.grad is None
    assert vae.scale.grad is None


def test_inactive_phi_skips_phi_vae_and_lpips() -> None:
    inputs = _loss_inputs()
    phi = _CountingPhi()
    vae = _CountingVAE()
    result = compute_phi_training_losses(
        phi=phi, predicted_z0=torch.randn(2, 4, 4, 4),
        target_z0=inputs["target_z0"], normalized_t=torch.tensor([0.8, 0.9]),
        target_rgb_minus_one_one=inputs["target_rgb_minus_one_one"],
        sample_zt=inputs["sample_zt"], timesteps=inputs["timesteps"],
        scheduler=_Scheduler(), vae=vae, lpips_model=_DifferentiableLPIPS(),
        timestep_range=[0.0, 0.2], lambda_z0=1.0, lambda_mu=1.0,
        lambda_lpips=1.0, scaling_factor=1.0,
    )
    assert result["loss"].item() == 0.0
    assert phi.calls == 0
    assert vae.decode_calls == 0


def test_latent_phi_save_load_round_trip(tmp_path) -> None:
    pytest.importorskip("safetensors")
    torch.manual_seed(23)
    phi = LatentPhi(latent_channels=4, hidden_channels=8, time_embed_dim=8, num_blocks=1)
    sample = torch.randn(2, 4, 5, 5)
    timesteps = torch.tensor([0.2, 0.7])
    expected = phi(sample, timesteps)
    weights, config = phi.save_pretrained(tmp_path)
    restored = LatentPhi.from_pretrained(tmp_path)
    assert weights.name == "latent_phi.safetensors"
    assert config.name == "latent_phi_config.json"
    assert restored.config == phi.config
    actual = restored(sample, timesteps)
    assert torch.allclose(actual.predicted_z0, expected.predicted_z0)
    assert torch.allclose(actual.mixing_weight, expected.mixing_weight)
