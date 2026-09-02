from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import torch
import yaml
from torch import nn

from src.diffusion_prediction import alpha_bar_for_timesteps, model_output_to_x0
from src.rgb_auxiliary_loss import compute_rgb_auxiliary_losses


class _CountingFrozenVAE(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.decoder = nn.Conv2d(4, 3, kernel_size=1, bias=False)
        self.decode_calls = 0
        self.requires_grad_(False)
        self.eval()

    def decode(self, latents: torch.Tensor, return_dict: bool = False):
        assert return_dict is False
        self.decode_calls += 1
        return (self.decoder(latents),)


class _FrozenLPIPS(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Conv2d(3, 2, kernel_size=1, bias=False)
        self.requires_grad_(False)
        self.eval()

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        difference = self.features(prediction) - self.features(target)
        return difference.square().mean(dim=(1, 2, 3), keepdim=True)


def test_disabled_rgb_auxiliary_branch_skips_vae_and_preserves_baseline_loss() -> None:
    vae = _CountingFrozenVAE()
    target = torch.zeros(1, 3, 4, 4)
    diffusion_loss = torch.tensor(1.25, requires_grad=True)

    results = compute_rgb_auxiliary_losses(
        predicted_z0=None,
        target_rgb_minus_one_one=target,
        vae=vae,
        lpips_model=None,
        enabled=False,
        lambda_l1=0.1,
        lambda_lpips_rgb=0.1,
        scaling_factor=0.18215,
    )
    total_loss = diffusion_loss + results["loss"]

    assert vae.decode_calls == 0
    assert total_loss.item() == diffusion_loss.item()
    total_loss.backward()
    assert diffusion_loss.grad is not None


def test_rgb_losses_backpropagate_through_frozen_vae_to_epsilon_prediction() -> None:
    torch.manual_seed(7)
    scheduler = SimpleNamespace(alphas_cumprod=torch.tensor([0.99, 0.7, 0.2]))
    timesteps = torch.tensor([1], dtype=torch.long)
    noisy_latents = torch.randn(1, 4, 4, 4)
    model_output = torch.randn(1, 4, 4, 4, requires_grad=True)
    alpha_bar = alpha_bar_for_timesteps(scheduler, timesteps)
    predicted_z0 = model_output_to_x0(
        model_output, noisy_latents, alpha_bar, "epsilon"
    )
    predicted_z0.retain_grad()

    vae = _CountingFrozenVAE()
    lpips_model = _FrozenLPIPS()
    target = torch.rand(1, 3, 4, 4) * 2.0 - 1.0
    results = compute_rgb_auxiliary_losses(
        predicted_z0=predicted_z0,
        target_rgb_minus_one_one=target,
        vae=vae,
        lpips_model=lpips_model,
        enabled=True,
        lambda_l1=0.1,
        lambda_lpips_rgb=0.1,
        scaling_factor=0.5,
    )
    # Keep the total-loss shape while ensuring any model_output gradient comes
    # exclusively from the RGB auxiliary branch in this test.
    diffusion_loss = model_output.sum() * 0.0
    total_loss = diffusion_loss + results["loss"]

    assert vae.decode_calls == 1
    assert torch.isfinite(total_loss)
    total_loss.backward()
    assert predicted_z0.grad is not None
    assert torch.count_nonzero(predicted_z0.grad).item() > 0
    assert model_output.grad is not None
    assert torch.count_nonzero(model_output.grad).item() > 0
    assert all(parameter.grad is None for parameter in vae.parameters())
    assert all(parameter.grad is None for parameter in lpips_model.parameters())


def test_l1_lpips_config_only_changes_the_declared_ablation_fields() -> None:
    baseline = yaml.safe_load(
        Path("configs/stage1_synthetic.yaml").read_text(encoding="utf-8")
    )
    experiment = yaml.safe_load(
        Path("configs/stage1_synthetic_l1_lpips.yaml").read_text(encoding="utf-8")
    )
    expected = dict(baseline)
    expected.update(
        {
            "output_dir": "outputs/stage1_synthetic_l1_lpips",
            "rgb_aux_loss_enabled": True,
            "lambda_l1": 0.1,
            "lambda_lpips_rgb": 0.1,
            "phi_enabled": False,
        }
    )
    assert experiment == expected
    training_source = Path("src/train_lora_upscaler.py").read_text(encoding="utf-8")
    assert 'config.get("rgb_aux_loss_enabled", False)' in training_source
