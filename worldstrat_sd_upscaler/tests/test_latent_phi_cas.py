from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
import torch
from PIL import Image
from torch import nn

from src.diffusion_prediction import mix_reverse_means, posterior_mean_from_x0
from src.infer_stage1_latent_phi import build_dataset, ddim_reverse_mean_step
from src.latent_phi import LatentPhi, PhiOutput, compute_phi_training_losses


class Scheduler:
    alphas_cumprod = torch.tensor([0.99, 0.92, 0.81, 0.65, 0.42])


def test_latent_phi_outputs_refined_z0_and_spatial_r() -> None:
    phi = LatentPhi(4, hidden_channels=8, time_embed_dim=8, num_blocks=1)
    output = phi(torch.randn(2, 4, 5, 6), torch.tensor([0.1, 0.8]))
    assert isinstance(output, PhiOutput)
    assert output.predicted_z0.shape == (2, 4, 5, 6)
    assert output.mixing_weight.shape == (2, 1, 5, 6)
    assert output.mixing_logits.shape == (2, 1, 5, 6)
    assert bool(((output.mixing_weight >= 0) & (output.mixing_weight <= 1)).all())


def test_posterior_mean_matches_manual_formula_for_batched_timesteps() -> None:
    scheduler = Scheduler()
    zt = torch.randn(3, 4, 2, 2)
    z0 = torch.randn_like(zt)
    timesteps = torch.tensor([0, 2, 4])
    actual = posterior_mean_from_x0(zt, z0, timesteps, scheduler)
    expected = []
    for index, timestep in enumerate(timesteps.tolist()):
        alpha_bar = scheduler.alphas_cumprod[timestep]
        previous = scheduler.alphas_cumprod[timestep - 1] if timestep else torch.tensor(1.0)
        alpha_t = alpha_bar / previous
        beta_t = 1.0 - alpha_t
        coefficient_z0 = previous.sqrt() * beta_t / (1.0 - alpha_bar)
        coefficient_zt = alpha_t.sqrt() * (1.0 - previous) / (1.0 - alpha_bar)
        expected.append(coefficient_z0 * z0[index] + coefficient_zt * zt[index])
    assert torch.allclose(actual, torch.stack(expected), atol=1e-6)
    assert torch.allclose(actual[0], z0[0], atol=1e-6)
    assert torch.isfinite(actual).all()


class CountingVAE(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(1.0), requires_grad=False)
        self.calls = 0

    def decode(self, latent, return_dict=False):
        self.calls += 1
        return (latent[:, :3] * self.scale,)


class CountingLPIPS(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.calls = 0

    def forward(self, prediction, target):
        self.calls += 1
        return (prediction - target).square().mean((1, 2, 3), keepdim=True)


class CountingPhi(LatentPhi):
    def __init__(self) -> None:
        super().__init__(4, hidden_channels=8, time_embed_dim=8, num_blocks=1)
        self.calls = 0

    def forward(self, predicted_z0, normalized_t):
        self.calls += 1
        return super().forward(predicted_z0, normalized_t)


def loss_inputs(normalized_t=None):
    torch.manual_seed(31)
    return {
        "predicted_z0": torch.randn(2, 4, 4, 4),
        "target_z0": torch.randn(2, 4, 4, 4),
        "sample_zt": torch.randn(2, 4, 4, 4),
        "timesteps": torch.tensor([1, 3]),
        "normalized_t": normalized_t if normalized_t is not None else torch.tensor([0.1, 0.8]),
        "target_rgb_minus_one_one": torch.rand(2, 3, 4, 4) * 2 - 1,
    }


def losses(phi, inputs, vae, lpips=None, timestep_range=(0.0, 1.0), z0=1.0, mu=1.0, perceptual=0.0):
    return compute_phi_training_losses(
        phi=phi,
        predicted_z0=inputs["predicted_z0"],
        target_z0=inputs["target_z0"],
        normalized_t=inputs["normalized_t"],
        target_rgb_minus_one_one=inputs["target_rgb_minus_one_one"],
        sample_zt=inputs["sample_zt"],
        timesteps=inputs["timesteps"],
        scheduler=Scheduler(),
        vae=vae,
        lpips_model=lpips,
        timestep_range=timestep_range,
        lambda_z0=z0,
        lambda_mu=mu,
        lambda_lpips=perceptual,
        scaling_factor=1.0,
    )


def test_l_mu_updates_r_head_and_shared_backbone_but_not_z0_head_or_theta() -> None:
    theta = nn.Parameter(torch.tensor(0.7))
    adapter = nn.Parameter(torch.tensor(0.4))
    inputs = loss_inputs()
    source = inputs["predicted_z0"]
    inputs["predicted_z0"] = theta * source + adapter * source.square()
    phi = CountingPhi()
    losses(phi, inputs, CountingVAE(), z0=0.0, mu=1.0)["loss"].backward()
    assert theta.grad is None and adapter.grad is None
    assert any(parameter.grad is not None for parameter in phi.mixing_head.parameters())
    assert any(parameter.grad is not None for parameter in phi.input_conv.parameters())
    assert all(parameter.grad is None for parameter in phi.z0_head.parameters())


def test_z0_and_lpips_reach_z0_head_through_frozen_vae() -> None:
    phi = CountingPhi()
    vae = CountingVAE()
    lpips = CountingLPIPS()
    inputs = loss_inputs()
    inputs["predicted_z0"].requires_grad_(True)
    losses(phi, inputs, vae, lpips, z0=1.0, mu=0.0, perceptual=1.0)["loss"].backward()
    assert vae.calls == 1 and lpips.calls == 1
    assert any(parameter.grad is not None for parameter in phi.z0_head.parameters())
    assert inputs["predicted_z0"].grad is None
    assert vae.scale.grad is None


def test_inactive_batch_skips_every_cas_component(monkeypatch) -> None:
    import src.latent_phi as module

    def forbidden(*args, **kwargs):
        raise AssertionError("posterior helper was called")

    monkeypatch.setattr(module, "posterior_mean_from_x0", forbidden)
    phi, vae, lpips = CountingPhi(), CountingVAE(), CountingLPIPS()
    result = losses(
        phi,
        loss_inputs(torch.tensor([0.8, 0.9])),
        vae,
        lpips,
        timestep_range=(0.0, 0.2),
        perceptual=1.0,
    )
    assert result["loss"].item() == 0.0
    assert phi.calls == vae.calls == lpips.calls == 0
    assert all(parameter.grad is None for parameter in phi.parameters())


def test_spatial_r_broadcast_and_extremes() -> None:
    base = torch.randn(2, 4, 3, 5)
    refined = torch.randn_like(base)
    for value, expected in ((0.0, base), (1.0, refined), (0.5, (base + refined) / 2)):
        r_t = torch.full((2, 1, 3, 5), value)
        assert torch.allclose(mix_reverse_means(base, refined, r_t), expected)
    spatial = torch.rand(2, 1, 3, 5)
    assert torch.allclose(
        mix_reverse_means(base, refined, spatial),
        spatial * refined + (1.0 - spatial) * base,
    )


class FakeDDIM:
    def __init__(self) -> None:
        self.calls = 0

    def step(self, model_output, timestep, sample, eta, return_dict):
        assert eta == 0.0 and return_dict is True
        self.calls += 1
        return SimpleNamespace(prev_sample=sample.float() - 0.25 * model_output.float())


def test_inactive_ddim_is_baseline_and_active_uses_two_same_sample_means() -> None:
    sample = torch.randn(1, 4, 3, 3)
    base_output = torch.randn_like(sample)
    phi_output = torch.randn_like(sample)
    scheduler = FakeDDIM()
    inactive = ddim_reverse_mean_step(scheduler, base_output, 7, sample)
    assert scheduler.calls == 1
    assert torch.equal(inactive, sample - 0.25 * base_output)
    for value in (0.0, 0.5, 1.0):
        scheduler = FakeDDIM()
        r_t = torch.full((1, 1, 3, 3), value)
        actual = ddim_reverse_mean_step(scheduler, base_output, 7, sample, phi_output, r_t)
        base_previous = sample - 0.25 * base_output
        phi_previous = sample - 0.25 * phi_output
        assert scheduler.calls == 2
        assert torch.allclose(actual, r_t * phi_previous + (1.0 - r_t) * base_previous)


def test_full_test_dataset_is_unlimited_unless_limit_is_explicit(tmp_path) -> None:
    lr_dir = tmp_path / "test" / "LR_bicubic"
    gt_dir = tmp_path / "test" / "GT_geo_rad_visual"
    lr_dir.mkdir(parents=True)
    gt_dir.mkdir(parents=True)
    for index in range(3):
        Image.new("RGB", (2, 2), (index, index, index)).save(lr_dir / f"sample_{index}.png")
        Image.new("RGB", (8, 8), (index, index, index)).save(gt_dir / f"sample_{index}.png")
    config = {
        "data_root": str(tmp_path),
        "val_lr_subdir": "LR_bicubic",
        "gt_subdir": "GT_geo_rad_visual",
        "gt_crop_size": 8,
        "scale": 4,
        "strict_pairs": False,
        "prompt_mode": "fixed",
    }
    full = build_dataset(config, split="test", limit=None)
    limited = build_dataset(config, split="test", limit=2)
    assert len(full) == 3
    assert len(limited) == 2
    assert [full[index]["filename"] for index in range(3)] == [
        "sample_0.png",
        "sample_1.png",
        "sample_2.png",
    ]


def test_save_load_restores_z0_and_r_heads_and_rejects_legacy(tmp_path) -> None:
    pytest.importorskip("safetensors")
    phi = LatentPhi(4, hidden_channels=8, time_embed_dim=8, num_blocks=1)
    phi.save_pretrained(tmp_path)
    restored = LatentPhi.from_pretrained(tmp_path)
    assert torch.equal(restored.z0_head.weight, phi.z0_head.weight)
    assert torch.equal(restored.mixing_head.weight, phi.mixing_head.weight)
    config_path = tmp_path / "latent_phi_config.json"
    config = json.loads(config_path.read_text())
    config.pop("outputs_mixing_weight")
    config_path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="legacy LatentPhi"):
        LatentPhi.from_pretrained(tmp_path)
