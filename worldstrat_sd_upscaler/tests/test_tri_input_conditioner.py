from __future__ import annotations

import json

import pytest
import torch

from src.local_spectral_checker import CheckerConfig
from src.tri_input_conditioner import TriInputConditioner


SURFACE_INDICES = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9)


def _model(*, checker_enabled: bool = False) -> TriInputConditioner:
    return TriInputConditioner(
        raw_mean=[0.0] * 12,
        raw_std=[1.0] * 12,
        surface_band_indices=SURFACE_INDICES,
        checker_config=CheckerConfig(enabled=checker_enabled, scale=4),
    )


def _inputs(batch: int = 2, height: int = 5, width: int = 7):
    torch.manual_seed(101)
    rgb = torch.rand(batch, 3, height, width) * 2.0 - 1.0
    raw = torch.rand(batch, 12, height, width)
    fractions = torch.rand(batch, 5, height, width)
    uncertainty = torch.rand(batch, 5, height, width)
    unmixing = torch.cat((fractions, uncertainty), dim=1).float()
    valid = torch.ones(batch, 1, height, width, dtype=torch.bool)
    return rgb, raw, unmixing, valid


def test_odd_non_square_shapes_and_softmax_contract() -> None:
    model = _model().eval()
    rgb, raw, unmixing, valid = _inputs()
    output = model(rgb, raw, unmixing, valid, aux_present=torch.tensor([True, False]))

    assert output.q0.shape == output.q_star.shape == (2, 8, 20, 28)
    assert output.flow_hr.shape == (2, 2, 20, 28)
    assert output.support.shape == (2, 1, 20, 28)
    assert output.context_lr.shape == (2, 64, 5, 7)
    assert output.preview.shape == (2, 3, 20, 28)
    assert [tuple(value.shape) for value in output.geometry] == [
        (2, 32, 20, 28),
        (2, 64, 10, 14),
        (2, 96, 5, 7),
        (2, 128, 3, 4),
    ]
    assert torch.allclose(output.q0.sum(dim=1), torch.ones(2, 20, 28), atol=1.0e-6)
    assert output.active_mask.tolist() == [True, False]
    assert output.checker_diagnostics.fallback_reasons == (
        "checker_disabled",
        "checker_disabled",
    )


def test_unmixing_is_immutable_and_detached_while_unknown_prior_keeps_raw_active() -> None:
    model = _model().train()
    rgb, raw, unmixing, valid = _inputs(batch=1)
    unknown = torch.cat(
        (torch.zeros_like(unmixing[:, :5]), torch.ones_like(unmixing[:, 5:])), dim=1
    ).requires_grad_(True)
    original = unknown.detach().clone()
    raw = raw.requires_grad_(True)

    output = model(rgb, raw, unknown, valid, aux_present=True)
    output.context_lr.square().mean().backward()

    assert torch.equal(unknown.detach(), original)
    assert unknown.grad is None
    assert raw.grad is not None and bool((raw.grad.abs() > 0).any())
    assert any(
        parameter.grad is not None and bool((parameter.grad.abs() > 0).any())
        for parameter in model.raw_encoder.parameters()
    )
    assert all(
        parameter.grad is None or bool((parameter.grad == 0).all())
        for parameter in model.prior_encoder.parameters()
    )

    with torch.no_grad():
        changed_raw = model(rgb, raw.detach() + 0.5, unknown.detach(), valid, True)
    assert not torch.allclose(output.context_lr.detach(), changed_raw.context_lr)


def test_conditioner_has_finite_gradients_with_known_prior() -> None:
    model = _model().train()
    rgb, raw, unmixing, valid = _inputs(batch=1, height=3, width=5)
    output = model(rgb, raw, unmixing, valid, aux_present=True)
    loss = (
        output.q_star[:, 0].mean()
        + output.preview.square().mean()
        + sum(features.square().mean() for features in output.geometry)
        + output.context_lr.square().mean()
    )
    assert torch.isfinite(loss)
    loss.backward()

    for module in (model.rgb_encoder, model.raw_encoder, model.prior_encoder, model.fusion):
        gradients = [parameter.grad for parameter in module.parameters() if parameter.grad is not None]
        assert gradients
        assert all(bool(torch.isfinite(gradient).all()) for gradient in gradients)
        assert any(bool((gradient.abs() > 0).any()) for gradient in gradients)


def test_real_checker_is_called_and_preserves_gradient_boundary() -> None:
    model = TriInputConditioner(
        raw_mean=[0.0] * 12,
        raw_std=[1.0] * 12,
        surface_band_indices=SURFACE_INDICES,
        checker_config=CheckerConfig(
            enabled=True,
            scale=4,
            common_factor=2,
            window=4,
            stride=4,
            min_fit_samples=2,
            ridge=1.0e-2,
        ),
    ).train()
    rgb, raw, unmixing, valid = _inputs(batch=1, height=8, width=10)
    output = model(rgb, raw, unmixing, valid, aux_present=True)

    assert output.scores.shape == (1, 5, 4, 5)
    assert output.checker_diagnostics.valid_window_count.shape == (1,)
    assert not output.flow_hr.requires_grad
    assert not output.support.requires_grad
    output.q_star[:, 0].mean().backward()
    assert any(parameter.grad is not None for parameter in model.layout_head.parameters())


def test_save_load_round_trip_is_strict(tmp_path) -> None:
    pytest.importorskip("safetensors")
    model = _model().eval()
    rgb, raw, unmixing, valid = _inputs(batch=1, height=3, width=4)
    with torch.no_grad():
        expected = model(rgb, raw, unmixing, valid, True)

    weights_path, config_path = model.save_pretrained(
        tmp_path, artifact_metadata={"band_names": [f"B{index}" for index in range(12)]}
    )
    restored = TriInputConditioner.from_pretrained(tmp_path).eval()
    with torch.no_grad():
        actual = restored(rgb, raw, unmixing, valid, True)

    assert weights_path.name == "tri_input_conditioner.safetensors"
    assert config_path.name == "tri_input_config.json"
    assert torch.equal(restored.raw_mean, model.raw_mean)
    assert torch.equal(restored.raw_std, model.raw_std)
    assert torch.allclose(actual.q0, expected.q0)
    assert torch.allclose(actual.context_lr, expected.context_lr)
    assert torch.allclose(actual.preview, expected.preview)
    assert restored.artifact_metadata["band_names"][0] == "B0"

    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["module_version"] = -1
    config_path.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(ValueError, match="Unsupported tri-input module version"):
        TriInputConditioner.from_pretrained(tmp_path)


def test_explicit_surface_mapping_and_unmixing_contracts() -> None:
    with pytest.raises(ValueError, match="ten unique"):
        TriInputConditioner([0] * 12, [1] * 12, [0] * 10)
    model = _model()
    rgb, raw, unmixing, valid = _inputs(batch=1)
    with pytest.raises(TypeError, match="float32"):
        model(rgb, raw, unmixing.double(), valid, True)
    broken = unmixing.clone()
    broken[:, 0, 0, 0] = float("nan")
    with pytest.raises(ValueError, match="NaN or Inf"):
        model(rgb, raw, broken, valid, True)


def test_raw_invalid_pixels_mask_raw_and_prior_features() -> None:
    model = _model().eval()
    rgb, raw, unmixing, valid = _inputs(batch=1)
    valid.zero_()
    with torch.no_grad():
        first = model(rgb, raw, unmixing, valid, True)
        second = model(rgb, raw + 100.0, 1.0 - unmixing, valid, True)
    assert torch.allclose(first.context_lr, second.context_lr)
