from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from src.dataset import PairedSatelliteDataset, validate_pair_directories
from src.tri_input_data import RawMultispectral, SENTINEL2_L2A_BANDS


def _save_pattern(path: Path, width: int, height: int) -> None:
    y, x = np.mgrid[:height, :width]
    image = np.stack((x % 256, y % 256, (x + y) % 256), axis=-1).astype(np.uint8)
    Image.fromarray(image, mode="RGB").save(path)


def _tree(tmp_path: Path) -> Path:
    for split in ("train", "val"):
        for subdir in ("GT", "LR", "LR_bicubic"):
            (tmp_path / split / subdir).mkdir(parents=True)
        _save_pattern(tmp_path / split / "GT" / "sample.png", 64, 64)
        _save_pattern(tmp_path / split / "LR" / "sample.png", 16, 16)
        _save_pattern(tmp_path / split / "LR_bicubic" / "sample.png", 16, 16)
    return tmp_path


def test_exact_pairing_and_aligned_crop(tmp_path: Path) -> None:
    root = _tree(tmp_path)
    dataset = PairedSatelliteDataset(
        root,
        "train",
        "LR",
        gt_crop_size=32,
        scale=4,
        training=False,
        strict_pairs=True,
        augment=False,
    )
    sample = dataset[0]
    assert sample["gt"].shape == (3, 32, 32)
    assert sample["lr"].shape == (3, 8, 8)
    assert sample["source_type"] == "real"
    # Center crop starts at LR (4,4), exactly GT (16,16).
    assert sample["lr"][0, 0, 0].item() == pytest.approx((4 / 255) * 2 - 1)
    assert sample["gt"][0, 0, 0].item() == pytest.approx((16 / 255) * 2 - 1)


def test_missing_and_scale_errors_are_logged(tmp_path: Path) -> None:
    gt, lr = tmp_path / "GT", tmp_path / "LR"
    gt.mkdir()
    lr.mkdir()
    _save_pattern(gt / "bad.png", 63, 64)
    _save_pattern(lr / "bad.png", 16, 16)
    _save_pattern(gt / "missing.png", 64, 64)
    log = tmp_path / "invalid.csv"
    with pytest.raises(RuntimeError, match="No valid image pairs"):
        validate_pair_directories(gt, lr, invalid_log_path=log)
    text = log.read_text(encoding="utf-8")
    assert "scale_mismatch" in text
    assert "missing_lr" in text


def test_too_small_crop_raises_with_filename(tmp_path: Path) -> None:
    root = _tree(tmp_path)
    dataset = PairedSatelliteDataset(root, "train", "LR", gt_crop_size=128, training=False)
    with pytest.raises(ValueError, match="sample.png.*smaller"):
        _ = dataset[0]


def test_synthetic_replay_is_selectable(tmp_path: Path) -> None:
    root = _tree(tmp_path)
    dataset = PairedSatelliteDataset(
        root,
        "train",
        "LR",
        gt_crop_size=32,
        synthetic_lr_subdir="LR_bicubic",
        synthetic_replay_probability=1.0,
    )
    assert dataset[0]["source_type"] == "bicubic"


def _auxiliary_fixture(root: Path) -> tuple[Path, Path]:
    raw_dir, prior_dir = root / "raw", root / "prior"
    raw_dir.mkdir()
    prior_dir.mkdir()
    (raw_dir / "sample.tiff").write_bytes(b"fixture")
    np.save(prior_dir / "sample.npy", np.zeros((10, 16, 16), dtype=np.float32))
    return raw_dir, prior_dir


def test_tri_input_crop_and_spatial_transform_are_shared(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _tree(tmp_path)
    raw_dir, prior_dir = _auxiliary_fixture(root)
    y, x = torch.meshgrid(torch.arange(16), torch.arange(16), indexing="ij")
    raw = torch.stack([x.float() + channel * 100 for channel in range(12)])
    prior = torch.stack(
        [((x + y + channel) % 17).float() / 16.0 for channel in range(10)]
    ).to(torch.float32)
    prior_before = prior.clone()

    def fake_load(pair, band_names, value_conversion):
        del band_names, value_conversion
        return (
            RawMultispectral(
                raw_ms=raw.clone(),
                raw_valid=torch.ones(1, 16, 16, dtype=torch.bool),
                band_names=SENTINEL2_L2A_BANDS,
                path=pair.raw_ms_path,
                crs=None,
                transform=None,
                nodata=None,
            ),
            prior.clone(),
        )

    monkeypatch.setattr("src.tri_input_data.load_auxiliary_pair", fake_load)
    dataset = PairedSatelliteDataset(
        root,
        "train",
        "LR",
        gt_crop_size=32,
        scale=4,
        training=True,
        augment=True,
        tri_input_enabled=True,
        raw_ms_dir=raw_dir,
        unmixing_dir=prior_dir,
        raw_band_names=SENTINEL2_L2A_BANDS,
        raw_value_conversion={"mode": "identity"},
    )
    monkeypatch.setattr(dataset, "_aligned_crop_boxes", lambda *_: ((2, 3, 10, 11), (8, 12, 40, 44)))
    monkeypatch.setattr(dataset, "_spatial_transform_parameters", lambda: (True, True, 90))

    sample = dataset[0]

    lr_red = ((sample["lr"][0] + 1.0) * 0.5 * 255.0).round()
    assert torch.equal(lr_red, sample["raw_ms"][0])
    assert sample["raw_ms"].shape == (12, 8, 8)
    assert sample["unmixing"].shape == (10, 8, 8)
    assert sample["raw_valid"].all()
    assert sample["aux_present"].item() is True
    assert sample["lr_crop_box"].tolist() == [2, 3, 10, 11]
    assert sample["spatial_transform"].tolist() == [1, 1, 1]
    assert torch.equal(prior, prior_before), "offline F/U input must never be modified"


def test_tri_input_synthetic_replay_requires_explicit_full_branch_disable(tmp_path: Path) -> None:
    root = _tree(tmp_path)
    raw_dir, prior_dir = _auxiliary_fixture(root)
    common = dict(
        data_root=root,
        split="train",
        lr_subdir="LR",
        gt_crop_size=32,
        synthetic_lr_subdir="LR_bicubic",
        synthetic_replay_probability=1.0,
        tri_input_enabled=True,
        raw_ms_dir=raw_dir,
        unmixing_dir=prior_dir,
        raw_band_names=SENTINEL2_L2A_BANDS,
        raw_value_conversion={"mode": "identity"},
    )
    with pytest.raises(ValueError, match="cannot pair real raw/unmixing"):
        PairedSatelliteDataset(**common)

    dataset = PairedSatelliteDataset(**common, synthetic_aux_policy="disable")
    sample = dataset[0]
    assert sample["source_type"] == "bicubic"
    assert sample["aux_present"].item() is False
    assert torch.count_nonzero(sample["raw_ms"]).item() == 0
    assert torch.count_nonzero(sample["unmixing"][:5]).item() == 0
    assert torch.all(sample["unmixing"][5:] == 1)
    assert not sample["raw_valid"].any()


def test_rgb_only_dataset_does_not_require_auxiliary_paths(tmp_path: Path) -> None:
    root = _tree(tmp_path)
    dataset = PairedSatelliteDataset(
        root,
        "train",
        "LR",
        gt_crop_size=32,
        tri_input_enabled=False,
    )
    sample = dataset[0]
    assert "raw_ms" not in sample
    assert "unmixing" not in sample
