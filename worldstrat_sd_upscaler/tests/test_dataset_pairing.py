from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from src.dataset import PairedSatelliteDataset, validate_pair_directories


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

