from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from PIL import Image

from scripts.evaluate_unfinetuned_base import (
    add_lr_consistency_metrics,
    pad_to_multiple,
    sample_seed,
    summarize,
)


def test_pad_to_multiple_preserves_rgb_content() -> None:
    array = np.arange(5 * 7 * 3, dtype=np.uint8).reshape(5, 7, 3)
    padded, original_size = pad_to_multiple(Image.fromarray(array, mode="RGB"), 4)
    assert original_size == (7, 5)
    assert padded.size == (8, 8)
    result = np.asarray(padded)
    assert np.array_equal(result[:5, :7], array)
    assert np.array_equal(result[:5, 7], array[:, -1])
    assert np.array_equal(result[5:, :7], np.repeat(array[-1:, :, :], 3, axis=0))


def test_summary_has_robust_statistics_and_ignores_metadata() -> None:
    summary = summarize(
        pd.DataFrame({"filename": ["a", "b"], "seed": [1, 2], "psnr": [20.0, 30.0]})
    )
    assert set(summary) == {"psnr"}
    assert summary["psnr"]["mean"] == 25.0
    assert summary["psnr"]["median"] == 25.0
    assert summary["psnr"]["count"] == 2


def test_sample_seed_is_stable_across_conditions() -> None:
    assert sample_seed("scene_001.png") == sample_seed("scene_001.png")
    assert sample_seed("scene_001.png") != sample_seed("scene_002.png")


def test_lr_consistency_is_exact_for_matching_bicubic_observation() -> None:
    lr = torch.rand(3, 4, 5)
    sr = torch.nn.functional.interpolate(
        lr.unsqueeze(0), size=(16, 20), mode="bicubic", align_corners=False
    ).clamp(0, 1)[0]
    expected_lr = torch.nn.functional.interpolate(
        sr.unsqueeze(0), size=(4, 5), mode="bicubic", align_corners=False, antialias=True
    ).clamp(0, 1)[0]
    metrics: dict[str, float] = {}
    add_lr_consistency_metrics(metrics, sr, expected_lr)
    assert metrics["low_frequency_mae"] < 1e-7
    assert metrics["low_frequency_psnr"] == float("inf")
    assert metrics["low_frequency_spectral_angle"] < 1e-5
