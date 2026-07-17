from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from src.infer_upscaler import _pad_to_multiple


def test_pad_to_multiple_preserves_content_and_records_original_size() -> None:
    array = np.arange(159 * 157 * 3, dtype=np.uint8).reshape(159, 157, 3)
    image = Image.fromarray(array, mode="RGB")

    padded, original_size = _pad_to_multiple(image, 4)

    assert original_size == (157, 159)
    assert padded.size == (160, 160)
    padded_array = np.asarray(padded)
    assert np.array_equal(padded_array[:159, :157], array)
    expected_right = np.repeat(array[:, -1:, :], repeats=3, axis=1)
    assert np.array_equal(padded_array[:159, 157:], expected_right)
    assert np.array_equal(padded_array[159:, :157], array[-1:, :, :])
    expected_corner = np.repeat(array[-1:, -1:, :], repeats=3, axis=1)
    assert np.array_equal(padded_array[159:, 157:], expected_corner)


def test_pad_to_multiple_leaves_aligned_image_unchanged() -> None:
    image = Image.new("RGB", (128, 132))

    padded, original_size = _pad_to_multiple(image, 4)

    assert padded is image
    assert original_size == image.size


def test_pad_to_multiple_rejects_invalid_multiple() -> None:
    with pytest.raises(ValueError, match="positive"):
        _pad_to_multiple(Image.new("RGB", (8, 8)), 0)
