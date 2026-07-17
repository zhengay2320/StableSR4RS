from __future__ import annotations

from pathlib import Path

import pytest

from src.infer_upscaler import _select_input_files


def _inputs(tmp_path: Path) -> Path:
    tmp_path.mkdir(parents=True, exist_ok=True)
    for name in ("a.png", "b.jpg", "c.tif", "ignored.txt"):
        (tmp_path / name).write_bytes(b"placeholder")
    return tmp_path


def test_select_specific_samples_preserves_requested_order(tmp_path: Path) -> None:
    input_dir = _inputs(tmp_path)

    selected = _select_input_files(input_dir, ["c", "a.png", "c.tif"], None, 0, None)

    assert [path.name for path in selected] == ["c.tif", "a.png"]


def test_select_samples_from_file_then_slice(tmp_path: Path) -> None:
    input_dir = _inputs(tmp_path / "images")
    sample_file = tmp_path / "samples.txt"
    sample_file.write_text("# chosen samples\na\nc.tif\nb\n", encoding="utf-8")

    selected = _select_input_files(input_dir, [], sample_file, start_index=1, limit=1)

    assert [path.name for path in selected] == ["c.tif"]


def test_missing_requested_sample_fails_clearly(tmp_path: Path) -> None:
    input_dir = _inputs(tmp_path)

    with pytest.raises(FileNotFoundError, match="missing"):
        _select_input_files(input_dir, ["missing"], None, 0, None)


def test_selection_rejects_invalid_slice(tmp_path: Path) -> None:
    input_dir = _inputs(tmp_path)

    with pytest.raises(ValueError, match="start_index"):
        _select_input_files(input_dir, [], None, -1, None)
    with pytest.raises(ValueError, match="limit"):
        _select_input_files(input_dir, [], None, 0, 0)
