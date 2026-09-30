"""Stage-1 RGB pairs remain independent of the optional multispectral inputs."""

from __future__ import annotations

import ast
import random
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from PIL import Image

from scripts.train_tri_input_warmup import _build_dataset
from scripts.validate_tri_input_data import _split_rgb_dir, validate_split
from src.dataset import PairedSatelliteDataset
from src.tri_input_data import SENTINEL2_L2A_BANDS


def _write_raw(path: Path, raw: np.ndarray) -> None:
    rasterio = pytest.importorskip("rasterio")
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=raw.shape[2],
        height=raw.shape[1],
        count=12,
        dtype="float32",
        transform=rasterio.transform.from_origin(0, 80, 10, 10),
        crs="EPSG:32648",
    ) as destination:
        destination.write(raw)


def _fixture(tmp_path: Path) -> tuple[dict, dict[str, np.ndarray]]:
    """Use RGB files, raw bands and distractor RGB directories with distinct values."""
    y, x = np.mgrid[:8, :10]
    rgb = np.stack((x * 9 + 11, y * 12 + 17, x * 5 + y * 3 + 21), axis=-1).astype(np.uint8)
    target = np.repeat(np.repeat(rgb + 23, 4, axis=0), 4, axis=1)
    raw = np.stack([(x + 3 * y + channel + 2) / 100 for channel in range(12)]).astype(np.float32)
    prior = np.stack([(x + y + channel) / 50 for channel in range(10)]).astype(np.float32)
    config = {
        "data_root": str(tmp_path / "rgb"),
        "train_lr_subdir": "LR_bicubic",
        "val_lr_subdir": "LR_bicubic",
        "test_lr_subdir": "LR_bicubic",
        "gt_subdir": "GT_geo_rad_visual",
        "gt_crop_size": 16,
        "scale": 4,
        "strict_pairs": True,
        "augment": False,
        "prompt_dropout_probability": 0.0,
        "synthetic_replay_probability": 0.0,
        "tri_input": {
            "enabled": True,
            "synthetic_aux_policy": "error",
            "band_names": list(SENTINEL2_L2A_BANDS),
            "raw_value_conversion": {"mode": "identity", "input_units": "fixture"},
            "data": {},
        },
    }
    for split in ("train", "val", "test"):
        split_root = Path(config["data_root"]) / split
        for name, pixels in (
            ("LR_bicubic", rgb),
            ("GT_geo_rad_visual", target),
            ("LR", np.full_like(rgb, 251)),
            ("GT", np.full_like(target, 3)),
        ):
            directory = split_root / name
            directory.mkdir(parents=True)
            Image.fromarray(pixels).save(directory / "scene.png")
        raw_dir = tmp_path / "auxiliary" / split / "raw"
        prior_dir = tmp_path / "auxiliary" / split / "prior"
        raw_dir.mkdir(parents=True)
        prior_dir.mkdir(parents=True)
        _write_raw(raw_dir / "scene.tiff", raw)
        np.save(prior_dir / "scene.npy", prior)
        config["tri_input"]["data"][split] = {
            "raw_ms_dir": str(raw_dir),
            "unmixing_dir": str(prior_dir),
        }
    return config, {"rgb": rgb, "target": target, "raw": raw, "prior": prior}


def _datasets(config: dict, output: Path, entrypoint: str) -> list:
    if entrypoint == "warmup":
        return [_build_dataset(config, output)]
    # Execute the actual dataset-builder body in CPU isolation. Importing the
    # whole training module requires Accelerate, which these data-only tests do
    # not exercise; no training runtime is mocked or claimed as tested here.
    path = Path(__file__).resolve().parents[1] / "src" / "train_lora_upscaler.py"
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    builder = next(
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "build_datasets"
    )
    namespace = {"Any": Any, "Path": Path, "PairedSatelliteDataset": PairedSatelliteDataset}
    isolated = ast.Module(body=[builder], type_ignores=[])
    exec(compile(isolated, str(path), "exec"), namespace)
    return list(namespace["build_datasets"](config, output, write_invalid_logs=False))


def _rgb_tensor(pixels: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(pixels.copy()).permute(2, 0, 1).float() / 127.5 - 1.0


def _expected_crop(value: torch.Tensor, sample: dict, scale: int = 1) -> torch.Tensor:
    left, top, right, bottom = sample["lr_crop_box"].tolist()
    value = value[:, top * scale : bottom * scale, left * scale : right * scale]
    mirror, flip, quarter_turns = sample["spatial_transform"].tolist()
    if mirror:
        value = value.flip(-1)
    if flip:
        value = value.flip(-2)
    return torch.rot90(value, quarter_turns, (-2, -1))


@pytest.mark.parametrize("entrypoint", ["diffusion", "warmup"])
def test_synthetic_training_reads_separate_rgb_and_target_files(
    tmp_path: Path, entrypoint: str
) -> None:
    config, arrays = _fixture(tmp_path)
    for dataset in _datasets(config, tmp_path / "outputs", entrypoint):
        random.seed(17)
        sample = dataset[0]
        assert sample["sample_id"] == "scene"
        assert sample["source_type"] == "bicubic"
        assert sample["aux_present"].item() is True
        assert sample["raw_valid"].all()
        assert dataset.records[0].lr_path.parent.name == "LR_bicubic"
        assert dataset.records[0].gt_path.parent.name == "GT_geo_rad_visual"
        assert Path(sample["raw_ms_path"]).name == "scene.tiff"
        assert Path(sample["unmixing_path"]).name == "scene.npy"
        torch.testing.assert_close(sample["lr"], _expected_crop(_rgb_tensor(arrays["rgb"]), sample))
        torch.testing.assert_close(sample["gt"], _expected_crop(_rgb_tensor(arrays["target"]), sample, 4))
        torch.testing.assert_close(sample["raw_ms"], _expected_crop(torch.from_numpy(arrays["raw"]), sample))
        torch.testing.assert_close(sample["unmixing"], _expected_crop(torch.from_numpy(arrays["prior"]), sample))

        # The auxiliary bands can change without replacing either RGB tensor.
        changed_raw = arrays["raw"] + 0.35
        _write_raw(Path(sample["raw_ms_path"]), changed_raw)
        random.seed(17)
        changed_sample = dataset[0]
        assert torch.equal(sample["lr"], changed_sample["lr"])
        assert torch.equal(sample["gt"], changed_sample["gt"])
        assert not torch.equal(sample["raw_ms"], changed_sample["raw_ms"])
        assert changed_sample["aux_present"].item() is True


def test_synthetic_rgb_and_auxiliary_crop_and_transform_stay_synchronized(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, arrays = _fixture(tmp_path)
    config["augment"] = True
    dataset = _build_dataset(config, tmp_path / "outputs")
    coordinates = iter((2, 1))
    monkeypatch.setattr("src.dataset.random.randint", lambda low, high: next(coordinates))
    monkeypatch.setattr(dataset, "_spatial_transform_parameters", lambda: (True, True, 90))
    sample = dataset[0]
    assert sample["lr_crop_box"].tolist() == [2, 1, 6, 5]
    assert sample["spatial_transform"].tolist() == [1, 1, 1]
    torch.testing.assert_close(sample["lr"], _expected_crop(_rgb_tensor(arrays["rgb"]), sample))
    torch.testing.assert_close(sample["gt"], _expected_crop(_rgb_tensor(arrays["target"]), sample, 4))
    torch.testing.assert_close(sample["raw_ms"], _expected_crop(torch.from_numpy(arrays["raw"]), sample))
    torch.testing.assert_close(sample["unmixing"], _expected_crop(torch.from_numpy(arrays["prior"]), sample))


@pytest.mark.parametrize("entrypoint", ["diffusion", "warmup"])
@pytest.mark.parametrize("missing_subdir,reason", [("LR_bicubic", "missing_lr"), ("GT_geo_rad_visual", "missing_gt")])
def test_missing_selected_rgb_never_falls_back_to_raw_bands(
    tmp_path: Path, entrypoint: str, missing_subdir: str, reason: str
) -> None:
    config, _ = _fixture(tmp_path)
    (Path(config["data_root"]) / "train" / missing_subdir / "scene.png").unlink()
    with pytest.raises(ValueError, match=rf"scene\.png.*{reason}"):
        _datasets(config, tmp_path / "outputs", entrypoint)


@pytest.mark.parametrize("split", ["train", "val", "test"])
def test_precheck_uses_configured_synthetic_rgb_for_each_split(tmp_path: Path, split: str) -> None:
    config, _ = _fixture(tmp_path)
    rgb_dir = _split_rgb_dir(config, split, None)
    assert rgb_dir == Path(config["data_root"]) / split / "LR_bicubic"
    report = validate_split(config, split, None, tmp_path / "check" / split, None)
    assert report["success_count"] == 1
    assert report["failure_count"] == 0
