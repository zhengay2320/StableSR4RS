from __future__ import annotations

import csv
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import src.tri_input_data as tri_data
from src.tri_input_data import (
    SENTINEL2_L2A_BANDS,
    AuxiliaryPair,
    RawMultispectral,
    RawValueConversion,
    build_auxiliary_pairs,
    compute_raw_band_stats,
    load_auxiliary_pair,
    load_raw_band_stats,
    load_raw_multispectral,
    load_unmixing_tensor,
    normalize_raw_multispectral,
    save_raw_band_stats,
    validate_sentinel2_l2a_band_names,
    write_auxiliary_manifest,
)


def _touch(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"fixture")
    return path


def test_unique_stem_pairing_supports_different_extensions_and_manifest(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    prior_dir = tmp_path / "prior"
    raw = _touch(raw_dir / "scene-01.tiff")
    prior_dir.mkdir()
    np.save(prior_dir / "scene-01.npy", np.zeros((10, 2, 3), dtype=np.float32))

    pairs = build_auxiliary_pairs(["scene-01"], raw_dir, prior_dir)

    assert pairs == [AuxiliaryPair("scene-01", raw.resolve(), (prior_dir / "scene-01.npy").resolve())]
    manifest = tmp_path / "outputs" / "manifest.csv"
    write_auxiliary_manifest(pairs, manifest)
    loaded = build_auxiliary_pairs(
        ["scene-01"], raw_dir, prior_dir, manifest_path=manifest
    )
    assert loaded == pairs


def test_manifest_is_the_only_renamed_file_mapping(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    prior_dir = tmp_path / "prior"
    _touch(raw_dir / "renamed.tif")
    prior_dir.mkdir()
    np.save(prior_dir / "another-name.npy", np.zeros((10, 1, 1), dtype=np.float32))
    with pytest.raises(FileNotFoundError, match="raw multispectral.*sample-a"):
        build_auxiliary_pairs(["sample-a"], raw_dir, prior_dir)

    manifest = tmp_path / "mapping.csv"
    with manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["sample_id", "raw_ms_path", "unmixing_path"])
        writer.writeheader()
        writer.writerow(
            {
                "sample_id": "sample-a",
                "raw_ms_path": "renamed.tif",
                "unmixing_path": "another-name.npy",
            }
        )
    pair = build_auxiliary_pairs(
        ["sample-a"], raw_dir, prior_dir, manifest_path=manifest
    )[0]
    assert pair.raw_ms_path.name == "renamed.tif"
    assert pair.unmixing_path.name == "another-name.npy"


def test_duplicate_stem_and_missing_prior_fail_with_paths(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    prior_dir = tmp_path / "prior"
    _touch(raw_dir / "duplicate.tif")
    _touch(raw_dir / "duplicate.tiff")
    prior_dir.mkdir()
    np.save(prior_dir / "duplicate.npy", np.zeros((10, 1, 1), dtype=np.float32))
    with pytest.raises(ValueError, match=r"duplicate.*\.tif.*\.tiff"):
        build_auxiliary_pairs(["duplicate"], raw_dir, prior_dir)

    (raw_dir / "duplicate.tiff").unlink()
    _touch(raw_dir / "missing.tif")
    with pytest.raises(FileNotFoundError, match=r"unmixing.*missing.*missing\.npy"):
        build_auxiliary_pairs(["missing"], raw_dir, prior_dir)


def test_band_layout_is_explicit_complete_and_order_preserving() -> None:
    configured = (
        "B4",
        "B3",
        "B2",
        "B8",
        "B1",
        "B9",
        "B5",
        "B6",
        "B7",
        "B8A",
        "B11",
        "B12",
    )
    layout = validate_sentinel2_l2a_band_names(configured)
    assert layout.band_names == configured
    assert layout.rgb_indices == (0, 1, 2)
    assert layout.b8_index == 3
    assert layout.b1_index == 4
    assert layout.b9_index == 5
    assert tuple(layout.band_names[index] for index in layout.surface_indices) == (
        "B2",
        "B3",
        "B4",
        "B5",
        "B6",
        "B7",
        "B8",
        "B8A",
        "B11",
        "B12",
    )
    with pytest.raises(ValueError, match="will not be guessed"):
        validate_sentinel2_l2a_band_names(None)
    with pytest.raises(ValueError, match="Duplicate"):
        validate_sentinel2_l2a_band_names((*SENTINEL2_L2A_BANDS[:-1], "B11"))
    with pytest.raises(ValueError, match="unexpected=.*B10"):
        validate_sentinel2_l2a_band_names((*SENTINEL2_L2A_BANDS[:-1], "B10"))


def test_raw_value_conversion_must_be_explicit_and_never_assumes_divide_10000() -> None:
    stored = np.array([0, 10000], dtype=np.uint16)
    with pytest.raises(ValueError, match="explicitly configured"):
        RawValueConversion.from_config(None)
    identity = RawValueConversion.from_config({"mode": "identity"})
    assert np.array_equal(identity.apply(stored), np.array([0.0, 10000.0], dtype=np.float32))
    linear = RawValueConversion.from_config(
        {
            "mode": "linear",
            "scale": 0.0001,
            "offset": -0.1,
            "input_units": "DN",
            "output_units": "reflectance",
        }
    )
    assert linear.apply(stored).tolist() == pytest.approx([-0.1, 0.9])


class _FakeRaster:
    count = 12
    height = 2
    width = 3
    crs = "EPSG:32648"
    transform = (10.0, 0.0, 500000.0, 0.0, -10.0, 2000000.0)
    nodata = -9999.0

    def __init__(self) -> None:
        self.values = np.arange(12 * 2 * 3, dtype=np.uint16).reshape(12, 2, 3)
        self.masks = np.full_like(self.values, 255, dtype=np.uint8)
        self.masks[3, 1, 2] = 0

    def __enter__(self) -> "_FakeRaster":
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def read(self, *, masked: bool) -> np.ndarray:
        assert masked is False
        return self.values.copy()

    def read_masks(self) -> np.ndarray:
        return self.masks.copy()


def test_rasterio_is_lazy_raw_valid_is_common_and_zero_is_not_nodata(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = _touch(tmp_path / "raw.tiff")
    fake = _FakeRaster()
    monkeypatch.setattr(tri_data, "_import_rasterio", lambda: SimpleNamespace(open=lambda _: fake))

    sample = load_raw_multispectral(
        path,
        SENTINEL2_L2A_BANDS,
        {"mode": "linear", "scale": 0.5, "offset": 1.0},
        expected_hw=(2, 3),
    )

    assert sample.raw_ms.shape == (12, 2, 3)
    assert sample.raw_ms.dtype == torch.float32
    assert sample.raw_valid.shape == (1, 2, 3)
    assert sample.raw_valid.dtype == torch.bool
    assert sample.raw_valid[0, 0, 0]  # stored zero remains a valid observation
    assert not sample.raw_valid[0, 1, 2]
    assert torch.equal(sample.raw_ms[:, 1, 2], torch.zeros(12))
    assert sample.raw_ms[0, 0, 0].item() == pytest.approx(1.0)
    assert sample.crs == "EPSG:32648"
    assert sample.band_descriptions is None
    with pytest.raises(ValueError, match="silent resize is forbidden"):
        load_raw_multispectral(
            path,
            SENTINEL2_L2A_BANDS,
            {"mode": "identity"},
            expected_hw=(3, 2),
        )


def test_raster_band_descriptions_verify_configured_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = _touch(tmp_path / "described.tif")
    fake = _FakeRaster()
    fake.descriptions = tuple(reversed(SENTINEL2_L2A_BANDS))
    monkeypatch.setattr(tri_data, "_import_rasterio", lambda: SimpleNamespace(open=lambda _: fake))
    with pytest.raises(ValueError, match="descriptions do not match configured order"):
        load_raw_multispectral(path, SENTINEL2_L2A_BANDS, {"mode": "identity"})


def test_unmixing_contract_preserves_f_u_and_unknown_encoding(tmp_path: Path) -> None:
    path = tmp_path / "prior.npy"
    array = np.zeros((10, 3, 4), dtype=np.float32)
    array[:5, 0, 0] = np.array([0.2, 0.3, 0.1, 0.0, 0.0], dtype=np.float32)
    array[5:, 0, 0] = np.array([0.1, 0.2, 0.3, 0.4, 0.5], dtype=np.float32)
    array[5:, 1, 1] = 1.0  # F=0,U=1 is retained as unknown, not renormalized.
    before = array.copy()
    np.save(path, array)

    loaded = load_unmixing_tensor(path, expected_hw=(3, 4))

    assert loaded.dtype == torch.float32
    assert torch.equal(loaded, torch.from_numpy(before))
    assert torch.equal(loaded[:5, 1, 1], torch.zeros(5))
    assert torch.equal(loaded[5:, 1, 1], torch.ones(5))
    assert np.array_equal(array, before)

    np.save(tmp_path / "wrong_dtype.npy", array.astype(np.float64))
    with pytest.raises(TypeError, match="float32"):
        load_unmixing_tensor(tmp_path / "wrong_dtype.npy")
    invalid = before.copy()
    invalid[0, 0, 0] = np.nan
    np.save(tmp_path / "nan.npy", invalid)
    with pytest.raises(ValueError, match="NaN or Inf"):
        load_unmixing_tensor(tmp_path / "nan.npy")
    invalid = before.copy()
    invalid[0, 0, 0] = 1.01
    np.save(tmp_path / "range.npy", invalid)
    with pytest.raises(ValueError, match=r"\[0,1\]"):
        load_unmixing_tensor(tmp_path / "range.npy")


def test_auxiliary_pair_requires_matching_raw_and_unmixing_grid(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    raw_path = _touch(tmp_path / "scene.tif")
    prior_path = tmp_path / "scene.npy"
    np.save(prior_path, np.zeros((10, 3, 2), dtype=np.float32))
    raw = RawMultispectral(
        raw_ms=torch.zeros(12, 2, 3),
        raw_valid=torch.ones(1, 2, 3, dtype=torch.bool),
        band_names=SENTINEL2_L2A_BANDS,
        path=raw_path,
        crs=None,
        transform=None,
        nodata=None,
    )
    monkeypatch.setattr(tri_data, "load_raw_multispectral", lambda *args, **kwargs: raw)
    with pytest.raises(ValueError, match="grid mismatch"):
        load_auxiliary_pair(
            AuxiliaryPair("scene", raw_path, prior_path),
            SENTINEL2_L2A_BANDS,
            {"mode": "identity"},
        )


def test_train_only_stats_ignore_invalid_pixels_and_round_trip(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = [tmp_path / "a.tif", tmp_path / "b.tif"]
    for path in paths:
        _touch(path)

    def fake_load(path: str | Path, band_names: object, conversion: object) -> RawMultispectral:
        index = 0 if Path(path).stem == "a" else 1
        base = torch.arange(12, dtype=torch.float32).view(12, 1, 1)
        values = base + torch.tensor([[[1.0 + index, 3.0 + index]]])
        valid = torch.tensor([[[True, index == 0]]])
        return RawMultispectral(
            raw_ms=values,
            raw_valid=valid,
            band_names=SENTINEL2_L2A_BANDS,
            path=Path(path),
            crs=None,
            transform=None,
            nodata=None,
        )

    monkeypatch.setattr(tri_data, "load_raw_multispectral", fake_load)
    conversion = {"mode": "linear", "scale": 0.0001, "offset": 0.0, "input_units": "DN"}
    with pytest.raises(ValueError, match="only.*train"):
        compute_raw_band_stats(
            paths,
            SENTINEL2_L2A_BANDS,
            conversion,
            split="val",
            data_source="fixture",
        )

    stats = compute_raw_band_stats(
        paths,
        SENTINEL2_L2A_BANDS,
        conversion,
        split="train",
        data_source="unit-test train fixtures",
    )

    # Valid values in band 0 are [1,3] from a and [2] from b.
    assert stats.mean[0] == pytest.approx(2.0)
    assert stats.std[0] == pytest.approx(np.sqrt(2.0 / 3.0))
    assert stats.valid_pixel_count == (3,) * 12
    assert stats.sample_count == 2
    destination = tmp_path / "raw_band_stats.json"
    save_raw_band_stats(stats, destination)
    restored = load_raw_band_stats(
        destination,
        expected_band_names=SENTINEL2_L2A_BANDS,
        expected_conversion=conversion,
    )
    assert restored.mean == pytest.approx(stats.mean)
    assert restored.std == pytest.approx(stats.std)
    assert restored.data_source == "unit-test train fixtures"


def test_normalization_uses_saved_stats_and_keeps_invalid_pixels_zero() -> None:
    conversion = RawValueConversion.from_config({"mode": "identity"})
    stats = tri_data.RawBandStats(
        format_version=1,
        split="train",
        band_names=SENTINEL2_L2A_BANDS,
        raw_value_conversion=conversion,
        mean=(1.0,) * 12,
        std=(2.0,) * 12,
        valid_pixel_count=(2,) * 12,
        sample_count=1,
        data_source="fixture",
        created_utc="2026-01-01T00:00:00+00:00",
        software_versions={},
    )
    raw = torch.full((12, 1, 2), 5.0)
    valid = torch.tensor([[[True, False]]])

    normalized = normalize_raw_multispectral(raw, valid, stats)

    assert torch.equal(normalized[:, 0, 0], torch.full((12,), 2.0))
    assert torch.equal(normalized[:, 0, 1], torch.zeros(12))
