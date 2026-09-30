"""Strict data contracts for the optional tri-input conditioning path.

This module deliberately does not import :mod:`rasterio` at module import time.
RGB-only training therefore keeps its existing dependency and import behaviour.
The helpers here never modify source rasters or offline unmixing arrays.
"""

from __future__ import annotations

import csv
import json
import platform
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch


RAW_RASTER_EXTENSIONS = frozenset({".tif", ".tiff"})
UNMIXING_EXTENSIONS = frozenset({".npy"})
SENTINEL2_L2A_BANDS = (
    "B1",
    "B2",
    "B3",
    "B4",
    "B5",
    "B6",
    "B7",
    "B8",
    "B8A",
    "B9",
    "B11",
    "B12",
)
SENTINEL2_SURFACE_BANDS = (
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
UNMIXING_COMPONENTS = ("vegetation", "water", "bare", "snow", "building")
RAW_STATS_FORMAT_VERSION = 1


@dataclass(frozen=True)
class Sentinel2BandLayout:
    """Validated mapping from configured raster positions to Sentinel-2 bands."""

    band_names: tuple[str, ...]
    rgb_indices: tuple[int, int, int]
    b8_index: int
    b1_index: int
    b9_index: int
    surface_indices: tuple[int, ...]


@dataclass(frozen=True)
class RawValueConversion:
    """Explicit affine conversion from stored values to network physical units.

    ``linear`` means ``converted = stored * scale + offset``.  ``identity`` is
    represented separately so callers cannot silently assume a 1/10000 scale.
    """

    mode: str
    scale: float
    offset: float
    input_units: str | None = None
    output_units: str | None = None

    @classmethod
    def from_config(cls, config: Mapping[str, Any] | None) -> "RawValueConversion":
        if config is None:
            raise ValueError(
                "raw_value_conversion must be explicitly configured; use "
                "{'mode': 'identity'} only when stored TIFF values already have the desired units"
            )
        if not isinstance(config, Mapping):
            raise TypeError(f"raw_value_conversion must be a mapping, got {type(config).__name__}")
        mode = str(config.get("mode", "")).strip().lower()
        allowed_keys = {"mode", "scale", "offset", "input_units", "output_units"}
        unknown = sorted(set(config) - allowed_keys)
        if unknown:
            raise ValueError(f"Unknown raw_value_conversion fields: {unknown}")
        if mode == "identity":
            scale = float(config.get("scale", 1.0))
            offset = float(config.get("offset", 0.0))
            if scale != 1.0 or offset != 0.0:
                raise ValueError("identity conversion cannot specify a non-identity scale or offset")
        elif mode == "linear":
            if "scale" not in config:
                raise ValueError("linear raw_value_conversion requires an explicit scale")
            scale = float(config["scale"])
            offset = float(config.get("offset", 0.0))
        else:
            raise ValueError("raw_value_conversion.mode must be 'identity' or 'linear'")
        if not np.isfinite(scale) or scale == 0.0:
            raise ValueError(f"raw conversion scale must be finite and nonzero, got {scale}")
        if not np.isfinite(offset):
            raise ValueError(f"raw conversion offset must be finite, got {offset}")
        return cls(
            mode=mode,
            scale=scale,
            offset=offset,
            input_units=_optional_string(config.get("input_units")),
            output_units=_optional_string(config.get("output_units")),
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "scale": self.scale,
            "offset": self.offset,
            "input_units": self.input_units,
            "output_units": self.output_units,
        }

    def apply(self, stored: np.ndarray) -> np.ndarray:
        values = np.asarray(stored, dtype=np.float32)
        converted = values * np.float32(self.scale) + np.float32(self.offset)
        return np.ascontiguousarray(converted, dtype=np.float32)


@dataclass(frozen=True)
class AuxiliaryPair:
    """One explicit sample-id mapping to raw multispectral and offline prior files."""

    sample_id: str
    raw_ms_path: Path
    unmixing_path: Path


@dataclass(frozen=True)
class RawMultispectral:
    """A converted twelve-band raster and its common valid-observation mask."""

    raw_ms: torch.Tensor
    raw_valid: torch.Tensor
    band_names: tuple[str, ...]
    path: Path
    crs: str | None
    transform: tuple[float, ...] | None
    nodata: float | None
    band_descriptions: tuple[str, ...] | None = None


@dataclass(frozen=True)
class RawBandStats:
    """Train-only population statistics for converted raw bands."""

    format_version: int
    split: str
    band_names: tuple[str, ...]
    raw_value_conversion: RawValueConversion
    mean: tuple[float, ...]
    std: tuple[float, ...]
    valid_pixel_count: tuple[int, ...]
    sample_count: int
    data_source: str
    created_utc: str
    software_versions: Mapping[str, str]

    def as_dict(self) -> dict[str, Any]:
        return {
            "format_version": self.format_version,
            "split": self.split,
            "band_names": list(self.band_names),
            "raw_value_conversion": self.raw_value_conversion.as_dict(),
            "mean": list(self.mean),
            "std": list(self.std),
            "valid_pixel_count": list(self.valid_pixel_count),
            "sample_count": self.sample_count,
            "data_source": self.data_source,
            "created_utc": self.created_utc,
            "software_versions": dict(self.software_versions),
        }


def _optional_string(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _normalise_band_name(name: Any) -> str:
    value = str(name).strip().upper()
    if not value:
        raise ValueError("Sentinel-2 band names cannot be empty")
    if value.startswith("B") and value[1:].isdigit():
        value = f"B{int(value[1:])}"
    return value


def validate_sentinel2_l2a_band_names(band_names: Sequence[str] | None) -> Sentinel2BandLayout:
    """Validate an explicit, ordered twelve-band L2A mapping.

    The order is intentionally not inferred from raster band count or filenames.
    B1 and B9 remain available to the context encoder, while the ten surface
    bands are identified separately for the local spectral checker.
    """

    if band_names is None:
        raise ValueError("band_names is required; raster band order will not be guessed")
    names = tuple(_normalise_band_name(name) for name in band_names)
    if len(names) != 12:
        raise ValueError(f"Expected exactly 12 Sentinel-2 L2A band names, got {len(names)}: {names}")
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        raise ValueError(f"Duplicate Sentinel-2 band names are not allowed: {duplicates}")
    expected = set(SENTINEL2_L2A_BANDS)
    missing = sorted(expected - set(names))
    unexpected = sorted(set(names) - expected)
    if missing or unexpected:
        raise ValueError(
            "Configured Sentinel-2 L2A bands must contain exactly "
            f"{SENTINEL2_L2A_BANDS}; missing={missing}, unexpected={unexpected}"
        )
    index = {name: position for position, name in enumerate(names)}
    return Sentinel2BandLayout(
        band_names=names,
        rgb_indices=(index["B4"], index["B3"], index["B2"]),
        b8_index=index["B8"],
        b1_index=index["B1"],
        b9_index=index["B9"],
        surface_indices=tuple(index[name] for name in SENTINEL2_SURFACE_BANDS),
    )


def _index_unique_stems(
    directory: str | Path,
    extensions: frozenset[str],
    label: str,
    *,
    recursive: bool,
) -> dict[str, Path]:
    root = Path(directory).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"{label} directory does not exist: {root}")
    iterator = root.rglob("*") if recursive else root.iterdir()
    paths = sorted(
        (path.resolve() for path in iterator if path.is_file() and path.suffix.lower() in extensions),
        key=lambda path: str(path),
    )
    by_stem: dict[str, Path] = {}
    spelling_by_folded: dict[str, str] = {}
    for path in paths:
        sample_id = path.stem
        folded = sample_id.casefold()
        if folded in spelling_by_folded:
            previous_id = spelling_by_folded[folded]
            previous_path = by_stem[previous_id]
            raise ValueError(
                f"Ambiguous {label} stem {sample_id!r}: multiple candidates "
                f"{previous_path} and {path}"
            )
        spelling_by_folded[folded] = sample_id
        by_stem[sample_id] = path
    return by_stem


def _validate_sample_ids(sample_ids: Iterable[str]) -> tuple[str, ...]:
    result: list[str] = []
    seen: dict[str, str] = {}
    for raw_id in sample_ids:
        sample_id = str(raw_id).strip()
        if not sample_id:
            raise ValueError("sample_id cannot be empty")
        folded = sample_id.casefold()
        if folded in seen:
            raise ValueError(f"Duplicate requested sample_id {sample_id!r}; first occurrence={seen[folded]!r}")
        seen[folded] = sample_id
        result.append(sample_id)
    if not result:
        raise ValueError("At least one sample_id is required")
    return tuple(result)


def _resolve_manifest_path(value: str, base_dir: Path, sample_id: str, field: str) -> Path:
    if not value.strip():
        raise ValueError(f"Manifest sample {sample_id!r} has an empty {field}")
    candidate = Path(value).expanduser()
    path = candidate.resolve() if candidate.is_absolute() else (base_dir / candidate).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Manifest sample {sample_id!r} {field} does not exist: {path}")
    return path


def _pairs_from_manifest(
    sample_ids: tuple[str, ...],
    manifest_path: Path,
    raw_ms_dir: Path,
    unmixing_dir: Path,
) -> list[AuxiliaryPair]:
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Auxiliary manifest does not exist: {manifest_path}")
    with manifest_path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        required = {"sample_id", "raw_ms_path", "unmixing_path"}
        missing_columns = sorted(required - set(reader.fieldnames or ()))
        if missing_columns:
            raise ValueError(f"Manifest {manifest_path} is missing columns: {missing_columns}")
        entries: dict[str, AuxiliaryPair] = {}
        folded_ids: dict[str, str] = {}
        for row_number, row in enumerate(reader, start=2):
            sample_id = str(row.get("sample_id", "")).strip()
            if not sample_id:
                raise ValueError(f"Manifest {manifest_path}:{row_number} has an empty sample_id")
            folded = sample_id.casefold()
            if folded in folded_ids:
                raise ValueError(
                    f"Manifest {manifest_path} contains duplicate sample_id {sample_id!r} "
                    f"(also {folded_ids[folded]!r})"
                )
            raw_path = _resolve_manifest_path(
                str(row.get("raw_ms_path", "")), raw_ms_dir, sample_id, "raw_ms_path"
            )
            unmixing_path = _resolve_manifest_path(
                str(row.get("unmixing_path", "")), unmixing_dir, sample_id, "unmixing_path"
            )
            if raw_path.suffix.lower() not in RAW_RASTER_EXTENSIONS:
                raise ValueError(f"Manifest sample {sample_id!r} raw path is not TIFF: {raw_path}")
            if unmixing_path.suffix.lower() not in UNMIXING_EXTENSIONS:
                raise ValueError(f"Manifest sample {sample_id!r} unmixing path is not NPY: {unmixing_path}")
            folded_ids[folded] = sample_id
            entries[sample_id] = AuxiliaryPair(sample_id, raw_path, unmixing_path)
    pairs: list[AuxiliaryPair] = []
    for sample_id in sample_ids:
        pair = entries.get(sample_id)
        if pair is None:
            raise FileNotFoundError(
                f"Manifest {manifest_path} has no auxiliary mapping for sample {sample_id!r}; "
                f"raw root={raw_ms_dir}, unmixing root={unmixing_dir}"
            )
        pairs.append(AuxiliaryPair(sample_id, pair.raw_ms_path, pair.unmixing_path))
    return pairs


def build_auxiliary_pairs(
    sample_ids: Iterable[str],
    raw_ms_dir: str | Path,
    unmixing_dir: str | Path,
    *,
    manifest_path: str | Path | None = None,
    recursive: bool = False,
) -> list[AuxiliaryPair]:
    """Pair RGB sample ids with raw TIFF and offline NPY without order matching.

    With no manifest, both auxiliary filenames must have the same unique stem as
    ``sample_id``; their extensions may differ from the RGB file.  A manifest is
    the only supported way to map renamed files.  Relative manifest paths are
    resolved against ``raw_ms_dir`` and ``unmixing_dir`` respectively.
    """

    ids = _validate_sample_ids(sample_ids)
    raw_root = Path(raw_ms_dir).expanduser().resolve()
    prior_root = Path(unmixing_dir).expanduser().resolve()
    if manifest_path is not None:
        manifest = Path(manifest_path).expanduser().resolve()
        return _pairs_from_manifest(ids, manifest, raw_root, prior_root)

    raw_by_stem = _index_unique_stems(raw_root, RAW_RASTER_EXTENSIONS, "raw multispectral", recursive=recursive)
    prior_by_stem = _index_unique_stems(prior_root, UNMIXING_EXTENSIONS, "unmixing", recursive=recursive)
    pairs: list[AuxiliaryPair] = []
    for sample_id in ids:
        raw_path = raw_by_stem.get(sample_id)
        prior_path = prior_by_stem.get(sample_id)
        if raw_path is None:
            expected = raw_root / f"{sample_id}.tif[f]"
            raise FileNotFoundError(
                f"Missing raw multispectral TIFF for sample {sample_id!r}; searched {raw_root}; "
                f"expected unique stem near {expected}"
            )
        if prior_path is None:
            expected = prior_root / f"{sample_id}.npy"
            raise FileNotFoundError(
                f"Missing offline unmixing NPY for sample {sample_id!r}; searched {prior_root}; "
                f"expected {expected}"
            )
        pairs.append(AuxiliaryPair(sample_id, raw_path, prior_path))
    return pairs


def write_auxiliary_manifest(pairs: Sequence[AuxiliaryPair], path: str | Path) -> None:
    """Write an auditable absolute-path manifest without touching source data."""

    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["sample_id", "raw_ms_path", "unmixing_path"])
        writer.writeheader()
        for pair in pairs:
            writer.writerow(
                {
                    "sample_id": pair.sample_id,
                    "raw_ms_path": str(pair.raw_ms_path.resolve()),
                    "unmixing_path": str(pair.unmixing_path.resolve()),
                }
            )


def _import_rasterio() -> Any:
    try:
        import rasterio
    except ImportError as error:
        raise ImportError(
            "Tri-input raw TIFF reading requires rasterio. Install rasterio in the training "
            "environment; RGB-only code does not require it."
        ) from error
    return rasterio


def load_raw_multispectral(
    path: str | Path,
    band_names: Sequence[str] | None,
    value_conversion: RawValueConversion | Mapping[str, Any] | None,
    *,
    expected_hw: tuple[int, int] | None = None,
) -> RawMultispectral:
    """Read a twelve-band TIFF, explicitly convert units, and preserve validity.

    ``raw_valid`` is true only where all twelve raster masks are valid and all
    converted values are finite.  Numerically zero reflectance is not treated as
    nodata.  Invalid output pixels are filled with zero but remain distinguishable
    through the returned ``[1,H,W]`` boolean mask.
    """

    raster_path = Path(path).expanduser().resolve()
    if not raster_path.is_file():
        raise FileNotFoundError(f"Raw multispectral TIFF does not exist: {raster_path}")
    if raster_path.suffix.lower() not in RAW_RASTER_EXTENSIONS:
        raise ValueError(f"Raw multispectral file must be .tif or .tiff: {raster_path}")
    layout = validate_sentinel2_l2a_band_names(band_names)
    conversion = (
        value_conversion
        if isinstance(value_conversion, RawValueConversion)
        else RawValueConversion.from_config(value_conversion)
    )
    rasterio = _import_rasterio()
    try:
        with rasterio.open(raster_path) as dataset:
            if int(dataset.count) != 12:
                raise ValueError(
                    f"Raw multispectral sample {raster_path} has {dataset.count} bands; expected 12 "
                    f"for configured order {layout.band_names}"
                )
            raw_descriptions = tuple(getattr(dataset, "descriptions", ()) or ())
            nonempty_descriptions = tuple(
                None if value is None or not str(value).strip() else _normalise_band_name(value)
                for value in raw_descriptions
            )
            if any(value is not None for value in nonempty_descriptions):
                if len(nonempty_descriptions) != 12 or any(
                    value is None for value in nonempty_descriptions
                ):
                    raise ValueError(
                        f"Raw raster {raster_path} has partial band descriptions "
                        f"{raw_descriptions}; all 12 must be present to verify configured order"
                    )
                described = tuple(str(value) for value in nonempty_descriptions)
                if described != layout.band_names:
                    raise ValueError(
                        f"Raw raster band descriptions do not match configured order for {raster_path}: "
                        f"metadata={described}, configured={layout.band_names}"
                    )
                band_descriptions: tuple[str, ...] | None = described
            else:
                band_descriptions = None
            hw = (int(dataset.height), int(dataset.width))
            if expected_hw is not None and hw != tuple(int(value) for value in expected_hw):
                raise ValueError(
                    f"Raw multispectral grid mismatch for {raster_path}: raster={hw}, expected={expected_hw}; "
                    "silent resize is forbidden"
                )
            stored = np.asarray(dataset.read(masked=False))
            masks = np.asarray(dataset.read_masks())
            if stored.shape != (12, *hw):
                raise ValueError(f"Unexpected raw raster shape for {raster_path}: {stored.shape}")
            if masks.shape != stored.shape:
                raise ValueError(f"Unexpected raster mask shape for {raster_path}: {masks.shape}")
            converted = conversion.apply(stored)
            valid = np.all(masks > 0, axis=0) & np.all(np.isfinite(converted), axis=0)
            converted[:, ~valid] = 0.0
            crs = None if dataset.crs is None else str(dataset.crs)
            transform = None if dataset.transform is None else tuple(float(item) for item in dataset.transform)
            nodata = None if dataset.nodata is None else float(dataset.nodata)
    except (OSError, RuntimeError) as error:
        raise RuntimeError(f"Failed to read raw multispectral TIFF {raster_path}: {error}") from error
    return RawMultispectral(
        raw_ms=torch.from_numpy(np.ascontiguousarray(converted, dtype=np.float32)),
        raw_valid=torch.from_numpy(np.ascontiguousarray(valid[None, ...])).bool(),
        band_names=layout.band_names,
        path=raster_path,
        crs=crs,
        transform=transform,
        nodata=nodata,
        band_descriptions=band_descriptions,
    )


def load_unmixing_tensor(
    path: str | Path,
    *,
    expected_hw: tuple[int, int] | None = None,
) -> torch.Tensor:
    """Load the immutable offline ``[F,U]`` contract as float32 ``[10,H,W]``."""

    prior_path = Path(path).expanduser().resolve()
    if not prior_path.is_file():
        raise FileNotFoundError(f"Offline unmixing result does not exist: {prior_path}")
    if prior_path.suffix.lower() != ".npy":
        raise ValueError(f"Offline unmixing result must be .npy: {prior_path}")
    try:
        array = np.load(prior_path, allow_pickle=False)
    except (OSError, ValueError) as error:
        raise RuntimeError(f"Failed to load offline unmixing result {prior_path}: {error}") from error
    if array.dtype != np.float32:
        raise TypeError(f"Unmixing sample {prior_path} must be float32, got {array.dtype}")
    if array.ndim != 3 or array.shape[0] != 10 or array.shape[1] == 0 or array.shape[2] == 0:
        raise ValueError(f"Unmixing sample {prior_path} must have non-empty shape [10,H,W], got {array.shape}")
    if expected_hw is not None and tuple(array.shape[1:]) != tuple(int(value) for value in expected_hw):
        raise ValueError(
            f"Unmixing grid mismatch for {prior_path}: prior={array.shape[1:]}, expected={expected_hw}; "
            "silent resize is forbidden"
        )
    if not np.isfinite(array).all():
        raise ValueError(f"Unmixing sample {prior_path} contains NaN or Inf")
    minimum = float(array.min())
    maximum = float(array.max())
    if minimum < 0.0 or maximum > 1.0:
        raise ValueError(
            f"Unmixing sample {prior_path} must remain in [0,1], got range [{minimum}, {maximum}]"
        )
    return torch.from_numpy(np.ascontiguousarray(array))


def load_auxiliary_pair(
    pair: AuxiliaryPair,
    band_names: Sequence[str] | None,
    value_conversion: RawValueConversion | Mapping[str, Any] | None,
) -> tuple[RawMultispectral, torch.Tensor]:
    """Load raw/prior data and enforce their common low-resolution grid."""

    raw = load_raw_multispectral(pair.raw_ms_path, band_names, value_conversion)
    prior = load_unmixing_tensor(pair.unmixing_path, expected_hw=tuple(raw.raw_ms.shape[-2:]))
    return raw, prior


def compute_raw_band_stats(
    raw_paths: Iterable[str | Path],
    band_names: Sequence[str] | None,
    value_conversion: RawValueConversion | Mapping[str, Any] | None,
    *,
    split: str,
    data_source: str,
) -> RawBandStats:
    """Compute population mean/std from valid train pixels only."""

    if str(split).strip().lower() != "train":
        raise ValueError(f"Raw band statistics may only be computed from split='train', got {split!r}")
    if not str(data_source).strip():
        raise ValueError("data_source must explicitly identify the training data used for statistics")
    layout = validate_sentinel2_l2a_band_names(band_names)
    conversion = (
        value_conversion
        if isinstance(value_conversion, RawValueConversion)
        else RawValueConversion.from_config(value_conversion)
    )
    paths = [Path(path).expanduser().resolve() for path in raw_paths]
    if not paths:
        raise ValueError("At least one train raw TIFF is required to compute statistics")
    sums = np.zeros(12, dtype=np.float64)
    squared_sums = np.zeros(12, dtype=np.float64)
    counts = np.zeros(12, dtype=np.int64)
    for path in paths:
        sample = load_raw_multispectral(path, layout.band_names, conversion)
        values = sample.raw_ms.detach().cpu().numpy().astype(np.float64, copy=False)
        valid = sample.raw_valid.detach().cpu().numpy().astype(bool, copy=False)[0]
        if not valid.any():
            continue
        for channel in range(12):
            selected = values[channel][valid]
            sums[channel] += selected.sum(dtype=np.float64)
            squared_sums[channel] += np.square(selected, dtype=np.float64).sum(dtype=np.float64)
            counts[channel] += int(selected.size)
    if np.any(counts == 0):
        empty = [layout.band_names[index] for index in np.flatnonzero(counts == 0)]
        raise ValueError(f"No valid train pixels were available for bands: {empty}")
    mean = sums / counts
    variance = np.maximum(squared_sums / counts - np.square(mean), 0.0)
    std = np.sqrt(variance)
    invalid_std = ~np.isfinite(std) | (std <= 0.0)
    if np.any(invalid_std):
        names = [layout.band_names[index] for index in np.flatnonzero(invalid_std)]
        raise ValueError(f"Train raw band standard deviation is non-positive or non-finite for: {names}")
    return RawBandStats(
        format_version=RAW_STATS_FORMAT_VERSION,
        split="train",
        band_names=layout.band_names,
        raw_value_conversion=conversion,
        mean=tuple(float(value) for value in mean),
        std=tuple(float(value) for value in std),
        valid_pixel_count=tuple(int(value) for value in counts),
        sample_count=len(paths),
        data_source=str(data_source),
        created_utc=datetime.now(timezone.utc).isoformat(),
        software_versions={
            "python": platform.python_version(),
            "numpy": np.__version__,
            "torch": torch.__version__,
            "rasterio": str(getattr(_import_rasterio(), "__version__", "unknown")),
        },
    )


def save_raw_band_stats(stats: RawBandStats, path: str | Path) -> None:
    if stats.split != "train":
        raise ValueError(f"Refusing to save non-train raw statistics: split={stats.split!r}")
    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        json.dump(stats.as_dict(), handle, indent=2, ensure_ascii=False)


def load_raw_band_stats(
    path: str | Path,
    *,
    expected_band_names: Sequence[str] | None = None,
    expected_conversion: RawValueConversion | Mapping[str, Any] | None = None,
) -> RawBandStats:
    stats_path = Path(path).expanduser().resolve()
    if not stats_path.is_file():
        raise FileNotFoundError(f"Raw band statistics file does not exist: {stats_path}")
    with stats_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, Mapping):
        raise ValueError(f"Raw band statistics must be a JSON object: {stats_path}")
    version = int(payload.get("format_version", -1))
    if version != RAW_STATS_FORMAT_VERSION:
        raise ValueError(
            f"Unsupported raw stats format_version={version} in {stats_path}; "
            f"expected {RAW_STATS_FORMAT_VERSION}"
        )
    if payload.get("split") != "train":
        raise ValueError(f"Raw stats {stats_path} were not computed from train split: {payload.get('split')!r}")
    layout = validate_sentinel2_l2a_band_names(payload.get("band_names"))
    conversion = RawValueConversion.from_config(payload.get("raw_value_conversion"))
    mean = tuple(float(value) for value in payload.get("mean", ()))
    std = tuple(float(value) for value in payload.get("std", ()))
    counts = tuple(int(value) for value in payload.get("valid_pixel_count", ()))
    if len(mean) != 12 or len(std) != 12 or len(counts) != 12:
        raise ValueError(f"Raw stats {stats_path} must contain 12 mean/std/count values")
    if not np.isfinite(mean).all() or not np.isfinite(std).all() or any(value <= 0 for value in std):
        raise ValueError(f"Raw stats {stats_path} contain non-finite values or non-positive std")
    if any(value <= 0 for value in counts):
        raise ValueError(f"Raw stats {stats_path} contain non-positive valid pixel counts")
    if expected_band_names is not None:
        expected_layout = validate_sentinel2_l2a_band_names(expected_band_names)
        if layout.band_names != expected_layout.band_names:
            raise ValueError(
                f"Raw stats band order {layout.band_names} does not match configured order "
                f"{expected_layout.band_names}"
            )
    if expected_conversion is not None:
        expected = (
            expected_conversion
            if isinstance(expected_conversion, RawValueConversion)
            else RawValueConversion.from_config(expected_conversion)
        )
        if conversion != expected:
            raise ValueError(
                f"Raw stats conversion {conversion.as_dict()} does not match configured conversion "
                f"{expected.as_dict()}"
            )
    return RawBandStats(
        format_version=version,
        split="train",
        band_names=layout.band_names,
        raw_value_conversion=conversion,
        mean=mean,
        std=std,
        valid_pixel_count=counts,
        sample_count=int(payload.get("sample_count", 0)),
        data_source=str(payload.get("data_source", "")),
        created_utc=str(payload.get("created_utc", "")),
        software_versions={str(key): str(value) for key, value in dict(payload.get("software_versions", {})).items()},
    )


def normalize_raw_multispectral(
    raw_ms: torch.Tensor,
    raw_valid: torch.Tensor,
    stats: RawBandStats,
) -> torch.Tensor:
    """Standardize converted raw bands and keep invalid pixels explicitly zero."""

    if raw_ms.ndim not in {3, 4} or raw_ms.shape[-3] != 12:
        raise ValueError(f"raw_ms must be [12,H,W] or [B,12,H,W], got {tuple(raw_ms.shape)}")
    expected_mask_shape = (1, *raw_ms.shape[-2:]) if raw_ms.ndim == 3 else (raw_ms.shape[0], 1, *raw_ms.shape[-2:])
    if tuple(raw_valid.shape) != expected_mask_shape:
        raise ValueError(f"raw_valid shape {tuple(raw_valid.shape)} does not match expected {expected_mask_shape}")
    if raw_valid.dtype != torch.bool:
        raise TypeError(f"raw_valid must be boolean, got {raw_valid.dtype}")
    prefix = (12, 1, 1) if raw_ms.ndim == 3 else (1, 12, 1, 1)
    mean = raw_ms.new_tensor(stats.mean).view(prefix)
    std = raw_ms.new_tensor(stats.std).view(prefix)
    normalized = (raw_ms - mean) / std
    return torch.where(raw_valid.expand_as(normalized), normalized, torch.zeros_like(normalized))
