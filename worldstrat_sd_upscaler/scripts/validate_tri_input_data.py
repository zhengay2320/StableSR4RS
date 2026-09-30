#!/usr/bin/env python
"""Validate RGB/raw-Sentinel-2/offline-prior pairing without modifying inputs."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

from PIL import Image

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.dataset import SUPPORTED_EXTENSIONS
from src.tri_input_data import (
    RawValueConversion,
    build_auxiliary_pairs,
    load_auxiliary_pair,
    validate_sentinel2_l2a_band_names,
    write_auxiliary_manifest,
)
from src.utils import configure_logging, load_yaml_config, resolve_project_path, save_json


LOGGER = logging.getLogger("validate_tri_input_data")
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "val", "test"), required=True)
    parser.add_argument("--rgb-dir", type=Path, default=None)
    parser.add_argument("--raw-ms-dir", type=Path, default=None)
    parser.add_argument("--unmixing-dir", type=Path, default=None)
    parser.add_argument("--manifest-path", type=Path, default=None)
    parser.add_argument("--recursive", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/tri_input_data_check"))
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def _unique_rgb_files(directory: Path) -> dict[str, Path]:
    if not directory.is_dir():
        raise FileNotFoundError(f"Processed RGB directory does not exist: {directory}")
    result: dict[str, Path] = {}
    duplicates: dict[str, list[Path]] = {}
    for path in sorted(directory.iterdir()):
        if not path.is_file() or path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue
        if path.stem in result:
            duplicates.setdefault(path.stem, [result[path.stem]]).append(path.resolve())
        else:
            result[path.stem] = path.resolve()
    if duplicates:
        detail = "; ".join(
            f"{stem}: {', '.join(str(path) for path in paths)}"
            for stem, paths in duplicates.items()
        )
        raise ValueError(f"Duplicate processed RGB stems are forbidden: {detail}")
    if not result:
        raise RuntimeError(f"No supported processed RGB images found under {directory}")
    return result


def _split_rgb_dir(config: dict[str, Any], split: str, override: Path | None) -> Path:
    if override is not None:
        return override.expanduser().resolve()
    subdir_key = {"train": "train_lr_subdir", "val": "val_lr_subdir", "test": "test_lr_subdir"}[split]
    subdir = config.get(subdir_key)
    if not subdir:
        raise ValueError(
            f"No {subdir_key} is configured; pass --rgb-dir explicitly for split={split!r}"
        )
    return resolve_project_path(config["data_root"], PROJECT_ROOT) / split / str(subdir)


def validate_split(
    config: dict[str, Any],
    split: str,
    rgb_dir_override: Path | None,
    output_dir: Path,
    limit: int | None,
    raw_ms_dir_override: Path | None = None,
    unmixing_dir_override: Path | None = None,
    manifest_path_override: Path | None = None,
    recursive_override: bool = False,
) -> dict[str, Any]:
    tri = config.get("tri_input")
    if not isinstance(tri, dict) or not bool(tri.get("enabled", False)):
        raise ValueError("Configuration must explicitly set tri_input.enabled=true")
    split_config = (tri.get("data") or {}).get(split)
    if not isinstance(split_config, dict):
        raise ValueError(f"tri_input.data.{split} must be configured")
    raw_dir = raw_ms_dir_override or split_config.get("raw_ms_dir")
    prior_dir = unmixing_dir_override or split_config.get("unmixing_dir")
    if not raw_dir or not prior_dir:
        raise ValueError(
            f"tri_input.data.{split}.raw_ms_dir and unmixing_dir are required; "
            "null paths are unresolved, not guessed"
        )
    layout = validate_sentinel2_l2a_band_names(tri.get("band_names"))
    conversion = RawValueConversion.from_config(tri.get("raw_value_conversion"))
    rgb_dir = _split_rgb_dir(config, split, rgb_dir_override)
    rgb_files = _unique_rgb_files(rgb_dir)
    sample_ids = list(rgb_files)
    if limit is not None:
        if limit <= 0:
            raise ValueError("--limit must be positive")
        sample_ids = sample_ids[:limit]
    pairs = build_auxiliary_pairs(
        sample_ids,
        raw_dir,
        prior_dir,
        manifest_path=manifest_path_override or split_config.get("manifest_path"),
        recursive=bool(recursive_override or split_config.get("recursive", False)),
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / f"{split}_tri_input_manifest.csv"
    write_auxiliary_manifest(pairs, manifest_path)
    rows: list[dict[str, Any]] = []
    success = 0
    failures = 0
    for pair in pairs:
        rgb_path = rgb_files[pair.sample_id]
        row: dict[str, Any] = {
            "sample_id": pair.sample_id,
            "rgb_path": str(rgb_path),
            "raw_ms_path": str(pair.raw_ms_path),
            "unmixing_path": str(pair.unmixing_path),
            "status": "error",
            "error": "",
        }
        try:
            with Image.open(rgb_path) as opened:
                rgb = opened.convert("RGB")
                rgb_hw = (rgb.height, rgb.width)
            raw, prior = load_auxiliary_pair(pair, layout.band_names, conversion)
            if tuple(raw.raw_ms.shape[-2:]) != rgb_hw:
                raise ValueError(
                    f"RGB/raw grid mismatch for {pair.sample_id!r}: RGB={rgb_hw} ({rgb_path}), "
                    f"raw={tuple(raw.raw_ms.shape[-2:])} ({pair.raw_ms_path}); silent resize is forbidden"
                )
            row.update(
                {
                    "status": "ok",
                    "height": rgb_hw[0],
                    "width": rgb_hw[1],
                    "valid_fraction": float(raw.raw_valid.float().mean()),
                    "crs": raw.crs,
                    "transform": raw.transform,
                    "band_order_verification": (
                        "raster_descriptions_match"
                        if raw.band_descriptions is not None
                        else "configured_order_only_raster_has_no_band_descriptions"
                    ),
                    "unmixing_min": float(prior.min()),
                    "unmixing_max": float(prior.max()),
                }
            )
            success += 1
        except Exception as error:  # report every corrupt sample, then fail overall
            row["error"] = f"{type(error).__name__}: {error}"
            failures += 1
        rows.append(row)

    report = {
        "split": split,
        "rgb_dir": str(rgb_dir),
        "raw_ms_dir": str(Path(raw_dir).expanduser().resolve()),
        "unmixing_dir": str(Path(prior_dir).expanduser().resolve()),
        "band_names": list(layout.band_names),
        "raw_value_conversion": conversion.as_dict(),
        "manifest": str(manifest_path),
        "sample_count": len(rows),
        "success_count": success,
        "failure_count": failures,
        "spatial_alignment_basis": (
            "sample_id plus identical stored pixel grid; processed RGB/NPY do not prove "
            "geographic co-registration, so the data-production convention must be documented"
        ),
        "quality_mask_limit": (
            "raw_valid uses raster masks/finite values only; cloud and shadow screening must "
            "be performed upstream"
        ),
        "samples": rows,
    }
    save_json(report, output_dir / f"{split}_tri_input_report.json")
    if failures:
        raise RuntimeError(
            f"Tri-input validation failed for {failures}/{len(rows)} samples; "
            f"see {output_dir / f'{split}_tri_input_report.json'}"
        )
    return report


def main() -> None:
    args = parse_args()
    configure_logging()
    config = load_yaml_config(args.config)
    report = validate_split(
        config,
        args.split,
        args.rgb_dir,
        args.output_dir.expanduser().resolve(),
        args.limit,
        args.raw_ms_dir,
        args.unmixing_dir,
        args.manifest_path,
        args.recursive,
    )
    LOGGER.info(
        "Validated %d tri-input samples for split=%s; report=%s",
        report["success_count"],
        args.split,
        args.output_dir,
    )


if __name__ == "__main__":
    main()
