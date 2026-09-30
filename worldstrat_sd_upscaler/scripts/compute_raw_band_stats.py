#!/usr/bin/env python
"""Compute twelve-band statistics from paired train observations only."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.validate_tri_input_data import _split_rgb_dir, _unique_rgb_files
from src.tri_input_data import (
    RawValueConversion,
    build_auxiliary_pairs,
    compute_raw_band_stats,
    save_raw_band_stats,
    validate_sentinel2_l2a_band_names,
)
from src.utils import configure_logging, load_yaml_config


LOGGER = logging.getLogger("compute_raw_band_stats")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--rgb-dir", type=Path, default=None)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/tri_input_data_check/raw_band_stats.json"),
    )
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_logging()
    config = load_yaml_config(args.config)
    tri = config.get("tri_input")
    if not isinstance(tri, dict) or not bool(tri.get("enabled", False)):
        raise ValueError("Configuration must explicitly set tri_input.enabled=true")
    train = (tri.get("data") or {}).get("train")
    if not isinstance(train, dict) or not train.get("raw_ms_dir") or not train.get("unmixing_dir"):
        raise ValueError("tri_input.data.train raw_ms_dir/unmixing_dir must be explicit")
    layout = validate_sentinel2_l2a_band_names(tri.get("band_names"))
    conversion = RawValueConversion.from_config(tri.get("raw_value_conversion"))
    rgb_dir = _split_rgb_dir(config, "train", args.rgb_dir)
    rgb_files = _unique_rgb_files(rgb_dir)
    sample_ids = list(rgb_files)
    if args.limit is not None:
        if args.limit <= 0:
            raise ValueError("--limit must be positive")
        sample_ids = sample_ids[: args.limit]
    pairs = build_auxiliary_pairs(
        sample_ids,
        train["raw_ms_dir"],
        train["unmixing_dir"],
        manifest_path=train.get("manifest_path"),
        recursive=bool(train.get("recursive", False)),
    )
    stats = compute_raw_band_stats(
        [pair.raw_ms_path for pair in pairs],
        layout.band_names,
        conversion,
        split="train",
        data_source=(
            f"paired train observations: RGB={rgb_dir}; raw_ms={Path(train['raw_ms_dir']).expanduser().resolve()}"
        ),
    )
    destination = args.output.expanduser().resolve()
    save_raw_band_stats(stats, destination)
    LOGGER.info(
        "Saved train-only raw band statistics from %d samples to %s",
        stats.sample_count,
        destination,
    )


if __name__ == "__main__":
    main()
