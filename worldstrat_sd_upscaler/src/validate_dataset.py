#!/usr/bin/env python
"""Validate exact WorldStrat LR/GT pairing and x4 geometry."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.dataset import validate_pair_directories
from src.utils import configure_logging

LOGGER = logging.getLogger("validate_dataset")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_root", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "val", "test"), default="train")
    parser.add_argument("--lr_subdir", default="LR")
    parser.add_argument("--gt_subdir", default="GT")
    parser.add_argument("--scale", type=int, default=4)
    parser.add_argument("--strict_pairs", action="store_true")
    parser.add_argument("--output_csv", type=Path, default=Path("outputs/data_validation/invalid_pairs.csv"))
    parser.add_argument("--max_samples", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_logging()
    root = args.data_root.expanduser().resolve()
    valid, invalid = validate_pair_directories(
        gt_dir=root / args.split / args.gt_subdir,
        lr_dir=root / args.split / args.lr_subdir,
        scale=args.scale,
        strict_pairs=args.strict_pairs,
        invalid_log_path=args.output_csv.expanduser().resolve(),
        max_samples=args.max_samples,
    )
    LOGGER.info(
        "Validation complete for split=%s, LR=%s: valid=%d invalid=%d log=%s",
        args.split,
        args.lr_subdir,
        len(valid),
        len(invalid),
        args.output_csv,
    )


if __name__ == "__main__":
    main()
