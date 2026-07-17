"""Strictly paired WorldStrat GT/LR dataset with aligned augmentation."""

from __future__ import annotations

import csv
import logging
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
import torch
from PIL import Image, ImageOps, UnidentifiedImageError
from torch.utils.data import Dataset

from .utils import FIXED_PROMPT, pil_to_tensor

LOGGER = logging.getLogger(__name__)
SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}


@dataclass(frozen=True)
class PairRecord:
    """Paths and identity for a validated LR/GT pair."""

    filename: str
    sample_id: str
    gt_path: Path
    lr_path: Path
    synthetic_lr_path: Path | None = None


@dataclass(frozen=True)
class InvalidPair:
    """Description of a rejected data pair."""

    filename: str
    reason: str
    gt_path: str
    lr_path: str


def _image_names(directory: Path) -> dict[str, Path]:
    if not directory.is_dir():
        raise FileNotFoundError(f"Image directory does not exist: {directory}")
    return {
        path.name: path
        for path in sorted(directory.iterdir())
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    }


def _write_invalid_log(rows: Iterable[InvalidPair], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["filename", "reason", "gt_path", "lr_path"])
        writer.writeheader()
        for row in rows:
            writer.writerow(row.__dict__)


def validate_pair_directories(
    gt_dir: Path,
    lr_dir: Path,
    scale: int = 4,
    strict_pairs: bool = False,
    invalid_log_path: Path | None = None,
    synthetic_lr_dir: Path | None = None,
    max_samples: int | None = None,
) -> tuple[list[PairRecord], list[InvalidPair]]:
    """Validate exact filenames, readability, and x4 geometry without modifying data."""
    gt_files = _image_names(gt_dir)
    lr_files = _image_names(lr_dir)
    synthetic_files = _image_names(synthetic_lr_dir) if synthetic_lr_dir else {}
    names = sorted(set(gt_files) | set(lr_files))
    if max_samples is not None:
        if max_samples <= 0:
            raise ValueError(f"max_samples must be positive, got {max_samples}")
        names = names[:max_samples]
    valid: list[PairRecord] = []
    invalid: list[InvalidPair] = []

    for name in names:
        gt_path = gt_files.get(name)
        lr_path = lr_files.get(name)
        reason: str | None = None
        if gt_path is None:
            reason = "missing_gt"
        elif lr_path is None:
            reason = "missing_lr"
        else:
            try:
                with Image.open(gt_path) as gt_image, Image.open(lr_path) as lr_image:
                    gt_image.verify()
                    lr_image.verify()
                with Image.open(gt_path) as gt_image, Image.open(lr_path) as lr_image:
                    gt_size, lr_size = gt_image.size, lr_image.size
                if gt_size != (lr_size[0] * scale, lr_size[1] * scale):
                    reason = f"scale_mismatch: gt={gt_size}, lr={lr_size}, expected_scale={scale}"
            except (OSError, UnidentifiedImageError, ValueError) as error:
                reason = f"unreadable_image: {type(error).__name__}: {error}"

        synthetic_path = synthetic_files.get(name) if synthetic_lr_dir else None
        if reason is None and synthetic_lr_dir and synthetic_path is None:
            reason = "missing_synthetic_lr"
        if reason is None and synthetic_path is not None and gt_path is not None:
            try:
                with Image.open(gt_path) as gt_image, Image.open(synthetic_path) as syn_image:
                    if gt_image.size != (syn_image.size[0] * scale, syn_image.size[1] * scale):
                        reason = (
                            f"synthetic_scale_mismatch: gt={gt_image.size}, synthetic={syn_image.size}, "
                            f"expected_scale={scale}"
                        )
            except (OSError, UnidentifiedImageError, ValueError) as error:
                reason = f"unreadable_synthetic_image: {type(error).__name__}: {error}"

        if reason is not None:
            item = InvalidPair(name, reason, str(gt_path or ""), str(lr_path or ""))
            invalid.append(item)
            if strict_pairs:
                if invalid_log_path is not None:
                    _write_invalid_log(invalid, invalid_log_path)
                raise ValueError(f"Invalid pair {name!r}: {reason}; GT={gt_path}; LR={lr_path}")
        else:
            assert gt_path is not None and lr_path is not None
            valid.append(
                PairRecord(name, Path(name).stem, gt_path, lr_path, synthetic_path)
            )

    if invalid_log_path is not None:
        _write_invalid_log(invalid, invalid_log_path)
    if not valid:
        raise RuntimeError(
            f"No valid image pairs found between GT={gt_dir} and LR={lr_dir}; "
            f"invalid_count={len(invalid)}"
        )
    LOGGER.info("Validated %d pairs; rejected %d pairs", len(valid), len(invalid))
    return valid, invalid


class PairedSatelliteDataset(Dataset[dict[str, Any]]):
    """WorldStrat pairs with exactly aligned LR/GT crops and spatial transforms."""

    def __init__(
        self,
        data_root: str | Path,
        split: str,
        lr_subdir: str,
        gt_subdir: str = "GT",
        gt_crop_size: int = 512,
        scale: int = 4,
        training: bool = True,
        strict_pairs: bool = False,
        invalid_log_path: str | Path | None = None,
        synthetic_lr_subdir: str | None = None,
        synthetic_replay_probability: float = 0.0,
        prompt_mode: str = "fixed",
        metadata_path: str | Path | None = None,
        prompt_dropout_probability: float = 0.0,
        augment: bool = True,
        validation_limit: int | None = None,
    ) -> None:
        self.data_root = Path(data_root)
        self.split = split
        self.gt_crop_size = int(gt_crop_size)
        self.scale = int(scale)
        self.lr_crop_size = self.gt_crop_size // self.scale
        self.training = training
        self.synthetic_replay_probability = float(synthetic_replay_probability)
        self.prompt_mode = prompt_mode
        self.prompt_dropout_probability = float(prompt_dropout_probability)
        self.augment = augment and training

        if self.gt_crop_size <= 0 or self.gt_crop_size % self.scale != 0:
            raise ValueError(
                f"gt_crop_size must be positive and divisible by scale: "
                f"gt_crop_size={self.gt_crop_size}, scale={self.scale}"
            )
        if not 0.0 <= self.synthetic_replay_probability <= 1.0:
            raise ValueError("synthetic_replay_probability must be in [0, 1]")
        if not 0.0 <= self.prompt_dropout_probability <= 1.0:
            raise ValueError("prompt_dropout_probability must be in [0, 1]")
        if prompt_mode not in {"fixed", "metadata"}:
            raise ValueError(f"prompt_mode must be fixed or metadata, got {prompt_mode!r}")

        gt_dir = self.data_root / split / gt_subdir
        lr_dir = self.data_root / split / lr_subdir
        synthetic_dir = (
            self.data_root / split / synthetic_lr_subdir
            if synthetic_lr_subdir and self.synthetic_replay_probability > 0
            else None
        )
        log_path = Path(invalid_log_path) if invalid_log_path else None
        self.records, self.invalid_pairs = validate_pair_directories(
            gt_dir=gt_dir,
            lr_dir=lr_dir,
            scale=self.scale,
            strict_pairs=strict_pairs,
            invalid_log_path=log_path,
            synthetic_lr_dir=synthetic_dir,
            max_samples=validation_limit,
        )
        self.metadata = self._load_metadata(metadata_path)

    def _load_metadata(self, metadata_path: str | Path | None) -> dict[str, dict[str, str]]:
        if self.prompt_mode != "metadata":
            return {}
        candidates = []
        if metadata_path:
            candidates.append(Path(metadata_path))
        candidates.extend(
            [
                self.data_root / "metadata_final" / "final_metadata.csv",
                self.data_root / "final_metadata.csv",
            ]
        )
        path = next((candidate for candidate in candidates if candidate.is_file()), None)
        if path is None:
            LOGGER.warning("Metadata prompt mode requested but no final_metadata.csv was found; using fixed prompts")
            return {}
        frame = pd.read_csv(path, dtype=str).fillna("")
        if "sample_id" not in frame.columns and "filename" not in frame.columns:
            LOGGER.warning("Metadata %s has neither sample_id nor filename; using fixed prompts", path)
            return {}
        result: dict[str, dict[str, str]] = {}
        for _, row in frame.iterrows():
            key = str(row.get("sample_id") or Path(str(row.get("filename", ""))).stem)
            if key:
                values = {str(column): str(value) for column, value in row.items()}
                result[key] = values
                filename = str(row.get("filename", "")).strip()
                if filename:
                    result[Path(filename).stem] = values
        LOGGER.info("Loaded metadata for %d samples from %s", len(result), path)
        return result

    def __len__(self) -> int:
        return len(self.records)

    def _prompt(self, sample_id: str) -> str:
        if self.training and random.random() < self.prompt_dropout_probability:
            return ""
        row = self.metadata.get(sample_id)
        if row:
            ipcc = row.get("IPCC Class", "").strip()
            smod = row.get("SMOD Class", "").strip()
            if ipcc and smod:
                return (
                    f"a high-resolution overhead satellite image of {ipcc}, {smod}, "
                    "with accurate geographic structures and natural colors"
                )
        return FIXED_PROMPT

    def _aligned_crop(self, gt: Image.Image, lr: Image.Image, filename: str) -> tuple[Image.Image, Image.Image]:
        lr_width, lr_height = lr.size
        if lr_width < self.lr_crop_size or lr_height < self.lr_crop_size:
            raise ValueError(
                f"Image {filename!r} is smaller than the requested crop: LR={lr.size}, "
                f"required LR crop=({self.lr_crop_size}, {self.lr_crop_size}); GT will not be upscaled"
            )
        if self.training:
            left = random.randint(0, lr_width - self.lr_crop_size)
            top = random.randint(0, lr_height - self.lr_crop_size)
        else:
            left = (lr_width - self.lr_crop_size) // 2
            top = (lr_height - self.lr_crop_size) // 2
        lr_box = (left, top, left + self.lr_crop_size, top + self.lr_crop_size)
        gt_box = tuple(coordinate * self.scale for coordinate in lr_box)
        return gt.crop(gt_box), lr.crop(lr_box)

    def _spatial_transform(self, gt: Image.Image, lr: Image.Image) -> tuple[Image.Image, Image.Image]:
        if not self.augment:
            return gt, lr
        if random.random() < 0.5:
            gt, lr = ImageOps.mirror(gt), ImageOps.mirror(lr)
        if random.random() < 0.5:
            gt, lr = ImageOps.flip(gt), ImageOps.flip(lr)
        rotation = random.choice((0, 90, 180, 270))
        if rotation:
            gt, lr = gt.rotate(rotation, expand=False), lr.rotate(rotation, expand=False)
        return gt, lr

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        use_synthetic = (
            self.training
            and record.synthetic_lr_path is not None
            and random.random() < self.synthetic_replay_probability
        )
        lr_path = record.synthetic_lr_path if use_synthetic else record.lr_path
        source_type = "bicubic" if (use_synthetic or "bicubic" in lr_path.parent.name.lower()) else "real"
        try:
            with Image.open(record.gt_path) as opened_gt, Image.open(lr_path) as opened_lr:
                gt = opened_gt.convert("RGB")
                lr = opened_lr.convert("RGB")
        except (OSError, UnidentifiedImageError) as error:
            raise RuntimeError(
                f"Failed to read pair {record.filename!r}: GT={record.gt_path}, LR={lr_path}: {error}"
            ) from error
        if gt.size != (lr.size[0] * self.scale, lr.size[1] * self.scale):
            raise ValueError(
                f"Pair geometry changed after validation for {record.filename!r}: GT={gt.size}, LR={lr.size}"
            )
        gt, lr = self._aligned_crop(gt, lr, record.filename)
        gt, lr = self._spatial_transform(gt, lr)
        return {
            "gt": pil_to_tensor(gt, "minus_one_one"),
            "lr": pil_to_tensor(lr, "minus_one_one"),
            "prompt": self._prompt(record.sample_id),
            "sample_id": record.sample_id,
            "filename": record.filename,
            "source_type": source_type,
        }
