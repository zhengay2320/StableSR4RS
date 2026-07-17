"""Shared configuration, reproducibility, image, and checkpoint helpers."""

from __future__ import annotations

import json
import logging
import os
import random
import shutil
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from PIL import Image

LOGGER = logging.getLogger(__name__)

FIXED_PROMPT = (
    "a high-resolution overhead satellite image with accurate geographic structures, "
    "natural colors, clear buildings, roads, vegetation, farmland and terrain"
)
NEGATIVE_PROMPT = (
    "cartoon, painting, fantasy, distorted buildings, duplicated roads, broken field "
    "boundaries, checkerboard artifacts, excessive sharpening, false textures, strong color cast"
)


def configure_logging(level: int = logging.INFO) -> None:
    """Configure concise process-aware logging once."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        force=True,
    )


def load_yaml_config(path: str | Path) -> dict[str, Any]:
    """Load and minimally validate a YAML mapping."""
    config_path = Path(path).expanduser().resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"Configuration file does not exist: {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle)
    if not isinstance(loaded, dict):
        raise ValueError(f"Configuration must be a YAML mapping: {config_path}")
    return loaded


def require_config(config: Mapping[str, Any], *keys: str) -> None:
    """Raise a useful error when required configuration keys are absent."""
    missing = [key for key in keys if key not in config]
    if missing:
        raise KeyError(f"Missing required configuration keys: {', '.join(missing)}")


def require_diffusers_version(expected: str = "0.39.0") -> None:
    """Fail early when runtime Diffusers differs from the pinned implementation."""
    import diffusers

    if diffusers.__version__ != expected:
        raise RuntimeError(
            f"This project requires diffusers=={expected}, found {diffusers.__version__}. "
            "Run scripts/setup_env.sh to install the pinned editable checkout."
        )


def normalize_tokenizer_max_length(tokenizer: Any, text_encoder: Any) -> int:
    """Use the text encoder's finite context length for locally copied tokenizers."""
    max_length = int(text_encoder.config.max_position_embeddings)
    if max_length <= 0:
        raise ValueError(f"Invalid text encoder max_position_embeddings: {max_length}")
    tokenizer.model_max_length = max_length
    return max_length


def resolve_project_path(value: str | Path, project_root: Path) -> Path:
    """Resolve a path relative to the project root without requiring it to exist."""
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (project_root / path).resolve()


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch random number generators."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def worker_init_fn(worker_id: int) -> None:
    """Seed Python and NumPy from the deterministic PyTorch worker seed."""
    del worker_id
    seed = torch.initial_seed() % (2**32)
    random.seed(seed)
    np.random.seed(seed)


def tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    """Convert a CHW tensor in [-1, 1] or [0, 1] to RGB PIL."""
    image = tensor.detach().float().cpu()
    if image.ndim == 4:
        if image.shape[0] != 1:
            raise ValueError(f"Expected one image, got tensor shape {tuple(image.shape)}")
        image = image[0]
    if image.min().item() < 0:
        image = (image + 1.0) / 2.0
    image = image.clamp(0, 1).permute(1, 2, 0).numpy()
    return Image.fromarray(np.round(image * 255.0).astype(np.uint8), mode="RGB")


def pil_to_tensor(image: Image.Image, value_range: str = "zero_one") -> torch.Tensor:
    """Convert RGB PIL to CHW float tensor in [0,1] or [-1,1]."""
    array = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    tensor = torch.from_numpy(array).permute(2, 0, 1).contiguous()
    if value_range == "minus_one_one":
        return tensor.mul(2.0).sub(1.0)
    if value_range != "zero_one":
        raise ValueError(f"Unsupported value_range: {value_range}")
    return tensor


def low_frequency_projection(
    sr: torch.Tensor, lr: torch.Tensor, alpha: float = 0.5, scale: int = 4
) -> torch.Tensor:
    """Project SR low frequencies toward the observed LR image.

    Both tensors must be BCHW or CHW in [0, 1]. The correction is bicubically
    upsampled and the result is clamped to the valid image range.
    """
    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"alpha must be in [0, 1], got {alpha}")
    squeeze = sr.ndim == 3
    sr_b = sr.unsqueeze(0) if squeeze else sr
    lr_b = lr.unsqueeze(0) if lr.ndim == 3 else lr
    if sr_b.ndim != 4 or lr_b.ndim != 4:
        raise ValueError("sr and lr must be CHW or BCHW tensors")
    expected = (lr_b.shape[-2] * scale, lr_b.shape[-1] * scale)
    if tuple(sr_b.shape[-2:]) != expected:
        raise ValueError(
            f"SR size {tuple(sr_b.shape[-2:])} is not LR size {tuple(lr_b.shape[-2:])} x{scale}"
        )
    down = F.interpolate(sr_b, size=lr_b.shape[-2:], mode="bicubic", align_corners=False)
    residual = lr_b - down
    correction = F.interpolate(residual, size=sr_b.shape[-2:], mode="bicubic", align_corners=False)
    projected = (sr_b + alpha * correction).clamp(0.0, 1.0)
    return projected.squeeze(0) if squeeze else projected


def save_yaml(data: Mapping[str, Any], path: Path) -> None:
    """Write a mapping as human-readable YAML."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(dict(data), handle, sort_keys=False, allow_unicode=True)


def save_json(data: Mapping[str, Any], path: Path) -> None:
    """Write a JSON mapping with stable formatting."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(dict(data), handle, indent=2, ensure_ascii=False)


def enforce_checkpoint_limit(output_dir: Path, limit: int | None) -> None:
    """Remove oldest numbered checkpoints while preserving final artifacts."""
    if not limit or limit <= 0:
        return
    checkpoints: list[tuple[int, Path]] = []
    for path in output_dir.glob("checkpoint-*"):
        try:
            checkpoints.append((int(path.name.split("-")[-1]), path))
        except ValueError:
            LOGGER.warning("Ignoring unrecognized checkpoint directory: %s", path)
    checkpoints.sort()
    for _, path in checkpoints[: max(0, len(checkpoints) - limit)]:
        LOGGER.info("Removing old checkpoint: %s", path)
        shutil.rmtree(path)


def find_latest_checkpoint(output_dir: Path) -> Path | None:
    """Return the checkpoint with the largest numeric step."""
    candidates: list[tuple[int, Path]] = []
    for path in output_dir.glob("checkpoint-*"):
        try:
            candidates.append((int(path.name.rsplit("-", 1)[1]), path))
        except (IndexError, ValueError):
            continue
    return max(candidates, default=(0, None), key=lambda item: item[0])[1]


def atomic_torch_save(state: Any, path: Path) -> None:
    """Write a torch state via a same-directory temporary file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    torch.save(state, temporary)
    temporary.replace(path)
