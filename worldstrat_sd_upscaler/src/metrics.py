"""Remote-sensing super-resolution image metrics."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from skimage.metrics import structural_similarity


def _as_hwc(image: np.ndarray | torch.Tensor) -> np.ndarray:
    array = image.detach().float().cpu().numpy() if isinstance(image, torch.Tensor) else np.asarray(image)
    if array.ndim == 4 and array.shape[0] == 1:
        array = array[0]
    if array.ndim == 3 and array.shape[0] in (1, 3, 4):
        array = np.moveaxis(array, 0, -1)
    if array.ndim != 3 or array.shape[-1] != 3:
        raise ValueError(f"Expected RGB HWC/CHW image, got shape {array.shape}")
    array = array.astype(np.float64)
    if array.max(initial=0.0) > 1.0:
        array /= 255.0
    return np.clip(array, 0.0, 1.0)


def psnr(prediction: np.ndarray | torch.Tensor, target: np.ndarray | torch.Tensor) -> float:
    """Compute RGB PSNR for images in [0,1] or [0,255]."""
    pred, ref = _as_hwc(prediction), _as_hwc(target)
    mse = float(np.mean((pred - ref) ** 2))
    return float("inf") if mse == 0 else -10.0 * math.log10(mse)


def ssim(prediction: np.ndarray | torch.Tensor, target: np.ndarray | torch.Tensor) -> float:
    """Compute channel-aware structural similarity."""
    pred, ref = _as_hwc(prediction), _as_hwc(target)
    min_side = min(pred.shape[0], pred.shape[1])
    win_size = min(7, min_side if min_side % 2 else min_side - 1)
    if win_size < 3:
        raise ValueError(f"SSIM requires images at least 3x3, got {pred.shape[:2]}")
    return float(structural_similarity(ref, pred, channel_axis=-1, data_range=1.0, win_size=win_size))


def spectral_angle_deg(prediction: np.ndarray | torch.Tensor, target: np.ndarray | torch.Tensor) -> float:
    """Return mean per-pixel RGB spectral angle in degrees."""
    pred, ref = _as_hwc(prediction), _as_hwc(target)
    dot = np.sum(pred * ref, axis=-1)
    norm = np.linalg.norm(pred, axis=-1) * np.linalg.norm(ref, axis=-1)
    valid = norm > 1e-12
    if not np.any(valid):
        return 0.0
    cosine = np.clip(dot[valid] / norm[valid], -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine)).mean())


def basic_metrics(prediction: np.ndarray | torch.Tensor, target: np.ndarray | torch.Tensor) -> dict[str, float]:
    """Compute full-resolution scalar and channel metrics."""
    pred, ref = _as_hwc(prediction), _as_hwc(target)
    error = pred - ref
    return {
        "psnr": psnr(pred, ref),
        "ssim": ssim(pred, ref),
        "rgb_mae": float(np.abs(error).mean()),
        "bias_r": float(error[..., 0].mean()),
        "bias_g": float(error[..., 1].mean()),
        "bias_b": float(error[..., 2].mean()),
        "spectral_angle_deg": spectral_angle_deg(pred, ref),
    }


def bicubic_downsample(image: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
    """Bicubically downsample CHW or BCHW tensor to ``(height, width)``."""
    batched = image.unsqueeze(0) if image.ndim == 3 else image
    result = F.interpolate(batched, size=size, mode="bicubic", align_corners=False, antialias=True).clamp(0, 1)
    return result.squeeze(0) if image.ndim == 3 else result


class OptionalLPIPS:
    """Lazy LPIPS metric that reports an actionable missing-dependency error."""

    def __init__(self, device: torch.device) -> None:
        try:
            import lpips  # type: ignore
        except ImportError as error:
            raise RuntimeError(
                "LPIPS is not installed. Install it with `pip install lpips`, or run evaluate.py "
                "with --skip_lpips to compute all remaining metrics."
            ) from error
        self.model: Any = lpips.LPIPS(net="alex").to(device).eval()
        self.device = device

    @torch.no_grad()
    def __call__(self, prediction: torch.Tensor, target: torch.Tensor) -> float:
        pred = prediction.unsqueeze(0) if prediction.ndim == 3 else prediction
        ref = target.unsqueeze(0) if target.ndim == 3 else target
        return float(self.model(pred.to(self.device) * 2 - 1, ref.to(self.device) * 2 - 1).item())

