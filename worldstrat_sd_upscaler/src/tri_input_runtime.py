"""Shared runtime helpers for tri-input training, validation, and inference.

The helpers in this module intentionally know nothing about Diffusers or
Accelerate.  They keep the three execution paths on the same definition of a
condition and make the explicit ``aux_present=False`` replay policy a real
bypass rather than a zero-filled pseudo observation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import torch
import torch.nn.functional as F
from torch import nn

from src.tri_input_bridge import PreparedTriResiduals, TriInputBridge, prepare_tri_residuals
from src.tri_input_conditioner import TriInputConditionOutput, TriInputConditioner


@dataclass(frozen=True)
class PreparedTriCondition:
    """Conditioner output and residuals prepared once for one batch/tile."""

    output: TriInputConditionOutput
    residuals: PreparedTriResiduals


def _batch_tensor(
    batch: Mapping[str, Any],
    name: str,
    *,
    device: torch.device,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    if name not in batch:
        raise KeyError(f"Tri-input batch is missing required field {name!r}")
    value = batch[name]
    if not isinstance(value, torch.Tensor):
        value = torch.as_tensor(value)
    return value.to(device=device, dtype=dtype) if dtype is not None else value.to(device=device)


def prepare_tri_condition(
    conditioner: TriInputConditioner | nn.Module,
    bridge: TriInputBridge | nn.Module,
    *,
    rgb_minus_one_one: torch.Tensor,
    raw_ms: torch.Tensor,
    unmixing: torch.Tensor,
    raw_valid: torch.Tensor,
    aux_present: torch.Tensor | bool,
    unet_sample: torch.Tensor,
    num_images_per_prompt: int = 1,
    do_classifier_free_guidance: bool = False,
) -> PreparedTriCondition | None:
    """Compute one reusable condition, or bypass an all-inactive batch.

    Partial batches are supported: the bridge masks inactive samples and the
    preview loss helper applies the same mask.  If every sample is inactive,
    neither the conditioner nor bridge is called.
    """

    present = torch.as_tensor(aux_present, device=rgb_minus_one_one.device).bool().reshape(-1)
    if present.numel() == 1 and rgb_minus_one_one.shape[0] != 1:
        present = present.expand(rgb_minus_one_one.shape[0])
    if present.numel() != rgb_minus_one_one.shape[0]:
        raise ValueError(
            "aux_present must provide one value per RGB sample: "
            f"got {present.numel()} for batch {rgb_minus_one_one.shape[0]}"
        )
    if not bool(present.any()):
        return None

    output = conditioner(
        rgb_minus_one_one,
        raw_ms,
        unmixing,
        raw_valid,
        present,
    )
    residuals = prepare_tri_residuals(
        bridge,
        output,
        unet_sample,
        active_mask=present,
        num_images_per_prompt=num_images_per_prompt,
        do_classifier_free_guidance=do_classifier_free_guidance,
    )
    return PreparedTriCondition(output=output, residuals=residuals)


def prepare_tri_condition_from_batch(
    conditioner: TriInputConditioner | nn.Module,
    bridge: TriInputBridge | nn.Module,
    batch: Mapping[str, Any],
    *,
    rgb_minus_one_one: torch.Tensor,
    unet_sample: torch.Tensor,
    device: torch.device,
    num_images_per_prompt: int = 1,
    do_classifier_free_guidance: bool = False,
) -> PreparedTriCondition | None:
    """Device-safe batch adapter shared by the trainer and validation."""

    return prepare_tri_condition(
        conditioner,
        bridge,
        rgb_minus_one_one=rgb_minus_one_one.float(),
        raw_ms=_batch_tensor(batch, "raw_ms", device=device, dtype=torch.float32),
        unmixing=_batch_tensor(batch, "unmixing", device=device, dtype=torch.float32),
        raw_valid=_batch_tensor(batch, "raw_valid", device=device).bool(),
        aux_present=_batch_tensor(batch, "aux_present", device=device).bool(),
        unet_sample=unet_sample,
        num_images_per_prompt=num_images_per_prompt,
        do_classifier_free_guidance=do_classifier_free_guidance,
    )


def masked_preview_l1(
    preview: torch.Tensor,
    target_rgb_minus_one_one: torch.Tensor,
    active_mask: torch.Tensor,
) -> torch.Tensor:
    """Full-batch averaged preview loss with inactive samples weighted zero.

    Dividing by the complete batch, rather than active count, deliberately
    avoids compensating for synthetic replay or other inactive observations.
    """

    if preview.shape != target_rgb_minus_one_one.shape:
        raise ValueError(
            f"preview/target shapes differ: {tuple(preview.shape)} vs "
            f"{tuple(target_rgb_minus_one_one.shape)}"
        )
    mask = torch.as_tensor(active_mask, device=preview.device).bool().reshape(-1)
    if mask.numel() != preview.shape[0]:
        raise ValueError("active_mask must contain one value per preview sample")
    per_sample = F.l1_loss(
        preview.float(), target_rgb_minus_one_one.float(), reduction="none"
    ).mean(dim=(1, 2, 3))
    return (per_sample * mask.to(per_sample.dtype)).mean()


def tri_diagnostic_metrics(
    prepared: PreparedTriCondition | TriInputConditionOutput | None,
) -> dict[str, torch.Tensor]:
    """Return compact numeric diagnostics; scores remain internal heuristics."""

    if prepared is None:
        zero = torch.zeros((), dtype=torch.float32)
        return {
            "tri_aux_present_fraction": zero,
            "checker_valid_windows": zero,
            "checker_accepted_fraction": zero,
            "checker_mean_shift_hr": zero,
            "checker_internal_gain": zero,
        }
    output = prepared.output if isinstance(prepared, PreparedTriCondition) else prepared
    active = output.active_mask.float()
    diagnostics = output.checker_diagnostics
    valid = diagnostics.valid_window_count.float()
    accepted = diagnostics.accepted_window_count.float()
    active_count = active.sum().clamp_min(1.0)
    accepted_fraction = torch.where(valid > 0, accepted / valid.clamp_min(1.0), torch.zeros_like(valid))
    flow_magnitude = torch.linalg.vector_norm(output.flow_hr.float(), dim=1)
    supported = output.support[:, 0].float()
    mean_shift = (flow_magnitude * supported).flatten(1).sum(1) / supported.flatten(1).sum(1).clamp_min(1.0)
    gain = diagnostics.internal_error_before.float() - diagnostics.internal_error_after.float()
    return {
        "tri_aux_present_fraction": active.mean(),
        "checker_valid_windows": (valid * active).sum() / active_count,
        "checker_accepted_fraction": (accepted_fraction * active).sum() / active_count,
        "checker_mean_shift_hr": (mean_shift * active).sum() / active_count,
        "checker_internal_gain": (gain * active).sum() / active_count,
    }
