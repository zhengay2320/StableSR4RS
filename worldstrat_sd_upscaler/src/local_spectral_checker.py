"""Finite-candidate local spectral consistency checker.

The scores in this module are internal heuristic diagnostics.  They are not
probabilities, calibrated confidence, or measurements of unknown HR truth.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn


CANDIDATE_NAMES = ("stationary", "left", "right", "up", "down")


@dataclass(frozen=True)
class CheckerConfig:
    enabled: bool = True
    scale: int = 4
    common_factor: int = 2
    window: int = 8
    stride: int = 4
    max_shift_hr: float = 1.0
    ridge: float = 1.0e-4
    movement_penalty: float = 1.0e-4
    min_relative_gain: float = 0.02
    min_relative_gap: float = 0.005
    absolute_gain_floor: float = 1.0e-7
    min_fit_samples: int = 8
    max_condition_number: float = 1.0e8

    def __post_init__(self) -> None:
        if self.scale <= 0 or self.common_factor <= 0:
            raise ValueError("checker scale and common_factor must be positive")
        if self.window <= 0 or self.stride <= 0:
            raise ValueError("checker window and stride must be positive")
        if self.max_shift_hr < 0 or self.ridge <= 0:
            raise ValueError("checker max_shift_hr must be non-negative and ridge positive")
        if min(
            self.movement_penalty,
            self.min_relative_gain,
            self.min_relative_gap,
            self.absolute_gain_floor,
        ) < 0:
            raise ValueError("checker thresholds and movement penalty must be non-negative")
        if self.min_fit_samples < 2:
            raise ValueError("checker min_fit_samples must be at least two")

    @classmethod
    def from_mapping(cls, value: dict[str, Any] | None) -> "CheckerConfig":
        return cls(**(dict(value) if value else {}))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class SpectralCheckOutput:
    q_star: torch.Tensor
    flow_hr: torch.Tensor
    support: torch.Tensor
    scores: torch.Tensor
    valid_window_count: torch.Tensor
    accepted_window_count: torch.Tensor
    internal_error_before: torch.Tensor
    internal_error_after: torch.Tensor
    fallback_reasons: tuple[str, ...]


def _base_grid(batch: int, height: int, width: int, device: torch.device) -> torch.Tensor:
    y = (torch.arange(height, device=device, dtype=torch.float32) + 0.5) * (2.0 / height) - 1.0
    x = (torch.arange(width, device=device, dtype=torch.float32) + 0.5) * (2.0 / width) - 1.0
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    return torch.stack((xx, yy), dim=-1).unsqueeze(0).expand(batch, -1, -1, -1)


def warp_internal_structure(q: torch.Tensor, flow_hr: torch.Tensor) -> torch.Tensor:
    """Warp Q using target-grid pixel displacement; positive x/y moves content right/down."""
    if q.ndim != 4 or flow_hr.shape != (q.shape[0], 2, q.shape[2], q.shape[3]):
        raise ValueError(
            f"Expected q=[B,K,H,W] and flow=[B,2,H,W], got {tuple(q.shape)} and {tuple(flow_hr.shape)}"
        )
    height, width = q.shape[-2:]
    grid = _base_grid(q.shape[0], height, width, q.device)
    displacement = torch.stack(
        (
            2.0 * flow_hr[:, 0].float() / float(width),
            2.0 * flow_hr[:, 1].float() / float(height),
        ),
        dim=-1,
    )
    # grid_sample asks where to read from.  Reading from x-dx moves content by +dx.
    return F.grid_sample(
        q,
        grid - displacement,
        mode="bilinear",
        padding_mode="border",
        align_corners=False,
    )


def constant_shift_flow(q: torch.Tensor, dx: float, dy: float) -> torch.Tensor:
    flow = q.new_zeros((q.shape[0], 2, q.shape[2], q.shape[3]), dtype=torch.float32)
    flow[:, 0].fill_(float(dx))
    flow[:, 1].fill_(float(dy))
    return flow


def five_shift_candidates(q: torch.Tensor, shift_hr: float) -> tuple[torch.Tensor, ...]:
    shifts = (
        (0.0, 0.0),
        (-shift_hr, 0.0),
        (shift_hr, 0.0),
        (0.0, -shift_hr),
        (0.0, shift_hr),
    )
    return tuple(warp_internal_structure(q, constant_shift_flow(q, dx, dy)) for dx, dy in shifts)


def _window_positions(length: int, window: int, stride: int) -> list[int]:
    if length <= 0:
        return []
    actual = min(length, window)
    positions = list(range(0, max(1, length - actual + 1), stride))
    final = length - actual
    if not positions or positions[-1] != final:
        positions.append(final)
    return positions


def _masked_common_observation(
    surface: torch.Tensor, valid: torch.Tensor, factor: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Approximate a common observation grid by masked average pooling.

    This is deliberately an observation-operator approximation, not a claim of
    a calibrated sensor PSF.
    """
    if factor == 1:
        return surface.float(), valid.bool()
    weights = valid.float()
    numerator = F.avg_pool2d(
        surface.float() * weights, factor, factor, ceil_mode=True, count_include_pad=False
    )
    denominator = F.avg_pool2d(
        weights, factor, factor, ceil_mode=True, count_include_pad=False
    )
    pooled = numerator / denominator.clamp_min(1.0e-12)
    return pooled, denominator > 0


def _fold_error(
    design: torch.Tensor,
    observation: torch.Tensor,
    valid: torch.Tensor,
    ridge: float,
    min_samples: int,
    max_condition_number: float,
) -> torch.Tensor | None:
    """Two-way interleaved fit/score using a shared local spectral matrix."""
    channels, height, width = design.shape
    if observation.ndim != 3 or observation.shape[1:] != (height, width):
        raise ValueError("design and observation must share spatial dimensions")
    yy, xx = torch.meshgrid(
        torch.arange(height, device=design.device),
        torch.arange(width, device=design.device),
        indexing="ij",
    )
    # Alternate complete 2x2 spatial blocks between folds.  Pixel-wise or
    # column-wise alternation would leak immediate neighbours into the scoring
    # fold and is not the requested interleaved block split.
    fold = ((yy // 2) + (xx // 2)) % 2 == 0
    valid_2d = valid.reshape(height, width).bool()
    errors: list[torch.Tensor] = []
    identity = torch.eye(channels, device=design.device, dtype=torch.float32)
    a_all = design.float().permute(1, 2, 0)
    y_all = observation.float().permute(1, 2, 0)
    with torch.autocast(device_type=design.device.type, enabled=False):
        for train_fold in (fold, ~fold):
            train_mask = valid_2d & train_fold
            score_mask = valid_2d & ~train_fold
            if int(train_mask.sum()) < max(min_samples, channels) or int(score_mask.sum()) < 1:
                return None
            a_train = a_all[train_mask]
            y_train = y_all[train_mask]
            count = float(a_train.shape[0])
            gram = a_train.T @ a_train / count + float(ridge) * identity
            rhs = a_train.T @ y_train / count
            condition = torch.linalg.cond(gram)
            if not bool(torch.isfinite(condition)) or float(condition) > max_condition_number:
                return None
            try:
                coefficients = torch.linalg.solve(gram, rhs)
            except RuntimeError:
                return None
            prediction = a_all[score_mask] @ coefficients
            error = (prediction - y_all[score_mask]).square().mean()
            if not bool(torch.isfinite(error)):
                return None
            errors.append(error)
    return torch.stack(errors).mean()


class LocalSpectralChecker(nn.Module):
    """Choose among stationary/left/right/up/down Q candidates per local window."""

    def __init__(self, config: CheckerConfig | dict[str, Any] | None = None) -> None:
        super().__init__()
        self.config = config if isinstance(config, CheckerConfig) else CheckerConfig.from_mapping(config)

    def _empty_output(self, q0: torch.Tensor, reason: str) -> SpectralCheckOutput:
        batch, _, height, width = q0.shape
        zero = q0.new_zeros((), dtype=torch.float32)
        return SpectralCheckOutput(
            q_star=q0,
            flow_hr=q0.new_zeros((batch, 2, height, width), dtype=torch.float32),
            support=q0.new_zeros((batch, 1, height, width), dtype=torch.float32),
            scores=q0.new_zeros((batch, 5, 1, 1), dtype=torch.float32),
            valid_window_count=q0.new_zeros((batch,), dtype=torch.float32),
            accepted_window_count=q0.new_zeros((batch,), dtype=torch.float32),
            internal_error_before=zero.expand(batch).clone(),
            internal_error_after=zero.expand(batch).clone(),
            fallback_reasons=tuple(reason for _ in range(batch)),
        )

    def forward(
        self,
        q0: torch.Tensor,
        surface_observation: torch.Tensor,
        raw_valid: torch.Tensor,
    ) -> SpectralCheckOutput:
        if not self.config.enabled:
            # Disabled means an immediate identity path with no checker-only size constraints.
            return self._empty_output(q0, "checker_disabled")
        if q0.ndim != 4:
            raise ValueError(f"q0 must be BCHW, got {tuple(q0.shape)}")
        if surface_observation.ndim != 4 or surface_observation.shape[0] != q0.shape[0]:
            raise ValueError("surface_observation must be BCHW with q0 batch size")
        if raw_valid.shape != (
            q0.shape[0],
            1,
            surface_observation.shape[-2],
            surface_observation.shape[-1],
        ):
            raise ValueError("raw_valid must have shape [B,1,H,W] matching the observation")
        expected_hr = (
            surface_observation.shape[-2] * self.config.scale,
            surface_observation.shape[-1] * self.config.scale,
        )
        if q0.shape[-2:] != expected_hr:
            raise ValueError(
                f"q0 spatial size {tuple(q0.shape[-2:])} must equal raw grid {tuple(surface_observation.shape[-2:])} "
                f"times scale={self.config.scale}"
            )

        # Candidate scoring and discrete decisions intentionally do not backpropagate.
        with torch.no_grad():
            observation, common_valid = _masked_common_observation(
                surface_observation, raw_valid, self.config.common_factor
            )
            check_height, check_width = observation.shape[-2:]
            if check_height < 2 or check_width < 2:
                return self._empty_output(q0, "check_grid_too_small")
            candidates_hr = five_shift_candidates(q0.float(), self.config.max_shift_hr)
            candidates = tuple(
                F.adaptive_avg_pool2d(candidate, (check_height, check_width))
                for candidate in candidates_hr
            )

            batch = q0.shape[0]
            flow = q0.new_zeros(q0.shape[0], 2, *q0.shape[-2:], dtype=torch.float32)
            flow_weight = q0.new_zeros(q0.shape[0], 1, *q0.shape[-2:], dtype=torch.float32)
            accepted_weight = torch.zeros_like(flow_weight)
            score_maps = q0.new_zeros((batch, 5, check_height, check_width), dtype=torch.float32)
            score_weights = q0.new_zeros((batch, 1, check_height, check_width), dtype=torch.float32)
            valid_windows = q0.new_zeros((batch,), dtype=torch.float32)
            accepted_windows = q0.new_zeros((batch,), dtype=torch.float32)
            before_values = q0.new_zeros((batch,), dtype=torch.float32)
            after_values = q0.new_zeros((batch,), dtype=torch.float32)
            reasons: list[str] = []

            actual_window_h = min(self.config.window, check_height)
            actual_window_w = min(self.config.window, check_width)
            tops = _window_positions(check_height, self.config.window, self.config.stride)
            lefts = _window_positions(check_width, self.config.window, self.config.stride)
            shifts = (
                (0.0, 0.0),
                (-self.config.max_shift_hr, 0.0),
                (self.config.max_shift_hr, 0.0),
                (0.0, -self.config.max_shift_hr),
                (0.0, self.config.max_shift_hr),
            )

            for batch_index in range(batch):
                sample_reason = "no_informative_window"
                for top in tops:
                    for left in lefts:
                        bottom = min(top + actual_window_h, check_height)
                        right = min(left + actual_window_w, check_width)
                        mask = common_valid[batch_index, :, top:bottom, left:right]
                        local_scores: list[torch.Tensor] = []
                        failed = False
                        for candidate in candidates:
                            value = _fold_error(
                                candidate[batch_index, :, top:bottom, left:right],
                                observation[batch_index, :, top:bottom, left:right],
                                mask,
                                self.config.ridge,
                                self.config.min_fit_samples,
                                self.config.max_condition_number,
                            )
                            if value is None:
                                failed = True
                                break
                            local_scores.append(value)
                        if failed:
                            continue
                        score_tensor = torch.stack(local_scores)
                        penalized = score_tensor.clone()
                        penalized[1:] += self.config.movement_penalty
                        order = torch.argsort(penalized)
                        best = int(order[0])
                        second = int(order[1])
                        baseline = penalized[0]
                        best_score = penalized[best]
                        scale = baseline.abs().clamp_min(self.config.absolute_gain_floor)
                        gain = baseline - best_score
                        relative_gain = gain / scale
                        relative_gap = (penalized[second] - best_score) / scale
                        accept = (
                            best != 0
                            and float(gain) >= self.config.absolute_gain_floor
                            and float(relative_gain) >= self.config.min_relative_gain
                            and float(relative_gap) >= self.config.min_relative_gap
                        )
                        valid_windows[batch_index] += 1
                        score_maps[batch_index, :, top:bottom, left:right] += score_tensor[:, None, None]
                        score_weights[batch_index, :, top:bottom, left:right] += 1

                        hr_top = int(round(top * q0.shape[-2] / check_height))
                        hr_bottom = int(round(bottom * q0.shape[-2] / check_height))
                        hr_left = int(round(left * q0.shape[-1] / check_width))
                        hr_right = int(round(right * q0.shape[-1] / check_width))
                        flow_weight[batch_index, :, hr_top:hr_bottom, hr_left:hr_right] += 1
                        if accept:
                            dx, dy = shifts[best]
                            flow[batch_index, 0, hr_top:hr_bottom, hr_left:hr_right] += dx
                            flow[batch_index, 1, hr_top:hr_bottom, hr_left:hr_right] += dy
                            accepted_weight[batch_index, :, hr_top:hr_bottom, hr_left:hr_right] += 1
                            accepted_windows[batch_index] += 1
                            sample_reason = "accepted"

                if int(valid_windows[batch_index]) == 0:
                    reasons.append("insufficient_valid_observations_or_ill_conditioned")
                    continue

                flow[batch_index] /= flow_weight[batch_index].clamp_min(1.0)
                flow[batch_index : batch_index + 1] = F.avg_pool2d(
                    flow[batch_index : batch_index + 1], 3, stride=1, padding=1
                )
                provisional = warp_internal_structure(
                    q0[batch_index : batch_index + 1].float(),
                    flow[batch_index : batch_index + 1],
                )
                provisional_check = F.adaptive_avg_pool2d(
                    provisional, (check_height, check_width)
                )[0]
                baseline_error = _fold_error(
                    candidates[0][batch_index],
                    observation[batch_index],
                    common_valid[batch_index],
                    self.config.ridge,
                    self.config.min_fit_samples,
                    self.config.max_condition_number,
                )
                after_error = _fold_error(
                    provisional_check,
                    observation[batch_index],
                    common_valid[batch_index],
                    self.config.ridge,
                    self.config.min_fit_samples,
                    self.config.max_condition_number,
                )
                if baseline_error is None or after_error is None:
                    flow[batch_index].zero_()
                    accepted_weight[batch_index].zero_()
                    accepted_windows[batch_index].zero_()
                    reasons.append("sample_rescore_unavailable")
                    continue
                before_values[batch_index] = baseline_error
                after_values[batch_index] = after_error
                if float(after_error) >= float(baseline_error) - self.config.absolute_gain_floor:
                    flow[batch_index].zero_()
                    accepted_weight[batch_index].zero_()
                    accepted_windows[batch_index].zero_()
                    after_values[batch_index] = baseline_error
                    reasons.append("sample_rescore_not_improved")
                else:
                    reasons.append(sample_reason)

            support = accepted_weight / flow_weight.clamp_min(1.0)
            score_maps = score_maps / score_weights.clamp_min(1.0)
            flow = flow.detach()
            support = support.detach()
            score_maps = score_maps.detach()

        # Only this final warp is differentiable with respect to Q0.  The
        # discrete candidate choice and flow estimation above stay no-grad.
        # Samples that conservatively fell back to zero flow keep the exact Q0
        # tensor values; an identity grid_sample can otherwise introduce tiny
        # interpolation roundoff and break strict no-op behaviour.
        moving = flow.abs().flatten(1).sum(1) > 0
        if bool(moving.any()):
            warped = warp_internal_structure(q0, flow)
            q_star = torch.where(moving[:, None, None, None], warped, q0)
        else:
            q_star = q0
        return SpectralCheckOutput(
            q_star=q_star,
            flow_hr=flow,
            support=support,
            scores=score_maps,
            valid_window_count=valid_windows,
            accepted_window_count=accepted_windows,
            internal_error_before=before_values,
            internal_error_after=after_values,
            fallback_reasons=tuple(reasons),
        )
