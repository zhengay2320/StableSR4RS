# -*- coding: utf-8 -*-
"""小型端元库的线性解混数值核心（中文注释）。

1. fcls_batch：显式枚举单纯形所有非空面，求真正的非负、和为一最小二乘。
2. build_models：按类别约束构建端元组合；保留全部较小子组合。
3. unmix_library：同一个面只求一次，等价于在这些组合上比较FCLS解。

不使用“无约束最小二乘→截负数→归一化”近似，不对像元做光谱归一化。
所有输入光谱必须是同一反射率空间；weights是所有类别共用的残差权重。
本文件只依赖NumPy、threadpoolctl；独立于前面的五个提取模块。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations
from collections import Counter
from typing import Callable
import numpy as np
from threadpoolctl import threadpool_limits


@dataclass
class SolverConfig:
    # 这些阈值是实验起点，并非通过真实数据标定的精度标准。
    max_endmembers: int = 3
    max_per_class: int = 1  # 改为2可允许一个像元包含同类的两条不同光谱。
    max_models: int = 2500  # 超预算明确报错，不静默随机删模型。
    block_pixels: int = 2048
    max_rmse: float = 0.03  # 原始反射率空间RMSE；不是百分数，也不是MSE。
    selection_delta_mse: float = 1e-5  # 与最佳加权MSE的允许差，不是RMSE差。
    ambiguity_delta_mse: float = 1e-5
    ambiguity_max_extra: int = 0  # 近优模型大小 <= 所选大小 + 此值。
    abundance_uncertainty_max: float = 0.20  # 近优模型类别丰度跨度上限。
    condition_max: float = 1e5  # 仿射差分矩阵的条件数门槛。
    rank_rtol: float = 1e-12
    feasibility_tol: float = 1e-10  # 只容许浮点舍入尺度的轻微负数。
    weights: tuple | None = None  # 权重先归一为RMS=1，防止整体权重改变误差尺度。


@dataclass
class SimplexModel:
    model_id: int
    indices: tuple[int, ...]
    classes: tuple[int, ...]
    E: np.ndarray       # [L,k]，原始反射率
    Ew: np.ndarray      # [L,k]，共同加权后的光谱
    pinv_difference: np.ndarray | None
    condition: float

    @property
    def size(self) -> int:
        return len(self.indices)


def validate_solver(cfg: SolverConfig, n_bands: int) -> np.ndarray:
    """检查配置并返回共同权重。所有波段均必须拥有正权重，不允许偷偷删波段。"""
    if not 1 <= cfg.max_endmembers <= 5:
        raise ValueError('max_endmembers支持1—5；小端元集合采用显式面枚举。')
    if not 1 <= cfg.max_per_class <= cfg.max_endmembers:
        raise ValueError('max_per_class必须位于[1,max_endmembers]。')
    for name in ('max_models', 'block_pixels'):
        if getattr(cfg, name) < 1:
            raise ValueError(f'{name}必须为正整数。')
    if not np.isfinite(cfg.max_rmse) or cfg.max_rmse <= 0:
        raise ValueError('max_rmse必须为有限正数。')
    for name in ('selection_delta_mse', 'ambiguity_delta_mse'):
        if not np.isfinite(getattr(cfg, name)) or getattr(cfg, name) < 0:
            raise ValueError(f'{name}必须非负且有限。')
    if cfg.ambiguity_max_extra < 0 or not 0 <= cfg.abundance_uncertainty_max <= 1:
        raise ValueError('不确定性配置不合法。')
    if not np.isfinite(cfg.condition_max) or cfg.condition_max < 1:
        raise ValueError('condition_max必须 >= 1且有限。')
    if not 0 < cfg.rank_rtol < 1 or not 0 < cfg.feasibility_tol <= 1e-6:
        raise ValueError('rank_rtol/feasibility_tol不合法。')
    w = np.ones(n_bands) if cfg.weights is None else np.asarray(cfg.weights, dtype=float)
    if w.shape != (n_bands,) or not np.isfinite(w).all() or np.any(w <= 0):
        raise ValueError('weights必须为每个光谱波段提供一个有限正数。')
    return w / np.sqrt(np.mean(w ** 2))


def _validate_arrays(E: np.ndarray, Y: np.ndarray | None = None) -> tuple:
    E = np.asarray(E, dtype=np.float64)
    if E.ndim != 2 or min(E.shape) == 0 or not np.isfinite(E).all():
        raise ValueError('E必须是非空且有限的[波段数,端元数]矩阵。')
    if Y is None:
        return (E,)
    Y = np.asarray(Y, dtype=np.float64)
    if Y.ndim == 1:
        Y = Y[None, :]
    if Y.ndim != 2 or Y.shape[1] != E.shape[0] or not np.isfinite(Y).all():
        raise ValueError('Y必须是有限的[像元数,波段数]矩阵，且与E使用相同波段。')
    return E, Y


def _make_model(E: np.ndarray, indexes: tuple, class_ids: np.ndarray,
                weights: np.ndarray, cfg: SolverConfig, model_id: int):
    e = E[:, indexes]
    ew = e * weights[:, None]
    condition = 1.0
    pinv = None
    if len(indexes) > 1:
        # 消去a0：a0=1-sum(a1,...), 差分D=[e1-e0,...]。
        difference = ew[:, 1:] - ew[:, [0]]
        u, s, vt = np.linalg.svd(difference, full_matrices=False)
        if len(s) != len(indexes) - 1 or s[0] == 0 or s[-1] <= s[0] * cfg.rank_rtol:
            return None, 'affine_rank_deficient'
        condition = float(s[0] / s[-1])
        if condition > cfg.condition_max:
            return None, 'ill_conditioned'
        pinv = (vt.T / s) @ u.T
    return SimplexModel(model_id, indexes, tuple(int(class_ids[i]) for i in indexes),
                        e, ew, pinv, condition), None


def build_models(E: np.ndarray, class_ids: np.ndarray, cfg: SolverConfig):
    """枚举1...K个端元的全部类别允许组合；先小后大，保证组合库对取子集封闭。

    共线组合本身不必使用：其较小面已经可以表示相同凸包；近共线组合按配置拒绝。
    不跨类别合并近似或相同端元，否则会掩盖语义不可分性。
    """
    E, = _validate_arrays(E)
    class_ids = np.asarray(class_ids, dtype=int)
    if class_ids.shape != (E.shape[1],) or np.any(class_ids < 0):
        raise ValueError('每条端元必须具有一个非负整数类别编号。')
    w = validate_solver(cfg, E.shape[0])
    models, rejected = [], []
    rejected_sets = []
    n_seen = 0
    for k in range(1, min(cfg.max_endmembers, E.shape[1]) + 1):
        for subset in combinations(range(E.shape[1]), k):
            if max(Counter(class_ids[list(subset)]).values()) > cfg.max_per_class:
                continue
            n_seen += 1
            if n_seen > cfg.max_models:
                raise ValueError(f'允许组合超过max_models={cfg.max_models}。'
                                 '请显式减少候选或提高预算，不会自动截断组合。')
            # 拒绝不稳定子面时，也拒绝包含它的更大组合，保证保留模型集合对子集封闭。
            # 否则大组合的FCLS最优边界可能落在一个被丢弃的面上，破坏共享面求解的等价性。
            if any(s.issubset(subset) for s in rejected_sets):
                model, reason = None, 'contains_rejected_subface'
            else:
                model, reason = _make_model(E, subset, class_ids, w, cfg, len(models))
            if reason:
                rejected_sets.append(frozenset(subset))
                rejected.append(dict(indices=list(subset), size=k, reason=reason))
            else:
                models.append(model)
    return models, rejected, w


def solve_affine_face(model: SimplexModel, Y: np.ndarray,
                      weights: np.ndarray, tol: float = 1e-10):
    """求一个面的仿射投影，并检查是否落在该面单纯形内。

    面外解直接设为无效，不进行大幅截负修复；它的真正FCLS解在较小边界面上。
    库中包含所有这些边界面，因此批量引擎只需将同一个面计算一次。
    """
    n = len(Y)
    if model.size == 1:
        a = np.ones((n, 1), dtype=np.float64)
    else:
        b = (Y * weights - model.Ew[:, 0]) @ model.pinv_difference.T
        a = np.column_stack((1 - b.sum(axis=1), b))
    feasible = np.all(a >= -tol, axis=1) & np.all(a <= 1 + tol, axis=1)
    # 只修正已通过可行性检查解中的10^-10级舍入误差。
    a = np.clip(a, 0, 1)
    a /= np.maximum(a.sum(axis=1, keepdims=True), 1e-300)
    residual = Y - a @ model.E.T
    raw_mse = np.mean(residual ** 2, axis=1)
    weighted_mse = np.mean((residual * weights) ** 2, axis=1)
    raw_mse[~feasible] = np.inf
    weighted_mse[~feasible] = np.inf
    return a, weighted_mse, raw_mse


def fcls_batch(E: np.ndarray, Y: np.ndarray, weights=None,
               feasibility_tol: float = 1e-10):
    """对同一个固定端元矩阵，逐像元求真正FCLS（最多5条端元）。

    所有非空支撑集S均尝试：在1^Ta=1下求仿射最小二乘，只接受a>=0的解。
    比较全部可行面后取最小残差，最优解必在这些面之一；这不是事后截断LS。
    返回[像元数,端元数]丰度和共同加权MSE。
    退化矩阵允许多个等价丰度解，返回其中一个，而不是声称其唯一。
    """
    E, Y = _validate_arrays(E, Y)
    if E.shape[1] > 5:
        raise ValueError('此小型精确求解器最多支持5条端元。')
    cfg = SolverConfig(max_endmembers=E.shape[1], max_per_class=E.shape[1],
                       weights=weights, condition_max=1e15, feasibility_tol=feasibility_tol)
    models, _, w = build_models(E, np.arange(E.shape[1]), cfg)
    abundance = np.zeros((len(Y), E.shape[1]))
    error = np.full(len(Y), np.inf)
    with threadpool_limits(limits=1):
        for model in models:
            a, cost, _ = solve_affine_face(model, Y, w, cfg.feasibility_tol)
            better = cost < error
            abundance[better] = 0
            abundance[np.ix_(better, model.indices)] = a[better]
            error[better] = cost[better]
    return abundance, error


def evaluate_block(Y: np.ndarray, models: list, w: np.ndarray, cfg: SolverConfig):
    """一个像元块的所有面解；避免保存全景×所有模型×全部系数的大数组。"""
    n_models, n = len(models), len(Y)
    max_k = max(m.size for m in models)
    costs = np.full((n_models, n), np.inf)
    raw = np.full((n_models, n), np.inf)
    coeff = np.zeros((n_models, n, max_k), dtype=np.float64)
    for j, model in enumerate(models):
        a, cost, raw_mse = solve_affine_face(model, Y, w, cfg.feasibility_tol)
        coeff[j, :, :model.size] = a
        costs[j] = cost
        raw[j] = raw_mse
    return costs, raw, coeff


def unmix_library(Y: np.ndarray, E: np.ndarray, class_ids: np.ndarray,
                  cfg: SolverConfig | None = None, n_classes: int = 5,
                  progress: Callable | None = None) -> dict:
    """MESMA式组合选择、类别丰度、近优模型范围与删除类别敏感性。

    返回的是数值模型结果；“无端元/确认不存在/来源未核查”等语义由外层管理。
    raw_mse门槛使用原始反射率；选择和delta均使用共同加权MSE。
    全部模型都差时仍保存最佳诊断解，但fit_ok=False，不能当有效丰度。
    """
    cfg = cfg or SolverConfig()
    E, Y = _validate_arrays(E, Y)
    class_ids = np.asarray(class_ids, dtype=int)
    if np.any(class_ids >= n_classes):
        raise ValueError('端元类别编号超出n_classes。')
    models, rejected, w = build_models(E, class_ids, cfg)
    n, M = len(Y), E.shape[1]
    result = dict(
        abundance=np.zeros((n, n_classes)), candidate_abundance=np.zeros((n, M)),
        model_id=np.full(n, -1, dtype=np.int32), model_size=np.zeros(n, np.uint8),
        weighted_mse=np.full(n, np.nan), best_weighted_mse=np.full(n, np.nan),
        rmse=np.full(n, np.nan), fit_ok=np.zeros(n, bool),
        abundance_min=np.full((n, n_classes), np.nan),
        abundance_max=np.full((n, n_classes), np.nan),
        uncertainty=np.full((n, n_classes), np.nan),
        drop_class_delta=np.full((n, n_classes), np.nan),
        near_model_count=np.zeros(n, np.int32), models=models, rejected_models=rejected,
        weights=w, model_selected_count=np.zeros(len(models), int),
        model_feasible_count=np.zeros(len(models), int))
    sizes = np.asarray([m.size for m in models])
    class_presence = np.asarray([[c in m.classes for c in range(n_classes)] for m in models])
    with threadpool_limits(limits=1):
        for start in range(0, n, cfg.block_pixels):
            stop = min(start + cfg.block_pixels, n)
            block = Y[start:stop]
            costs, raw, coeff = evaluate_block(block, models, w, cfg)
            best_ids = np.argmin(costs, axis=0)
            px = np.arange(len(block))
            best = costs[best_ids, px]
            # 模型必须既通过原始RMSE门槛，也接近最优加权MSE。
            acceptable = ((raw <= cfg.max_rmse ** 2)
                          & (costs <= best[None, :] + cfg.selection_delta_mse))
            k_min = np.min(np.where(acceptable, sizes[:, None], 99), axis=0)
            eligible = acceptable & (sizes[:, None] == k_min[None, :])
            selection = np.argmin(np.where(eligible, costs, np.inf), axis=0)
            has_acceptable = acceptable.any(axis=0)
            selection[~has_acceptable] = best_ids[~has_acceptable]
            local_A = np.zeros((len(block), M))
            for j, model in enumerate(models):
                selected = selection == j
                if selected.any():
                    local_A[np.ix_(selected, model.indices)] = coeff[j, selected, :model.size]
            class_A = np.column_stack([local_A[:, class_ids == c].sum(axis=1)
                                       for c in range(n_classes)])
            sl = slice(start, stop)
            result['candidate_abundance'][sl] = local_A
            result['abundance'][sl] = class_A
            result['model_id'][sl] = selection
            result['model_size'][sl] = sizes[selection]
            result['weighted_mse'][sl] = costs[selection, px]
            result['best_weighted_mse'][sl] = best
            result['rmse'][sl] = np.sqrt(raw[selection, px])
            result['fit_ok'][sl] = has_acceptable
            result['model_selected_count'] += np.bincount(selection, minlength=len(models))
            result['model_feasible_count'] += np.isfinite(costs).sum(axis=1)
            # 近优模型集合不限于“第二名”。类别敏感性比端元编号变化更重要。
            near = ((costs <= best[None, :] + cfg.ambiguity_delta_mse)
                    & (raw <= cfg.max_rmse ** 2)
                    & (sizes[:, None] <= sizes[selection][None, :] + cfg.ambiguity_max_extra))
            # 所选有效模型也属于解释集合，防止两个容差不同导致其被排除。
            near[selection[has_acceptable], px[has_acceptable]] = True
            lower = np.full((len(block), n_classes), np.inf)
            upper = np.full((len(block), n_classes), -np.inf)
            for j, model in enumerate(models):
                yes = near[j]
                if not yes.any():
                    continue
                grouped = np.zeros((yes.sum(), n_classes))
                for t, c in enumerate(model.classes):
                    grouped[:, c] += coeff[j, yes, t]
                lower[yes] = np.minimum(lower[yes], grouped)
                upper[yes] = np.maximum(upper[yes], grouped)
            missing = ~near.any(axis=0)
            lower[missing] = np.nan
            upper[missing] = np.nan
            result['abundance_min'][sl] = lower
            result['abundance_max'][sl] = upper
            result['uncertainty'][sl] = upper - lower
            result['near_model_count'][sl] = near.sum(axis=0)
            # 去除一整类后最佳重建MSE的增量。没有可比较模型时保持NaN。
            for c in range(n_classes):
                allowed = ~class_presence[:, c]
                if not allowed.any() or not np.any(class_ids == c):
                    continue
                without = np.min(costs[allowed], axis=0)
                good = np.isfinite(without)
                result['drop_class_delta'][start:stop, c][good] = np.maximum(without[good] - best[good], 0)
            if progress:
                progress(stop, n)
    result['reconstruction'] = result['candidate_abundance'] @ E.T
    result['residual'] = Y - result['reconstruction']
    return result


def fixed_class_baseline(Y: np.ndarray, E: np.ndarray, class_ids: np.ndarray,
                         cfg: SolverConfig, n_classes: int = 5) -> dict:
    """每类使用被端元库选择器排在最前的一条候选，固定FCLS对照。
    最多五类；不允许与主模型使用不同的权重或不同的反射率缩放。
    """
    chosen = [int(np.flatnonzero(class_ids == c)[0]) for c in range(n_classes)
              if np.any(class_ids == c)]
    A = np.zeros((len(Y), len(chosen)))
    cost = np.full(len(Y), np.nan)
    for start in range(0, len(Y), cfg.block_pixels):
        stop = min(start + cfg.block_pixels, len(Y))
        A[start:stop], cost[start:stop] = fcls_batch(E[:, chosen], Y[start:stop], cfg.weights)
    grouped = np.zeros((len(Y), n_classes))
    for t, j in enumerate(chosen):
        grouped[:, class_ids[j]] += A[:, t]
    reconstruction = A @ E[:, chosen].T
    return dict(abundance=grouped, rmse=np.sqrt(np.mean((Y-reconstruction)**2, axis=1)),
                weighted_mse=cost, indices=chosen)
