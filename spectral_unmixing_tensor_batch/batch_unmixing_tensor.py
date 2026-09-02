# -*- coding: utf-8 -*-
r"""
WorldStrat 批量光谱解混 -> 单一深度学习张量
================================================

目标
----
对 INPUT_DIR 中的每一张 TIFF：
1. 复用当前文件夹中的五类端元提取模块；
2. 复用 spectral_unmixing_pipeline.py / unmixing_core.py 完成组合 FCLS 解混；
3. 不永久保存端元提取中间图、CSV、GeoTIFF、HTML 等文件；
4. 每张输入影像最终只保存一个 .npy 张量。

输出张量
--------
默认采用 PyTorch 常用的 CHW 排列：shape = [10, H, W]，dtype=float32。

通道固定为：
    0  vegetation_abundance
    1  water_abundance
    2  bare_abundance
    3  snow_abundance
    4  building_abundance
    5  vegetation_uncertainty
    6  water_uncertainty
    7  bare_uncertainty
    8  snow_uncertainty
    9  building_uncertainty

丰度范围原则上为 [0, 1]。
不确定性采用当前解混框架中的“近优模型类别丰度跨度”：
    uncertainty_c = max(F_c) - min(F_c)
因此也是 [0, 1]，但它是模型选择敏感性，不是概率或置信区间。

为了便于直接作为深度学习输入/监督信号，本脚本不在最终张量中保留 NaN：
- 当前类别/像元有可靠条件性估计：保留丰度与不确定性；
- 类别端元不可用、像元拟合失败、NoData/质量无效：
      abundance = 0
      uncertainty = 1
这样可以区分：
- “可信的零丰度”      -> abundance≈0, uncertainty≈0
- “不知道/无法估计”  -> abundance=0, uncertainty=1

依赖文件
--------
请将本脚本与以下文件放在同一目录：
    vegetation_candidates_all_bands.py
    water_candidates_all_bands.py
    bare_candidates_all_bands.py
    snow_candidates_all_bands.py
    building_candidates_all_bands.py
    unmixing_core.py
    spectral_unmixing_pipeline.py

运行
----
    python batch_unmixing_tensor.py
"""

from __future__ import annotations

from contextlib import redirect_stdout, redirect_stderr
from pathlib import Path
import gc
import io
import os
import shutil
import tempfile
import traceback

import numpy as np

import spectral_unmixing_pipeline as pipeline
from spectral_unmixing_pipeline import Config, ClassSpec
from unmixing_core import SolverConfig


# ============================================================================
# 1. 用户配置
# ============================================================================

HERE = Path(__file__).resolve().parent

INPUT_DIR = Path(r"/data/zhengay/EDiffSR-main/data/new_star/train/lr")
OUTPUT_DIR = Path(r"/data/zhengay/EDiffSR-main/data/new_star/test/train_unmixing")

# False：只扫描 INPUT_DIR 当前一级；True：递归扫描子目录。
RECURSIVE = False

# 已确认当前 WorldStrat 数据就是 float 反射率时使用 1.0 / 0.0。
DATA_CONFIRMED = True
SCALE = 1.0
OFFSET = 0.0

# 已存在同名 .npy 时是否跳过。
SKIP_EXISTING = True

# 输出布局：
#   "CHW" -> [10, H, W]，适合 PyTorch；
#   "HWC" -> [H, W, 10]，更接近普通图像数组。
OUTPUT_LAYOUT = "CHW"

# 是否把内部五类提取器/解混器的大量控制台输出隐藏。
# False 时每景只打印开始/完成/失败信息。
VERBOSE_INTERNAL = False

# ---------------------------------------------------------------------------
# 组合 FCLS 设置
# ---------------------------------------------------------------------------
SOLVER = SolverConfig(
    max_endmembers=3,
    max_per_class=1,
    max_rmse=0.03,
    selection_delta_mse=1e-5,
    ambiguity_delta_mse=1e-5,
    block_pixels=2048,
)

# ---------------------------------------------------------------------------
# 五类端元提取参数覆盖。
# 默认不修改原五个脚本的 CFG / Config。
# ---------------------------------------------------------------------------
EXTRACTION_OVERRIDES = {
    "vegetation": {},
    "water": {},
    "bare": {},
    "snow": {},
    "building": {},

    # 例如做建筑腐蚀消融实验：
    # "building": {"erosion_radius": 0},
}

# 类别是否已被外部证据确认不存在。
# 默认全部 unknown，不能因为提取失败就自动记为 absent。
CLASS_STATES = {
    "vegetation": ("unknown", ""),
    "water": ("unknown", ""),
    "bare": ("unknown", ""),
    "snow": ("unknown", ""),
    "building": ("unknown", ""),
}

CLASSES = ("vegetation", "water", "bare", "snow", "building")


# ============================================================================
# 2. 工具函数
# ============================================================================

def discover_tiffs(input_dir: Path) -> list[Path]:
    """按确定顺序扫描 TIFF。"""
    if not input_dir.is_dir():
        raise FileNotFoundError(f"找不到输入目录：{input_dir}")

    iterator = input_dir.rglob("*") if RECURSIVE else input_dir.iterdir()
    files = [
        p for p in iterator
        if p.is_file() and p.suffix.lower() in {".tif", ".tiff"}
    ]
    files.sort(key=lambda p: str(p.relative_to(input_dir)).lower())

    # 输出使用 stem.npy，避免同 stem 冲突。
    seen = {}
    for p in files:
        key = p.stem.lower()
        if key in seen:
            raise ValueError(
                "发现同名 TIFF，输出文件会冲突：\n"
                f"  {seen[key]}\n  {p}"
            )
        seen[key] = p
    return files


def make_config(input_file: Path, temp_root: Path) -> Config:
    """构造单景配置；所有中间文件都写入临时目录。"""
    class_specs = {}
    for name in CLASSES:
        state, reason = CLASS_STATES[name]
        class_specs[name] = ClassSpec(
            source="AUTO",
            scene_state=state,
            state_reason=reason,
            reviewed=False,
            max_candidates=3,
            extraction_overrides=dict(EXTRACTION_OVERRIDES.get(name, {})),
        )

    return Config(
        input_path=input_file,
        script_dir=HERE,
        output_root=temp_root,
        run_name="work",
        product_level="AUTO",
        band_map=None,
        data_confirmed=DATA_CONFIRMED,
        scale=SCALE,
        offset=OFFSET,
        clear_mask_path=None,
        scl_path=None,
        classes=class_specs,
        solver=SOLVER,

        # 自动批处理中允许未人工核查候选进入 conditional 解。
        allow_unreviewed_candidates=True,
        continue_on_extractor_error=False,

        # 后续深度学习只需要最终张量，因此全部关闭。
        run_baseline=False,
        show_figures=False,
        make_plots=False,
        make_html=False,
        save_pixel_csv=False,
        save_example_pixels=0,
        blas_threads=1,
    )


def _scatter(values: np.ndarray, valid: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """把 [N,C] 结果还原为 [C,H,W]；未参与解混的位置填 NaN。"""
    values = np.asarray(values)
    if values.ndim != 2:
        raise ValueError(f"期望二维 [N,C] 数组，实际 {values.shape}")
    if values.shape[0] != int(valid.sum()):
        raise ValueError("解混像元数量与 valid 掩膜不一致。")

    out = np.full((values.shape[1], *shape), np.nan, dtype=np.float32)
    out[:, valid] = values.T.astype(np.float32, copy=False)
    return out


def build_training_tensor(data: dict) -> np.ndarray:
    """
    将 pipeline 返回值压缩为一个深度学习张量。

    返回
    ----
    CHW 模式：float32 [10,H,W]
    HWC 模式：float32 [H,W,10]

    0~4 为五类丰度，5~9 为对应类别不确定性。
    """
    scene = data["scene"]
    valid = np.asarray(scene["valid"], dtype=bool)
    shape = tuple(scene["shape"])

    # conditional_abundance 已经做过：
    # - fit 不通过 -> NaN
    # - 类别端元不可用 -> 该类别 NaN
    # - 用户明确确认不存在 -> 该类别 0
    abundance = _scatter(data["conditional_abundance"], valid, shape)

    # uncertainty 是近优模型集合中，各类别丰度的 max-min。
    # 对 unavailable 类，pipeline 已将该列设为 NaN。
    uncertainty = _scatter(data["uncertainty"], valid, shape)

    if abundance.shape[0] != 5 or uncertainty.shape[0] != 5:
        raise ValueError(
            f"预期五类丰度/不确定性，实际 {abundance.shape}, {uncertainty.shape}"
        )

    # ---------------------------------------------------------------------
    # 深度学习友好的缺失值编码
    # ---------------------------------------------------------------------
    # 若丰度或其不确定性任一不可用，则把这一类在该像元标记为“未知”：
    #   abundance = 0
    #   uncertainty = 1
    # 这与“可信的0丰度（0, 低 uncertainty）”不同。
    unknown = ~np.isfinite(abundance) | ~np.isfinite(uncertainty)

    abundance = np.clip(np.nan_to_num(abundance, nan=0.0, posinf=0.0, neginf=0.0), 0.0, 1.0)
    uncertainty = np.clip(np.nan_to_num(uncertainty, nan=1.0, posinf=1.0, neginf=1.0), 0.0, 1.0)

    abundance[unknown] = 0.0
    uncertainty[unknown] = 1.0

    # 对原始无效/质量排除位置，所有类别统一设为完全未知。
    invalid = ~valid
    abundance[:, invalid] = 0.0
    uncertainty[:, invalid] = 1.0

    tensor = np.concatenate([abundance, uncertainty], axis=0).astype(np.float32, copy=False)

    # 最终安全检查：不允许 NaN/Inf 流入深度学习数据。
    if not np.isfinite(tensor).all():
        raise RuntimeError("最终训练张量仍包含 NaN/Inf。")
    if tensor.min() < -1e-6 or tensor.max() > 1 + 1e-6:
        raise RuntimeError("最终训练张量超出 [0,1]。")

    if OUTPUT_LAYOUT.upper() == "CHW":
        return tensor
    if OUTPUT_LAYOUT.upper() == "HWC":
        return np.moveaxis(tensor, 0, -1)
    raise ValueError("OUTPUT_LAYOUT 只能是 'CHW' 或 'HWC'。")


def process_one(input_file: Path, index: int, total: int) -> bool:
    """处理一张影像，最终只留下一个 .npy。"""
    output_file = OUTPUT_DIR / f"{input_file.stem}.npy"

    if SKIP_EXISTING and output_file.is_file():
        print(f"[{index}/{total}] 已存在，跳过：{output_file.name}")
        return True

    print(f"[{index}/{total}] 解混：{input_file.name}")

    # 临时目录位于输出目录下，便于 Windows 环境中管理磁盘位置。
    # 正常或异常结束都会尝试删除。
    temp_parent = OUTPUT_DIR / ".unmixing_tmp"
    temp_parent.mkdir(parents=True, exist_ok=True)
    temp_dir = Path(tempfile.mkdtemp(prefix=input_file.stem + "__", dir=temp_parent))

    try:
        cfg = make_config(input_file, temp_dir)

        if VERBOSE_INTERNAL:
            data = pipeline.main(cfg)
        else:
            # 不把五个提取模块的大量中间打印保留下来。
            with open(os.devnull, "w", encoding="utf-8") as devnull:
                with redirect_stdout(devnull), redirect_stderr(devnull):
                    data = pipeline.main(cfg)

        tensor = build_training_tensor(data)

        # 原子式保存：先写临时 .npy，再替换正式文件。
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        tmp_output = output_file.with_name(output_file.name + ".tmp.npy")
        np.save(tmp_output, tensor, allow_pickle=False)
        os.replace(tmp_output, output_file)

        print(f"           -> {output_file.name}  shape={tuple(tensor.shape)}")

        del tensor
        del data
        gc.collect()
        return True

    except Exception as exc:
        print(f"           !! 失败：{type(exc).__name__}: {exc}")
        if VERBOSE_INTERNAL:
            traceback.print_exc()
        return False

    finally:
        # 删除端元提取和解混过程产生的全部中间结果。
        shutil.rmtree(temp_dir, ignore_errors=True)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    files = discover_tiffs(INPUT_DIR)
    if not files:
        print(f"没有找到 TIFF：{INPUT_DIR}")
        return

    print(f"输入目录：{INPUT_DIR}")
    print(f"输出目录：{OUTPUT_DIR}")
    print(f"影像数量：{len(files)}")
    print(f"输出布局：{OUTPUT_LAYOUT}；每景只保留一个10通道 .npy")
    print("通道 0~4：vegetation, water, bare, snow, building 丰度")
    print("通道 5~9：对应五类不确定性")
    print()

    success = 0
    failed = []

    for i, input_file in enumerate(files, start=1):
        ok = process_one(input_file, i, len(files))
        if ok:
            success += 1
        else:
            failed.append(input_file.name)

    # 正常结束后，如果临时父目录已经空了，就一并删除。
    temp_parent = OUTPUT_DIR / ".unmixing_tmp"
    if temp_parent.is_dir():
        try:
            temp_parent.rmdir()
        except OSError:
            pass

    print("\n批处理完成")
    print(f"成功：{success}/{len(files)}")
    print(f"失败：{len(failed)}")
    if failed:
        print("失败文件：")
        for name in failed:
            print("  -", name)


if __name__ == "__main__":
    main()
