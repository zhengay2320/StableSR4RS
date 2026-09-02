# -*- coding: utf-8 -*-
r"""
WorldStrat 验证集批量光谱解混
================================

使用方法
--------
1. 将本文件放到以下文件所在的同一目录：
   - vegetation_candidates_all_bands.py
   - water_candidates_all_bands.py
   - bare_candidates_all_bands.py
   - snow_candidates_all_bands.py
   - building_candidates_all_bands.py
   - unmixing_core.py
   - spectral_unmixing_pipeline.py

2. 确认顶部“用户配置区”中的路径和反射率设置。
3. 运行：
       python batch_unmixing.py

输出结构
--------
E:\开源数据集\word_star\new_star\val\lr_unmixing\
    batch_manifest.csv
    batch_status.json
    batch_logs\
        Landcover-xxxxxx.log
    Landcover-xxxxxx\
        run_status.json
        01_class_status.csv
        01_endmember_library.csv
        results\
            abundance_conditional.tif
            abundance_screened.tif
            rmse.tif
            quality_flags.tif
            unmixing_result.npz
            ...
        figures\
        report.html
    ...

重要语义
--------
- 类别提取失败 != 该类别不存在。
- 条件性丰度 abundance_conditional 是在“当前可用端元库”下的结果。
- 缺少可靠端元的类别保留 NaN，不伪装成 0。
- screened 结果要求更严格；自动批处理中候选没有经过人工来源核查时，
  screened 可能大量为 NaN，这是设计行为，不是批处理失败。
"""

from __future__ import annotations

from contextlib import redirect_stdout, redirect_stderr
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
import csv
import gc
import io
import json
import os
import shutil
import sys
import time
import traceback

import numpy as np

from spectral_unmixing_pipeline import Config, ClassSpec, main as run_unmixing
from unmixing_core import SolverConfig


# ============================================================================
# 1. 用户配置区
# ============================================================================

HERE = Path(__file__).resolve().parent

# 输入：WorldStrat val/lr 下的全部 TIFF。
INPUT_DIR = Path(r"E:\开源数据集\word_star\new_star\val\lr")

# 输出：每张影像一个独立子目录。
OUTPUT_DIR = Path(r"E:\开源数据集\word_star\new_star\val\lr_unmixing")

# False：只扫描 INPUT_DIR 当前一级。
# True ：递归扫描所有子目录。
RECURSIVE = False

# --------------------------------------------------------------------------
# 反射率设置
# --------------------------------------------------------------------------
# 你前面的标准 WorldStrat 文件已经按 float 反射率直接读取，因此这里默认：
# rho = DN * 1.0 + 0.0
# 如果换了另一批预处理方式不同的数据，必须同步修改。
DATA_CONFIRMED = True
SCALE = 1.0
OFFSET = 0.0

# --------------------------------------------------------------------------
# 批量运行时的输出与速度设置
# --------------------------------------------------------------------------
# 可视化图和 report.html 对排查错误很有用，因此默认保留。
MAKE_PLOTS = True
MAKE_HTML = True

# pixel_trace.csv 每个像元一行；数据很多时会明显增加磁盘占用。
SAVE_PIXEL_CSV = False

# 单景额外保存多少个像元级光谱检查图。批处理建议 0 或 1。
SAVE_EXAMPLE_PIXELS = 0

# 是否运行“每类一条固定代表端元”的 FCLS 对照。
RUN_BASELINE = True

# 是否额外保存最方便后续算法调用的 NumPy 文件：
#   abundance_conditional.npy : [5, H, W]
#   rmse.npy                  : [H, W]
#   class_order.json          : 五个通道顺序
SAVE_NUMPY_ARRAYS = True

# --------------------------------------------------------------------------
# 断点续跑策略
# --------------------------------------------------------------------------
# True：已经正常完成的场景直接跳过。
SKIP_FINISHED = True

# 若某个场景目录已经存在，但状态是 failed/running 等不完整状态：
# True  -> 先重命名旧目录保留现场，再重新处理；
# False -> 跳过该场景。
RETRY_INCOMPLETE = True

# 这些状态认为“本次运行已经结束”。
# no_available_endmembers 虽然没有可用端元，但它仍是一个完成的诊断结果。
FINISHED_STATES = {"completed", "no_available_endmembers", "inspection_only"}

# --------------------------------------------------------------------------
# 单景解混模型设置
# --------------------------------------------------------------------------
SOLVER = SolverConfig(
    max_endmembers=3,          # 首版每像元尝试 1~3 个端元。
    max_per_class=1,           # 每个类别在单个模型中最多用 1 条候选光谱。
    max_rmse=0.03,             # 原始反射率单位，后续应结合验证结果标定。
    selection_delta_mse=1e-5,  # 接近最优时优先选择更简单的模型。
    ambiguity_delta_mse=1e-5,
    block_pixels=2048,
)

# --------------------------------------------------------------------------
# 五类提取模块的参数覆盖
# --------------------------------------------------------------------------
# 默认为空：继续使用五个原脚本各自 CFG / Config 中的设置。
# 如果要做统一的批处理实验，可以只在这里集中修改。
EXTRACTION_OVERRIDES = {
    "vegetation": {},
    "water": {},
    "bare": {},
    "snow": {},
    "building": {},

    # 例如，仅进行“关闭建筑腐蚀”的诊断实验时可以写：
    # "building": {"erosion_radius": 0},
}

# --------------------------------------------------------------------------
# 类别状态
# --------------------------------------------------------------------------
# 批处理时不能因为某个提取器没找到端元，就自动宣称该类“不存在”。
# 因此默认全部 unknown。
# 如果你有外部可靠规则能对整个批次确认某类不存在，可以在这里修改，
# 但 absent 必须同时填写 state_reason。
CLASS_STATES = {
    "vegetation": ("unknown", ""),
    "water": ("unknown", ""),
    "bare": ("unknown", ""),
    "snow": ("unknown", ""),
    "building": ("unknown", ""),
}


# ============================================================================
# 2. 辅助函数
# ============================================================================

CLASSES = ("vegetation", "water", "bare", "snow", "building")
CLASS_ORDER_CN = ["绿色植被", "水体", "裸土/砂地", "雪/冰候选", "建筑/人工表面候选"]


class Tee(io.TextIOBase):
    """同时把 stdout/stderr 输出到终端和单景日志文件。"""

    def __init__(self, *streams):
        self.streams = streams

    def write(self, text):
        for stream in self.streams:
            stream.write(text)
            stream.flush()
        return len(text)

    def flush(self):
        for stream in self.streams:
            stream.flush()



def _jsonable(value):
    """把 Path / NumPy 类型转换成 JSON 可写类型。"""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(v) for v in value]
    return value



def save_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_jsonable(obj), ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )



def load_json(path: Path, default=None):
    if not path.is_file():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except Exception:
        return default



def discover_tiffs(input_dir: Path) -> list[Path]:
    """扫描 tif/tiff，并进行确定性排序。"""
    if not input_dir.is_dir():
        raise FileNotFoundError(f"找不到输入文件夹：{input_dir}")

    iterator = input_dir.rglob("*") if RECURSIVE else input_dir.iterdir()
    files = [
        p for p in iterator
        if p.is_file() and p.suffix.lower() in {".tif", ".tiff"}
    ]
    files.sort(key=lambda p: str(p.relative_to(input_dir)).lower())

    # 当前输出使用 stem 作为场景目录名。防止 a.tif 与 a.tiff 意外冲突。
    seen = {}
    for p in files:
        key = p.stem.lower()
        if key in seen:
            raise ValueError(
                "发现同名场景，无法安全使用文件名 stem 作为输出目录：\n"
                f"  {seen[key]}\n  {p}\n"
                "请先重命名，或自行修改 make_scene_id()。"
            )
        seen[key] = p
    return files



def make_scene_id(path: Path) -> str:
    """单景输出目录名。WorldStrat 文件名本身可直接使用 stem。"""
    return path.stem



def archive_existing_directory(path: Path) -> Path:
    """保留失败/中断现场，再让同名正式目录重新运行。"""
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = path.with_name(path.name + f"__previous_{stamp}")
    target = base
    k = 1
    while target.exists():
        target = path.with_name(base.name + f"_{k}")
        k += 1
    path.rename(target)
    return target



def make_scene_config(input_file: Path, scene_id: str) -> Config:
    """为每张影像创建一份完全独立的解混配置。"""
    class_specs = {}
    for name in CLASSES:
        state, reason = CLASS_STATES[name]
        class_specs[name] = ClassSpec(
            source="AUTO",            # 没有指定旧结果目录时，调用对应提取器。
            scene_state=state,
            state_reason=reason,
            reviewed=False,           # 批处理中不自动冒充“人工核查通过”。
            max_candidates=3,
            extraction_overrides=dict(EXTRACTION_OVERRIDES.get(name, {})),
        )

    cfg = Config(
        input_path=input_file,
        script_dir=HERE,
        output_root=OUTPUT_DIR,
        run_name=scene_id,
        product_level="AUTO",
        band_map=None,
        data_confirmed=DATA_CONFIRMED,
        scale=SCALE,
        offset=OFFSET,

        # 如果你的 val 集有统一、同网格的质量层，可以在这里扩展成按文件名查找。
        clear_mask_path=None,
        scl_path=None,

        classes=class_specs,
        solver=SOLVER,

        # 自动处理保留未人工核查候选，供 conditional 丰度使用；
        # screened 输出依然会根据质量标记进行更严格筛查。
        allow_unreviewed_candidates=True,

        # 某一类别“候选不足”由主流程安全处理；真正的软件错误仍使该景失败。
        continue_on_extractor_error=False,

        show_figures=False,
        make_plots=MAKE_PLOTS,
        make_html=MAKE_HTML,
        run_baseline=RUN_BASELINE,
        save_pixel_csv=SAVE_PIXEL_CSV,
        save_example_pixels=SAVE_EXAMPLE_PIXELS,
        blas_threads=1,
    )
    return cfg



def save_numpy_shortcuts(scene_dir: Path) -> None:
    """
    从 pipeline 已保存的 NPZ 中再输出三个简单文件，便于其他算法直接读取。

    abundance_conditional.npy 的固定通道顺序：
      0 vegetation
      1 water
      2 bare
      3 snow
      4 building
    """
    if not SAVE_NUMPY_ARRAYS:
        return

    npz_path = scene_dir / "results" / "unmixing_result.npz"
    if not npz_path.is_file():
        return

    with np.load(npz_path, allow_pickle=False) as d:
        abundance = np.asarray(d["conditional_abundance"], dtype=np.float32)
        rmse = np.asarray(d["rmse"], dtype=np.float32)
        classes = [str(x) for x in d["classes"].tolist()]

    if abundance.ndim != 3 or abundance.shape[0] != 5:
        raise ValueError(
            f"{npz_path} 中 conditional_abundance 形状异常：{abundance.shape}"
        )

    np.save(scene_dir / "results" / "abundance_conditional.npy", abundance)
    np.save(scene_dir / "results" / "rmse.npy", rmse)
    save_json(
        scene_dir / "results" / "class_order.json",
        {
            "shape": list(abundance.shape),
            "class_order": classes,
            "class_order_cn": CLASS_ORDER_CN,
            "note": "NaN 表示当前类别/像元没有可靠条件性估计，不等于丰度为0。",
        },
    )



def read_scene_result_summary(scene_dir: Path) -> dict:
    """从单景输出中读取适合批处理汇总的少量指标。"""
    status = load_json(scene_dir / "run_status.json", {}) or {}
    coverage = load_json(scene_dir / "03_coverage_summary.json", {}) or {}

    return {
        "state": status.get("state", "unknown"),
        "fit_ok_pixels": status.get("fit_ok_pixels", coverage.get("fit_ok_pixels")),
        "screened_ok_pixels": status.get(
            "screened_ok_pixels", coverage.get("screened_ok_pixels")
        ),
        "unresolved_classes": "|".join(status.get(
            "unresolved_classes", coverage.get("unresolved_classes", [])
        ) or []),
        "unmixing_domain_pixels": coverage.get("unmixing_domain_pixels"),
    }



def write_manifest(path: Path, rows: list[dict]) -> None:
    """每处理一景就重写一次，异常中断时仍保留已经完成的进度。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "index",
        "total",
        "scene_id",
        "input_file",
        "status",
        "action",
        "output_dir",
        "elapsed_seconds",
        "fit_ok_pixels",
        "screened_ok_pixels",
        "unmixing_domain_pixels",
        "unresolved_classes",
        "error",
    ]
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


# ============================================================================
# 3. 单景处理
# ============================================================================


def process_one(input_file: Path, index: int, total: int) -> dict:
    scene_id = make_scene_id(input_file)
    scene_dir = OUTPUT_DIR / scene_id
    log_dir = OUTPUT_DIR / "batch_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{scene_id}.log"

    row = {
        "index": index,
        "total": total,
        "scene_id": scene_id,
        "input_file": str(input_file),
        "output_dir": str(scene_dir),
        "status": "pending",
        "action": "",
        "elapsed_seconds": None,
        "fit_ok_pixels": None,
        "screened_ok_pixels": None,
        "unmixing_domain_pixels": None,
        "unresolved_classes": "",
        "error": "",
    }

    # ----------------------------------------------------------------------
    # 断点续跑判断
    # ----------------------------------------------------------------------
    if scene_dir.exists():
        old_status = load_json(scene_dir / "run_status.json", {}) or {}
        old_state = old_status.get("state", "unknown")

        if old_state in FINISHED_STATES and SKIP_FINISHED:
            row.update(read_scene_result_summary(scene_dir))
            row["status"] = old_state
            row["action"] = "skip_finished"
            print(f"[{index}/{total}] 跳过已完成：{scene_id} ({old_state})")
            return row

        if not RETRY_INCOMPLETE:
            row["status"] = old_state
            row["action"] = "skip_existing_incomplete"
            print(f"[{index}/{total}] 跳过已有未完成目录：{scene_id} ({old_state})")
            return row

        archived = archive_existing_directory(scene_dir)
        row["action"] = f"retry_after_archive:{archived.name}"
    else:
        row["action"] = "process"

    # ----------------------------------------------------------------------
    # 正式执行
    # ----------------------------------------------------------------------
    print("\n" + "=" * 88)
    print(f"[{index}/{total}] 开始：{input_file.name}")
    print(f"输出：{scene_dir}")
    print("=" * 88)

    start = time.perf_counter()
    cfg = make_scene_config(input_file, scene_id)

    # 每次覆盖单景日志；旧失败现场如果存在已经随整个目录归档。
    with log_path.open("w", encoding="utf-8") as log_file:
        tee_out = Tee(sys.stdout, log_file)
        tee_err = Tee(sys.stderr, log_file)

        try:
            with redirect_stdout(tee_out), redirect_stderr(tee_err):
                data = run_unmixing(cfg)

            # main() 正常结束后，读取磁盘上的最终状态；不要仅凭 Python 返回值猜。
            summary = read_scene_result_summary(scene_dir)
            row.update(summary)
            row["status"] = summary["state"]

            # 额外生成方便后续 NumPy 调用的五通道结果与误差图。
            save_numpy_shortcuts(scene_dir)

            # 及时释放单景大数组，避免长批次内存逐渐上涨。
            del data
            gc.collect()

        except KeyboardInterrupt:
            row["status"] = "interrupted"
            row["error"] = "KeyboardInterrupt"
            row["elapsed_seconds"] = round(time.perf_counter() - start, 3)
            print(f"\n用户中断：{scene_id}")
            raise

        except Exception as exc:
            # 单景失败写入日志，但不让整个验证集停止。
            row["status"] = "failed"
            row["error"] = repr(exc)
            print("\n[单景失败]", repr(exc))
            traceback.print_exc(file=tee_err)
            gc.collect()

    row["elapsed_seconds"] = round(time.perf_counter() - start, 3)

    print(
        f"[{index}/{total}] 结束：{scene_id}；状态={row['status']}；"
        f"耗时={row['elapsed_seconds']} s"
    )
    return row


# ============================================================================
# 4. 批处理主入口
# ============================================================================


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    files = discover_tiffs(INPUT_DIR)
    if not files:
        raise FileNotFoundError(f"{INPUT_DIR} 中没有找到 .tif/.tiff 文件。")

    print(f"输入目录：{INPUT_DIR}")
    print(f"输出目录：{OUTPUT_DIR}")
    print(f"共发现 {len(files)} 张 TIFF。")
    print("处理顺序：顺序执行；单景失败不会终止整个批次。")

    # 保存本次批处理全局配置，便于复现实验。
    save_json(
        OUTPUT_DIR / "batch_config.json",
        {
            "input_dir": INPUT_DIR,
            "output_dir": OUTPUT_DIR,
            "recursive": RECURSIVE,
            "data_confirmed": DATA_CONFIRMED,
            "scale": SCALE,
            "offset": OFFSET,
            "make_plots": MAKE_PLOTS,
            "make_html": MAKE_HTML,
            "save_pixel_csv": SAVE_PIXEL_CSV,
            "save_example_pixels": SAVE_EXAMPLE_PIXELS,
            "run_baseline": RUN_BASELINE,
            "save_numpy_arrays": SAVE_NUMPY_ARRAYS,
            "skip_finished": SKIP_FINISHED,
            "retry_incomplete": RETRY_INCOMPLETE,
            "solver": asdict(SOLVER),
            "extraction_overrides": EXTRACTION_OVERRIDES,
            "class_states": CLASS_STATES,
            "script_dir": HERE,
            "started_at": datetime.now().isoformat(timespec="seconds"),
        },
    )

    rows: list[dict] = []
    manifest_path = OUTPUT_DIR / "batch_manifest.csv"
    batch_start = time.perf_counter()

    try:
        for i, input_file in enumerate(files, start=1):
            row = process_one(input_file, i, len(files))
            rows.append(row)
            write_manifest(manifest_path, rows)

            # 运行状态每景刷新一次，异常停机后也容易看到进度。
            save_json(
                OUTPUT_DIR / "batch_status.json",
                {
                    "state": "running",
                    "processed": len(rows),
                    "total": len(files),
                    "last_scene": row["scene_id"],
                    "status_counts": {
                        state: sum(r["status"] == state for r in rows)
                        for state in sorted({r["status"] for r in rows})
                    },
                    "updated_at": datetime.now().isoformat(timespec="seconds"),
                },
            )

    except KeyboardInterrupt:
        write_manifest(manifest_path, rows)
        save_json(
            OUTPUT_DIR / "batch_status.json",
            {
                "state": "interrupted",
                "processed": len(rows),
                "total": len(files),
                "elapsed_seconds": round(time.perf_counter() - batch_start, 3),
                "updated_at": datetime.now().isoformat(timespec="seconds"),
            },
        )
        raise

    total_seconds = round(time.perf_counter() - batch_start, 3)
    status_counts = {
        state: sum(r["status"] == state for r in rows)
        for state in sorted({r["status"] for r in rows})
    }

    save_json(
        OUTPUT_DIR / "batch_status.json",
        {
            "state": "completed",
            "processed": len(rows),
            "total": len(files),
            "status_counts": status_counts,
            "elapsed_seconds": total_seconds,
            "finished_at": datetime.now().isoformat(timespec="seconds"),
            "manifest": manifest_path,
        },
    )

    print("\n" + "=" * 88)
    print("批处理完成")
    print(f"总影像数：{len(files)}")
    print(f"状态统计：{status_counts}")
    print(f"总耗时：{total_seconds} s")
    print(f"汇总表：{manifest_path}")
    print("=" * 88)


if __name__ == "__main__":
    main()
