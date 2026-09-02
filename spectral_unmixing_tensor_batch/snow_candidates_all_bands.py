# -*- coding: utf-8 -*-
"""WorldStrat单景雪地候选端元：NDSI/亮度/质量筛选、空间均质性、全波段聚类。

本文件独立运行，不依赖植被/水体脚本。适用于小幅裁剪影像，不是超大整景分块实现。
目标是暴露、较明亮、局部均质的雪覆盖表面候选，不是完整雪域、雪深或雪丰度。
不能仅凭本方法可靠区分积雪与所有裸冰，也不能仅凭指数排除全部云/冰云。
NDSI(B3,B11)与此前水体MNDWI(B3,B11)公式相同，不能当成两个独立证据。
所有12/13个光谱波段参与聚类和候选输出，质量层和指数不进入聚类距离。
默认参数为研究起点；程序没有访问用户本地E盘，也没有对真实输入做精度标定。

参考：
USGS NDSI: https://www.usgs.gov/landsat-missions/normalized-difference-snow-index
Sentinel-2 NDSI: https://custom-scripts.sentinel-hub.com/custom-scripts/sentinel-2/ndsi/
ESA/Copernicus SCL、云雪检测与B10高海拔限制：
https://sentiwiki.copernicus.eu/web/s2-processing
WorldStrat科学波段排列：
https://raw.githubusercontent.com/worldstrat/worldstrat/main/dataset_generation/SentinelDownloader.py
本脚本的阈值、局部均质性和候选组合是实验方案，不是Sen2Cor的完整复现。
"""
from dataclasses import dataclass
from pathlib import Path
import argparse
import json
import re
import warnings
import platform
from importlib.metadata import version, PackageNotFoundError

import numpy as np
import pandas as pd
import rasterio
from scipy import ndimage as ndi
from skimage.filters import threshold_otsu
from sklearn.cluster import kmeans_plusplus
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

# ==================== 0. 配置 ====================
# 科学波段名与TIFF通道编号不同。Rasterio通道从1开始。
# 标准顺序来自WorldStrat官方SentinelDownloader.set_bands()：
# https://raw.githubusercontent.com/worldstrat/worldstrat/main/dataset_generation/SentinelDownloader.py
WORLDSTRAT_L2A_BANDS = (
    "B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B9", "B11", "B12"
)
WORLDSTRAT_L1C_BANDS = (
    "B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B9", "B10", "B11", "B12"
)
WORLDSTRAT_L2A_MAP = {b: i+1 for i, b in enumerate(WORLDSTRAT_L2A_BANDS)}
WORLDSTRAT_L1C_MAP = {b: i+1 for i, b in enumerate(WORLDSTRAT_L1C_BANDS)}


@dataclass
class Config:
    # 保留此前输入路径作为可运行的配置模板；不意味着该图中已确认存在积雪。
    input_path: Path = Path(r"E:\开源数据集\word_star\new_star\train\lr\Landcover-770359.tiff")
    output_dir: Path = Path(r"D:\program_myself\data_pro\SpectralUnmixing\snow_results_all_bands\Landcover-770359")
    product_level: str = "AUTO"          # 仅在已知标准完整格式时按12/13通道解析。
    band_map: dict | None = None          # 手填时必须一一覆盖全部科学波段。
    feature_bands: str | tuple = "ALL"   # 所有光谱波段用于聚类，不允许只用四波段。
    purity_bands: str | tuple = ("B2", "B3", "B4", "B8")

    # 反射率 = 原值*scale+offset。不能猜测所有文件都应除以10000。
    data_confirmed: bool = True
    scale: float | dict | None = None
    offset: float | dict | None = None
    # 可选0/1质量掩膜。1表示允许分析，应保留有效雪像元；不能直接传SCL编码。
    clear_mask_path: Path | None = None
    # 可选L2A的SCL分类图；必须与输入同网格、同景同日期，分类层只能最近邻对齐。
    # 标准多光谱TIFF不一定自带SCL；本程序不会凭空生成该文件。
    scl_path: Path | None = None
    # 0无数据、1饱和/坏像元、2暗像元/投影阴影、3云影、8/9云、10卷云。
    # 保留11(雪/冰)，也不要求全部候选都必须被SCL判为11，避免循环验证。
    scl_excluded_classes: tuple = (0, 1, 2, 3, 8, 9, 10)
    show_figures: bool = False

    # -------- 雪地识别参数：均为待真实数据验证的实验设置 --------
    min_index_sum: float = 0.02          # NDSI分母下限，反射率单位。
    min_ndvi_sum: float = 0.02
    ndsi_base_min: float = 0.40
    ndsi_base_max: float = 0.50          # 初筛阈值=clip(Otsu,0.40,0.50)。
    min_green: float = 0.20              # 区分暗水与较明亮雪面的辅助条件。
    min_nir: float = 0.15                # 不沿用水体版的近红外上限。
    use_ndvi_gate: bool = True
    max_ndvi: float = 0.30               # 只研究暴露雪面，排除明显绿色植被信号。

    # -------- 空间内部支持与较强NDSI --------
    min_patch_pixels: int = 25           # 8邻接连通域最少像元数。
    erosion_radius: int = 2              # 5×5掩膜腐蚀；不平滑原始反射率。
    ndsi_high_min: float = 0.50
    ndsi_high_max: float = 0.70          # 避免全雪场景中过度追逐趋近1的NDSI。
    ndsi_high_quantile: float = 0.40
    window_size: int = 3                 # 局部统计3×3，区别于上述5×5腐蚀。
    brightness_floor: float = 0.10       # H=U/max(M,0.10)；不是仪器噪声估计。
    homogeneous_abs_max: float = 0.03    # 比暗水体使用更宽的绝对反射率波动门槛。
    homogeneous_quantile: float = 0.50
    homogeneous_min: float = 0.03
    homogeneous_max: float = 0.10
    min_seed_pixels: int = 30

    # -------- 全波段K-means，与植被/水体使用一致的框架 --------
    max_clusters: int = 5
    n_init: int = 10
    max_iter: int = 200
    random_seed: int = 42
    feature_scale_floor: float = 0.02     # 逐波段IQR的下限，不删除低变化波段。
    min_cluster_pixels: int = 10
    min_cluster_fraction: float = 0.02
    min_cluster_component: int = 3
    silhouette_sample_size: int = 1200
    silhouette_min: float = 0.25
    silhouette_tolerance: float = 0.02
    representative_fraction: float = 0.50
    trace_sample_pixels: int = 6


CFG = Config()


class CandidateUnavailable(RuntimeError):
    """候选证据不足，不等于确认影像无雪；main据此输出insufficient_candidates状态。"""


def validate_config(cfg: Config) -> None:
    """检查输入参数，不偷偷修改用户设定的阈值或波段。"""
    cfg.input_path, cfg.output_dir = Path(cfg.input_path), Path(cfg.output_dir)
    for name in ("clear_mask_path", "scl_path"):
        if getattr(cfg, name) is not None:
            setattr(cfg, name, Path(getattr(cfg, name)))
    positive = ("min_index_sum", "min_ndvi_sum", "brightness_floor", "homogeneous_abs_max",
                "feature_scale_floor", "min_green", "min_nir")
    for name in positive:
        if not np.isfinite(getattr(cfg, name)) or getattr(cfg, name) <= 0:
            raise ValueError(f"{name}必须是正有限数。")
    for name in ("ndsi_high_quantile", "homogeneous_quantile", "representative_fraction"):
        if not 0 < getattr(cfg, name) <= 1:
            raise ValueError(f"{name}必须位于(0,1]。")
    for name in ("min_patch_pixels", "erosion_radius", "min_seed_pixels", "max_clusters", "n_init",
                 "max_iter", "min_cluster_pixels", "min_cluster_component", "trace_sample_pixels",
                 "silhouette_sample_size"):
        value = getattr(cfg, name)
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 1:
            raise ValueError(f"{name}必须为正整数。")
    if not isinstance(cfg.window_size, int) or cfg.window_size < 3 or cfg.window_size % 2 != 1:
        raise ValueError("window_size须为不小于3的奇数。")
    if cfg.silhouette_sample_size < 3 or not 0 < cfg.min_cluster_fraction <= 1:
        raise ValueError("轮廓系数样本应不少于3，簇比例须在(0,1]。")
    if not np.isfinite(cfg.silhouette_tolerance) or cfg.silhouette_tolerance < 0:
        raise ValueError("silhouette_tolerance应为非负有限数。")
    for name in ("ndsi_base_min", "ndsi_base_max", "max_ndvi", "ndsi_high_min", "ndsi_high_max", "silhouette_min"):
        if not -1 <= getattr(cfg, name) <= 1:
            raise ValueError(f"{name}应在[-1,1]。")
    if not (cfg.ndsi_base_min <= cfg.ndsi_base_max <= cfg.ndsi_high_min <= cfg.ndsi_high_max):
        raise ValueError("NDSI边界应满足base_min<=base_max<=high_min<=high_max。")
    if not 0 < cfg.homogeneous_min <= cfg.homogeneous_max:
        raise ValueError("均质性边界应满足0<min<=max。")
    if any(isinstance(x, bool) or not isinstance(x, (int, np.integer)) or x not in range(12)
           for x in cfg.scl_excluded_classes):
        raise ValueError("scl_excluded_classes只能包含0..11的整数。")
    if 11 in cfg.scl_excluded_classes:
        raise ValueError("雪地提取不能把SCL=11(雪/冰)整类作为无效数据排除。")


def save_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


# ==================== 1. 输出工具、完整波段映射与反射率读取 ====================
def save_table(table: pd.DataFrame, path: Path, print_table: bool = False) -> None:
    """CSV采用UTF-8 BOM，方便在Windows中打开中文列名。"""
    table.to_csv(path, index=False, encoding="utf-8-sig")
    if print_table:
        print(table.to_string(index=False))



def finish_plot(cfg: Config, filename: str) -> None:
    """一张图一个文件，不拼成拥挤的小图；只做可视化，不修改科学数据。"""
    plt.tight_layout()
    plt.savefig(cfg.output_dir / filename, dpi=160, bbox_inches="tight")
    if cfg.show_figures:
        plt.show()
    plt.close()



def save_map(data: dict, name: str, array: np.ndarray, title: str) -> None:
    """同时保存数值TIFF和预览PNG。掩膜0/1；聚类0=非种子，1..K=簇编号。"""
    cfg = data["cfg"]
    arr = np.asarray(array, dtype=np.float64).copy()
    arr[~data.get("numeric_valid", data["valid"])] = np.nan
    profile = dict(driver="GTiff", height=arr.shape[0], width=arr.shape[1],
                   count=1, dtype="float32", nodata=-9999.0, compress="deflate",
                   transform=data["transform"], crs=data["crs"])
    with rasterio.open(cfg.output_dir / f"{name}.tif", "w", **profile) as dst:
        dst.write(np.where(np.isfinite(arr), arr, -9999).astype("float32"), 1)
        dst.set_band_description(1, name)
    plt.figure(figsize=(7, 6))
    plt.imshow(np.ma.masked_invalid(arr))
    plt.colorbar()
    plt.title(title)
    plt.xlabel("Column (0-based)")
    plt.ylabel("Row (0-based)")
    finish_plot(cfg, f"{name}.png")



def record_mask(data: dict, name: str, mask: np.ndarray) -> None:
    """每完成一道筛选就记录保留像元数；即使后续失败，中间文件也已经保存。"""
    data["masks"][name] = mask
    count = int(mask.sum())
    data["counts"].append({"stage": name, "pixels": count})
    save_table(pd.DataFrame(data["counts"]), data["cfg"].output_dir / "stage_counts.csv")
    print(f"{name}: {count} 个像元")
    save_map(data, name, mask, name.replace("_", " "))



def save_pixel_trace(data: dict) -> None:
    """逐有效像元记录完整光谱、指数、逐项门槛、空间筛选、簇号及核心样本标记。"""
    rows, cols = np.where(data.get("numeric_valid", data["valid"]))
    table = pd.DataFrame({"row": rows, "col": cols})
    for i, name in enumerate(data["bands"]):
        table[name] = data["cube"][i, rows, cols]
    for name in ("valid", "quality_mask", "scl", "ndsi", "ndvi", "index_valid", "ndvi_valid", "heterogeneity",
                 "local_abs_rms", "local_mean_rms", "boundary_distance_pixels", "cluster_map", "used_for_mean"):
        if name in data:
            table[name] = data[name][rows, cols]
    for name, gate in data.get("gates", {}).items():
        table["gate_" + name] = gate[rows, cols].astype(np.uint8)
    for name, mask in data["masks"].items():
        table[name] = mask[rows, cols].astype(np.uint8)
    save_table(table, data["cfg"].output_dir / "pixel_trace.csv")


def save_thresholds(data: dict) -> None:
    save_json(data["cfg"].output_dir / "thresholds.json", data["thresholds"])


def require_pixels(data: dict, mask: np.ndarray, stage: str) -> None:
    """候选不足时停止，并保留像元追踪；不能把停止解释为确定无雪。"""
    count, required = int(mask.sum()), data["cfg"].min_seed_pixels
    if count < required:
        save_pixel_trace(data)
        raise CandidateUnavailable(
            f"{stage}只有{count}个像元，少于{required}。请检查门槛统计、质量掩膜与中间图。"
            "本程序不自动放宽阈值凑候选，也不能据此断言整图无雪。")


def inspect_tiff(cfg: Config) -> dict:
    if not cfg.input_path.is_file():
        raise FileNotFoundError(f"找不到文件：{cfg.input_path}\n请在存有该文件的电脑上运行。")
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    with rasterio.open(cfg.input_path) as src:
        meta = {"path": str(cfg.input_path), "count": src.count,
                "height": src.height, "width": src.width,
                "crs": str(src.crs), "transform": list(src.transform),
                "resolution": list(src.res), "descriptions": list(src.descriptions),
                "scales": list(src.scales), "offsets": list(src.offsets),
                "nodata": src.nodata, "tags": src.tags(),
                "band_tags": {str(i): src.tags(i) for i in src.indexes}}
        for i in src.indexes:
            # 必须先转浮点，再做运算；不在uint16上做减法。
            band = src.read(i, masked=True).astype("float64").filled(np.nan)
            values = band[np.isfinite(band)]
            quantiles = np.quantile(values, [0, .02, .5, .98, 1]) if values.size else [np.nan]*5
            rows.append(dict(channel=i, description=src.descriptions[i-1],
                             dtype=src.dtypes[i-1], scale=src.scales[i-1],
                             offset=src.offsets[i-1], valid_pixels=values.size,
                             minimum=quantiles[0], p02=quantiles[1], median=quantiles[2],
                             p98=quantiles[3], maximum=quantiles[4]))
    (cfg.output_dir / "00_metadata.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(f"\n影像：{meta['count']}个通道，{meta['height']}行×{meta['width']}列")
    print(f"CRS={meta['crs']}，TIFF已有像元尺寸={meta['resolution']}")
    save_table(pd.DataFrame(rows), cfg.output_dir / "00_band_statistics.csv", True)
    resolved = resolve_worldstrat_bands(meta["count"], tuple(meta["descriptions"]), cfg)
    meta["resolved_product_level"] = resolved["level"]
    meta["resolved_band_map"] = resolved["mapping"]
    table = pd.DataFrame({
        "channel": range(1, meta["count"] + 1),
        "band": resolved["bands"],
        "used_for_ndsi": [b in ("B3", "B11") for b in resolved["bands"]],
        "used_for_brightness": [b in ("B3", "B8") for b in resolved["bands"]],
        "used_for_ndvi": [b in ("B4", "B8") for b in resolved["bands"]],
        "used_for_purity": [b in resolved["purity_bands"] for b in resolved["bands"]],
        "used_for_clustering": True,
    })
    print(f"\n按已确认的标准WorldStrat格式解析为{resolved['level']}：")
    save_table(table, cfg.output_dir / "00_resolved_band_map.csv", True)
    (cfg.output_dir / "00_metadata.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    return meta



def normalize_band_name(name: str) -> str:
    """统一B01/B1、B08/B8的写法，但严格区分B8与B8A。"""
    name = str(name).strip().upper()
    if name == "B8A":
        return name
    if re.fullmatch(r"B(?:0?[1-9]|1[0-2])", name):
        return f"B{int(name[1:])}"
    raise ValueError(f"不是支持的Sentinel-2科学波段名称：{name}")



def resolve_worldstrat_bands(count: int, descriptions: tuple, cfg: Config) -> dict:
    """在用户已确认标准WorldStrat全波段格式的前提下，解析并校验完整映射。

    12通道 -> L2A；13通道 -> L1C。其他数量直接报错，不猜QA、重复时相或RGB排列。
    显式映射可以说明重新排列过的通道，但必须包含该级别全部科学波段且覆盖每个通道。
    """
    level = cfg.product_level.upper()
    if level == "AUTO":
        if count not in (12, 13):
            raise ValueError(f"读到{count}个通道，不是标准WorldStrat全波段的12/13通道。")
        level = "L2A" if count == 12 else "L1C"
    if level not in ("L2A", "L1C"):
        raise ValueError("product_level只能是AUTO、L2A或L1C。")
    expected = WORLDSTRAT_L2A_BANDS if level == "L2A" else WORLDSTRAT_L1C_BANDS
    if count != len(expected):
        raise ValueError(f"指定{level}应有{len(expected)}个光谱波段，但文件有{count}个通道。")

    if cfg.band_map is None:
        mapping = {band: i + 1 for i, band in enumerate(expected)}
    else:
        mapping = {}
        for name, channel in cfg.band_map.items():
            band = normalize_band_name(name)
            if band in mapping:
                raise ValueError(f"band_map中{name}与另一个名称重复指向科学波段{band}。")
            if isinstance(channel, bool) or not isinstance(channel, (int, np.integer)):
                raise ValueError(f"{band}的TIFF通道编号必须是从1开始的整数。")
            mapping[band] = int(channel)
        if set(mapping) != set(expected):
            missing = sorted(set(expected) - set(mapping))
            extra = sorted(set(mapping) - set(expected))
            raise ValueError(f"band_map必须覆盖{level}全部波段。缺少={missing}；多余={extra}。")
    if sorted(mapping.values()) != list(range(1, count + 1)):
        raise ValueError("band_map必须一一覆盖全部TIFF通道，不能重复、漏选或越界。")

    # 可读描述若与映射冲突，应停止，而不是带着错误波段名继续计算NDVI。
    pattern = r"(?<![A-Z0-9])B(8A|0?[1-9]|1[0-2])(?![A-Z0-9])"
    forbidden = r"(?<![A-Z0-9])(SCL|QA60|AOT|WVP|CLM|CLP|DATAMASK)(?![A-Z0-9])"
    inverse = {index: band for band, index in mapping.items()}
    for channel, description in enumerate(descriptions, start=1):
        text = (description or "").upper()
        if re.search(forbidden, text):
            raise ValueError(f"第{channel}通道描述为{description}，包含质量/辅助层，不可当反射率聚类。")
        hits = re.findall(pattern, text)
        if len(hits) == 1:
            declared = normalize_band_name("B" + hits[0])
            if declared != inverse[channel]:
                raise ValueError(
                    f"第{channel}通道描述为{declared}，但映射写成{inverse[channel]}；请检查完整band_map。"
                )

    # 读取顺序固定为TIFF存储顺序，后续CSV、中心列及端元矩阵全部沿用这个顺序。
    names = [inverse[i] for i in range(1, count + 1)]
    if isinstance(cfg.feature_bands, str):
        if cfg.feature_bands.upper() != "ALL":
            raise ValueError('全波段版的feature_bands请设置为"ALL"。')
    else:
        specified = [normalize_band_name(b) for b in cfg.feature_bands]
        if len(specified) != count or set(specified) != set(names):
            raise ValueError('本版聚类必须包含全部波段；请设feature_bands="ALL"，不能保留四波段子集。')
    if isinstance(cfg.purity_bands, str):
        if cfg.purity_bands.upper() != "ALL":
            raise ValueError('purity_bands应是科学波段元组或"ALL"。')
        purity_names = names.copy()
    else:
        purity_names = [normalize_band_name(b) for b in cfg.purity_bands]
        if not purity_names or len(purity_names) != len(set(purity_names)):
            raise ValueError("purity_bands不能为空或重复。")
        if not set(purity_names).issubset(names):
            raise ValueError("purity_bands中有当前文件不存在的波段。")
    return dict(level=level, mapping=mapping, bands=names, purity_bands=purity_names)



def calibration_vector(spec, metadata_values, names, indexes) -> np.ndarray:
    """读取反射率转换参数；字典缺少任何一个波段时明确报错，不沿用四波段配置。"""
    if spec is None:
        values = [metadata_values[i - 1] for i in indexes]
    elif isinstance(spec, dict):
        normalized = {normalize_band_name(k): v for k, v in spec.items()}
        missing = [band for band in names if band not in normalized]
        if missing:
            raise ValueError(f"逐波段scale/offset字典缺少{missing}；本版必须校准全部光谱波段。")
        values = [normalized[band] for band in names]
    else:
        values = [float(spec)] * len(names)
    return np.asarray(values, dtype=np.float64)



def read_aligned_quality_layer(path: Path, shape: tuple, crs, transform) -> tuple:
    """只读取同网格单通道质量层。不自动重采样，不把SCL变成连续数值。"""
    with rasterio.open(path) as src:
        if src.count != 1:
            raise ValueError("质量层应为单通道。")
        if src.shape != shape or src.crs != crs or not src.transform.almost_equals(transform):
            raise ValueError("质量层与输入网格不一致。需先用最近邻对齐，并确认同一景/日期。")
        arr = src.read(1).astype(np.float64)
        ok = (src.read_masks(1) > 0) & np.isfinite(arr)
    return arr, ok


def snow_quality_mask(numeric_valid: np.ndarray, cfg: Config, crs, transform) -> dict:
    """交集使用可选0/1掩膜与SCL；保留SCL=11，不要求SCL先判为雪。"""
    allowed = numeric_valid.copy()
    result = {"quality_supplied": cfg.clear_mask_path is not None or cfg.scl_path is not None}
    if cfg.clear_mask_path is not None:
        q, ok = read_aligned_quality_layer(cfg.clear_mask_path, allowed.shape, crs, transform)
        if not np.all(np.isin(q[ok], [0, 1])):
            raise ValueError("clear_mask_path必须是0/1掩膜；原始SCL请通过scl_path输入。")
        allowed &= ok & (q == 1)
    if cfg.scl_path is not None:
        scl, ok = read_aligned_quality_layer(cfg.scl_path, allowed.shape, crs, transform)
        if not np.all(np.isin(scl[ok], np.arange(12))):
            raise ValueError("SCL含0..11以外的有效类别或非整数；检查分类图及对齐方法。")
        scl[~ok] = np.nan
        allowed &= ok & ~np.isin(scl, cfg.scl_excluded_classes)
        result["scl"] = scl
        table = pd.DataFrame([
            {"scl_class": c, "pixels_numeric_valid": int((numeric_valid & (scl == c)).sum()),
             "pixels_allowed": int((allowed & (scl == c)).sum()),
             "excluded_by_scl_config": c in cfg.scl_excluded_classes}
            for c in range(12)])
        save_table(table, cfg.output_dir / "01_scl_statistics.csv", True)
        snow_before = numeric_valid & (scl == 11)
        if cfg.clear_mask_path is not None and np.any(snow_before & ~allowed):
            warnings.warn("外部0/1掩膜排除了部分SCL=11位置。请检查去云掩膜是否误删了雪/冰。")
    if not result["quality_supplied"]:
        warnings.warn("未提供云/云影/饱和质量信息：仅靠NDSI不能排除所有冰云、卷云。结果需人工核查。")
    result["quality_mask"] = allowed
    result["quality_mode"] = ("SCL+binary" if cfg.scl_path and cfg.clear_mask_path else
                              "SCL" if cfg.scl_path else "binary" if cfg.clear_mask_path else "none")
    return result


def read_reflectance(cfg: Config) -> dict:
    """完整读取全部波段，保留原反射率；质量剔除和指数筛选使用独立掩膜。"""
    validate_config(cfg)
    if not cfg.data_confirmed:
        raise ValueError("确认反射率单位及scale/offset后再设data_confirmed=True。")
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    with rasterio.open(cfg.input_path) as src:
        resolved = resolve_worldstrat_bands(src.count, src.descriptions, cfg)
        names, mapping = resolved["bands"], resolved["mapping"]
        indexes = [mapping[b] for b in names]
        raw = src.read(indexes).astype(np.float64)
        numeric_valid = np.all(src.read_masks(indexes) > 0, axis=0)
        scales = calibration_vector(cfg.scale, src.scales, names, indexes)
        offsets = calibration_vector(cfg.offset, src.offsets, names, indexes)
        if not np.all(np.isfinite(scales)) or np.any(scales <= 0) or not np.all(np.isfinite(offsets)):
            raise ValueError("校准参数须有限，且scale须大于0。")
        cube = raw*scales[:,None,None] + offsets[:,None,None]
        numeric_valid &= np.all(np.isfinite(cube), axis=0)
        transform, crs = src.transform, src.crs
    quality = snow_quality_mask(numeric_valid, cfg, crs, transform)
    valid = quality["quality_mask"]
    cube[:, ~numeric_valid] = np.nan  # 不截断>1或<0；质量剔除位置的原光谱仍可追踪。
    data = dict(cfg=cfg, cube=cube, bands=names, numeric_valid=numeric_valid, valid=valid,
                thresholds={}, gates={}, masks={}, counts=[], transform=transform, crs=crs,
                product_level=resolved["level"], purity_bands=resolved["purity_bands"], **quality)
    save_map(data, "01_quality_allowed", valid, "Quality-allowed pixels (snow retained)")
    if "scl" in quality:
        save_map(data, "01_scl", quality["scl"], "SCL: class 11 is snow or ice")
    save_json(cfg.output_dir / "01_quality_summary.json",
              dict(mode=quality["quality_mode"], numeric_valid_pixels=int(numeric_valid.sum()),
                   allowed_pixels=int(valid.sum()), scl_excluded_classes=cfg.scl_excluded_classes,
                   note="A supplied mask is not proof of perfect cloud removal; verify scene/date."))
    if not valid.any():
        save_pixel_trace(data)
        raise CandidateUnavailable("所有数值有效像元均被质量掩膜排除；不能据此认定无雪。")
    table = pd.DataFrame({"band": names, "channel": indexes, "scale": scales, "offset": offsets})
    table["used_for_clustering"] = True
    table["used_for_purity"] = [b in resolved["purity_bands"] for b in names]
    save_table(table, cfg.output_dir / "01_band_mapping.csv", True)
    print(f"完整反射率形状={cube.shape}；质量允许像元={valid.sum()}")
    print("均质性波段：", resolved["purity_bands"], "；聚类波段：", names)
    if resolved["level"] == "L1C":
        warnings.warn("L1C候选为TOA光谱；全部波段保留B10。未使用无地形校正的固定B10阈值排云。")
    statistics = []
    for i, name in enumerate(names):
        values = cube[i, valid]
        p02, med, p98 = np.quantile(values, [.02,.5,.98])
        statistics.append(dict(band=name, minimum=values.min(), p02=p02, median=med,
                               p98=p98, maximum=values.max(), negative_count=int((values<0).sum())))
    save_table(pd.DataFrame(statistics), cfg.output_dir / "01_reflectance_statistics.csv", True)
    if max(x["median"] for x in statistics) > 2:
        raise ValueError("某波段中位数>2，与当前反射率阈值体系不符；检查scale/offset。未自动除以10000。")
    save_json(cfg.output_dir / "01_config.json", vars(cfg))
    # RGB只用于检查来源位置。共用线性显示拉伸，不把显示值送入任何科学计算。
    rgb = np.moveaxis(cube[[names.index(b) for b in ("B4","B3","B2")]],0,-1)
    low, high = np.quantile(rgb[numeric_valid], [.02,.98])
    # 均一场景中退化的分位数区间改用0..max反射率，只影响显示。
    if high-low < 1e-8:
        low, high = 0.0, max(1.0, float(high))
    data["rgb"] = np.nan_to_num(np.clip((rgb-low)/(high-low),0,1))
    plt.figure(figsize=(7,6))
    plt.imshow(data["rgb"])
    plt.title("RGB preview: common display stretch only")
    finish_plot(cfg, "01_rgb_preview.png")
    return data


# ==================== 2. 计算雪地指数，保留分子分母与反射率有效域 ====================
def band(data: dict, name: str) -> np.ndarray:
    """按科学波段名称返回二维反射率，避免B8A与第9通道混淆。"""
    return data["cube"][data["bands"].index(name)]


def safe_normalized_difference(a: np.ndarray, b: np.ndarray,
                               valid: np.ndarray, min_sum: float) -> tuple:
    """计算(a-b)/(a+b)。负值或小分母位置设NaN；不裁剪指数、不修改反射率。"""
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    denominator = a + b
    ok = valid & np.isfinite(a) & np.isfinite(b) & (a >= 0) & (b >= 0) & (denominator >= min_sum)
    result = np.full(a.shape, np.nan, dtype=np.float64)
    np.divide(a - b, denominator, out=result, where=ok)
    return result, ok


def compute_snow_indices(data: dict) -> dict:
    """计算NDSI和植被排除用NDVI，并保存真实像元的分子、分母。"""
    cfg, valid = data["cfg"], data["valid"]
    green, red, nir, swir1 = (band(data,b) for b in ("B3","B4","B8","B11"))
    ndsi, ndsi_ok = safe_normalized_difference(green, swir1, valid, cfg.min_index_sum)
    ndvi, ndvi_ok = safe_normalized_difference(nir, red, valid, cfg.min_ndvi_sum)
    # NDSI与MNDWI是同一个绿光-SWIR1比值；不再计算一张相同图充当第二证据。
    # NDVI关闭时不以其有效性限制候选，但仍要求识别使用波段非负。
    nonnegative = np.all(np.stack([green, red, nir, swir1]) >= 0, axis=0)
    index_valid = valid & ndsi_ok & nonnegative
    if cfg.use_ndvi_gate:
        index_valid &= ndvi_ok
    data.update(ndsi=ndsi, ndvi=ndvi, ndvi_valid=ndvi_ok, index_valid=index_valid)
    for name in ("ndsi","ndvi"):
        save_map(data, "02_"+name, data[name], name.upper())
    save_map(data, "02_index_valid", index_valid, "Valid snow-index domain")
    rr,cc = np.where(data.get("numeric_valid",valid))
    ids = np.linspace(0,len(rr)-1,min(12,len(rr)),dtype=int)
    r,c = rr[ids],cc[ids]
    examples = pd.DataFrame({"row":r,"col":c,"quality_allowed":valid[r,c],
        "B3_green":green[r,c],"B4_red":red[r,c],"B8_nir":nir[r,c],"B11_swir1":swir1[r,c],
        "ndsi_numerator":(green-swir1)[r,c],"ndsi_denominator":(green+swir1)[r,c],
        "ndsi":ndsi[r,c],"ndvi_numerator":(nir-red)[r,c],"ndvi_denominator":(nir+red)[r,c],
        "ndvi":ndvi[r,c],"index_valid":index_valid[r,c]})
    save_table(examples,cfg.output_dir / "02_index_examples.csv",True)
    save_pixel_trace(data)
    require_pixels(data,index_valid,"可可靠计算雪地指数的数值域")
    return {n:data[n] for n in ("ndsi","ndvi","index_valid")}


def plot_ndsi_histogram(data: dict) -> None:
    """初筛后即保存直方图；后续失败也有阈值诊断图。"""
    cfg = data["cfg"]
    plt.figure(figsize=(8, 4))
    plt.hist(data["ndsi"][data["index_valid"]], bins=80)
    for key, text, style in (("base_ndsi", "Base", "--"), ("high_ndsi", "High", ":")):
        if key in data["thresholds"]:
            value = data["thresholds"][key]
            plt.axvline(value, linestyle=style, label=f"{text}={value:.3f}")
    plt.xlabel("NDSI")
    plt.ylabel("Pixel count")
    plt.legend()
    finish_plot(cfg, "02_ndsi_histogram.png")


# ==================== 3. NDSI、绝对亮度与可选植被排除 ====================
def screen_initial_snow(data: dict) -> np.ndarray:
    """NDSI只提供比值特征；用绝对亮度和可选NDVI排除条件形成候选。"""
    cfg,ok = data["cfg"],data["index_valid"]
    require_pixels(data,ok,"雪地指数计算域")
    values = data["ndsi"][ok]
    otsu = float(threshold_otsu(values)) if np.ptp(values)>1e-12 else float(values[0])
    t0 = float(np.clip(otsu,cfg.ndsi_base_min,cfg.ndsi_base_max))
    data["thresholds"].update(otsu_ndsi=otsu,base_ndsi=t0,
        base_ndsi_bounds=[cfg.ndsi_base_min,cfg.ndsi_base_max],min_green=cfg.min_green,
        min_nir=cfg.min_nir,use_ndvi_gate=cfg.use_ndvi_gate,max_ndvi=cfg.max_ndvi)
    save_thresholds(data)
    plot_ndsi_histogram(data)
    gates = {"index_valid":ok, "ndsi":data["ndsi"]>=t0,
        "green_floor":band(data,"B3")>=cfg.min_green,
        "nir_floor":band(data,"B8")>=cfg.min_nir,
        "not_vegetation":(data["ndvi_valid"] & (data["ndvi"]<=cfg.max_ndvi))
                          if cfg.use_ndvi_gate else data["valid"].copy()}
    data["gates"] = gates
    record_mask(data,"03_ndsi_only",ok & gates["ndsi"])
    initial = data["valid"].copy()
    stats = []
    for name,gate in gates.items():
        before = int(initial.sum())
        initial &= gate
        stats.append(dict(gate=name,enabled=name!="not_vegetation" or cfg.use_ndvi_gate,
            pass_alone_among_quality_allowed=int((data["valid"] & gate).sum()),
            before_cumulative=before,after_cumulative=int(initial.sum()),
            removed_at_this_step=before-int(initial.sum())))
    save_table(pd.DataFrame(stats),cfg.output_dir / "03_gate_statistics.csv",True)
    record_mask(data,"03_initial_snow",initial)
    data["initial_snow"] = initial
    # 联合查看NDSI与NIR，能直观看到高NDSI暗水被亮度门槛拒绝的原因。
    rr,cc = np.where(ok)
    rng = np.random.default_rng(cfg.random_seed)
    ids = rng.choice(len(rr),min(5000,len(rr)),replace=False)
    plt.figure(figsize=(7,5))
    plt.scatter(data["ndsi"][rr[ids],cc[ids]],band(data,"B8")[rr[ids],cc[ids]],s=6)
    plt.axvline(t0,linestyle="--",label=f"NDSI >= {t0:.3f}")
    plt.axhline(cfg.min_nir,linestyle=":",label=f"B8 >= {cfg.min_nir:.3f}")
    plt.xlabel("NDSI (same formula as MNDWI)")
    plt.ylabel("B8 reflectance")
    plt.legend()
    finish_plot(cfg,"03_ndsi_vs_nir.png")
    save_pixel_trace(data)
    require_pixels(data,initial,"NDSI、亮度和植被排除初筛")
    return initial


# ==================== 4. 雪区内部支持：小斑块去除、雪区边界缓冲 ====================
def select_snow_interior(data: dict) -> np.ndarray:
    cfg, initial = data["cfg"], data["initial_snow"]
    components, count = ndi.label(initial, structure=np.ones((3, 3), dtype=bool))
    areas = np.bincount(components.ravel())
    keep = areas >= cfg.min_patch_pixels
    keep[0] = False
    cleaned = keep[components]
    component_table = pd.DataFrame({"component": np.arange(1, count+1),
        "pixels": areas[1:], "kept": keep[1:]})
    save_table(component_table, cfg.output_dir / "04_components.csv")
    record_mask(data, "04_large_snow_patches", cleaned)

    # 不做孔洞填充或闭运算，避免把雪区内部的岩石、道路等非雪地主动填成雪地。
    # 半径r的正方形意味着中心周围(2r+1)^2位置必须均属于初筛雪地。
    r = cfg.erosion_radius
    interior = ndi.binary_erosion(cleaned, structure=np.ones((2*r+1, 2*r+1), dtype=bool),
                                  iterations=1, border_value=0)
    # 将影像外视为未知且不用于候选，显式补一圈0再计算棋盘距离。
    padded = np.pad(cleaned, 1, mode="constant", constant_values=False)
    distance = ndi.distance_transform_cdt(padded, metric="chessboard")[1:-1, 1:-1].astype(float)
    data["boundary_distance_pixels"] = distance
    save_map(data, "05_boundary_distance_pixels", distance,
             "Distance to candidate-mask boundary (pixels; not a verified snow boundary)")
    record_mask(data, "05_snow_interior", interior)
    data["interior"] = interior
    data["thresholds"].update(min_patch_pixels=cfg.min_patch_pixels, erosion_radius=r)
    save_thresholds(data)
    save_pixel_trace(data)
    require_pixels(data, interior, "有足够雪区边界缓冲的雪地内部")
    return interior


# ==================== 5. 较明亮雪面的局部均质性：相对/绝对波动同时检查 ====================
def local_snow_statistics(cube: np.ndarray, valid: np.ndarray,
                           window_size: int = 3, brightness_floor: float = 0.10) -> dict:
    """返回各波段局部均值/标准差，以及相对和绝对RMS光谱波动。

    U(p)=sqrt(mean_b[var_b(p)])             # 反射率单位的绝对波动
    M(p)=sqrt(mean_b[mean_b(p)^2])          # 局部平均光谱亮度
    H(p)=U(p)/max(M(p), brightness_floor)   # 有亮度下限的相对异质性

    brightness_floor是实验参数，不是噪声估计或雪深信息。
    不完整邻域一律设NaN；临时填0仅用于滤波，不污染最终有效统计。
    """
    cube = np.asarray(cube, dtype=np.float64)
    if cube.ndim != 3 or cube.shape[1:] != valid.shape or cube.shape[0] < 1:
        raise ValueError("cube应为[B,H,W]，valid应为[H,W]。")
    if window_size < 3 or window_size % 2 != 1 or brightness_floor <= 0:
        raise ValueError("局部窗口应为>=3的奇数，brightness_floor须为正。")
    valid = valid & np.all(np.isfinite(cube), axis=0)
    coverage = ndi.uniform_filter(valid.astype(np.float64), size=window_size,
                                  mode="constant", cval=0.0)
    full = coverage >= 1 - 1e-10
    means, variances = [], []
    for image in cube:
        safe = np.where(valid, image, 0.0)
        mean = ndi.uniform_filter(safe, size=window_size, mode="constant", cval=0.0)
        second = ndi.uniform_filter(safe**2, size=window_size, mode="constant", cval=0.0)
        variance = np.maximum(second - mean**2, 0.0)
        means.append(mean)
        variances.append(variance)
    means, variances = np.stack(means), np.stack(variances)
    abs_rms = np.sqrt(variances.mean(axis=0))
    mean_rms = np.sqrt((means**2).mean(axis=0))
    heterogeneity = abs_rms / np.maximum(mean_rms, brightness_floor)
    for image in (abs_rms, mean_rms, heterogeneity):
        image[~full] = np.nan
    means[:, ~full], variances[:, ~full] = np.nan, np.nan
    return dict(mean=means, std=np.sqrt(variances), abs_rms=abs_rms,
                mean_rms=mean_rms, heterogeneity=heterogeneity, full=full)


def save_local_snow_example(data: dict, row: int, col: int) -> None:
    """把一个实际种子的邻域、每个波段的均值与标准差写出，核对H的全部计算。"""
    cfg, local = data["cfg"], data["local_stats"]
    r, c = int(row), int(col)
    half = cfg.window_size // 2
    neighborhood = []
    for rr in range(r-half, r+half+1):
        for cc in range(c-half, c+half+1):
            record = {"row": rr, "col": cc, "ndsi": data["ndsi"][rr,cc]}
            for name in data["purity_bands"]:
                record[name] = band(data, name)[rr,cc]
            neighborhood.append(record)
    table = pd.DataFrame({"band": data["purity_bands"], "local_mean": local["mean"][:,r,c],
                          "local_std_ddof0": local["std"][:,r,c],
                          "local_variance": local["std"][:,r,c]**2})
    save_table(pd.DataFrame(neighborhood), cfg.output_dir / "07_example_neighborhood.csv")
    save_table(table, cfg.output_dir / "07_local_example.csv", True)
    formula = dict(row=r, col=c, window_size=cfg.window_size, purity_bands=data["purity_bands"],
        abs_rms=float(local["abs_rms"][r,c]), mean_rms=float(local["mean_rms"][r,c]),
        brightness_floor=cfg.brightness_floor,
        used_denominator=float(max(local["mean_rms"][r,c], cfg.brightness_floor)),
        heterogeneity=float(local["heterogeneity"][r,c]))
    save_json(cfg.output_dir / "07_local_example.json", formula)
    print("\n一个实际入选雪地像元的局部计算：", formula)


def filter_snow_seeds(data: dict) -> np.ndarray:
    cfg, interior, ndsi = data["cfg"], data["interior"], data["ndsi"]
    require_pixels(data, interior, "雪地内部像元")
    raw_high = float(np.quantile(ndsi[interior], cfg.ndsi_high_quantile))
    # 只要求足够强的雪地指数，不以最高NDSI作为绝对纯度排序。
    # 当两种雪地的指数都很高时，这个上限可以避免其中一组在聚类前被整组删除。
    high_threshold = max(data["thresholds"]["base_ndsi"],
        float(np.clip(raw_high, cfg.ndsi_high_min, cfg.ndsi_high_max)))
    high = interior & (ndsi >= high_threshold)
    data["thresholds"].update(high_ndsi=high_threshold, raw_high_quantile_value=raw_high,
        high_quantile=cfg.ndsi_high_quantile, high_ndsi_bounds=[cfg.ndsi_high_min,cfg.ndsi_high_max])
    save_thresholds(data)
    plot_ndsi_histogram(data)
    record_mask(data, "06_high_ndsi", high)

    indexes = [data["bands"].index(b) for b in data["purity_bands"]]
    local = local_snow_statistics(data["cube"][indexes], data["index_valid"],
                                   cfg.window_size, cfg.brightness_floor)
    data["local_stats"] = local
    data["heterogeneity"] = local["heterogeneity"]
    data["local_abs_rms"], data["local_mean_rms"] = local["abs_rms"], local["mean_rms"]
    for key, name in (("heterogeneity", "07_relative_heterogeneity"),
                      ("local_abs_rms", "07_absolute_variation"),
                      ("local_mean_rms", "07_mean_spectral_brightness")):
        save_map(data, name, data[key], name.replace("_", " "))

    # 先通过绝对波动上限，再从这批可靠候选中估计相对波动分位数。
    eligible = high & local["full"] & (local["abs_rms"] <= cfg.homogeneous_abs_max)
    record_mask(data, "07_absolute_homogeneous", eligible)
    require_pixels(data, eligible, "高NDSI且具有较小绝对光谱波动的内部像元")
    raw_h = float(np.quantile(local["heterogeneity"][eligible], cfg.homogeneous_quantile))
    # 均质性已充分好时不继续追逐最低相对波动，避免仅因微小噪声差异继续删减已均质的雪地。
    # min/max均为经验边界，不是统计置信度或仪器噪声估计。
    h_threshold = float(np.clip(raw_h, cfg.homogeneous_min, cfg.homogeneous_max))
    seeds = eligible & (local["heterogeneity"] <= h_threshold)
    data["seeds"] = seeds
    data["thresholds"].update(max_relative_heterogeneity=h_threshold,
        raw_heterogeneity_quantile_value=raw_h, relative_heterogeneity_bounds=[cfg.homogeneous_min,cfg.homogeneous_max],
        max_absolute_variation=cfg.homogeneous_abs_max, brightness_floor=cfg.brightness_floor,
        window_size=cfg.window_size, homogeneous_quantile=cfg.homogeneous_quantile)
    save_thresholds(data)
    record_mask(data, "08_snow_seeds", seeds)
    save_pixel_trace(data)
    require_pixels(data, seeds, "最终高可信雪地种子")

    rr, cc = np.where(seeds)
    middle = len(rr)//2
    save_local_snow_example(data, int(rr[middle]), int(cc[middle]))
    plt.figure(figsize=(7, 6))
    plt.imshow(data["rgb"])
    plt.scatter(cc, rr, s=4, marker=".", label="Snow candidate seeds")
    plt.legend()
    plt.title("Snow seeds on RGB: source-location inspection")
    finish_plot(cfg, "08_snow_seeds_on_rgb.png")
    print("\n最终雪地种子数量：", int(seeds.sum()), "；实际阈值：", data["thresholds"])
    return seeds


# ==================== 6. 全光谱聚类及逐轮诊断（不使用指数/空间坐标作特征） ====================
def prepare_all_band_features(data: dict) -> dict:
    """在同一批种子位置提取全部波段，形成N×B矩阵，再建立仅用于聚类的缩放副本。"""
    cfg = data["cfg"]
    if cfg.feature_scale_floor <= 0:
        raise ValueError("feature_scale_floor必须大于0。")
    rows, cols = np.where(data["seeds"])
    # 冒号表示全部波段：不能改成B2/B3/B4/B8索引。
    # 原cube为[B,H,W]，索引后为[B,N]，转置得到[N,B]。
    X = data["cube"][:, rows, cols].T.copy()
    names = data["bands"]
    if X.ndim != 2 or X.shape[1] != len(names) or X.shape[0] == 0:
        raise ValueError("全波段种子矩阵维数错误或没有种子。")
    if not np.isfinite(X).all():
        raise ValueError("种子光谱含NaN/Inf；不能补0或丢弃该波段后继续聚类。")

    # 缩放参数仅由入选的雪地种子估计，不用全图植被/裸地来决定雪地类内距离。
    median = np.median(X, axis=0)
    q25, q75 = np.quantile(X, [.25, .75], axis=0)
    iqr = q75 - q25
    scale = np.maximum(iqr, cfg.feature_scale_floor)
    Z = (X - median) / scale
    # 这是可逆的逐波段仿射变换：没有PCA、没有逐像元归一化、没有删除波段。
    # 常数波段仍保留，它自然不贡献样本之间的距离。
    constant = np.ptp(X, axis=0) <= 1e-12
    scaling = pd.DataFrame({
        "feature_column_0based": np.arange(len(names)),
        "band": names, "median": median, "q25": q25, "q75": q75, "iqr": iqr,
        "used_scale": scale, "raw_space_distance_weight": 1 / scale**2,
        "constant_among_seeds": constant,
    })
    print(f"\n全波段聚类：X.shape={X.shape}；Z.shape={Z.shape}")
    print("每个样本的全部特征列：", names)
    save_table(scaling, cfg.output_dir / "09_feature_scaling.csv", True)
    if constant.any():
        print("以下波段在种子中近似常数，仍保留，不删除：", np.asarray(names)[constant].tolist())

    raw_table = pd.DataFrame(X, columns=names)
    feature_table = pd.DataFrame(Z, columns=[f"{b}_z" for b in names])
    for table in (raw_table, feature_table):
        table.insert(0, "col", cols)
        table.insert(0, "row", rows)
        table.insert(0, "seed_index", np.arange(len(rows)))
    save_table(raw_table, cfg.output_dir / "09_seed_reflectance_all_bands.csv")
    save_table(feature_table, cfg.output_dir / "09_seed_features_all_bands.csv")
    print("\n前5个种子的全部原始反射率：\n", raw_table.head().to_string(index=False))
    print("\n前5个种子的全部缩放后特征：\n", feature_table.head().to_string(index=False))
    layout = dict(n_samples=len(X), n_features=X.shape[1], bands=names,
                  purity_bands=data["purity_bands"],
                  note="ALL spectral channels; no indices/coordinates/PCA in features")
    (cfg.output_dir / "09_feature_layout.json").write_text(
        json.dumps(layout, ensure_ascii=False, indent=2), encoding="utf-8")
    np.savez_compressed(cfg.output_dir / "09_all_band_features.npz", X=X, Z=Z,
                        median=median, scale=scale, bands=np.asarray(names), rows=rows, cols=cols)
    return dict(X=X, Z=Z, median=median, scale=scale, rows=rows, cols=cols)



def kmeans_with_trace(X: np.ndarray, k: int, n_init: int = 10,
                      max_iter: int = 200, seed: int = 42,
                      feature_names: list | None = None,
                      trace_sample_pixels: int = 6) -> dict:
    """显式执行全维Lloyd迭代，而不是用一句KMeans.fit隐藏计算过程。

    1. k-means++选择初始中心。
    2. 对每个像元计算到所有中心的全波段平方距离，分配到最近中心。
    3. 用簇内所有光谱的逐波段平均更新中心。
    4. 标签不再变化时收敛；多次初始化中保留SSE最小的收敛结果。

    history中的中心/SSE是本轮更新之后的值。
    assignments中的距离是本轮分配时（中心更新之前）的值。
    第一轮分配采用initial_centers，此后采用上一轮history中的中心。
    """
    X = np.asarray(X, dtype=np.float64)
    if n_init < 1 or max_iter < 1 or trace_sample_pixels < 1:
        raise ValueError("n_init、max_iter和trace_sample_pixels必须为正整数。")
    if X.ndim != 2 or not np.isfinite(X).all() or len(X) < k or k < 1:
        raise ValueError("X须为有限二维数组，并且1<=K<=样本数。")
    if len(np.unique(X, axis=0)) < k:
        raise ValueError("不同光谱数量少于K，不能强行生成K个中心。")
    names = feature_names if feature_names is not None else [f"feature_{i+1}" for i in range(X.shape[1])]
    if len(names) != X.shape[1] or len(set(names)) != len(names):
        raise ValueError("feature_names必须逐列对应X，且名称不能重复。")
    trace_ids = np.linspace(0, len(X)-1, min(trace_sample_pixels, len(X)), dtype=int)
    best, restarts = None, []
    for run in range(n_init):
        centers, _ = kmeans_plusplus(X, n_clusters=k, random_state=seed+run)
        initial_centers = centers.copy()
        previous_labels = np.full(len(X), -1, dtype=int)
        history, assignments = [], []
        converged, failure, sse = False, "maximum iterations", np.nan
        for iteration in range(1, max_iter + 1):
            # [N,1,B] - [1,K,B] -> [N,K,B]，沿全部B个波段求平方和。
            # distances[i,j]就是第i个种子到第j个中心的完整光谱平方距离。
            distances = ((X[:, None, :] - centers[None, :, :])**2).sum(axis=2)
            labels = distances.argmin(axis=1)
            sizes = np.bincount(labels, minlength=k)
            if np.any(sizes == 0):
                failure = "empty cluster"
                break

            # 只抽少量种子保存逐轮距离，避免CSV膨胀；全体样本仍参与每轮计算。
            for sample_id in trace_ids:
                item = {"iteration": iteration, "seed_index": int(sample_id),
                        "assigned_cluster": int(labels[sample_id]) + 1,
                        "previous_cluster": int(previous_labels[sample_id]) + 1,
                        "changed_label": bool(labels[sample_id] != previous_labels[sample_id])}
                item.update({f"distance2_center_{j+1}": float(distances[sample_id, j])
                             for j in range(k)})
                assignments.append(item)

            # 更新中心时沿样本维求平均，不在波段维上求平均。
            new_centers = np.stack([X[labels == j].mean(axis=0) for j in range(k)])
            sse = float(((X - new_centers[labels])**2).sum())
            shift = float(np.linalg.norm(new_centers - centers))
            changes = int(np.count_nonzero(labels != previous_labels))
            entry = {"iteration": iteration, "sse": sse,
                     "center_shift": shift, "changed_labels": changes}
            entry.update({f"n_{j+1}": int(sizes[j]) for j in range(k)})
            # 用科学波段名标注中心，避免feature_9被误读为B9；L2A第9列实际是B8A。
            entry.update({f"center_{j+1}_{band}_z": float(new_centers[j, b])
                          for j in range(k) for b, band in enumerate(names)})
            history.append(entry)
            centers = new_centers
            if changes == 0:
                converged = True
                break
            previous_labels = labels.copy()
        restarts.append({"run": run+1, "seed": seed+run, "converged": converged,
                         "sse": sse if converged else np.nan,
                         "iterations": iteration, "failure": "" if converged else failure})
        if converged and (best is None or sse < best["sse"]):
            best = dict(centers=centers.copy(), initial_centers=initial_centers,
                        labels=labels.copy(), sse=sse, history=pd.DataFrame(history),
                        assignments=pd.DataFrame(assignments), selected_run=run+1)
    if best is None:
        raise RuntimeError(f"K={k}的全部初始化均失败，请检查数据或增加max_iter。")
    best["restarts"] = pd.DataFrame(restarts)
    return best



def save_full_band_diagnostics(data: dict, features: dict, model: dict, k: int) -> None:
    """保存最终距离的逐波段分解、各波段SSE贡献及可还原的逐轮中心。"""
    cfg, names = data["cfg"], data["bands"]
    X, Z = features["X"], features["Z"]
    labels, centers = model["labels"], model["centers"]
    rows, cols = features["rows"], features["cols"]

    # 簇内SSE = 对每个波段的SSE求和。这里可核对每个波段是否真的参与。
    within_band_sse = ((Z - centers[labels])**2).sum(axis=0)
    total_band_sse = ((Z - Z.mean(axis=0))**2).sum(axis=0)
    between_band_sse = np.maximum(total_band_sse - within_band_sse, 0)
    sse_sum = float(within_band_sse.sum())
    share = within_band_sse / sse_sum if sse_sum > 0 else np.zeros(len(names))
    contribution = pd.DataFrame({"band": names,
        "within_cluster_sse": within_band_sse, "within_sse_fraction": share,
        "between_cluster_sse": between_band_sse, "total_sse": total_band_sse})
    print("\n各波段对最终簇内SSE的贡献（不是物理重要性排名）：")
    save_table(contribution, cfg.output_dir / "10_band_contributions.csv", True)
    assert np.isclose(sse_sum, model["sse"], rtol=1e-8, atol=1e-8)
    plt.figure(figsize=(10, 4))
    plt.bar(names, within_band_sse)
    plt.xlabel("All spectral bands")
    plt.ylabel("Contribution to within-cluster SSE")
    finish_plot(cfg, "10_band_contributions.png")

    # 实际种子的逐波段计算示例：差值平方沿B个波段相加=全光谱平方距离。
    sample_ids = np.linspace(0, len(Z)-1, min(cfg.trace_sample_pixels, len(Z)), dtype=int)
    examples = []
    for sample_id in sample_ids:
        for j in range(k):
            delta = Z[sample_id] - centers[j]
            total = float((delta**2).sum())
            for b, band in enumerate(names):
                examples.append({"seed_index": int(sample_id), "row": int(rows[sample_id]),
                    "col": int(cols[sample_id]), "compared_cluster": j+1,
                    "assigned_cluster": int(labels[sample_id])+1, "band": band,
                    "pixel_z": Z[sample_id, b], "center_z": centers[j, b],
                    "delta_z": delta[b], "squared_contribution": delta[b]**2,
                    "total_squared_distance": total})
    example_table = pd.DataFrame(examples)
    save_table(example_table, cfg.output_dir / "10_final_distances_by_band.csv")
    first = example_table[(example_table.seed_index == sample_ids[0]) &
                          (example_table.compared_cluster == 1)]
    print("\n一个真实种子到第1个最终中心的全部波段距离分解：")
    print(first[["band", "pixel_z", "center_z", "squared_contribution"]].to_string(index=False))
    print("平方距离合计=", first.squared_contribution.sum())

    # 保存全体种子的最终标签和到所属中心的距离，不仅保存抽样点。
    final = pd.DataFrame({"seed_index": np.arange(len(X)), "row": rows, "col": cols,
                          "cluster": labels+1,
                          "distance2_assigned_center": ((Z-centers[labels])**2).sum(axis=1)})
    save_table(final, cfg.output_dir / "10_final_seed_labels.csv")

    # 将逐轮特征中心逆变换为反射率，便于直接看每条中心光谱如何变化。
    # 这是整个簇的均值中心，不等于后面只用核心样本平均得到的候选端元。
    center_rows = []
    for entry in model["history"].to_dict("records"):
        for j in range(k):
            for b, band in enumerate(names):
                value_z = entry[f"center_{j+1}_{band}_z"]
                value_rho = value_z * features["scale"][b] + features["median"][b]
                center_rows.append(dict(iteration=entry["iteration"], cluster=j+1, band=band,
                                        center_z=value_z, center_reflectance=value_rho))
    save_table(pd.DataFrame(center_rows), cfg.output_dir / "10_selected_centers_reflectance.csv")

    assignments = model["assignments"].copy()
    ids = assignments.seed_index.to_numpy(dtype=int)
    assignments.insert(2, "row", rows[ids])
    assignments.insert(3, "col", cols[ids])
    save_table(assignments, cfg.output_dir / "10_selected_assignments.csv")



def cluster_snow_seeds(data: dict) -> dict:
    cfg, seeds = data["cfg"], data["seeds"]
    features = prepare_all_band_features(data)
    Z, rows, cols = features["Z"], features["rows"], features["cols"]
    rng = np.random.default_rng(cfg.random_seed)
    # 所有K使用同一批评价样本；实际聚类则始终使用全部种子和全部波段。
    evaluation = rng.choice(len(Z), min(len(Z), cfg.silhouette_sample_size), replace=False)
    save_table(pd.DataFrame({"seed_index": evaluation, "row": rows[evaluation], "col": cols[evaluation]}),
               cfg.output_dir / "10_silhouette_samples.csv")
    minimum_size = max(cfg.min_cluster_pixels, int(np.ceil(cfg.min_cluster_fraction*len(Z))))
    if minimum_size < 1:
        raise ValueError("最小簇大小必须为正。")
    max_k = min(cfg.max_clusters, len(np.unique(Z, axis=0)), len(Z)//minimum_size)
    if max_k < 1:
        raise CandidateUnavailable("种子数量不足以满足最小簇大小。")
    models, results = {}, []
    for k in range(1, max_k+1):
        print(f"\n正在比较K={k}：{len(Z)}个种子 × {Z.shape[1]}个光谱波段")
        try:
            model = kmeans_with_trace(
                Z, k, cfg.n_init, cfg.max_iter, cfg.random_seed,
                feature_names=data["bands"], trace_sample_pixels=cfg.trace_sample_pixels)
        except (ValueError, RuntimeError) as error:
            # 某个K失败不应让可用的K=1或其他K一起消失；明确记录失败原因。
            results.append(dict(k=k, sse=np.nan, silhouette=np.nan,
                min_cluster_size=0, min_largest_component=0, supported=False, error=str(error)))
            continue
        models[k] = model
        labels = model["labels"]
        sizes = np.bincount(labels, minlength=k)
        largest_components = []
        for j in range(k):
            mask = np.zeros(seeds.shape, dtype=bool)
            mask[rows[labels == j], cols[labels == j]] = True
            components, _ = ndi.label(mask, structure=np.ones((3, 3), dtype=bool))
            areas = np.bincount(components.ravel())[1:]
            largest_components.append(int(areas.max()) if areas.size else 0)
        supported = bool(sizes.min() >= minimum_size
                         and min(largest_components) >= cfg.min_cluster_component)
        score, note = np.nan, ""
        eval_labels = labels[evaluation]
        # 评价子样本必须覆盖所有簇，否则不给该K一个看似完整的轮廓系数。
        if supported and k >= 2:
            if len(np.unique(eval_labels)) == k and k < len(evaluation):
                score = float(silhouette_score(Z[evaluation], eval_labels))
            else:
                note = "evaluation sample does not cover all clusters"
        results.append(dict(k=k, sse=model["sse"], silhouette=score,
            min_cluster_size=int(sizes.min()), min_largest_component=min(largest_components),
            supported=supported, error=note))
        save_table(model["history"], cfg.output_dir / f"10_kmeans_K{k}_iterations.csv")
        save_table(model["restarts"], cfg.output_dir / f"10_kmeans_K{k}_restarts.csv")
        initial_table = pd.DataFrame(model["initial_centers"], columns=[f"{b}_z" for b in data["bands"]])
        initial_table.insert(0, "cluster", np.arange(1, k+1))
        save_table(initial_table, cfg.output_dir / f"10_kmeans_K{k}_initial_centers.csv")

    table = pd.DataFrame(results)
    save_table(table, cfg.output_dir / "10_k_selection.csv", True)
    accepted = table[(table.k >= 2) & table.supported & (table.silhouette >= cfg.silhouette_min)]
    if accepted.empty:
        row_one = table.loc[table.k == 1]
        if 1 not in models or row_one.empty or not bool(row_one.supported.iloc[0]):
            raise CandidateUnavailable("连K=1都没有足够支持；检查种子空间分布，不能据此断言无雪。")
        chosen_k = 1
    else:
        # 得分接近时优先少量候选，避免为了增加数量而人为细分光谱。
        best_score = accepted.silhouette.max()
        chosen_k = int(accepted.loc[
            accepted.silhouette >= best_score-cfg.silhouette_tolerance, "k"].min())
    table["selected"] = table.k == chosen_k
    save_table(table, cfg.output_dir / "10_k_selection.csv")
    model = models[chosen_k]
    print(f"\n最终保留{chosen_k}个雪地光谱簇；最佳初始化编号={model['selected_run']}")
    columns = ["iteration", "sse", "center_shift", "changed_labels"] + [f"n_{j+1}" for j in range(chosen_k)]
    print(model["history"][columns].to_string(index=False))
    save_table(model["history"], cfg.output_dir / "10_selected_iterations.csv")
    save_full_band_diagnostics(data, features, model, chosen_k)
    plt.figure(figsize=(7, 4))
    plt.plot(model["history"].iteration, model["history"].sse, marker="o")
    plt.xlabel("Lloyd iteration")
    plt.ylabel("Within-cluster sum of squares (all bands)")
    finish_plot(cfg, "10_selected_convergence.png")
    cluster_map = np.zeros(seeds.shape, dtype=np.int16)
    cluster_map[rows, cols] = model["labels"] + 1
    data["cluster_map"] = cluster_map
    save_map(data, "11_seed_clusters", cluster_map, "Snow seed clusters: all-band features")
    save_pixel_trace(data)
    result = dict(**features, model=model, k=chosen_k)
    data["clustering"] = result
    return result



# ==================== 7. 每个雪地光谱簇的核心样本与完整候选端元 ====================
def extract_candidate_spectra(data: dict) -> np.ndarray:
    cfg, cluster = data["cfg"], data["clustering"]
    X, Z = cluster["X"], cluster["Z"]
    labels, centers = cluster["model"]["labels"], cluster["model"]["centers"]
    rows, cols = cluster["rows"], cluster["cols"]
    if X.shape[1] != len(data["bands"]):
        raise ValueError("候选光谱维数与完整波段列表不一致，不能输出部分波段端元。")
    spectra, actual_spectra, core_stds, summaries = [], [], [], []
    used_for_mean = np.zeros(data["valid"].shape, dtype=np.uint8)
    for j in range(cluster["k"]):
        members = np.flatnonzero(labels == j)
        distance = np.linalg.norm(Z[members]-centers[j], axis=1)
        # 舍弃离簇中心最远的一半；在原始反射率上求剩余核心样本的平均。
        # 这是按整条光谱选样本后平均，不是分别截断各波段。
        core_count = min(len(members), max(3, int(np.ceil(cfg.representative_fraction*len(members)))))
        core = members[np.argsort(distance, kind="stable")[:core_count]]
        spectrum = X[core].mean(axis=0)
        spectra.append(spectrum)
        core_stds.append(X[core].std(axis=0, ddof=1 if core_count > 1 else 0))
        used_for_mean[rows[core], cols[core]] = 1
        # 标出核心样本中最接近这个平均光谱的实际像元，供RGB上核查。
        # 它是“最接近稳健均值的实际像元”，不是严格定义的medoid。
        target = (spectrum-cluster["median"])/cluster["scale"]
        chosen = core[np.argmin(np.linalg.norm(Z[core]-target, axis=1))]
        r, c = int(rows[chosen]), int(cols[chosen])
        actual_spectra.append(X[chosen])
        x, y = rasterio.transform.xy(data["transform"], r, c, offset="center")
        summaries.append({"candidate": f"snow_{j+1}", "cluster_pixels": len(members),
                          "averaged_pixels": core_count, "row": r, "col": c,
                          "x": x, "y": y, "crs": str(data["crs"]),
                          "representative_ndsi": data["ndsi"][r,c],
                          "representative_green": band(data,"B3")[r,c],
                          "representative_nir": band(data,"B8")[r,c],
                          "quality_mode": data["quality_mode"],
                          "representative_scl": data["scl"][r,c] if "scl" in data else np.nan,
                          "representative_ndvi": data["ndvi"][r,c],
                          "median_ndsi": np.median(data["ndsi"][rows[members],cols[members]]),
                          "median_swir1": np.median(band(data, "B11")[rows[members],cols[members]]),
                          "median_swir2": np.median(band(data, "B12")[rows[members],cols[members]]),
                          "negative_values_in_core_any_band": int((X[core] < 0).sum()),
                          "median_heterogeneity": np.median(data["heterogeneity"][rows[members], cols[members]])})
    # E形状=(波段数,候选数)，与后续线性解混中的端元矩阵方向保持一致。
    E = np.stack(spectra, axis=1)
    actual = np.stack(actual_spectra, axis=1)
    spectrum_table = pd.DataFrame(E, columns=[f"snow_{j+1}" for j in range(E.shape[1])])
    spectrum_table.insert(0, "band", data["bands"])
    print("\n候选雪地光谱（反射率，不是标准化特征）：")
    save_table(spectrum_table, cfg.output_dir / "12_candidate_spectra.csv", True)
    actual_table = pd.DataFrame(actual, columns=[f"snow_{j+1}" for j in range(E.shape[1])])
    actual_table.insert(0, "band", data["bands"])
    save_table(actual_table, cfg.output_dir / "12_actual_pixel_spectra.csv")
    # 核心样本的逐波段标准差是离散程度，不是置信区间或纯度概率。
    std_table = pd.DataFrame(np.stack(core_stds, axis=1), columns=spectrum_table.columns[1:])
    std_table.insert(0, "band", data["bands"])
    save_table(std_table, cfg.output_dir / "12_candidate_core_std.csv")
    save_table(pd.DataFrame(summaries), cfg.output_dir / "12_candidate_summary.csv", True)
    data["used_for_mean"] = used_for_mean
    save_pixel_trace(data)
    np.savez_compressed(cfg.output_dir / "12_candidates.npz", E=E, actual_pixel_spectra=actual,
                        bands=np.asarray(data["bands"]), product_level=data["product_level"],
                        core_std=np.stack(core_stds, axis=1),
                        purity_bands=np.asarray(data["purity_bands"]), ndsi=data["ndsi"],
                        ndvi=data["ndvi"], quality_mode=data["quality_mode"],
                        quality_allowed=data["valid"],
                        absolute_variation=data["local_abs_rms"],
                        heterogeneity=data["heterogeneity"], seeds=data["seeds"],
                        cluster_map=data["cluster_map"], used_for_mean=used_for_mean)
    plt.figure(figsize=(10, 4))
    for j in range(E.shape[1]):
        plt.plot(data["bands"], E[:, j], marker="o", label=f"snow_{j+1}")
    plt.xlabel("All spectral bands (categorical axis, not wavelength spacing)")
    plt.ylabel("Reflectance")
    plt.legend()
    finish_plot(cfg, "12_candidate_spectra.png")
    plt.figure(figsize=(7, 6))
    plt.imshow(data["rgb"])
    for item in summaries:
        plt.scatter(item["col"], item["row"], marker="x", s=80)
        plt.annotate(item["candidate"], (item["col"]+2, item["row"]+2))
    plt.title("Actual representative pixels of snow candidates")
    finish_plot(cfg, "12_candidate_locations.png")
    data["E"] = E
    data["candidate_summary"] = pd.DataFrame(summaries)
    print(f"\n处理完成。候选端元矩阵E的形状={E.shape}；输出目录：{cfg.output_dir}")
    return E



# ==================== 8. 任意像元追踪与整套运行入口 ====================
def trace_pixel(data: dict, row: int, col: int) -> pd.DataFrame:
    """在Notebook中调用trace_pixel(data,行号,列号)，查看为何保留/剔除。"""
    if not (0 <= row < data["valid"].shape[0] and 0 <= col < data["valid"].shape[1]):
        raise IndexError("像元坐标越界；行列号从0开始。")
    records = [{"item": "valid", "value": bool(data["valid"][row,col])}]
    for i, b in enumerate(data["bands"]):
        records.append({"item": b, "value": data["cube"][i,row,col]})
    for name in ("quality_mask", "scl", "ndsi", "ndvi", "ndvi_valid", "index_valid", "heterogeneity",
                 "local_abs_rms", "local_mean_rms", "boundary_distance_pixels", "cluster_map", "used_for_mean"):
        if name in data:
            records.append({"item": name, "value": data[name][row,col]})
    for group in ("gates", "masks"):
        for name, mask in data.get(group, {}).items():
            records.append({"item": group+"_"+name, "value": bool(mask[row,col])})
    result = pd.DataFrame(records)
    print(result.to_string(index=False))
    return result


def main(cfg: Config | None = None) -> dict | None:
    """顺序执行；失败时保存阶段和原因，不输出虚构的候选或空的成功文件。"""
    cfg = CFG if cfg is None else cfg
    validate_config(cfg)
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    status = {"input_path": str(cfg.input_path), "state": "running", "stage": "inspect_tiff"}
    status_path = cfg.output_dir / "run_status.json"
    save_json(status_path, status)
    data = None
    try:
        # 每次运行前仅删除本程序已知的上轮最终文件，避免失败后误读过期端元。
        # 不删除目录，不使用通配符清理任何用户文件。
        for name in ("12_candidate_spectra.csv", "12_actual_pixel_spectra.csv", "12_candidate_core_std.csv",
                     "12_candidate_summary.csv", "12_candidates.npz", "12_candidate_spectra.png",
                     "12_candidate_locations.png"):
            old = cfg.output_dir / name
            if old.is_file():
                old.unlink()
        inspect_tiff(cfg)
        if not cfg.data_confirmed:
            status.update(state="inspection_only", stage="await_reflectance_confirmation")
            save_json(status_path, status)
            print("\n完整波段映射已解析。确认反射率单位后设置data_confirmed=True继续。")
            return None
        environment = {"python": platform.python_version()}
        for package in ("numpy", "pandas", "rasterio", "scipy", "scikit-image", "scikit-learn", "matplotlib"):
            try:
                environment[package] = version(package)
            except PackageNotFoundError:
                environment[package] = "unknown"
        save_json(cfg.output_dir / "00_environment.json", environment)
        status["stage"] = "read_reflectance"
        save_json(status_path, status)
        data = read_reflectance(cfg)
        operations = (compute_snow_indices, screen_initial_snow, select_snow_interior,
                      filter_snow_seeds, cluster_snow_seeds, extract_candidate_spectra)
        for operation in operations:
            status["stage"] = operation.__name__
            save_json(status_path, status)
            operation(data)
        status.update(state="completed", product_level=data["product_level"],
                      cluster_bands=data["bands"], candidate_matrix_shape=list(data["E"].shape),
                      seed_pixels=int(data["seeds"].sum()),quality_mode=data["quality_mode"],
                      needs_source_inspection=True)
    except Exception as error:
        status.update(state="insufficient_candidates" if isinstance(error, CandidateUnavailable) else "failed",
                      error=str(error))
        save_json(status_path, status)
        if data is not None:
            save_pixel_trace(data)
        raise
    save_json(status_path, status)
    return data


def cli() -> None:
    """命令行参数可覆盖配置；不填写则使用文件顶部的CFG。"""
    parser = argparse.ArgumentParser(description="WorldStrat雪地候选提取：全部光谱波段聚类")
    parser.add_argument("--input", type=Path, help="输入TIFF")
    parser.add_argument("--output", type=Path, help="输出目录")
    parser.add_argument("--confirmed", action="store_true", help="已确认反射率单位")
    parser.add_argument("--scale", type=float, help="反射率=原值*scale+offset")
    parser.add_argument("--offset", type=float)
    parser.add_argument("--scl", type=Path, help="同网格同日期SCL分类层；保留11雪/冰")
    parser.add_argument("--clear-mask", type=Path, help="同网格0/1质量掩膜；1允许且应保留雪")
    parser.add_argument("--show", action="store_true", help="逐张显示图像")
    parser.add_argument("--purity-all", action="store_true", help="均质性也用全波段；聚类一直用全波段")
    args = parser.parse_args()
    for name in ("input", "output", "scale", "offset"):
        value = getattr(args, name)
        if value is not None:
            setattr(CFG, {"input": "input_path", "output": "output_dir"}.get(name, name), value)
    if args.scl is not None:
        CFG.scl_path = args.scl
    if args.clear_mask is not None:
        CFG.clear_mask_path = args.clear_mask
    if args.confirmed:
        CFG.data_confirmed = True
    if args.show:
        CFG.show_figures = True
    if args.purity_all:
        CFG.purity_bands = "ALL"
    main(CFG)


if __name__ == "__main__":
    cli()
