# -*- coding: utf-8 -*-
"""WorldStrat单景植被候选端元：全光谱波段聚类，中文注释与逐步记录。

输入：标准WorldStrat全波段TIFF（L2A=12个光谱波段，L1C=13个光谱波段）。
流程：NDVI -> 植被内部/均质性筛选 -> 全波段K-means -> 全波段候选光谱。
注意：这里仍然只是植被类内候选，不是最终三端元解混或物理纯度证明。
程序整体读取一张裁剪影像，不适用于超大整景的分块计算。
"""
from dataclasses import dataclass
from pathlib import Path
import json
import re
import warnings

import numpy as np
import pandas as pd
import rasterio
from scipy import ndimage as ndi
from skimage.filters import threshold_otsu
from sklearn.cluster import kmeans_plusplus
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt


# ==================== 0. 配置区：通常只需要修改这里 ====================
# 波段名不带前导0；顺序来自WorldStrat官方SentinelDownloader.set_bands()。
# 官方代码：https://raw.githubusercontent.com/worldstrat/worldstrat/main/dataset_generation/SentinelDownloader.py
WORLDSTRAT_L2A_BANDS = (
    "B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B9", "B11", "B12"
)
WORLDSTRAT_L1C_BANDS = (
    "B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B9", "B10", "B11", "B12"
)
WORLDSTRAT_L2A_MAP = {band: i + 1 for i, band in enumerate(WORLDSTRAT_L2A_BANDS)}
WORLDSTRAT_L1C_MAP = {band: i + 1 for i, band in enumerate(WORLDSTRAT_L1C_BANDS)}


@dataclass
class Config:
    input_path: Path = Path(r"E:\开源数据集\word_star\new_star\train\lr\Landcover-255756.tiff")
    output_dir: Path = Path(r"D:\program_myself\data_pro\SpectralUnmixing\vegetation_results_all_bands\Landcover-255756")

    # 已知输入保持WorldStrat官方全波段格式，AUTO才可按12/13通道选择L2A/L1C。
    # 这不是适用于任意12/13通道TIFF的通用波段猜测规则。
    product_level: str = "AUTO"     # "AUTO"、"L2A"或"L1C"。
    band_map: dict | None = None    # None=上述官方完整映射；手填时必须覆盖全部通道。

    # 单独控制聚类输入。ALL表示全部12/13个科学光谱波段，不能只填写四个波段。
    # 同时支持显式列出完整波段元组，但不允许少选或静默丢弃波段。
    feature_bands: str | tuple = "ALL"

    # NDVI仍然只使用B4/B8；均质性筛选与聚类波段相互独立。
    # 默认用原生10m的四波段检查局部边界；它不会限制后面的全波段聚类。
    # 需要均质性也遍历全部波段时，改为 "ALL"。
    purity_bands: str | tuple = ("B2", "B3", "B4", "B8")

    # 数值确认不是波段确认：仅确认标准排列，仍不能确定new_star是否另做了数值变换。
    # 第一次运行保持False，只输出元信息和自动完整映射；确认单位后再改True。
    data_confirmed: bool = True
    # 反射率 = 文件原始值 * scale + offset。
    # None：使用TIFF的逐波段scales/offsets，它们可能只是默认1和0。
    # 原值就是反射率：scale=1.0, offset=0.0。
    # 仅在已确认原值=反射率*10000且无额外偏移时：scale=0.0001, offset=0.0。
    # 填写字典时必须给出所有光谱波段，而不是仅给B2/B3/B4/B8。
    scale: float | dict | None = None
    offset: float | dict | None = None

    # 同网格0/1有效质量掩膜；不是SCL/QA原始类别编号。
    clear_mask_path: Path | None = None
    show_figures: bool = False

    # 以下阈值都是首轮实验参数，不是本图已验证的通用标准。
    min_red_nir_sum: float = 0.02
    min_nir: float = 0.10
    ndvi_base_min: float = 0.35
    ndvi_high_min: float = 0.60
    ndvi_high_quantile: float = 0.60
    min_patch_pixels: int = 9
    window_size: int = 3
    homogeneous_quantile: float = 0.50
    homogeneous_max: float = 0.10
    min_seed_pixels: int = 30

    # K是植被类内部候选数，不是整张图的地物类别数，也不是物种数。
    max_clusters: int = 5
    n_init: int = 10
    max_iter: int = 200
    random_seed: int = 42
    feature_scale_floor: float = 0.02   # IQR下限，单位为反射率，防止近常数波段放大噪声。
    min_cluster_pixels: int = 10
    min_cluster_fraction: float = 0.02
    min_cluster_component: int = 3
    silhouette_sample_size: int = 1200
    silhouette_min: float = 0.25
    silhouette_tolerance: float = 0.02
    representative_fraction: float = 0.50
    # 逐轮追踪这些种子到所有中心的距离；最终标签和光谱仍保存所有种子。
    trace_sample_pixels: int = 6


CFG = Config()


# ==================== 通用输出函数：不参与算法决策 ====================
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
    arr[~data["valid"]] = np.nan
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
    """为每个输入有效像元保留行列号、反射率、指标和各阶段是否通过。"""
    rows, cols = np.where(data["valid"])
    result = pd.DataFrame({"row": rows, "col": cols})
    for i, band in enumerate(data["bands"]):
        result[band] = data["cube"][i, rows, cols]
    for name in ("ndvi", "heterogeneity", "cluster_map", "used_for_mean"):
        if name in data:
            result[name] = data[name][rows, cols]
    for name, mask in data["masks"].items():
        result[name] = mask[rows, cols].astype(np.uint8)
    save_table(result, data["cfg"].output_dir / "pixel_trace.csv")


# ==================== 1. 只读取元数据与原始值，核对通道 ====================
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


def read_reflectance(cfg: Config) -> dict:
    if not cfg.data_confirmed:
        raise ValueError("标准WorldStrat映射已可自动解析；确认反射率单位及scale/offset后，再设data_confirmed=True。")
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    with rasterio.open(cfg.input_path) as src:
        resolved = resolve_worldstrat_bands(src.count, src.descriptions, cfg)
        mapping, names = resolved["mapping"], resolved["bands"]
        indexes = [mapping[b] for b in names]
        # names已覆盖全部通道；这里不再使用四波段feature_bands去切片。
        assert len(indexes) == src.count
        raw = src.read(indexes).astype(np.float64)
        valid = np.all(src.read_masks(indexes) > 0, axis=0)
        scales = calibration_vector(cfg.scale, src.scales, names, indexes)
        offsets = calibration_vector(cfg.offset, src.offsets, names, indexes)
        if not np.all(np.isfinite(scales)) or np.any(scales <= 0) or not np.all(np.isfinite(offsets)):
            raise ValueError("反射率转换参数必须有限，且scale必须大于0。")
        cube = raw * scales[:, None, None] + offsets[:, None, None]
        valid &= np.all(np.isfinite(cube), axis=0)
        transform, crs = src.transform, src.crs
    if cfg.clear_mask_path is not None:
        with rasterio.open(cfg.clear_mask_path) as mask_src:
            if (mask_src.shape != valid.shape or mask_src.crs != crs
                    or not mask_src.transform.almost_equals(transform)):
                raise ValueError("质量掩膜与影像网格不一致。请先对齐，掩膜只能采用最近邻重采样。")
            quality = mask_src.read(1, masked=True).filled(0)
            if not np.all(np.isin(quality, [0, 1])):
                raise ValueError("clear_mask必须是0/1图，不可直接传原始SCL分类编号图。")
            valid &= quality == 1
    else:
        warnings.warn("未提供质量掩膜：只排除了NoData/非有限数值，并不代表已去云、云影。")
    if not valid.any():
        raise ValueError("所选波段和质量掩膜没有共同有效像元。")
    cube[:, ~valid] = np.nan
    data = dict(cfg=cfg, cube=cube, bands=names, valid=valid,
                product_level=resolved["level"], purity_bands=resolved["purity_bands"],
                transform=transform, crs=crs, masks={}, counts=[])
    table = pd.DataFrame({"band": names, "channel": indexes, "scale": scales, "offset": offsets})
    table["used_for_clustering"] = True
    table["used_for_purity"] = [b in resolved["purity_bands"] for b in names]
    print("\n全部光谱波段与实际反射率转换：")
    save_table(table, cfg.output_dir / "01_band_mapping.csv", True)
    print(f"完整反射率数组形状={cube.shape}，共同有效像元={valid.sum()}")
    print("局部均质性波段：", resolved["purity_bands"])
    print("聚类与候选端元波段：", names)
    if resolved["level"] == "L1C":
        warnings.warn("L1C全波段聚类保留B10等大气敏感波段；得到的是本图TOA光谱候选，不等于地表反射率端元。")
    statistics = []
    for i, band in enumerate(names):
        values = cube[i, valid]
        p02, median, p98 = np.quantile(values, [.02, .50, .98])
        statistics.append(dict(band=band, p02=p02, median=median, p98=p98,
                               minimum=values.min(), maximum=values.max()))
    save_table(pd.DataFrame(statistics), cfg.output_dir / "01_reflectance_statistics.csv", True)
    if max(item["median"] for item in statistics) > 2:
        warnings.warn("某波段中位数大于2，请重点核对是否遗漏了反射率缩放；程序不会自动除以10000。")
    (cfg.output_dir / "01_config.json").write_text(
        json.dumps(vars(cfg), ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    # 真彩色预览只用于检查位置。各通道的2%~98%拉伸结果绝不送入后续计算。
    rgb = np.moveaxis(cube[[names.index(b) for b in ("B4", "B3", "B2")]], 0, -1)
    preview = np.zeros_like(rgb)
    for j in range(3):
        low, high = np.quantile(rgb[..., j][valid], [.02, .98])
        preview[..., j] = np.clip((rgb[..., j] - low) / max(high-low, 1e-12), 0, 1)
    data["rgb"] = np.nan_to_num(preview)
    plt.figure(figsize=(7, 6))
    plt.imshow(data["rgb"])
    plt.title("RGB preview: display stretch only")
    finish_plot(cfg, "01_rgb_preview.png")
    return data


# ==================== 2. NDVI：保存分子、分母与像元计算实例 ====================
def compute_ndvi(data: dict) -> np.ndarray:
    cfg = data["cfg"]
    red = data["cube"][data["bands"].index("B4")]
    nir = data["cube"][data["bands"].index("B8")]
    denominator = nir + red
    # 本实验端元搜索采用非负红光/近红外，并排除过暗分母。
    # 不把无效NDVI填成0；否则会改变直方图和自动阈值。
    ok = data["valid"] & (red >= 0) & (nir >= 0) & (denominator >= cfg.min_red_nir_sum)
    ndvi = np.full(red.shape, np.nan, dtype=np.float64)
    np.divide(nir-red, denominator, out=ndvi, where=ok)
    if ok.sum() < cfg.min_seed_pixels:
        raise ValueError("可计算NDVI的有效像元太少，请检查波段、缩放、NoData和质量掩膜。")
    data["ndvi"], data["ndvi_valid"] = ndvi, ok
    save_map(data, "02_ndvi", ndvi, "NDVI")
    # 使用真实影像的12个位置演示NDVI的逐像元计算，不使用预设演示数字。
    rows, cols = np.where(ok)
    sample = np.linspace(0, len(rows)-1, min(12, len(rows)), dtype=int)
    r, c = rows[sample], cols[sample]
    table = pd.DataFrame({"row": r, "col": c, "Red_B4": red[r, c],
                          "NIR_B8": nir[r, c], "NIR-Red": nir[r, c]-red[r, c],
                          "NIR+Red": denominator[r, c], "NDVI": ndvi[r, c]})
    print("\n真实像元的NDVI计算：")
    save_table(table, cfg.output_dir / "02_ndvi_examples.csv", True)
    print("NDVI分位数[0,25,50,75,100]%：", np.quantile(ndvi[ok], [0,.25,.5,.75,1]))
    return ndvi


# ==================== 3. 高NDVI + 内部像元 + 光谱均质性 ====================
def local_spectral_heterogeneity(cube: np.ndarray, valid: np.ndarray,
                                 window_size: int = 3) -> np.ndarray:
    """H = sqrt(各波段局部方差之和) / sqrt(各波段局部均值平方之和)。

    H越小，邻域中的光谱越一致。它不是植被丰度，也不是纯度概率。
    无效位置暂用0填充以便滤波；最后严格排除邻域不完整或含无效值的位置。
    """
    if window_size < 3 or window_size % 2 != 1:
        raise ValueError("window_size须为不小于3的奇数。")
    coverage = ndi.uniform_filter(valid.astype(np.float64), size=window_size,
                                  mode="constant", cval=0.0)
    variance_sum = np.zeros(valid.shape, dtype=np.float64)
    mean_square_sum = np.zeros(valid.shape, dtype=np.float64)
    for band in cube:
        safe = np.where(valid, band, 0.0)
        mean = ndi.uniform_filter(safe, size=window_size, mode="constant", cval=0.0)
        second = ndi.uniform_filter(safe**2, size=window_size, mode="constant", cval=0.0)
        variance_sum += np.maximum(second-mean**2, 0.0)
        mean_square_sum += mean**2
    score = np.sqrt(variance_sum) / np.maximum(np.sqrt(mean_square_sum), 1e-12)
    score[coverage < 1-1e-10] = np.nan
    return score


def filter_vegetation_seeds(data: dict) -> np.ndarray:
    cfg, ndvi, ok = data["cfg"], data["ndvi"], data["ndvi_valid"]
    nir = data["cube"][data["bands"].index("B8")]
    values = ndvi[ok]
    otsu = float(threshold_otsu(values)) if np.ptp(values) > 1e-12 else float(values[0])
    base_threshold = max(cfg.ndvi_base_min, otsu)
    print(f"\nOtsu阈值={otsu:.6f}，实际初筛NDVI阈值={base_threshold:.6f}，NIR下限={cfg.min_nir}")
    data["thresholds"] = dict(otsu=otsu, base_ndvi=base_threshold, min_nir=cfg.min_nir)
    (cfg.output_dir / "thresholds.json").write_text(
        json.dumps(data["thresholds"], ensure_ascii=False, indent=2), encoding="utf-8")
    initial = ok & (ndvi >= base_threshold) & (nir >= cfg.min_nir)
    record_mask(data, "03_initial_vegetation", initial)

    # 8邻接连通域：面积太小的孤立斑块不用于端元搜索。
    components, _ = ndi.label(initial, structure=np.ones((3, 3), dtype=bool))
    areas = np.bincount(components.ravel())
    keep = areas >= cfg.min_patch_pixels
    keep[0] = False  # 0是背景，不可保留。
    cleaned = keep[components]
    record_mask(data, "04_large_patches", cleaned)

    # 只腐蚀较宽松的植被掩膜，不先腐蚀稀疏的高NDVI点集。
    # 3×3窗口腐蚀一次：中心及其8邻域都属于植被时才保留中心。
    interior = ndi.binary_erosion(cleaned, structure=np.ones((cfg.window_size, cfg.window_size)),
                                 iterations=1, border_value=0)
    record_mask(data, "05_interior", interior)
    if interior.sum() < cfg.min_seed_pixels:
        save_pixel_trace(data)
        raise ValueError("植被内部像元不足。检查波段、掩膜和阈值；程序不自动降标准凑端元。")

    high_threshold = max(cfg.ndvi_high_min,
                         float(np.quantile(ndvi[interior], cfg.ndvi_high_quantile)))
    print(f"高NDVI阈值={high_threshold:.6f}")
    data["thresholds"]["high_ndvi"] = high_threshold
    (cfg.output_dir / "thresholds.json").write_text(
        json.dumps(data["thresholds"], ensure_ascii=False, indent=2), encoding="utf-8")
    high = interior & (ndvi >= high_threshold)
    record_mask(data, "06_high_ndvi", high)
    purity_indexes = [data["bands"].index(b) for b in data["purity_bands"]]
    purity_cube = data["cube"][purity_indexes]
    print("本次局部均质性检查波段：", data["purity_bands"])
    # 仅此处使用purity_bands；下游聚类直接用data["cube"]的全部波段。
    score = local_spectral_heterogeneity(purity_cube, ok, cfg.window_size)
    data["heterogeneity"] = score
    save_map(data, "07_heterogeneity", score, "Local spectral heterogeneity: lower is better")
    eligible = high & np.isfinite(score)
    if eligible.sum() < cfg.min_seed_pixels:
        save_pixel_trace(data)
        raise ValueError("高NDVI且具有完整有效邻域的像元不足，不能可靠提取候选。")
    score_threshold = min(cfg.homogeneous_max,
                          float(np.quantile(score[eligible], cfg.homogeneous_quantile)))
    seeds = eligible & (score <= score_threshold)
    record_mask(data, "08_pure_candidates", seeds)
    data["seeds"] = seeds
    thresholds = dict(otsu=otsu, base_ndvi=base_threshold, high_ndvi=high_threshold,
                      max_heterogeneity=score_threshold, min_nir=cfg.min_nir)
    data["thresholds"] = thresholds
    print("\n本图实际计算出的阈值：", thresholds)
    (cfg.output_dir / "thresholds.json").write_text(
        json.dumps(thresholds, ensure_ascii=False, indent=2), encoding="utf-8")
    plt.figure(figsize=(8, 4))
    plt.hist(values, bins=80)
    plt.axvline(base_threshold, linestyle="--", label=f"Base={base_threshold:.3f}")
    plt.axvline(high_threshold, linestyle=":", label=f"High={high_threshold:.3f}")
    plt.xlabel("NDVI")
    plt.ylabel("Pixel count")
    plt.legend()
    finish_plot(cfg, "02_ndvi_histogram.png")
    save_pixel_trace(data)
    if seeds.sum() < cfg.min_seed_pixels:
        raise ValueError("最终高可信种子不足，已保存中间结果。不能把相对最好的差像元当纯端元。")

    # 选择实际入选的一个位置，把邻域均值、标准差及H计算全部打印出来。
    positions = np.argwhere(seeds)
    r, c = positions[len(positions)//2]
    radius = cfg.window_size // 2
    patch = purity_cube[:, r-radius:r+radius+1, c-radius:c+radius+1]
    means, stds = patch.mean(axis=(1, 2)), patch.std(axis=(1, 2), ddof=0)
    table = pd.DataFrame({"row": int(r), "col": int(c), "band": data["purity_bands"],
                          "local_mean": means, "local_std": stds, "local_variance": stds**2})
    print(f"\n局部计算示例：row={r}, col={c}（均从0开始）")
    print("该位置的NDVI邻域：\n", ndvi[r-radius:r+radius+1, c-radius:c+radius+1])
    save_table(table, cfg.output_dir / "07_local_example.csv", True)
    direct_score = np.linalg.norm(stds) / max(np.linalg.norm(means), 1e-12)
    print(f"H = {np.linalg.norm(stds):.8f} / {np.linalg.norm(means):.8f} = {direct_score:.8f}")
    example = (f"row={r}, col={c}, window_size={cfg.window_size}\n"
               + "NDVI neighborhood:\n"
               + np.array2string(ndvi[r-radius:r+radius+1, c-radius:c+radius+1], precision=8)
               + f"\nH={np.linalg.norm(stds):.10f}/{np.linalg.norm(means):.10f}={direct_score:.10f}\n")
    (cfg.output_dir / "07_local_example.txt").write_text(example, encoding="utf-8")
    assert np.isclose(direct_score, score[r, c], rtol=1e-5, atol=1e-7)
    return seeds


# ==================== 4. 全波段聚类：矩阵、逐轮距离、中心和波段贡献 ====================
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

    # 缩放参数仅由入选的植被种子估计，不用全图水体/裸地来决定植被类内距离。
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
                  note="ALL spectral channels; no NDVI/coordinates/PCA in features")
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


def cluster_vegetation_seeds(data: dict) -> dict:
    cfg, seeds = data["cfg"], data["seeds"]
    features = prepare_all_band_features(data)
    Z, rows, cols = features["Z"], features["rows"], features["cols"]
    rng = np.random.default_rng(cfg.random_seed)
    # 所有K使用同一批评价样本；实际聚类则始终使用全部种子和全部波段。
    evaluation = rng.choice(len(Z), min(len(Z), cfg.silhouette_sample_size), replace=False)
    minimum_size = max(cfg.min_cluster_pixels, int(np.ceil(cfg.min_cluster_fraction*len(Z))))
    if minimum_size < 1:
        raise ValueError("最小簇大小必须为正。")
    max_k = min(cfg.max_clusters, len(np.unique(Z, axis=0)), len(Z)//minimum_size)
    if max_k < 1:
        raise ValueError("种子数量不足以满足最小簇大小。")
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
            raise ValueError("连K=1都没有足够支持；已保存K比较结果，请检查种子空间分布。")
        chosen_k = 1
    else:
        # 得分接近时优先少量候选，避免为了增加数量而人为细分光谱。
        best_score = accepted.silhouette.max()
        chosen_k = int(accepted.loc[
            accepted.silhouette >= best_score-cfg.silhouette_tolerance, "k"].min())
    table["selected"] = table.k == chosen_k
    save_table(table, cfg.output_dir / "10_k_selection.csv")
    model = models[chosen_k]
    print(f"\n最终保留{chosen_k}个植被光谱簇；最佳初始化编号={model['selected_run']}")
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
    save_map(data, "11_seed_clusters", cluster_map, "Vegetation seed clusters: all-band features")
    save_pixel_trace(data)
    result = dict(**features, model=model, k=chosen_k)
    data["clustering"] = result
    return result


# ==================== 5. 每个光谱簇 -> 一个稳健候选光谱与实际来源像元 ====================
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
        summaries.append({"candidate": f"veg_{j+1}", "cluster_pixels": len(members),
                          "averaged_pixels": core_count, "row": r, "col": c,
                          "x": x, "y": y, "crs": str(data["crs"]),
                          "representative_ndvi": data["ndvi"][r, c],
                          "median_ndvi": np.median(data["ndvi"][rows[members], cols[members]]),
                          "median_heterogeneity": np.median(data["heterogeneity"][rows[members], cols[members]])})
    # E形状=(波段数,候选数)，与后续线性解混中的端元矩阵方向保持一致。
    E = np.stack(spectra, axis=1)
    actual = np.stack(actual_spectra, axis=1)
    spectrum_table = pd.DataFrame(E, columns=[f"veg_{j+1}" for j in range(E.shape[1])])
    spectrum_table.insert(0, "band", data["bands"])
    print("\n候选植被光谱（反射率，不是标准化特征）：")
    save_table(spectrum_table, cfg.output_dir / "12_candidate_spectra.csv", True)
    actual_table = pd.DataFrame(actual, columns=[f"veg_{j+1}" for j in range(E.shape[1])])
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
                        purity_bands=np.asarray(data["purity_bands"]), ndvi=data["ndvi"],
                        heterogeneity=data["heterogeneity"], seeds=data["seeds"],
                        cluster_map=data["cluster_map"], used_for_mean=used_for_mean)
    plt.figure(figsize=(10, 4))
    for j in range(E.shape[1]):
        plt.plot(data["bands"], E[:, j], marker="o", label=f"veg_{j+1}")
    plt.xlabel("All spectral bands (categorical axis, not wavelength spacing)")
    plt.ylabel("Reflectance")
    plt.legend()
    finish_plot(cfg, "12_candidate_spectra.png")
    plt.figure(figsize=(7, 6))
    plt.imshow(data["rgb"])
    for item in summaries:
        plt.scatter(item["col"], item["row"], marker="x", s=80)
        plt.annotate(item["candidate"], (item["col"]+2, item["row"]+2))
    plt.title("Actual representative pixels of vegetation candidates")
    finish_plot(cfg, "12_candidate_locations.png")
    data["E"] = E
    print(f"\n处理完成。候选端元矩阵E的形状={E.shape}；输出目录：{cfg.output_dir}")
    return E


def main(cfg: Config = CFG) -> dict | None:
    inspect_tiff(cfg)
    if not cfg.data_confirmed:
        print("\n目前已完成元信息与完整波段映射检查。确认scale/offset后设data_confirmed=True再运行。")
        return None
    if not (0 < cfg.representative_fraction <= 1):
        raise ValueError("representative_fraction应在(0,1]内。")
    if cfg.feature_scale_floor <= 0 or cfg.trace_sample_pixels < 1:
        raise ValueError("feature_scale_floor和trace_sample_pixels必须为正。")
    status_path = cfg.output_dir / "run_status.json"
    status = dict(input_path=str(cfg.input_path), state="running", stage="read_reflectance")
    status_path.write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8")
    try:
        data = read_reflectance(cfg)
        for operation in (compute_ndvi, filter_vegetation_seeds,
                          cluster_vegetation_seeds, extract_candidate_spectra):
            status["stage"] = operation.__name__
            status_path.write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8")
            operation(data)
        status.update(state="completed", product_level=data["product_level"],
                      cluster_bands=data["bands"], candidate_matrix_shape=list(data["E"].shape))
    except Exception as error:
        status.update(state="failed", error=str(error))
        status_path.write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8")
        raise
    status_path.write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8")
    return data


if __name__ == "__main__":
    main()
