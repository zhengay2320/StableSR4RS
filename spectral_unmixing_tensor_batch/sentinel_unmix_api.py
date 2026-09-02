# -*- coding: utf-8 -*-
"""供其他算法 import 调用的 Sentinel-2 五类线性解混接口。

唯一的业务入口::

    abundance, error = unmix_image(r"D:\\data\\scene.tif")

返回约定
--------
* abundance: np.ndarray, float32, (H, W, 5)，保持输入文件的像素网格。
* 五通道永远为：水体、绿色植被、裸露矿物地表、人工不透水面、雪冰。
* error: np.ndarray, float32, (H, W)，十个分析波段的反射率重建 RMSE。
* 成功求解的像元丰度非负、五类之和约为 1；未激活类别的整通道为 0。
* 无效观测位置的五通道置 0，error 置 NaN。请用 np.isfinite(error) 作掩膜。
* “未激活类别”不等于经过独立验证的“现实中绝对不存在”。发现某类有
  支持证据但无法提取近纯端元时，抛 MissingEndmemberError，不以 0 掩盖失败。

计算流程
--------
读取及校验 -> 复用本景端元缓存，或本景候选提取 -> FCLS -> 类别聚合 -> RMSE。
不调用网络，不绘图，不写输出文件，不在 import 时执行任务。

可选的同名伴随文件（均自动由唯一输入路径推导；不是函数的附加参数）
----------------------------------------------------------------------
scene.unmix.json      已确认的波段顺序/scale/offset。用于缺少元数据的影像。
scene.valid.tif       已对齐的有效观测掩膜：1 有效，0 无效。
scene.scl.tif         已对齐的 Sentinel-2 SCL；保留雪冰，剔除云及云影等。
scene.seeds.tif       已有提取方法的候选标签：0 未标注，1~5 对应上述五类。
scene.prior.tif       已对齐的五通道类别支持度，顺序同输出，不是丰度真值。
scene.endmembers.npz  已有建库方法的结果；存在时优先使用，格式见使用说明。

没有提供你的完整旧工程源码，因此本文件不假装调用未知的工程函数。
内置的是“逐景代表端元 + FCLS”基线，不是前述完整 MESMA/Dynamic World 下载器。
已有提取器可通过 seeds/endmembers 文件复用；也可以只替换下面私有函数
_get_endmembers()，保持 unmix_image() 及返回契约不变。

科学边界
--------
仅靠单景的指数不能可靠区分全球所有裸土与人工地表。因此无先验/缓存时，
仅在水体、绿色植被、雪冰等规则有支持且没有明显裸地/人工地表疑似区域时
尝试解混。遇到疑似裸地/人工地表，默认要求上述 seeds/prior/endmembers，
而不是用 NDBI 正负强行标为建筑。规则阈值只是保守工程起点，未做全球校准。
本函数不自动改变空间分辨率：需要 20 m 分析时，请先准备共同网格影像。

依赖：Python >= 3.10；numpy、scipy、rasterio。
参考接口：
https://rasterio.readthedocs.io/en/stable/api/rasterio.io.html
https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.binary_erosion.html
https://pysptools.sourceforge.io/abundance_maps.html
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
import json
import re
import warnings

import numpy as np
import rasterio
from scipy.ndimage import binary_erosion, uniform_filter

__all__ = [
    "unmix_image", "CLASS_NAMES", "ANALYSIS_BANDS", "UnmixingError",
    "InputMetadataError", "MissingEndmemberError", "UnmixingWarning",
]

# 这个元组是 API 协议，不能按当前图像出现的类别随意重排！
CLASS_NAMES = (
    "water", "green_vegetation", "bare_mineral", "impervious", "snow_ice"
)
ANALYSIS_BANDS = (
    "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12"
)
_L2A_12 = (
    "B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08",
    "B8A", "B09", "B11", "B12"
)

# 仅为无标签文件提供的明确布局约定，不声称“任意12波段TIFF都是此顺序”。
# 对已重排的数据，请通过 TIFF descriptions 或 scene.unmix.json 声明真实顺序。
_ALLOW_CONVENTIONAL_UNLABELLED_LAYOUT = True
_MIN_SEED_PIXELS = 16
_MAX_SEED_PIXELS = 2000
_EROSION_ITERATIONS = 1
_MAX_LOCAL_STD = 0.025   # 反射率单位；工程初值，不是全球校准阈值。
_PRIOR_MIN = 0.85
_PRIOR_MARGIN = 0.20
_MAX_ENDMEMBERS = 8     # 全支持集 FCLS 的计算上限；内置提取器最多给出5条。
_CHUNK_SIZE = 4096
_RANDOM_SEED = 42
_FEASIBILITY_TOL = 1e-9


class UnmixingError(RuntimeError):
    """不能产生合格的解混候选结果；调用方应捕获并记录，而不是填零当成功。"""


class InputMetadataError(UnmixingError):
    """输入波段、单位、空间对齐或缓存格式不明确/不正确。"""


class MissingEndmemberError(UnmixingError):
    """疑似存在某类别，但没有足够可靠端元；不等于该类别不存在。"""


class UnmixingWarning(UserWarning):
    """数据假设或方法适用边界提醒。生产调用方可通过 warnings 统一收集。"""


@dataclass
class _Scene:
    path: Path
    cube: np.ndarray       # H,W,B，统一为反射率，计算采用 float64。
    valid: np.ndarray      # H,W，所有分析波段均有效。
    transform: object
    crs: object
    shape: tuple[int, int]


def _warn(message: str) -> None:
    warnings.warn(message, UnmixingWarning, stacklevel=3)


def _sidecar(path: Path, suffix: str) -> Path:
    """scene.tiff -> scene.endmembers.npz 等；不修改输入文件。"""
    return path.with_name(path.stem + suffix)


def _band_name(value: object) -> str:
    """把 B2/B02/Sentinel-2 B02 等规范到 B02；保留 SCL/CLM 等特殊名称。"""
    text = str(value or "").strip().upper()
    matches = re.findall(r"(?<![A-Z0-9])B(0?[1-9]|1[0-2]|8A)(?![A-Z0-9])", text)
    if len(matches) == 1:
        return "B8A" if matches[0] == "8A" else f"B{int(matches[0]):02d}"
    return text


def _load_metadata(path: Path) -> dict:
    meta_path = _sidecar(path, ".unmix.json")
    if not meta_path.exists():
        return {}
    with meta_path.open("r", encoding="utf-8-sig") as stream:
        meta = json.load(stream)
    allowed = {"band_names", "scale", "offset", "units", "product_level"}
    if not isinstance(meta, dict) or set(meta) - allowed:
        raise InputMetadataError(f"{meta_path.name} 必须是JSON对象，允许键为 {sorted(allowed)}")
    if "offset" in meta and "scale" not in meta:
        raise InputMetadataError("JSON 中声明 offset 时也应声明 scale，防止混用缩放来源。")
    if "units" in meta and meta["units"] != "reflectance":
        raise InputMetadataError("当前接口只接收/转换为 reflectance，不接收显示拉伸或标准化特征。")
    return meta


def _per_band_parameter(value: object, count: int, key: str) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float64)
    if arr.ndim == 0:
        arr = np.full(count, float(arr))
    if arr.shape != (count,) or not np.isfinite(arr).all():
        raise InputMetadataError(f"{key} 必须为有限标量，或长度等于文件全部波段数 {count} 的数组。")
    return arr


def _read_aligned(path: Path, scene: _Scene) -> tuple[np.ndarray, np.ndarray, tuple]:
    """伴随栅格必须事先严格配准。宁可报错，也不悄悄用普通resize错配先验。"""
    with rasterio.open(path) as ds:
        if ds.shape != scene.shape or ds.crs != scene.crs or not ds.transform.almost_equals(scene.transform):
            raise InputMetadataError(f"伴随文件 {path.name} 的尺寸/CRS/transform 与输入不一致。")
        data = ds.read(masked=True, out_dtype="float64")
        values = data.filled(np.nan)
        valid = (~np.ma.getmaskarray(data)).all(axis=0) & np.isfinite(values).all(axis=0)
        return values, valid, ds.descriptions


def _read_scene(path: Path) -> _Scene:
    meta = _load_metadata(path)
    with rasterio.open(path) as ds:
        level = str(meta.get("product_level", ds.tags().get("product_level", ""))).upper()
        if level in {"L1C", "LEVEL-1C"}:
            raise InputMetadataError("检测到 L1C；本基线的候选规则要求已确认的 L2A 地表反射率。")
        if "band_names" in meta:
            if not isinstance(meta["band_names"], list) or len(meta["band_names"]) != ds.count:
                raise InputMetadataError("band_names 必须为数组，按文件顺序覆盖全部波段。")
            names = tuple(_band_name(v) for v in meta["band_names"])
        else:
            names = tuple(_band_name(v) for v in ds.descriptions)
            if not set(ANALYSIS_BANDS).issubset(names):
                # 有明确但不完整/矛盾的波段描述时不覆盖它们；只允许完全未标注的标准布局。
                if any(names) or not _ALLOW_CONVENTIONAL_UNLABELLED_LAYOUT:
                    raise InputMetadataError("波段描述缺少所需十波段；请在 .unmix.json 中声明真实 band_names。")
                if ds.count == 10:
                    names = ANALYSIS_BANDS
                elif ds.count == 12:
                    names = _L2A_12
                else:
                    raise InputMetadataError("无波段描述时仅支持约定的10/12波段布局；RGB图不能执行此解混。")
                _warn(f"{path.name} 无波段描述，暂按约定顺序 {names} 读取；须确认数据未重排。")
        if any(names.count(b) != 1 for b in ANALYSIS_BANDS):
            raise InputMetadataError("每个分析波段必须恰好出现一次；检查重复或缺失波段。")
        positions = np.asarray([names.index(b) for b in ANALYSIS_BANDS])
        indices = (positions + 1).tolist()  # rasterio band index 从1开始。
        raw = ds.read(indices, masked=True, out_dtype="float64")
        data = raw.filled(np.nan)
        valid = (~np.ma.getmaskarray(raw)).all(axis=0) & np.isfinite(data).all(axis=0)

        # 缩放优先级：显式JSON -> 非默认TIFF scale/offset -> 浮点反射率假设。
        # 对无缩放信息的整数DN不猜“/10000”，因为偏移量可能同样不可忽略。
        if "scale" in meta:
            scale = _per_band_parameter(meta["scale"], ds.count, "scale")[positions]
            offset = _per_band_parameter(meta.get("offset", 0.0), ds.count, "offset")[positions]
        else:
            scale = np.asarray(ds.scales, dtype=np.float64)[positions]
            offset = np.asarray(ds.offsets, dtype=np.float64)[positions]
            explicit = not (np.all(scale == 1.0) and np.all(offset == 0.0))
            if not explicit and np.issubdtype(np.dtype(ds.dtypes[indices[0] - 1]), np.integer):
                raise InputMetadataError(
                    "整数影像没有已确认的scale/offset。请核实后在同名 .unmix.json 中设置；不自动猜测除以10000。"
                )
            if not explicit and meta.get("units") != "reflectance":
                _warn("无显式反射率单位，按浮点地表反射率使用；不适用于拉伸值、z-score或RGB显示图。")
        if not np.isfinite(scale).all() or not np.isfinite(offset).all() or np.any(scale <= 0):
            raise InputMetadataError("scale必须为有限正值，offset必须有限。")
        data = data * scale[:, None, None] + offset[:, None, None]
        if valid.any():
            finite_values = data[:, valid]
            if np.percentile(finite_values, 99) > 2.0 or np.percentile(finite_values, 1) < -0.25:
                raise InputMetadataError("反射率范围明显异常；请检查缩放、偏移和输入是否做过显示归一化。")
        # 不是把数值硬裁剪到[0,1]；仅剔除极端异常和全波段零填充像元。
        valid &= np.isfinite(data).all(axis=0)
        valid &= (data >= -0.25).all(axis=0) & (data <= 2.0).all(axis=0)
        valid &= np.any(data != 0, axis=0)
        cloud_mask_found = False
        if "SCL" in names:
            scl_ma = ds.read(names.index("SCL") + 1, masked=True)
            scl = scl_ma.filled(0)
            valid &= ~np.isin(scl, [0, 1, 3, 8, 9, 10])  # 11=雪冰，保留。
            cloud_mask_found = True
        if "DATAMASK" in names:
            dm = ds.read(names.index("DATAMASK") + 1, masked=True).filled(0)
            valid &= dm > 0
        if "CLM" in names:
            clm = ds.read(names.index("CLM") + 1, masked=True).filled(255)
            valid &= clm == 0
            cloud_mask_found = True
        scene = _Scene(path, np.moveaxis(data, 0, -1), valid, ds.transform, ds.crs, ds.shape)

    for suffix in (".valid.tif", ".scl.tif"):
        mask_path = _sidecar(path, suffix)
        if mask_path.exists():
            mask, mask_valid, _ = _read_aligned(mask_path, scene)
            if mask.shape[0] != 1:
                raise InputMetadataError(f"{mask_path.name} 应为单通道。")
            if suffix == ".valid.tif":
                scene.valid &= mask_valid & (mask[0] > 0)
            else:
                scene.valid &= mask_valid & ~np.isin(mask[0], [0, 1, 3, 8, 9, 10])
            cloud_mask_found = True
    if not cloud_mask_found:
        _warn("未找到云/有效观测掩膜；输入须事先去云。反射率规则不等于可靠云检测。")
    return scene


def _load_support(scene: _Scene) -> np.ndarray | None:
    """复用已有五类候选提取结果；先 seeds，后五通道 prior。不自动联网下载DW。"""
    seed_path = _sidecar(scene.path, ".seeds.tif")
    if seed_path.exists():
        data, valid, _ = _read_aligned(seed_path, scene)
        if data.shape[0] != 1 or not np.isin(data[0][valid], np.arange(6)).all():
            raise InputMetadataError("seeds必须单通道：0未标注，1水、2植被、3裸地、4人工地表、5雪。")
        return np.stack([(data[0] == c + 1) & valid for c in range(5)], axis=-1).astype(float)
    prior_path = _sidecar(scene.path, ".prior.tif")
    if prior_path.exists():
        data, valid, descriptions = _read_aligned(prior_path, scene)
        if data.shape[0] != 5 or tuple(descriptions) != CLASS_NAMES:
            raise InputMetadataError(f"prior必须为5通道，且band descriptions依次为 {CLASS_NAMES}。")
        if np.any(data[:, valid] < 0) or np.any(data[:, valid] > 1):
            raise InputMetadataError("prior支持度必须在[0,1]，不是DN或未经转换的标签。")
        if np.any(data[:, valid].sum(axis=0) > 1.0001):
            raise InputMetadataError("五类支持度之和不能超过1；不要对混淆类别重复计数。")
        data[:, ~valid] = 0
        return np.moveaxis(data, 0, -1)
    return None


def _normalised_difference(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    result = np.zeros_like(a, dtype=float)
    # 极暗或出现负和的像元不靠指数比值提供纯像元证据。
    np.divide(a - b, a + b, out=result, where=(a + b) > 1e-6)
    return result


def _candidate_masks(scene: _Scene, support: np.ndarray | None) -> tuple[list, list]:
    """类别先验只限制端元候选，不限制后续像元可以分配给哪些材料。"""
    x = np.where(scene.valid[..., None], scene.cube, 0.0)
    blue, green, red, nir, swir1 = (x[..., i] for i in (0, 1, 2, 6, 8))
    ndvi = _normalised_difference(nir, red)
    green_swir = _normalised_difference(green, swir1)
    water = (green_swir > 0.20) & (nir < 0.14) & (swir1 < 0.08) & (green < 0.30) & (ndvi < 0.25)
    vegetation = (ndvi > 0.65) & (nir > 0.16)
    snow = (green_swir > 0.40) & (green > 0.35) & (nir > 0.25) & (swir1 < 0.25)
    dry = (ndvi < 0.35) & (swir1 > 0.02) & ~(water | snow)
    physical = [water, vegetation, dry, dry, snow]
    if support is None:
        ambiguous = scene.valid & dry
        if np.count_nonzero(ambiguous) >= 4:
            raise MissingEndmemberError(
                "当前图含裸露地表/人工地表疑似像元，仅用光谱阈值无法可靠赋予这两类语义。"
                "请先复用已有提取方法生成同名 .seeds.tif 或 .endmembers.npz，"
                "也可提供五类 .prior.tif；函数调用仍只有图像路径。"
            )
        _warn("未提供类别先验：本次只采用保守的水体/绿色植被/雪冰规则；未检出不等于真实不存在。")
        masks = [water, vegetation, np.zeros(scene.shape, bool), np.zeros(scene.shape, bool), snow]
        evidence = [m & scene.valid for m in masks]
    else:
        masks, evidence = [], []
        for c in range(5):
            other = np.max(np.delete(support, c, axis=-1), axis=-1)
            evidence.append(scene.valid & (support[..., c] >= 0.50))
            masks.append(
                physical[c] & (support[..., c] >= _PRIOR_MIN)
                & ((support[..., c] - other) >= _PRIOR_MARGIN)
            )
    return [m & scene.valid for m in masks], evidence


def _extract_local_endmembers(scene: _Scene, support: np.ndarray | None) -> tuple[np.ndarray, np.ndarray]:
    """逐景自适应基线：每类筛选近纯内部像元，取靠近稳健中心的真实光谱。

    这是可替换的提取环节。你已有的五类提取方法可返回相同的 (U, ids)：
    U为(K,10)反射率，ids为(K,)且数值0~4。后面的FCLS和五通道封装无需修改。
    本基线不冒充端元束聚类/MESMA；缓存入口可复用更成熟的多端元提取结果。
    """
    masks, evidence = _candidate_masks(scene, support)
    valid_float = scene.valid.astype(float)
    local_weight = uniform_filter(valid_float, size=3, mode="constant", cval=0.0)
    variance = np.zeros(scene.shape, dtype=float)
    denominator = np.maximum(local_weight, 1e-12)
    for b in range(scene.cube.shape[-1]):
        v = np.where(scene.valid, scene.cube[..., b], 0.0)
        mean = uniform_filter(v, size=3, mode="constant") / denominator
        second = uniform_filter(v * v, size=3, mode="constant") / denominator
        variance += np.maximum(second - mean * mean, 0.0)
    local_std = np.sqrt(variance / scene.cube.shape[-1])
    rng = np.random.default_rng(_RANDOM_SEED)
    endmembers, class_ids, missing = [], [], []
    for class_id, mask in enumerate(masks):
        inside = binary_erosion(mask, structure=np.ones((3, 3), bool), iterations=_EROSION_ITERATIONS)
        good = inside & (local_weight >= 0.99) & (local_std <= _MAX_LOCAL_STD)
        coords = np.argwhere(good)
        if len(coords) < _MIN_SEED_PIXELS:
            if np.any(evidence[class_id]):
                missing.append(f"{CLASS_NAMES[class_id]}（近纯候选{len(coords)}个）")
            continue
        if len(coords) > _MAX_SEED_PIXELS:
            coords = coords[rng.choice(len(coords), _MAX_SEED_PIXELS, replace=False)]
        samples = scene.cube[coords[:, 0], coords[:, 1]]
        centre = np.median(samples, axis=0)
        distance = np.mean((samples - centre) ** 2, axis=1)
        endmembers.append(samples[int(np.argmin(distance))].copy())
        class_ids.append(class_id)
    if missing:
        raise MissingEndmemberError(
            "有类别存在证据但未取得足够近纯端元：" + ", ".join(missing)
            + "。请补充同景/同日邻域端元，不应将这些类别当作不存在直接填零。"
        )
    if not endmembers:
        raise MissingEndmemberError("本景没有提取到任何可靠端元；请检查掩膜、尺度或提供已验证端元。")
    return np.asarray(endmembers, dtype=float), np.asarray(class_ids, dtype=int)


def _get_endmembers(scene: _Scene) -> tuple[np.ndarray, np.ndarray]:
    """复用已有方法的统一适配点。优先读本景缓存，否则调用内置自适应提取。"""
    cache_path = _sidecar(scene.path, ".endmembers.npz")
    if cache_path.exists():
        with np.load(cache_path, allow_pickle=False) as cache:
            required = {"spectra", "class_names", "band_names", "units", "source_image"}
            if not required.issubset(cache.files):
                raise InputMetadataError(f"端元缓存必须包含 {sorted(required)}，禁止靠数组下标猜类别。")
            if str(cache["units"].item()) != "reflectance":
                raise InputMetadataError("缓存端元必须为 reflectance，与已转换影像使用相同单位。")
            if str(cache["source_image"].item()) != scene.path.name:
                raise InputMetadataError("缓存source_image与输入文件名不一致；不要复用另一个图块的端元。")
            band_names = tuple(_band_name(v) for v in cache["band_names"].tolist())
            if len(set(band_names)) != len(band_names) or not set(ANALYSIS_BANDS).issubset(band_names):
                raise InputMetadataError("端元缓存波段名重复或不完整。")
            raw = np.asarray(cache["spectra"], dtype=float)
            classes = cache["class_names"].tolist()
            if not isinstance(classes, list) or raw.ndim != 2 or raw.shape != (len(classes), len(band_names)):
                raise InputMetadataError("缓存要求spectra=(K,B)，class_names=(K,)，band_names=(B,)。")
            if any(c not in CLASS_NAMES for c in classes):
                raise InputMetadataError(f"未知端元类别；只能使用 {CLASS_NAMES}。")
            U = raw[:, [band_names.index(b) for b in ANALYSIS_BANDS]]
            ids = np.asarray([CLASS_NAMES.index(c) for c in classes], dtype=int)
    else:
        U, ids = _extract_local_endmembers(scene, _load_support(scene))
    if U.ndim != 2 or U.shape[1] != len(ANALYSIS_BANDS) or not (1 <= len(U) <= _MAX_ENDMEMBERS):
        raise InputMetadataError(f"本接口接受1~{_MAX_ENDMEMBERS}条端元；较大端元库应先精简或使用MESMA。")
    if not np.isfinite(U).all() or np.any(U < -0.25) or np.any(U > 2.0):
        raise InputMetadataError("端元包含非有限值/明显异常反射率。")
    for i, j in combinations(range(len(U)), 2):
        if ids[i] != ids[j] and np.linalg.norm(U[i] - U[j]) < 1e-6:
            raise MissingEndmemberError("不同类别具有几乎相同的端元光谱，无法辨别类别比例。")
    if len(U) > 1:
        singular = np.linalg.svd(U[1:] - U[0], compute_uv=False)
        if singular[-1] <= max(singular[0], 1e-12) * 1e-6:
            _warn("端元仿射分离度低，丰度可能不稳定/不唯一；低RMSE不能证明类别比例正确。")
    absent = [name for c, name in enumerate(CLASS_NAMES) if c not in ids]
    if absent:
        _warn("本次端元库未激活这些类别，其输出通道为0：" + ", ".join(absent)
              + "。这是当前模型的未检出状态，不是独立验证的绝对不存在。")
    return U, ids


def _prepare_faces(U: np.ndarray) -> list[tuple[np.ndarray, np.ndarray | None]]:
    """FCLS凸二次问题的支持集枚举：每个非空支持集上解等式约束最小二乘。

    若U_s以第一条为参考，则 a0=1-sum(z)，重建为 U_s[0]+z@(U_s[1:]-U_s[0])。
    只接收非负可行解，再从所有支持集中选择误差最小者。不是NNLS后归一化。
    仿射相关的面可跳过：其凸包中的点也能由更小的独立支持集表示。
    """
    faces = []
    for size in range(1, len(U) + 1):
        for indices in combinations(range(len(U)), size):
            idx = np.asarray(indices, dtype=int)
            if size == 1:
                faces.append((idx, None))
                continue
            D = U[idx[1:]] - U[idx[0]]
            singular = np.linalg.svd(D, compute_uv=False)
            if singular[-1] <= max(singular[0], 1e-12) * 1e-10:
                continue
            faces.append((idx, np.linalg.pinv(D, rcond=1e-10)))
    return faces


def _fcls_batch(X: np.ndarray, U: np.ndarray, faces: list | None = None) -> tuple[np.ndarray, np.ndarray]:
    """数值核心：(N,B)+(K,B) -> (N,K)丰度、(N,)光谱RMSE。

    支持集解已满足FCLS等式和非负可行性；下方max/归一化仅清除1e-9以内的
    浮点舍入残差，不是把任意非约束解事后裁剪为丰度。
    """
    X, U = np.asarray(X, float), np.asarray(U, float)
    if X.ndim != 2 or U.ndim != 2 or X.shape[1] != U.shape[1] or not len(U):
        raise ValueError("FCLS要求X=(N,B)、U=(K,B)、K>=1。")
    if not np.isfinite(X).all() or not np.isfinite(U).all():
        raise ValueError("FCLS只能接收有限值；无效像元必须先排除。")
    faces = _prepare_faces(U) if faces is None else faces
    best = np.zeros((len(X), len(U)), dtype=float)
    best_error = np.full(len(X), np.inf)
    for idx, inverse in faces:
        if len(idx) == 1:
            a = np.ones((len(X), 1), dtype=float)
        else:
            z = (X - U[idx[0]]) @ inverse
            a = np.column_stack((1.0 - z.sum(axis=1), z))
        feasible = np.all(a >= -_FEASIBILITY_TOL, axis=1)
        rows = np.flatnonzero(feasible)
        if not len(rows):
            continue
        aa = np.maximum(a[rows], 0.0)
        aa /= aa.sum(axis=1, keepdims=True)
        residual = X[rows] - aa @ U[idx]
        mse = np.mean(residual * residual, axis=1)
        improve = mse < best_error[rows]
        selected = rows[improve]
        best[selected] = 0.0
        best[np.ix_(selected, idx)] = aa[improve]
        best_error[selected] = mse[improve]
    if not np.isfinite(best_error).all() or not np.allclose(best.sum(axis=1), 1.0, atol=1e-7):
        raise UnmixingError("FCLS未得到有限可行解；不返回全零伪结果。")
    return best, np.sqrt(best_error)


def unmix_image(image_path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """读取一幅影像，自适应线性解混，返回五类丰度和独立误差图。

    Parameters
    ----------
    image_path : str | pathlib.Path
        唯一输入。已对齐的多波段Sentinel-2地表反射率TIFF路径，而非RGB预览图。
        支持必需的十个分析波段；波段位置优先由TIFF描述或同名JSON给出。
        不要求传入类别、端元、阈值、输出目录或模型实例。

    Returns
    -------
    abundance : numpy.ndarray
        float32, shape=(H,W,5)，通道顺序严格对应CLASS_NAMES。
        多条端元属于同类时先求端元丰度再相加，不改变输出通道顺序。
        未激活类别的整个通道为0；无效观测位置的所有通道也为0，必须结合error辨别。
    error : numpy.ndarray
        float32, shape=(H,W)。error[p]=sqrt(mean_b((x[p,b]-x_hat[p,b])**2))。
        单位为反射率；这里没有对波段进行单独标准化。
        无效观测为NaN；它是重建残差，不是丰度真值误差/概率置信度。

    Raises
    ------
    FileNotFoundError
        图像路径不存在。
    InputMetadataError
        波段/单位不明确，文件或缓存格式错误，伴随影像不对齐。
    MissingEndmemberError
        类别语义无法确定，或有存在证据但近纯端元不足。
    UnmixingError
        求解失败。

    Notes
    -----
    全图无有效观测时，返回全0丰度和全NaN误差，并发出警告；不推断地物缺失。
    正常像元的丰度之和约为1。绝不通过“异常捕获后全0”掩盖解混失败。
    不把高RMSE自动改为0，也不删掉其误差；由下游用经验证的阈值进一步筛选。
    返回的是原图网格上的线性拟合候选丰度，尚不是校准过的真实面积百分比产品。
    """
    if not isinstance(image_path, (str, Path)):
        raise TypeError("image_path 必须是 str 或 pathlib.Path。")
    path = Path(image_path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"图像不存在：{path}")
    try:
        scene = _read_scene(path)
    except UnmixingError:
        raise
    except (OSError, ValueError, TypeError, rasterio.errors.RasterioError) as exc:
        raise InputMetadataError(f"读取影像/元数据失败：{path.name}；{exc}") from exc
    h, w = scene.shape
    abundance = np.zeros((h, w, len(CLASS_NAMES)), dtype=np.float32)
    error = np.full((h, w), np.nan, dtype=np.float32)
    positions = np.flatnonzero(scene.valid.ravel())
    if not len(positions):
        _warn("全图没有有效观测：返回全0丰度和全NaN误差；不能解释为五类都不存在。")
        return abundance, error
    try:
        U, class_ids = _get_endmembers(scene)
    except UnmixingError:
        raise
    except (OSError, ValueError, TypeError, KeyError, rasterio.errors.RasterioError) as exc:
        raise InputMetadataError(f"读取或构建端元库失败：{path.name}；{exc}") from exc
    faces = _prepare_faces(U)
    image_flat = scene.cube.reshape(-1, len(ANALYSIS_BANDS))
    abundance_flat = abundance.reshape(-1, len(CLASS_NAMES))
    error_flat = error.ravel()
    for start in range(0, len(positions), _CHUNK_SIZE):
        pixels = positions[start:start + _CHUNK_SIZE]
        a, rmse = _fcls_batch(image_flat[pixels], U, faces)
        # class_ids可以不按顺序，且可以重复。逐端元相加保证同类多端元正确聚合。
        for j, class_id in enumerate(class_ids):
            abundance_flat[pixels, class_id] += a[:, j].astype(np.float32)
        error_flat[pixels] = rmse.astype(np.float32)
    return np.ascontiguousarray(abundance), np.ascontiguousarray(error)
