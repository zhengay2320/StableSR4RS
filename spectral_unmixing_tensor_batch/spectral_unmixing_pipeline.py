# -*- coding: utf-8 -*-
"""WorldStrat五类候选端元 → MESMA式组合FCLS → 状态、诊断与可视化。

将本文件、unmixing_core.py、run_unmixing.py放到原五个*_all_bands.py旁边即可。
原模块不被修改：EXTRACT/AUTO时复用其Config/CFG/main，EXISTING时严格检查旧输出。

重要区别：提取种子只用于构造端元库；解混覆盖全部质量允许的像元。
提取失败不等于类别不存在；0、NaN、条件性拟合和质量筛查输出分别保存。
默认保留未核查候选供探索，但它们不会自动变成“已验证端元”。
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict, fields
from pathlib import Path
from datetime import datetime, timezone
from importlib import util, metadata
from contextlib import redirect_stdout, redirect_stderr
import copy
import hashlib
import html
import json
import os
import platform
import sys
import traceback
import uuid
import warnings

import numpy as np
import pandas as pd
import rasterio
from rasterio.crs import CRS as RasterCRS
from scipy import ndimage as ndi
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm
from threadpoolctl import threadpool_limits

from unmixing_core import (SolverConfig, unmix_library, fixed_class_baseline,
                           evaluate_block, validate_solver)

CLASSES = ('vegetation', 'water', 'bare', 'snow', 'building')
CLASS_NAMES = {'vegetation': '绿色植被', 'water': '水体', 'bare': '裸土/砂地',
               'snow': '雪/冰候选', 'building': '建筑/人工表面候选'}
MODULE_NAMES = {c: f'{c}_candidates_all_bands' for c in CLASSES}
QUALITY_BITS = {
    1: 'numeric_invalid', 2: 'quality_excluded', 4: 'fit_not_acceptable',
    8: 'class_abundance_ambiguous', 16: 'scene_library_incomplete',
    32: 'selected_candidate_unreviewed', 64: 'very_dark_observation',
    128: 'provenance_unverified', 256: 'no_available_model',
}


@dataclass
class ClassSpec:
    # AUTO：有output_dir就读已有结果，否则调用本类脚本；不搜索任意旧文件夹。
    source: str = 'AUTO'  # AUTO / EXTRACT / EXISTING / SKIP
    output_dir: Path | None = None
    # unknown表示尚未确定；present表示用户有存在证据；absent须填写证据说明。
    scene_state: str = 'unknown'  # unknown / present / absent
    state_reason: str = ''
    # 仅在实际检查候选位置和类别含义后设为True。不是由提取成功自动设True。
    reviewed: bool = False
    reviewed_candidates: tuple = ()
    selected_candidates: tuple | None = None  # 指定本类CSV中的候选列名；None=自动选。
    max_candidates: int = 3
    # 覆盖原脚本里的阈值/屋顶先验等；默认保留用户原CFG/Config设置。
    extraction_overrides: dict = field(default_factory=dict)


@dataclass
class Config:
    input_path: Path = Path(r'E:\开源数据集\word_star\new_star\train\lr\Landcover-118405.tiff')
    script_dir: Path = field(default_factory=lambda: Path(__file__).resolve().parent)
    output_root: Path = field(default_factory=lambda: Path(__file__).resolve().parent / 'unmixing_results')
    run_name: str | None = None  # None创建时间戳+随机短码目录，永不覆盖旧实验。
    product_level: str = 'AUTO'
    band_map: dict | None = None
    data_confirmed: bool = False
    scale: float | dict | None = None
    offset: float | dict | None = None
    # 共同质量掩膜，不是端元种子掩膜；不能将水或雪当作全图无效类别排除。
    clear_mask_path: Path | None = None
    scl_path: Path | None = None
    scl_excluded_classes: tuple = (0, 1, 2, 3, 8, 9, 10)
    classes: dict = field(default_factory=lambda: {c: ClassSpec() for c in CLASSES})
    # 老模块输出没有输入内容哈希时，默认拒绝直接复用。
    # 确认其来自同一个、未修改过的输入后，可明确设True；仍检查波段、单位、网格、核心均值。
    allow_legacy_results: bool = False
    allow_unreviewed_candidates: bool = True  # 保留供条件性诊断，严格图仍不接受。
    continue_on_extractor_error: bool = False  # 程序/文件错误默认中止；真正候选不足可继续。
    within_class_duplicate_rmse: float = 1e-4
    cross_class_warning_rmse: float = 0.005
    solver: SolverConfig = field(default_factory=SolverConfig)
    run_baseline: bool = True
    dark_rms_threshold: float = 0.0  # 默认关闭；需要暗观测警示时明确设置。不能用暗=阴影误删真实水体。
    detection_abundance: float = 0.05  # 仅用于统计和显示“模型响应”，不改变原丰度。
    dominant_min: float = 0.60
    show_figures: bool = False
    make_plots: bool = True
    make_html: bool = True
    save_pixel_csv: bool = True
    save_example_pixels: int = 3
    blas_threads: int = 1


def _clean_json(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return _clean_json(value.tolist())
    if isinstance(value, np.generic):
        return _clean_json(value.item())
    if isinstance(value, dict):
        return {str(k): _clean_json(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_clean_json(v) for v in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def save_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_clean_json(value), ensure_ascii=False, indent=2,
                               allow_nan=False), encoding='utf-8')


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding='utf-8-sig'))


def save_csv(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    table = rows if isinstance(rows, pd.DataFrame) else pd.DataFrame(rows)
    table.to_csv(path, index=False, encoding='utf-8-sig')


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def normalized_path(path) -> str:
    # Windows同盘下大小写不敏感；不使用文件名相同作为“同景”的证明。
    return os.path.normcase(str(Path(path).expanduser().resolve()))


def load_extractor(script_dir: Path, class_name: str):
    """以绝对路径加载原脚本。插入sys.modules，使dataclass在动态导入时正常解析。"""
    path = Path(script_dir) / (MODULE_NAMES[class_name] + '.py')
    if not path.is_file():
        raise FileNotFoundError(f'缺少原脚本：{path}。请把五个完整全波段文件放到script_dir。')
    name = '_unmix_legacy_' + class_name + '_' + sha256_file(path)[:12]
    if name in sys.modules:
        return sys.modules[name]
    spec = util.spec_from_file_location(name, path)
    module = util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(name, None)
        raise
    for required in ('Config', 'main', 'resolve_worldstrat_bands', 'calibration_vector'):
        if not hasattr(module, required):
            raise ValueError(f'{path.name}缺少接口{required}，不是本任务对应的完整全波段版本。')
    return module


def validate_config(cfg: Config) -> None:
    cfg.input_path, cfg.script_dir = Path(cfg.input_path), Path(cfg.script_dir)
    cfg.output_root = Path(cfg.output_root)
    if not cfg.input_path.is_file():
        raise FileNotFoundError(f'找不到输入TIFF：{cfg.input_path}')
    if set(cfg.classes) != set(CLASSES):
        raise ValueError(f'classes需要且只能含{CLASSES}。')
    for c in CLASSES:
        s = cfg.classes[c]
        if isinstance(s, dict):
            s = cfg.classes[c] = ClassSpec(**s)
        s.source = s.source.upper()
        if s.source not in ('AUTO', 'EXTRACT', 'EXISTING', 'SKIP'):
            raise ValueError(f'{c}的source无效。')
        if s.scene_state not in ('unknown', 'present', 'absent'):
            raise ValueError(f'{c}的scene_state无效。')
        if s.scene_state == 'absent' and not s.state_reason.strip():
            raise ValueError(f'{c}设为absent时必须填写state_reason；提取失败不是缺失证据。')
        if s.max_candidates < 1:
            raise ValueError('max_candidates必须>=1。')
        if s.source == 'EXISTING' and s.output_dir is None:
            raise ValueError(f'{c}的EXISTING模式必须指定output_dir。')
    for name in ('within_class_duplicate_rmse', 'cross_class_warning_rmse', 'dark_rms_threshold'):
        if not np.isfinite(getattr(cfg, name)) or getattr(cfg, name) < 0:
            raise ValueError(f'{name}应为非负有限值。')
    if not 0 <= cfg.detection_abundance <= 1 or not 0 <= cfg.dominant_min <= 1:
        raise ValueError('丰度显示/统计门槛必须在[0,1]。')
    if cfg.run_name and (Path(cfg.run_name).name != cfg.run_name or cfg.run_name in ('.', '..')):
        raise ValueError('run_name只能是目录名，不能包含路径。')
    if cfg.save_example_pixels < 0 or cfg.blas_threads < 1:
        raise ValueError('示例数量或线程数量不合法。')


def prepare_run(cfg: Config) -> Path:
    """每次运行单独目录，原结果始终保留；不递归删除用户目录。"""
    name = cfg.run_name or datetime.now().strftime('%Y%m%d_%H%M%S_') + uuid.uuid4().hex[:6]
    root = cfg.output_root / name
    if root.exists():
        raise FileExistsError(f'实验目录已存在：{root}。请用新的run_name，防止旧结果混入。')
    root.mkdir(parents=True)
    env = dict(python=platform.python_version(), platform=platform.platform())
    for p in ('numpy', 'pandas', 'rasterio', 'scipy', 'matplotlib', 'scikit-image',
              'scikit-learn', 'pyproj', 'threadpoolctl'):
        try:
            env[p] = metadata.version(p)
        except metadata.PackageNotFoundError:
            env[p] = 'not_installed'
    save_json(root / '00_config.json', asdict(cfg))
    save_json(root / '00_environment.json', env)
    return root


def write_raster(path: Path, array: np.ndarray, scene: dict,
                 descriptions=None, dtype='float32', nodata=np.nan, **tags) -> None:
    """保留CRS/变换；浮点NoData使用NaN，让“不可估计”与0丰度保持区别。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    array = np.asarray(array)
    if array.ndim == 2:
        array = array[None]
    if array.ndim != 3 or array.shape[1:] != scene['shape']:
        raise ValueError('输出数组网格不一致。')
    profile = dict(driver='GTiff', height=array.shape[1], width=array.shape[2],
                   count=array.shape[0], dtype=dtype, transform=scene['transform'],
                   crs=scene['crs'], nodata=nodata, compress='deflate',
                   predictor=3 if np.dtype(dtype).kind == 'f' else 2)
    with rasterio.open(path, 'w', **profile) as dst:
        dst.write(array.astype(dtype))
        if descriptions:
            if len(descriptions) != array.shape[0]:
                raise ValueError('输出通道说明数量不一致。')
            dst.descriptions = tuple(descriptions)
        dst.update_tags(**{k: str(v) for k, v in tags.items()})


def read_aligned_layer(path: Path, scene: dict, binary=False):
    with rasterio.open(path) as src:
        if (src.count != 1 or src.shape != scene['shape'] or src.crs != scene['crs']
                or not src.transform.almost_equals(scene['transform'])):
            raise ValueError(f'辅助图层必须为同网格单通道：{path}')
        data = src.read(1).astype(float)
        known = (src.read_masks(1) > 0) & np.isfinite(data)
        if binary and not np.isin(data[known], [0, 1]).all():
            raise ValueError(f'{path}不是0/1掩膜；不能直接传分类概率或SCL编号。')
        return data, known


def read_scene(cfg: Config, root: Path) -> dict:
    """复用原building模块的完整波段解析和校准，不复用其“只适合建筑”的有效掩膜。"""
    module = load_extractor(cfg.script_dir, 'building')
    legacy = module.Config()
    legacy.product_level, legacy.band_map = cfg.product_level, cfg.band_map
    legacy.feature_bands = 'ALL'
    with rasterio.open(cfg.input_path) as src:
        resolved = module.resolve_worldstrat_bands(src.count, src.descriptions, legacy)
        names = list(resolved['bands'])
        indexes = [resolved['mapping'][b] for b in names]
        raw = src.read(indexes).astype(np.float64)
        numeric = np.all(src.read_masks(indexes) > 0, axis=0) & np.all(np.isfinite(raw), axis=0)
        scales = module.calibration_vector(cfg.scale, src.scales, names, indexes)
        offsets = module.calibration_vector(cfg.offset, src.offsets, names, indexes)
        if not np.isfinite(scales).all() or np.any(scales <= 0) or not np.isfinite(offsets).all():
            raise ValueError('scale必须为有限正数，offset必须有限。')
        scene = dict(shape=src.shape, transform=src.transform, crs=src.crs, bands=names,
                     level=resolved['level'], indexes=indexes, scales=scales, offsets=offsets)
        scene['metadata'] = dict(path=str(cfg.input_path.resolve()), count=src.count,
                                 width=src.width, height=src.height, crs=str(src.crs),
                                 transform=list(src.transform), descriptions=list(src.descriptions))
    save_json(root / '00_scene_metadata.json', scene['metadata'])
    mapping = pd.DataFrame(dict(band=names, channel=indexes, scale=scales, offset=offsets))
    save_csv(root / '00_band_mapping.csv', mapping)
    print('\n完整波段与共同校准：\n' + mapping.to_string(index=False))
    if not cfg.data_confirmed:
        scene['inspection_only'] = True
        return scene
    cube = raw * scales[:, None, None] + offsets[:, None, None]
    numeric &= np.all(np.isfinite(cube), axis=0)
    cube[:, ~numeric] = np.nan
    quality = np.ones(scene['shape'], bool)
    if cfg.clear_mask_path:
        q, known = read_aligned_layer(Path(cfg.clear_mask_path), scene, binary=True)
        quality &= known & (q == 1)
    if cfg.scl_path:
        scl, known = read_aligned_layer(Path(cfg.scl_path), scene)
        if not np.isin(scl[known], np.arange(12)).all():
            raise ValueError('SCL必须是0—11的原始类别编码。')
        quality &= known & ~np.isin(scl, cfg.scl_excluded_classes)
    if cfg.clear_mask_path is None and cfg.scl_path is None:
        warnings.warn('没有共同质量掩膜：数值有效不表示已去云/阴影。水体和雪地不会自动从解混区域删除。')
    valid = numeric & quality
    if not valid.any():
        raise ValueError('没有全部波段共同有效且质量允许的像元。')
    scene.update(cube=cube, numeric_valid=numeric, quality_allowed=quality, valid=valid,
                 input_sha256=sha256_file(cfg.input_path), input_path=cfg.input_path.resolve())
    # 全景科学计算仅以valid为域，绝不取五种提取seed掩膜的交集或并集。
    quality_path = root / '00_common_quality_mask.tif'
    write_raster(quality_path, valid, scene, ['allowed_for_unmixing'], 'uint8', None)
    scene['common_quality_path'] = quality_path
    rgb = np.moveaxis(cube[[names.index(b) for b in ('B4', 'B3', 'B2')]], 0, -1)
    low, high = np.quantile(rgb[valid].ravel(), [.02, .98])
    if high - low < 1e-12:
        high = low + 1e-12
    scene['rgb_stretch'] = dict(low=float(low), high=float(high), shared_between_original_reconstruction=True)
    scene['rgb'] = np.nan_to_num(np.clip((rgb - low)/(high - low), 0, 1))
    save_json(root / '00_rgb_stretch.json', scene['rgb_stretch'])
    save_json(root / '00_input_fingerprint.json', dict(input_path=str(scene['input_path']),
              sha256=scene['input_sha256'], bands=names, level=scene['level'], scales=scales, offsets=offsets))
    return scene


def _candidate_shortage(module, error: Exception) -> bool:
    """兼容早期vegetation/water用ValueError表达不足的接口，但不吞掉任意程序错误。"""
    if hasattr(module, 'CandidateUnavailable') and isinstance(error, module.CandidateUnavailable):
        return True
    text = str(error)
    return isinstance(error, (ValueError, RuntimeError)) and (
        '像元不足' in text or '种子不足' in text or '候选不足' in text
        or ('个像元' in text and '少于' in text)
        or '没有满足' in text and ('聚类' in text or '簇' in text)
        or '不能可靠提取候选' in text
        or '种子数量不足以满足最小簇大小' in text
        or '连K=1都没有足够支持' in text)


def run_extractor(cfg: Config, class_name: str, spec: ClassSpec,
                  scene: dict, root: Path) -> tuple[Path | None, str | None]:
    """复用原CFG中的调参；全局路径、反射率校准和全波段要求由主控统一。
    屋顶先验等旧CFG设置保留。不会偷偷改变建筑提取方法或自动降低纯净度标准。
    """
    module = load_extractor(cfg.script_dir, class_name)
    legacy = copy.deepcopy(getattr(module, 'CFG', module.Config()))
    allowed = {f.name for f in fields(legacy)}
    protected = {'input_path', 'output_dir', 'data_confirmed', 'feature_bands',
                 'scale', 'offset', 'product_level', 'band_map', 'show_figures'}
    bad = set(spec.extraction_overrides) - allowed
    if bad or set(spec.extraction_overrides) & protected:
        raise ValueError(f'{class_name}提取覆盖参数无效或试图覆盖全局保护参数：{bad or protected & set(spec.extraction_overrides)}')
    for key, value in spec.extraction_overrides.items():
        setattr(legacy, key, value)
    legacy.input_path = cfg.input_path.resolve()
    legacy.output_dir = root / 'extraction' / class_name
    legacy.output_dir.mkdir(parents=True)
    legacy.data_confirmed, legacy.feature_bands, legacy.show_figures = True, 'ALL', False
    legacy.product_level, legacy.band_map = scene['level'], {b: i for b, i in zip(scene['bands'], scene['indexes'])}
    legacy.scale = dict(zip(scene['bands'], scene['scales'].tolist()))
    legacy.offset = dict(zip(scene['bands'], scene['offsets'].tolist()))
    # 保留原类自己的质量支持，但它不会反过来限制整图解混。
    own_clear = getattr(legacy, 'clear_mask_path', None)
    if own_clear:
        q, known = read_aligned_layer(Path(own_clear), scene, binary=True)
        own_path = legacy.output_dir / 'adapter_combined_quality.tif'
        write_raster(own_path, scene['valid'] & known & (q == 1), scene,
                     ['allowed_for_this_extractor'], 'uint8', None)
        legacy.clear_mask_path = own_path
    else:
        legacy.clear_mask_path = scene['common_quality_path']
    if hasattr(legacy, 'scl_path') and cfg.scl_path is not None:
        legacy.scl_path = Path(cfg.scl_path)
    save_json(legacy.output_dir / 'adapter_actual_config.json', vars(legacy))
    log_path = legacy.output_dir / 'adapter_console.log'
    print(f'[{class_name}] 调用原模块；过程日志 → {log_path.name}')
    # 旧版植被在全图NDVI都不可计算时会对空数组求ptp；这里显式报告证据不足。
    if class_name == 'vegetation':
        red, nir = (scene['cube'][scene['bands'].index(b)] for b in ('B4', 'B8'))
        evidence = scene['valid'] & (red >= 0) & (nir >= 0) & ((red + nir) >= legacy.min_red_nir_sum)
        if not evidence.any():
            message = 'NDVI没有有效计算位置；未知是否存在植被，不运行空数组统计。'
            save_json(legacy.output_dir / 'run_status.json', dict(state='insufficient_candidates',
                stage='adapter_ndvi_precheck', input_path=str(legacy.input_path), error=message))
            return None, message
    try:
        with log_path.open('w', encoding='utf-8') as log, redirect_stdout(log), redirect_stderr(log):
            with threadpool_limits(limits=cfg.blas_threads):
                data = module.main(legacy)
        if data is None or 'E' not in data:
            raise RuntimeError('提取main未返回完整端元；检查所用脚本版本和运行状态。')
    except Exception as error:
        if _candidate_shortage(module, error):
            return None, str(error)
        with log_path.open('a', encoding='utf-8') as log:
            log.write('\nADAPTER TRACEBACK\n' + traceback.format_exc())
        if cfg.continue_on_extractor_error:
            return None, 'EXTRACTOR_ERROR_NOT_ABSENCE: ' + repr(error)
        raise RuntimeError(f'{class_name}模块程序/输入错误，不能当成地物不存在。日志：{log_path}') from error
    # 包装器为新结果增加输入内容哈希和输出文件哈希，避免下次读到过期结果。
    files = ['12_candidate_spectra.csv', '12_candidate_summary.csv', '12_candidates.npz',
             '12_candidate_core_std.csv', '01_band_mapping.csv', '00_metadata.json']
    manifest = dict(schema='worldstrat_candidate_binding_v1', input_sha256=scene['input_sha256'],
                    input_path=str(scene['input_path']), class_name=class_name,
                    module_sha256=sha256_file(cfg.script_dir / (MODULE_NAMES[class_name] + '.py')),
                    hashes={name: sha256_file(legacy.output_dir/name) for name in files if (legacy.output_dir/name).is_file()})
    save_json(legacy.output_dir / 'unmix_source.json', manifest)
    return legacy.output_dir, None


def load_candidate_directory(directory: Path, class_name: str, cfg: Config,
                             scene: dict, spec: ClassSpec) -> list:
    """只接收本流程定义的成功输出；不将孤立CSV默认为本景端元。
    校验波段名称、顺序、尺度、网格、完成状态、NPZ一致性、来源核心均值。
    """
    directory = Path(directory)
    status_path = directory / 'run_status.json'
    if not status_path.is_file():
        raise ValueError(f'{directory}缺少run_status.json，无法排除旧文件/半成品。')
    status = load_json(status_path)
    if status.get('state') != 'completed':
        raise ValueError(f'已有{class_name}输出不是completed：{status.get("state")}。不得读取残留端元。')
    required = ['12_candidate_spectra.csv', '12_candidate_summary.csv', '12_candidates.npz',
                '01_band_mapping.csv', '00_metadata.json']
    for name in required:
        if not (directory/name).is_file():
            raise FileNotFoundError(f'缺少候选配套文件：{directory/name}')
    meta = load_json(directory/'00_metadata.json')
    same_path = normalized_path(meta['path']) == normalized_path(cfg.input_path)
    binding = directory/'unmix_source.json'
    if binding.is_file():
        manifest = load_json(binding)
        if manifest.get('input_sha256') != scene['input_sha256'] or manifest.get('class_name') != class_name:
            raise ValueError(f'{class_name}输出的输入内容哈希/类别与当前影像不符。')
        for name, expected in manifest['hashes'].items():
            if not (directory/name).is_file() or sha256_file(directory/name) != expected:
                raise ValueError(f'{class_name}已绑定的输出文件发生变更：{name}')
        provenance = 'content_hash_bound'
    else:
        if not cfg.allow_legacy_results:
            raise ValueError('旧结果没有输入哈希。确认来自同一未修改TIFF后，设置allow_legacy_results=True；'
                             '或使用EXTRACT重新提取以生成输入绑定。')
        if not same_path or normalized_path(status.get('input_path', '')) != normalized_path(cfg.input_path):
            raise ValueError(f'{class_name}旧结果输入路径不符，不能仅按文件名相同来认定同景。')
        provenance = 'legacy_path_metadata_user_acknowledged'
    if (meta['height'], meta['width']) != scene['shape'] or meta['count'] != len(scene['bands']):
        raise ValueError(f'{class_name}候选的影像形状/波段数不符。')
    if not np.allclose(meta['transform'], list(scene['transform']), atol=1e-12, rtol=0):
        raise ValueError(f'{class_name}候选网格变换不符。')
    crs_old = None if meta['crs'] in (None, 'None', '') else RasterCRS.from_user_input(meta['crs'])
    if crs_old != scene['crs']:
        raise ValueError(f'{class_name}候选CRS不符。')
    table = pd.read_csv(directory/'12_candidate_spectra.csv')
    if 'band' not in table or table['band'].duplicated().any():
        raise ValueError('端元CSV需要唯一band列，不能把缩放特征表当端元。')
    if set(table['band']) != set(scene['bands']):
        raise ValueError(f'{class_name}候选必须包含本景全部12/13波段，不能补0或丢波段。')
    columns = list(table.columns[1:])
    if len(columns) == 0 or table.columns[0] != 'band' or len(set(columns)) != len(columns):
        raise ValueError('端元CSV列格式错误。')
    table = table.set_index('band').loc[scene['bands']]
    E = table.to_numpy(dtype=float)
    if not np.isfinite(E).all():
        raise ValueError(f'{class_name}端元含非有限值。')
    mapping = pd.read_csv(directory/'01_band_mapping.csv').set_index('band').loc[scene['bands']]
    if not np.array_equal(mapping['channel'].to_numpy(), scene['indexes']):
        raise ValueError('端元与影像通道位置不一致。')
    if not np.allclose(mapping['scale'], scene['scales'], atol=1e-12, rtol=1e-9) or not np.allclose(mapping['offset'], scene['offsets'], atol=1e-12, rtol=1e-9):
        raise ValueError('端元和影像的scale/offset不一致；不能分别归一化后混用。')
    summary = pd.read_csv(directory/'12_candidate_summary.csv')
    if 'candidate' not in summary or summary['candidate'].duplicated().any() or set(columns) != set(summary['candidate']):
        raise ValueError('候选光谱列与来源汇总表不一致。')
    summary = summary.set_index('candidate').loc[columns]
    with np.load(directory/'12_candidates.npz', allow_pickle=False) as pack:
        order = [list(pack['bands'].astype(str)).index(b) for b in scene['bands']]
        if len(order) != len(pack['bands']) or not np.allclose(pack['E'][order], E, atol=1e-8, rtol=1e-6):
            raise ValueError('CSV与NPZ端元不一致。')
        if str(pack['product_level']) != scene['level'] or pack['seeds'].shape != scene['shape']:
            raise ValueError('NPZ产品层级或来源网格不一致。')
        # 用当前原始反射率重算旧输出中记录的核心均值，防止单位或来源被误替换。
        if 'cluster_map' in pack and 'used_for_mean' in pack:
            cm, used = pack['cluster_map'], pack['used_for_mean']
            for j in range(len(columns)):
                core = (cm == j+1) & (used > 0)
                if not core.any() or not np.allclose(scene['cube'][:, core].mean(axis=1), E[:, j], atol=2e-7, rtol=1e-5):
                    raise ValueError(f'{class_name}/{columns[j]}与当前影像记录核心像元的均值不符。')
        core_std = pack['core_std'][order] if 'core_std' in pack else np.full_like(E, np.nan)
    if spec.selected_candidates is not None and not set(spec.selected_candidates).issubset(columns):
        raise ValueError(f'{class_name} selected_candidates中有不存在的列名。实际：{columns}')
    if not set(spec.reviewed_candidates).issubset(columns):
        raise ValueError(f'{class_name} reviewed_candidates中有不存在的列名。')
    out = []
    for j, name in enumerate(columns):
        if spec.selected_candidates is not None and name not in spec.selected_candidates:
            continue
        row = summary.loc[name].to_dict()
        r, c = int(row['row']), int(row['col'])
        if not (0 <= r < scene['shape'][0] and 0 <= c < scene['shape'][1]):
            raise ValueError('候选位置超出当前影像。')
        reviewed = bool(spec.reviewed or name in spec.reviewed_candidates)
        if not reviewed and not cfg.allow_unreviewed_candidates:
            continue
        out.append(dict(id=f'{class_name}:{name}', class_name=class_name, class_id=CLASSES.index(class_name),
                        name=name, spectrum=E[:, j], core_std=core_std[:, j], reviewed=reviewed,
                        provenance=provenance, source_dir=str(directory), source_metadata=row,
                        row=r, col=c, averaged_pixels=int(row.get('averaged_pixels', 0)),
                        cluster_pixels=int(row.get('cluster_pixels', 0)),
                        semantic_status=row.get('semantic_status', 'candidate_source_not_verified')))
    return out


def select_representative_candidates(candidates: list, spec: ClassSpec,
                                     duplicate_rmse: float):
    """类内去重后保留光谱多样性：来源支持较好者作起点，之后最远点选样。
    这是计算预算内的代表性抽样，不是纯度优化；不同类别之间绝不自动去重。
    """
    candidates = sorted(candidates, key=lambda c: (-int(c['reviewed']), -c['averaged_pixels'], c['id']))
    unique, excluded = [], []
    for cand in candidates:
        duplicate = next((u for u in unique if np.sqrt(np.mean((cand['spectrum']-u['spectrum'])**2)) <= duplicate_rmse), None)
        if duplicate:
            excluded.append(dict(candidate=cand['id'], reason='within_class_duplicate', kept=duplicate['id']))
        else:
            unique.append(cand)
    if len(unique) <= spec.max_candidates:
        return unique, excluded
    selected = [unique.pop(0)]
    while unique and len(selected) < spec.max_candidates:
        distances = [min(np.sqrt(np.mean((u['spectrum']-s['spectrum'])**2)) for s in selected) for u in unique]
        selected.append(unique.pop(int(np.argmax(distances))))
    excluded.extend(dict(candidate=u['id'], reason='explicit_candidate_budget', kept=None) for u in unique)
    return selected, excluded


def collect_endmembers(cfg: Config, scene: dict, root: Path) -> dict:
    """构建固定五类状态表，只有明确absent才是0；失败、跳过、缺失都是未知。"""
    candidates, states, rejected = [], [], []
    for name in CLASSES:
        spec = cfg.classes[name]
        state = dict(class_name=name, label=CLASS_NAMES[name], asserted_scene_state=spec.scene_state,
                     assertion_reason=spec.state_reason, extraction_state='not_attempted',
                     status='unavailable', candidates=0, reviewed_candidates=0, reason='')
        if spec.scene_state == 'absent':
            state.update(status='confirmed_absent_by_user', extraction_state='skipped_confirmed_absent', reason=spec.state_reason)
            states.append(state)
            print(f'[{name}] 按明确场景声明不建模：{spec.state_reason}')
            continue
        if spec.source == 'SKIP':
            state.update(extraction_state='skipped_unknown', reason='用户跳过；不能解释为不存在。')
            states.append(state)
            continue
        read_existing = spec.source == 'EXISTING' or (spec.source == 'AUTO' and spec.output_dir is not None)
        if read_existing:
            directory = Path(spec.output_dir)
            # 有失败状态时，明确返回缺失端元，不读取同目录旧的成功光谱文件。
            sp = directory/'run_status.json'
            if sp.is_file() and load_json(sp).get('state') != 'completed':
                old = load_json(sp)
                state.update(extraction_state='existing_not_completed',
                             reason=str(old.get('error', old.get('state'))), source_dir=str(directory))
                states.append(state)
                continue
            state['extraction_state'] = 'loaded_existing'
        else:
            directory, error = run_extractor(cfg, name, spec, scene, root)
            if error:
                state.update(extraction_state='insufficient_candidates' if not error.startswith('EXTRACTOR_ERROR') else 'extractor_error', reason=error)
                states.append(state)
                print(f'[{name}] 无可用端元：{error[:140]}')
                save_csv(root/'01_class_status.csv', states)
                continue
            state['extraction_state'] = 'completed_now'
        items = load_candidate_directory(directory, name, cfg, scene, spec)
        items, removed = select_representative_candidates(items, spec, cfg.within_class_duplicate_rmse)
        rejected.extend(removed)
        state.update(candidates=len(items), reviewed_candidates=sum(i['reviewed'] for i in items),
                     source_dir=str(directory))
        if items:
            state['status'] = 'available_reviewed' if all(i['reviewed'] for i in items) else 'available_unreviewed'
            state['reason'] = '端元可计算；类别是否存在不由提取成功单独证明。'
        else:
            state['reason'] = '没有通过端元库检查/人工选择/审阅策略的候选。'
        states.append(state)
        candidates.extend(items)
        print(f'[{name}] {state["status"]}；入库{len(items)}条')
    E = np.column_stack([c['spectrum'] for c in candidates]) if candidates else np.zeros((len(scene['bands']), 0))
    class_ids = np.array([c['class_id'] for c in candidates], dtype=int)
    # 不能把类内去重逻辑用于跨类：跨类近似光谱要作为冲突报告。
    conflicts = []
    for i, a in enumerate(candidates):
        for j in range(i+1, len(candidates)):
            b = candidates[j]
            if a['class_id'] == b['class_id']:
                continue
            rmse = float(np.sqrt(np.mean((a['spectrum']-b['spectrum'])**2)))
            denom = np.linalg.norm(a['spectrum'])*np.linalg.norm(b['spectrum'])
            angle = float(np.degrees(np.arccos(np.clip(a['spectrum']@b['spectrum']/denom, -1, 1)))) if denom else None
            conflicts.append(dict(candidate_1=a['id'], candidate_2=b['id'], rmse=rmse,
                                  spectral_angle_deg=angle, close_spectra_warning=rmse<=cfg.cross_class_warning_rmse))
    save_csv(root/'01_class_status.csv', states)
    save_json(root/'01_class_status.json', states)
    save_csv(root/'01_candidate_exclusions.csv', pd.DataFrame(rejected, columns=['candidate','reason','kept']))
    save_csv(root/'01_cross_class_similarity.csv', pd.DataFrame(conflicts, columns=['candidate_1','candidate_2','rmse','spectral_angle_deg','close_spectra_warning']))
    library_table = pd.DataFrame(E, columns=[c['id'] for c in candidates])
    library_table.insert(0, 'band', scene['bands'])
    save_csv(root/'01_endmember_library.csv', library_table)
    manifest = [{k:v for k,v in c.items() if k not in ('spectrum', 'core_std')} for c in candidates]
    save_json(root/'01_endmember_metadata.json', manifest)
    incomplete = [s['class_name'] for s in states if s['status'] == 'unavailable']
    save_json(root/'01_library_warning.json', dict(unresolved_classes=incomplete,
        missing_class_not_absence=True, all_abundances_conditional_on_library=True,
        no_automatic_library_completion=True))
    return dict(E=E, class_ids=class_ids, candidates=candidates, states=states,
                unresolved_classes=incomplete, conflicts=conflicts)


def scatter_to_image(values: np.ndarray, scene: dict, fill=np.nan, dtype=float) -> np.ndarray:
    """将所有质量有效像元上的[N]或[N,C]结果还原到原始网格，不插值不平滑。"""
    values = np.asarray(values)
    valid = scene['valid']
    if values.shape[0] != int(valid.sum()):
        raise ValueError('像元结果数量与解混有效域不一致。')
    if values.ndim == 1:
        image = np.full(scene['shape'], fill, dtype=dtype)
        image[valid] = values
    elif values.ndim == 2:
        image = np.full((values.shape[1], *scene['shape']), fill, dtype=dtype)
        image[:, valid] = values.T
    else:
        raise ValueError('只支持[N]或[N,C]。')
    return image


def apply_class_states(A: np.ndarray, states: list) -> np.ndarray:
    """将数值求解器内部“缺失列的0”替换为语义上的NaN；明确不存在才保持0。"""
    A = np.array(A, dtype=float, copy=True)
    for c, state in enumerate(states):
        if state['status'] == 'unavailable':
            A[:, c] = np.nan
        elif state['status'] == 'confirmed_absent_by_user':
            A[:, c] = 0.0
    return A


def empty_model_result(Y: np.ndarray, n_classes=5) -> dict:
    n, L = Y.shape
    return dict(abundance=np.zeros((n,n_classes)), candidate_abundance=np.zeros((n,0)),
                model_id=np.full(n,-1,np.int32), model_size=np.zeros(n,np.uint8),
                weighted_mse=np.full(n,np.nan), best_weighted_mse=np.full(n,np.nan),
                rmse=np.full(n,np.nan), fit_ok=np.zeros(n,bool),
                abundance_min=np.full((n,n_classes),np.nan), abundance_max=np.full((n,n_classes),np.nan),
                uncertainty=np.full((n,n_classes),np.nan), drop_class_delta=np.full((n,n_classes),np.nan),
                near_model_count=np.zeros(n,np.int32), models=[], rejected_models=[],
                weights=np.ones(L), model_selected_count=np.zeros(0,int), model_feasible_count=np.zeros(0,int),
                reconstruction=np.full_like(Y,np.nan), residual=np.full_like(Y,np.nan))


def solve_scene(cfg: Config, scene: dict, library: dict, root: Path) -> dict:
    Y = scene['cube'][:, scene['valid']].T
    if library['E'].shape[1] == 0:
        print('没有任何可用端元：仍输出类别状态与NaN诊断图，不虚构解混。')
        numerical = empty_model_result(Y)
    else:
        def progress(done, total):
            if done == total or done % max(cfg.solver.block_pixels * 5, 1) == 0:
                print(f'解混像元：{done}/{total}')
        numerical = unmix_library(Y, library['E'], library['class_ids'], cfg.solver, 5, progress)
    model_table = []
    for model in numerical['models']:
        model_table.append(dict(model_id=model.model_id, n_endmembers=model.size,
                                candidate_ids=' | '.join(library['candidates'][j]['id'] for j in model.indices),
                                class_names=' | '.join(CLASSES[c] for c in model.classes),
                                condition=model.condition,
                                selected_pixels=int(numerical['model_selected_count'][model.model_id]),
                                feasible_pixels=int(numerical['model_feasible_count'][model.model_id])))
    save_csv(root/'02_model_library.csv', pd.DataFrame(model_table, columns=[
        'model_id','n_endmembers','candidate_ids','class_names','condition','selected_pixels','feasible_pixels']))
    save_json(root/'02_rejected_models.json', numerical['rejected_models'])
    save_json(root/'02_solver_summary.json', dict(
        allowed_models=len(numerical['models']), rejected_models=len(numerical['rejected_models']),
        solver='exact_active_face_enumeration', model_set_closed_under_subsets=True,
        weights=numerical['weights'], settings=asdict(cfg.solver),
        note='单个面只计算一次；完整模型的边界FCLS解由库中的较小面覆盖。不做丰度事后大幅截断。'))
    # 默认保存固定每类一候选的对照。对照和主方法使用同一反射率和共同权重。
    if cfg.run_baseline and library['E'].shape[1]:
        numerical['baseline'] = fixed_class_baseline(Y, library['E'], library['class_ids'], cfg.solver)
    n = len(Y)
    flags = np.zeros(n, np.uint16)
    flags[~numerical['fit_ok']] |= 4
    # 近优模型丰度范围只表示对模型选择的敏感性，不是校准过的概率。
    span = numerical['uncertainty']
    max_span = np.max(np.where(np.isfinite(span), span, 0), axis=1)
    flags[max_span > cfg.solver.abundance_uncertainty_max] |= 8
    if library['unresolved_classes']:
        flags |= 16  # 缺失类的位置未知，因此对全图保守标记，而非乱猜只影响某片区域。
    unreviewed = np.asarray([not c['reviewed'] for c in library['candidates']])
    if unreviewed.any():
        used_unreviewed = (numerical['candidate_abundance'][:, unreviewed] > 1e-8).any(axis=1)
        flags[used_unreviewed] |= 32
    if cfg.dark_rms_threshold > 0:
        flags[np.sqrt(np.mean(Y**2,axis=1)) < cfg.dark_rms_threshold] |= 64
    if not numerical['models']:
        flags |= 256
    screened_ok = flags == 0
    raw_A = apply_class_states(numerical['abundance'], library['states'])
    conditional_A = raw_A.copy()
    conditional_A[~numerical['fit_ok']] = np.nan
    screened_A = raw_A.copy()
    screened_A[~screened_ok] = np.nan
    # 未知类的模型敏感性同样是NaN，而不是“零不确定性”。
    for c, state in enumerate(library['states']):
        if state['status'] == 'unavailable':
            for key in ('uncertainty', 'abundance_min', 'abundance_max', 'drop_class_delta'):
                numerical[key][:,c] = np.nan
    numerical.update(raw_abundance=raw_A, conditional_abundance=conditional_A,
                     screened_abundance=screened_A, quality_flags=flags,
                     screened_ok=screened_ok, max_class_span=max_span,
                     Y=Y, library=library, scene=scene, cfg=cfg, root=root)
    return numerical


def pixel_areas_m2(scene: dict) -> tuple[np.ndarray, str]:
    """像元角点变换至经纬度，再计算WGS84椭球四边形面积。
    不把EPSG:4326的度²当m²；四角测地线是小栅格下的面积近似。
    无CRS时返回NaN，并仍允许按“像元等效和”统计。
    """
    if scene['crs'] is None:
        return np.full(scene['shape'], np.nan), 'no_crs_area_unknown'
    from pyproj import Transformer, Geod
    transformer = Transformer.from_crs(scene['crs'], 'EPSG:4326', always_xy=True)
    geod = Geod(ellps='WGS84')
    h,w = scene['shape']
    rows, cols = np.indices((h,w))
    lons, lats = [], []
    for dr,dc in ((0,0),(0,1),(1,1),(1,0)):
        x,y = scene['transform'] * (cols+dc, rows+dr)
        lon,lat = transformer.transform(np.asarray(x).ravel().tolist(), np.asarray(y).ravel().tolist())
        lons.append(np.asarray(lon).ravel())
        lats.append(np.asarray(lat).ravel())
    lons, lats = np.stack(lons,axis=1), np.stack(lats,axis=1)
    areas = np.full(h*w,np.nan)
    for i,(lon,lat) in enumerate(zip(lons,lats)):
        if np.isfinite(lon).all() and np.isfinite(lat).all():
            area,_ = geod.polygon_area_perimeter(lon,lat)
            areas[i] = abs(area)
    return areas.reshape(h,w), 'WGS84_geodesic_four_corner_approximation'


def export_results(data: dict) -> None:
    """结果明确分为raw、conditional、screened三层，不用nan_to_num偷换语义。"""
    cfg, scene, library, root = (data[k] for k in ('cfg','scene','library','root'))
    out, figs = root/'results', root/'figures'
    out.mkdir(exist_ok=True)
    figs.mkdir(exist_ok=True)
    names = list(CLASSES)
    for label, key in [('raw_model','raw_abundance'), ('conditional','conditional_abundance'),
                       ('screened','screened_abundance')]:
        image = scatter_to_image(data[key], scene)
        write_raster(out/f'abundance_{label}.tif', image, scene, names,
                     units='linear_model_fraction', layer=label,
                     unavailable_class='NaN_not_zero', class_order=','.join(names))
        for c,name in enumerate(names):
            write_raster(out/label/f'{name}.tif', image[c], scene, [name],
                         units='linear_model_fraction', layer=label)
    flags_img = np.zeros(scene['shape'],np.uint16)
    flags_img[~scene['numeric_valid']] |= 1
    flags_img[scene['numeric_valid'] & ~scene['quality_allowed']] |= 2
    flags_img[scene['valid']] = data['quality_flags']
    data['quality_image'] = flags_img
    write_raster(out/'quality_flags.tif',flags_img,scene,['bitwise_quality_flags'],'uint16',None)
    save_json(out/'quality_legend.json', dict(bits=QUALITY_BITS,
        screened_rule='flags == 0', note='位可叠加；screened仅表示通过配置检查，不是地面真值验证。'))
    write_raster(out/'fit_ok.tif',scatter_to_image(data['fit_ok'],scene,0,np.uint8),scene,['fit_ok'],'uint8',None)
    write_raster(out/'screened_ok.tif',scatter_to_image(data['screened_ok'],scene,0,np.uint8),scene,['screened_ok'],'uint8',None)
    for key in ('rmse','weighted_mse','best_weighted_mse','max_class_span'):
        write_raster(out/f'{key}.tif',scatter_to_image(data[key],scene),scene,[key])
    for key in ('uncertainty','abundance_min','abundance_max','drop_class_delta'):
        write_raster(out/f'{key}.tif',scatter_to_image(data[key],scene),scene,names)
    write_raster(out/'model_id.tif',scatter_to_image(data['model_id'],scene,-1,np.int32),scene,['model_id'],'int32',-1)
    write_raster(out/'model_size.tif',scatter_to_image(data['model_size'],scene,0,np.uint8),scene,['number_of_endmembers'],'uint8',None)
    write_raster(out/'near_model_count.tif',scatter_to_image(data['near_model_count'],scene,0,np.int32),scene,['near_model_count'],'int32',None)
    reconstruction = scatter_to_image(data['reconstruction'],scene)
    write_raster(out/'reconstruction_all_bands.tif',reconstruction,scene,scene['bands'])
    write_raster(out/'residual_all_bands.tif',scatter_to_image(data['residual'],scene),scene,scene['bands'])
    # 主导类别只是辅助图：0无拟合，1混合/歧义，2..6对应固定五类。
    dominant = np.ones(len(data['Y']),np.uint8)
    best_c = np.argmax(data['abundance'],axis=1)
    best_a = np.max(data['abundance'],axis=1)
    dominant[(best_a>=cfg.dominant_min) & (data['max_class_span']<=cfg.solver.abundance_uncertainty_max)] = best_c[(best_a>=cfg.dominant_min) & (data['max_class_span']<=cfg.solver.abundance_uncertainty_max)] + 2
    dominant[~data['fit_ok']] = 0
    data['dominant_conditional'] = scatter_to_image(dominant,scene,0,np.uint8)
    write_raster(out/'dominant_conditional.tif',data['dominant_conditional'],scene,['dominant_under_current_library'],'uint8',None)
    save_json(out/'dominant_legend.json',{0:'no_acceptable_fit_or_invalid',1:'mixed_or_ambiguous',**{i+2:c for i,c in enumerate(CLASSES)}})
    np.savez_compressed(out/'unmixing_result.npz',
                        bands=np.asarray(scene['bands']),classes=np.asarray(CLASSES),
                        candidate_ids=np.asarray([c['id'] for c in library['candidates']],dtype=str),
                        E=library['E'],valid=scene['valid'],
                        raw_abundance=scatter_to_image(data['raw_abundance'],scene),
                        conditional_abundance=scatter_to_image(data['conditional_abundance'],scene),
                        screened_abundance=scatter_to_image(data['screened_abundance'],scene),
                        candidate_abundance=data['candidate_abundance'],
                        quality_flags=flags_img,rmse=scatter_to_image(data['rmse'],scene),
                        selected_model=scatter_to_image(data['model_id'],scene,-1,np.int32))
    if 'baseline' in data:
        base = data['baseline']
        base_A = apply_class_states(base['abundance'],library['states'])
        base_A[base['rmse']>cfg.solver.max_rmse]=np.nan
        write_raster(out/'baseline_conditional_abundance.tif',scatter_to_image(base_A,scene),scene,names)
        write_raster(out/'baseline_rmse.tif',scatter_to_image(base['rmse'],scene),scene,['baseline_rmse'])
        save_json(root/'02_baseline.json',dict(candidate_ids=[library['candidates'][j]['id'] for j in base['indices']],
              note='每类第一条候选固定FCLS对照，图也只代表条件性拟合。'))
    # 逐波段误差不能被一个总RMSE掩盖。
    band_stats=[]
    for b,name in enumerate(scene['bands']):
        errors=data['residual'][:,b]
        finite=np.isfinite(errors)
        band_stats.append(dict(band=name,mean_residual=float(errors[finite].mean()) if finite.any() else None,
            rmse=float(np.sqrt(np.mean(errors[finite]**2))) if finite.any() else None))
    save_csv(root/'03_band_residual_statistics.csv',band_stats)
    if cfg.save_pixel_csv:
        rr,cc=np.where(scene['valid'])
        table=dict(row=rr,col=cc,model_id=data['model_id'],model_size=data['model_size'],
                   rmse=data['rmse'],weighted_mse=data['weighted_mse'],fit_ok=data['fit_ok'],
                   screened_ok=data['screened_ok'],quality_flags=data['quality_flags'],
                   near_model_count=data['near_model_count'])
        for c,name in enumerate(CLASSES):
            table[f'{name}_conditional']=data['conditional_abundance'][:,c]
            table[f'{name}_screened']=data['screened_abundance'][:,c]
            table[f'{name}_model_span']=data['uncertainty'][:,c]
            table[f'{name}_drop_delta_mse']=data['drop_class_delta'][:,c]
        save_csv(out/'pixel_trace.csv',pd.DataFrame(table))
    areas, area_method=pixel_areas_m2(scene)
    area_values=areas[scene['valid']]
    scene_rows=[]
    for c,state in enumerate(library['states']):
        a=data['conditional_abundance'][:,c]
        screened=data['screened_abundance'][:,c]
        finite=np.isfinite(a)
        trusted=np.isfinite(screened)
        detection=finite & (a>=cfg.detection_abundance)
        detmap=scatter_to_image(detection,scene,False,bool)
        labeled,_=ndi.label(detmap,np.ones((3,3),bool))
        counts=np.bincount(labeled.ravel())[1:]
        if state['status']=='confirmed_absent_by_user':
            inference='confirmed_absent_by_user'
        elif state['status']=='unavailable':
            inference='not_estimable_missing_endmember'
        elif not detection.any():
            inference=('asserted_present_but_not_detected_by_model' if state.get('asserted_scene_state')=='present'
                       else 'not_detected_under_current_model_not_proof_of_absence')
        else:
            inference='conditional_model_response_requires_source_review'
        # 没有有效估计时保持null，不把nansum(empty)=0冒充面积结果。
        has_area=np.isfinite(area_values).all()
        scene_rows.append(dict(class_name=CLASSES[c],state=state['status'],inference=inference,
            conditional_pixels=int(finite.sum()),screened_pixels=int(trusted.sum()),
            responding_pixels=int(detection.sum()),largest_response_component=int(counts.max()) if len(counts) else 0,
            conditional_pixel_equivalents=float(a[finite].sum()) if finite.any() else None,
            screened_pixel_equivalents=float(screened[trusted].sum()) if trusted.any() else None,
            conditional_area_m2=float(np.sum(a[finite]*area_values[finite])) if finite.any() and has_area else None,
            screened_area_m2=float(np.sum(screened[trusted]*area_values[trusted])) if trusted.any() and has_area else None,
            conditional_mean=float(a[finite].mean()) if finite.any() else None))
    save_csv(root/'03_scene_class_summary.csv',scene_rows)
    coverage=dict(total_pixels=int(np.prod(scene['shape'])),numeric_valid_pixels=int(scene['numeric_valid'].sum()),
        unmixing_domain_pixels=int(scene['valid'].sum()),fit_ok_pixels=int(data['fit_ok'].sum()),
        screened_ok_pixels=int(data['screened_ok'].sum()),
        unresolved_classes=library['unresolved_classes'],area_method=area_method,
        unmixing_domain_area_m2=float(area_values.sum()) if np.isfinite(area_values).all() else None,
        quality_flag_counts={meaning:int(((flags_img&bit)!=0).sum()) for bit,meaning in QUALITY_BITS.items()},
        important='0=当前模型零贡献或用户明确缺失；NaN=未估计。screened不等于精度已验证。')
    save_json(root/'03_coverage_summary.json',coverage)
    data['coverage_summary']=coverage
    data['scene_summary']=scene_rows


def _finish_figure(cfg: Config, path: Path):
    plt.tight_layout()
    plt.savefig(path,dpi=150,bbox_inches='tight')
    if cfg.show_figures:
        plt.show()
    plt.close()


def plot_map(cfg: Config, path: Path, array: np.ndarray, title: str,
             vmin=None,vmax=None,colorbar_label=None):
    """每张图单独输出；丰度固定0—1；没有有效数值时明确写NaN而非画全零。"""
    height = max(3.6, min(8.2, 6.2 * array.shape[0] / array.shape[1] + 1.2))
    plt.figure(figsize=(7,height))
    image=plt.imshow(np.ma.masked_invalid(array),vmin=vmin,vmax=vmax,interpolation='nearest')
    plt.title(title)
    plt.xlabel('Column (0-based)')
    plt.ylabel('Row (0-based)')
    plt.colorbar(image,label=colorbar_label,fraction=.04,pad=.035)
    if not np.isfinite(array).any():
        plt.text(.5,.5,'No estimate (NaN)\nSee class status / quality flags',
                 transform=plt.gca().transAxes,ha='center',va='center')
    _finish_figure(cfg,path)


def plot_results(data: dict) -> None:
    cfg,scene,library,root=(data[k] for k in ('cfg','scene','library','root'))
    figdir=root/'figures'
    figdir.mkdir(exist_ok=True)
    plt.figure(figsize=(7,6))
    plt.imshow(scene['rgb'])
    plt.title('Original RGB: shared fixed display stretch')
    _finish_figure(cfg,figdir/'00_original_rgb.png')
    if np.isfinite(data['reconstruction']).any():
        channels=[scene['bands'].index(b) for b in ('B4','B3','B2')]
        reconstructed=scatter_to_image(data['reconstruction'][:,channels],scene)
        rgb=np.moveaxis(reconstructed,0,-1)
        low,high=(scene['rgb_stretch'][k] for k in ('low','high'))
        plt.figure(figsize=(7,6))
        plt.imshow(np.nan_to_num(np.clip((rgb-low)/(high-low),0,1)))
        plt.title('Reconstructed RGB: same stretch as original')
        _finish_figure(cfg,figdir/'01_reconstructed_rgb.png')
    if library['candidates']:
        plt.figure(figsize=(7,6))
        plt.imshow(scene['rgb'])
        for cand in library['candidates']:
            plt.scatter(cand['col'],cand['row'],marker='x',s=60)
            plt.annotate(cand['id'],(cand['col']+1,cand['row']+1),fontsize=7)
        plt.title('Endmember source positions (not target unmixing domain)')
        _finish_figure(cfg,figdir/'02_endmember_locations.png')
    for c,name in enumerate(CLASSES):
        state=library['states'][c]['status']
        if np.any(library['class_ids']==c):
            plt.figure(figsize=(10,4))
            for cand in library['candidates']:
                if cand['class_id']==c:
                    plt.plot(scene['bands'],cand['spectrum'],marker='o',
                             linestyle='-' if cand['reviewed'] else '--',label=cand['name'])
            plt.xlabel('All bands: categorical axis, not wavelength spacing')
            plt.ylabel('Reflectance')
            plt.title(f'{name} candidate spectra: {state}')
            plt.legend()
            _finish_figure(cfg,figdir/f'03_spectra_{name}.png')
        for prefix,key in [('conditional','conditional_abundance'),('screened','screened_abundance')]:
            plot_map(cfg,figdir/f'04_{prefix}_{name}.png',scatter_to_image(data[key][:,c],scene),
                     f'{name} | {prefix}\n{state}',0,1,'Linear-model fraction')
        plot_map(cfg,figdir/f'05_span_{name}.png',scatter_to_image(data['uncertainty'][:,c],scene),
                 f'{name}: range across near-optimal models',0,1,'Abundance range (not probability)')
        delta=data['drop_class_delta'][:,c]
        plot_map(cfg,figdir/f'06_drop_delta_{name}.png',scatter_to_image(delta,scene),
                 f'{name}: increase in MSE when class is excluded',0,None,'Weighted MSE increase')
    plot_map(cfg,figdir/'07_rmse.png',scatter_to_image(data['rmse'],scene),
             'Reconstruction RMSE: raw reflectance',0,None,'RMSE')
    if 'baseline' in data:
        # 主方案和对照误差图使用相同数值上限，避免视觉伸缩造成虚假改进。
        base=data['baseline']['rmse']
        vals=np.concatenate([data['rmse'][np.isfinite(data['rmse'])],base[np.isfinite(base)]])
        top=max(cfg.solver.max_rmse,float(np.quantile(vals,.98))) if len(vals) else cfg.solver.max_rmse
        plot_map(cfg,figdir/'07_main_rmse_same_scale.png',scatter_to_image(data['rmse'],scene),
                 'MESMA-style RMSE (shared scale)',0,top,'RMSE')
        plot_map(cfg,figdir/'07_baseline_rmse_same_scale.png',scatter_to_image(base,scene),
                 'One candidate per class: fixed FCLS RMSE',0,top,'RMSE')
    size_for_plot = np.where(data['model_id'] >= 0, data['model_size'].astype(float), np.nan)
    plot_map(cfg,figdir/'08_model_size.png',scatter_to_image(size_for_plot,scene),
             'Number of selected endmembers',1,cfg.solver.max_endmembers,'Count')
    # 位图各位不能用一个连续色标解释；分别绘制每个质量标记。
    for bit,label in QUALITY_BITS.items():
        if bit in (1,2):
            arr=((data['quality_image']&bit)!=0).astype(float)
        else:
            arr=scatter_to_image(((data['quality_flags']&bit)!=0).astype(float),scene)
        plot_map(cfg,figdir/f'09_quality_{bit}_{label}.png',arr,label,0,1,'Flag')
    plt.figure(figsize=(7,6))
    im=plt.imshow(data['dominant_conditional'],norm=BoundaryNorm(np.arange(-.5,7.5,1),256),interpolation='nearest')
    cb=plt.colorbar(im,ticks=np.arange(7))
    cb.ax.set_yticklabels(['no fit','mixed/ambiguous',*CLASSES])
    plt.title('Dominant class: conditional on current library')
    _finish_figure(cfg,figdir/'10_dominant_conditional.png')
    finite=np.isfinite(data['rmse'])
    if finite.any():
        plt.figure(figsize=(8,4))
        plt.hist(data['rmse'][finite],bins=60)
        plt.axvline(cfg.solver.max_rmse,linestyle='--',label='configured acceptance threshold')
        plt.xlabel('Reflectance RMSE')
        plt.ylabel('Pixel count')
        plt.legend()
        _finish_figure(cfg,figdir/'11_rmse_histogram.png')


def inspect_pixel(data: dict, row: int, col: int, top_models: int = 10,
                  save: bool = True, plot: bool = True) -> dict:
    """Notebook入口：查看某个实际像元的模型、丰度、光谱和每波段残差。
    top_models表使用每个组合真正的FCLS解（含其子面），不把面外仿射解当可行解。
    """
    scene,library,cfg,root=(data[k] for k in ('scene','library','cfg','root'))
    row,col=int(row),int(col)
    if top_models < 1:
        raise ValueError('top_models必须为正整数。')
    if not 0<=row<scene['shape'][0] or not 0<=col<scene['shape'][1]:
        raise ValueError('row/col超出影像。')
    if not scene['valid'][row,col]:
        raise ValueError('该像元不在共同质量有效域内。')
    if not data['models']:
        raise ValueError('没有可用端元模型；请先检查类别状态。')
    dense_index=np.full(scene['shape'],-1,int)
    dense_index[scene['valid']]=np.arange(scene['valid'].sum())
    i=int(dense_index[row,col])
    Y=data['Y'][i:i+1]
    costs,raw,coeff=evaluate_block(Y,data['models'],data['weights'],cfg.solver)
    rows=[]
    # 对一个组合的边界解，取所有包含在其中的面之最优解。
    # 图像主求解缓存面解；这里还原“完整组合的FCLS结果”供人工核对。
    for model in data['models']:
        subsets=[m.model_id for m in data['models'] if set(m.indices).issubset(model.indices)]
        face_id=min(subsets,key=lambda j:costs[j,0])
        face=data['models'][face_id]
        grouped=np.zeros(5)
        for t,c in enumerate(face.classes):
            grouped[c]+=coeff[face_id,0,t]
        item=dict(model_id=model.model_id,requested_endmembers=model.size,
                  candidates=' | '.join(library['candidates'][j]['id'] for j in model.indices),
                  fcls_active_face_id=face_id,weighted_mse=float(costs[face_id,0]),
                  rmse=float(np.sqrt(raw[face_id,0])),selected=model.model_id==data['model_id'][i])
        grouped = apply_class_states(grouped[None, :], library['states'])[0]
        item.update({c:float(grouped[k]) for k,c in enumerate(CLASSES)})
        rows.append(item)
    ranked=pd.DataFrame(rows).sort_values(['weighted_mse','requested_endmembers','model_id'])
    table=ranked.head(top_models).copy()
    # 简单模型可能略高于最低误差而被选择；不能让它在竞争表中消失。
    if not table['selected'].any():
        table=pd.concat([ranked[ranked['selected']],table],ignore_index=True)
    abund=pd.DataFrame(dict(class_name=CLASSES,conditional=data['conditional_abundance'][i],
                           screened=data['screened_abundance'][i],near_model_min=data['abundance_min'][i],
                           near_model_max=data['abundance_max'][i],range=data['uncertainty'][i],
                           drop_delta_mse=data['drop_class_delta'][i]))
    spectra=pd.DataFrame(dict(band=scene['bands'],observed=Y[0],reconstructed=data['reconstruction'][i],
                             residual=data['residual'][i],weighted_residual=data['residual'][i]*data['weights']))
    used=np.flatnonzero(data['candidate_abundance'][i]>1e-8)
    selected=pd.DataFrame([dict(candidate=library['candidates'][j]['id'],
                               abundance=float(data['candidate_abundance'][i,j]),
                               reviewed=library['candidates'][j]['reviewed']) for j in used])
    info=dict(row=row,col=col,model_id=int(data['model_id'][i]),rmse=float(data['rmse'][i]),
              quality_flags=int(data['quality_flags'][i]),
              flag_names=[meaning for bit,meaning in QUALITY_BITS.items() if data['quality_flags'][i]&bit])
    dest=root/'pixel_inspection'/f'r{row}_c{col}'
    if save:
        dest.mkdir(parents=True,exist_ok=True)
        save_json(dest/'pixel_info.json',info)
        for name,frame in [('competing_models',table),('class_abundance',abund),('spectra',spectra),('used_candidates',selected)]:
            save_csv(dest/f'{name}.csv',frame)
    if plot:
        dest.mkdir(parents=True,exist_ok=True)
        plt.figure(figsize=(10,4))
        plt.plot(scene['bands'],Y[0],marker='o',label='observed')
        plt.plot(scene['bands'],data['reconstruction'][i],marker='x',linestyle='--',label='reconstructed')
        for j in used:
            plt.plot(scene['bands'],library['E'][:,j],linestyle=':',alpha=.6,label=library['candidates'][j]['id'])
        plt.ylabel('Reflectance')
        plt.xlabel('All spectral bands')
        plt.title(f'Pixel ({row},{col}): spectra, RMSE={data["rmse"][i]:.5f}')
        plt.legend(fontsize=8)
        _finish_figure(cfg,dest/'spectrum_fit.png')
        plt.figure(figsize=(8,4))
        vals=data['raw_abundance'][i]
        ok=np.isfinite(vals)
        plt.bar(np.arange(5)[ok],vals[ok])
        for c in np.flatnonzero(~ok):
            plt.text(c,.05,'NaN',ha='center')
        plt.xticks(np.arange(5),CLASSES,rotation=15)
        plt.ylim(0,1)
        plt.ylabel('Conditional model fraction')
        plt.title(f'Pixel ({row},{col}); flags={data["quality_flags"][i]}')
        _finish_figure(cfg,dest/'class_abundance.png')
        plt.figure(figsize=(10,4))
        plt.bar(scene['bands'],data['residual'][i])
        plt.axhline(0,linestyle='--')
        plt.ylabel('Observed minus reconstructed reflectance')
        plt.title(f'Pixel ({row},{col}): residual by band')
        _finish_figure(cfg,dest/'residual.png')
    return dict(info=info,abundance=abund,spectra=spectra,models=table,used_candidates=selected)


def write_html_report(data: dict) -> Path:
    """离线HTML只读取本地PNG和表格，不依赖网络、JS服务或浏览器插件。"""
    root=data['root']
    def esc(x):return html.escape(str(x))
    sections=[]
    sections.append('<h1>场景自适应端元组合光谱解混</h1>')
    sections.append('<p><strong>conditional</strong>是现有端元库下的拟合；'
                    '<strong>screened</strong>仅保留通过配置质量与来源检查的像元。'
                    '缺少端元、未核查候选、低拟合误差不能自动证明地物不存在或丰度准确。</p>')
    sections.append('<h2>类别状态</h2>'+pd.DataFrame(data['library']['states']).to_html(index=False,escape=True))
    sections.append('<h2>覆盖与警告</h2><pre>'+esc(json.dumps(_clean_json(data['coverage_summary']),ensure_ascii=False,indent=2))+'</pre>')
    sections.append('<h2>场景统计（只统计相应有效区域）</h2>'+pd.DataFrame(data['scene_summary']).to_html(index=False,escape=True))
    sections.append('<h2>可视化</h2><p>所有丰度图统一0—1。NaN空白不代表零。原图与重建图使用相同拉伸。'
                    '质量标记分图显示，避免把位编码误读成连续数值。</p><div class="gallery">')
    for path in sorted((root/'figures').glob('*.png')):
        rel=path.relative_to(root).as_posix()
        sections.append(f'<figure><a href="{esc(rel)}"><img src="{esc(rel)}" loading="lazy"></a><figcaption>{esc(path.name)}</figcaption></figure>')
    sections.append('</div><h2>像元检查</h2><div class="gallery">')
    for path in sorted((root/'pixel_inspection').glob('*/*.png')):
        rel=path.relative_to(root).as_posix()
        sections.append(f'<figure><img src="{esc(rel)}" loading="lazy"><figcaption>{esc(rel)}</figcaption></figure>')
    sections.append('</div><p>数学与软件验证见代码包；此报告不能替代独立地面/高分辨率参考验证。</p>')
    css='body{font-family:system-ui,sans-serif;margin:28px;line-height:1.6;max-width:1500px}table{border-collapse:collapse;display:block;overflow:auto}th,td{padding:7px;border:1px solid #ccc;text-align:left;font-size:13px}pre{white-space:pre-wrap;overflow-wrap:anywhere}.gallery{display:grid;grid-template-columns:repeat(auto-fit,minmax(330px,1fr));gap:20px}figure{margin:0}img{width:100%;height:auto}figcaption{font-size:12px;overflow-wrap:anywhere}'
    page='<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>光谱解混报告</title><style>'+css+'</style><body>'+''.join(sections)+'</body></html>'
    path=root/'report.html'
    path.write_text(page,encoding='utf-8')
    return path


def main(cfg: Config | None = None) -> dict | None:
    """一键入口。逐步Notebook也可以单独调用各阶段函数。"""
    cfg=cfg or Config()
    validate_config(cfg)
    root=prepare_run(cfg)
    status=dict(state='running',input_path=str(cfg.input_path),output_dir=str(root),stage='read_scene',
                started_utc=datetime.now(timezone.utc).isoformat())
    save_json(root/'run_status.json',status)
    print(f'本次独立实验目录：{root}')
    try:
        scene=read_scene(cfg,root)
        if scene.get('inspection_only'):
            status.update(state='inspection_only',stage='await_reflectance_confirmation')
            save_json(root/'run_status.json',status)
            print('当前只检查波段和校准配置。确认反射率单位后，将data_confirmed改为True。')
            return None
        validate_solver(cfg.solver,len(scene['bands']))
        status['stage']='collect_endmembers';save_json(root/'run_status.json',status)
        library=collect_endmembers(cfg,scene,root)
        status['stage']='solve_scene';save_json(root/'run_status.json',status)
        data=solve_scene(cfg,scene,library,root)
        status['stage']='export_results';save_json(root/'run_status.json',status)
        export_results(data)
        if cfg.make_plots:
            plot_results(data)
        if cfg.save_example_pixels and data['models']:
            coords=np.argwhere(scene['valid'])
            # 确定性选取不同位置，仅作查看示例，不作为验证集或人为挑选好看的像元。
            selected=np.unique(np.linspace(0,len(coords)-1,min(cfg.save_example_pixels,len(coords)),dtype=int))
            for j in selected:
                r,c=coords[j]
                inspect_pixel(data,int(r),int(c),save=True,plot=cfg.make_plots)
        if cfg.make_html:
            write_html_report(data)
        state='completed' if data['models'] else 'no_available_endmembers'
        status.update(state=state,stage='finished',fit_ok_pixels=int(data['fit_ok'].sum()),
                      screened_ok_pixels=int(data['screened_ok'].sum()),
                      unresolved_classes=library['unresolved_classes'],
                      finished_utc=datetime.now(timezone.utc).isoformat())
        save_json(root/'run_status.json',status)
        print('\n处理结束：'+str(root))
        print(f'数值拟合通过：{data["fit_ok"].sum()}；严格配置筛查通过：{data["screened_ok"].sum()}')
        if library['unresolved_classes']:
            print('以下类别缺少端元、不是确认不存在：'+', '.join(library['unresolved_classes']))
        return data
    except Exception as error:
        status.update(state='failed',error=repr(error),traceback=traceback.format_exc())
        save_json(root/'run_status.json',status)
        raise


if __name__=='__main__':
    main()
