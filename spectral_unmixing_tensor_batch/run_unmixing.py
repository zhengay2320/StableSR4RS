# -*- coding: utf-8 -*-
"""用户运行入口：与之前五个提取脚本放在同一文件夹，然后修改本文件配置。"""
from pathlib import Path
from spectral_unmixing_pipeline import Config, ClassSpec, main
from unmixing_core import SolverConfig

HERE = Path(__file__).resolve().parent


def make_config() -> Config:
    cfg = Config(
        input_path=Path(r'E:\开源数据集\word_star\new_star\train\lr\UNHCR-COLs003235.tiff'),
        script_dir=HERE,
        output_root=HERE / 'unmixing_results',
        # 首次检查数据：只有确认TIFF原值已经是反射率时，才采用1和0并改True。
        data_confirmed=True,
        scale=1.0,
        offset=0.0,
        clear_mask_path=None,  # 同网格0/1；1有效。不要填端元seeds掩膜！
        scl_path=None,         # 可选同景SCL；默认不排除水体6和雪地11。
        classes={
            'vegetation': ClassSpec(),
            'water': ClassSpec(),
            'bare': ClassSpec(),
            'snow': ClassSpec(),
            'building': ClassSpec(),
        },
        solver=SolverConfig(
            max_endmembers=3,       # 初始尝试1、2、3端元模型。
            max_per_class=1,        # 每类最多用1个候选；设2可允许同类多候选混合。
            max_rmse=0.03,         # 原始反射率单位，须结合真实数据标定。
            selection_delta_mse=1e-5,
            ambiguity_delta_mse=1e-5,
            block_pixels=2048,
        ),
        show_figures=False,  # 图都保存；用report.html一起查看，不弹几十个窗口。
        make_plots=True,
        make_html=True,
        run_baseline=True,
    )

    # --------------------------------------------------------
    # 方式A（默认）：运行之前五个模块，保留各自CFG/Config中的阈值与先验。
    # 类别失败不会自动被声明不存在；真正程序错误默认仍中止并写日志。
    # 每次使用新的输出目录，原始脚本和既有实验都不被覆盖。
    # --------------------------------------------------------

    # 需要显式覆盖某个提取参数，可按实际研究结果使用以下写法。
    # 下例仅是此前“关闭腐蚀”的诊断选项，不代表已经修复建筑语义混淆：
    # cfg.classes['building'].extraction_overrides = {'erosion_radius': 0}

    # --------------------------------------------------------
    # 方式B：已有端元结果时，填写实际结果目录即可，不需要重复提取。
    # 目录必须包含12_candidate_spectra.csv、12_candidates.npz、
    # 12_candidate_summary.csv、01_band_mapping.csv、00_metadata.json、run_status.json。
    # 一个类别可读旧结果，另一个类别仍运行提取；不要求全部同一种来源方式。
    # --------------------------------------------------------
    # cfg.classes['vegetation'].source = 'EXISTING'
    # cfg.classes['vegetation'].output_dir = Path(r'E:\实际植被结果目录')
    # cfg.classes['water'].source = 'EXISTING'
    # cfg.classes['water'].output_dir = Path(r'E:\实际水体结果目录')
    #
    # 仅对尚无新包装器输入哈希的旧结果，需要人工确认来自同一未修改TIFF后设True。
    # cfg.allow_legacy_results = True

    # --------------------------------------------------------
    # 场景类别与端元核查不是同一件事；不要因提取失败就设置absent。
    # --------------------------------------------------------
    # 已核查确实不含某类后才填写：
    # cfg.classes['snow'].scene_state = 'absent'
    # cfg.classes['snow'].state_reason = '填写本景实际核查依据/日期/区域，而非“提取失败”。'
    #
    # 已知建筑存在，但未必能提取成功：
    # cfg.classes['building'].scene_state = 'present'
    # cfg.classes['building'].state_reason = '填写建筑来源核查依据。'
    #
    # 来源光谱与类别核查通过后才设reviewed=True。
    # cfg.classes['vegetation'].reviewed = True
    # 或只认可某些候选，名称使用原CSV中的列名：
    # cfg.classes['building'].reviewed_candidates = ('roof_candidate_1',)
    #
    # 明确只保留指定候选：
    # cfg.classes['vegetation'].selected_candidates = ('veg_1', 'veg_2')

    return cfg


if __name__ == '__main__':
    main(make_config())
