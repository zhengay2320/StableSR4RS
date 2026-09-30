from __future__ import annotations

import json
import sys
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
import yaml
from PIL import Image

from scripts.train_tri_input_warmup import main as warmup_main
from scripts.validate_tri_input_data import validate_split
from src.tri_input_conditioner import TriInputConditioner
from src.tri_input_data import (
    SENTINEL2_L2A_BANDS,
    compute_raw_band_stats,
    save_raw_band_stats,
)


def _fixture(tmp_path: Path) -> tuple[dict, Path]:
    rasterio = pytest.importorskip("rasterio")
    data = tmp_path / "data"
    lr_dir = data / "train" / "LR"
    gt_dir = data / "train" / "GT"
    raw_dir = tmp_path / "raw"
    prior_dir = tmp_path / "prior"
    for directory in (lr_dir, gt_dir, raw_dir, prior_dir):
        directory.mkdir(parents=True)
    y, x = np.mgrid[:8, :8]
    lr = np.stack((x * 20, y * 20, (x + y) * 10), axis=-1).astype(np.uint8)
    Image.fromarray(lr, mode="RGB").save(lr_dir / "scene.png")
    Image.fromarray(
        np.repeat(np.repeat(lr, 4, axis=0), 4, axis=1), mode="RGB"
    ).save(gt_dir / "scene.png")
    raw = np.stack(
        [x.astype(np.float32) + 2 * y.astype(np.float32) + channel for channel in range(12)]
    )
    with rasterio.open(
        raw_dir / "scene.tiff",
        "w",
        driver="GTiff",
        width=8,
        height=8,
        count=12,
        dtype="float32",
        transform=rasterio.transform.from_origin(0, 80, 10, 10),
        crs="EPSG:32648",
    ) as destination:
        destination.write(raw)
    prior = np.zeros((10, 8, 8), dtype=np.float32)
    prior[:5] = 0.2
    prior[5:] = 0.25
    np.save(prior_dir / "scene.npy", prior)
    stats = compute_raw_band_stats(
        [raw_dir / "scene.tiff"],
        SENTINEL2_L2A_BANDS,
        {"mode": "identity", "input_units": "fixture"},
        split="train",
        data_source="CPU fixture train only",
    )
    stats_path = tmp_path / "raw_band_stats.json"
    save_raw_band_stats(stats, stats_path)
    config = {
        "data_root": str(data),
        "train_lr_subdir": "LR",
        "val_lr_subdir": "LR",
        "gt_subdir": "GT",
        "gt_crop_size": 32,
        "scale": 4,
        "train_batch_size": 1,
        "num_workers": 0,
        "strict_pairs": True,
        "augment": False,
        "synthetic_replay_probability": 0.0,
        "seed": 7,
        "tri_input": {
            "enabled": True,
            "components": 8,
            "band_names": list(SENTINEL2_L2A_BANDS),
            "raw_value_conversion": {"mode": "identity", "input_units": "fixture"},
            "raw_stats_path": str(stats_path),
            "data": {
                "train": {
                    "raw_ms_dir": str(raw_dir),
                    "unmixing_dir": str(prior_dir),
                    "manifest_path": None,
                }
            },
            "checker": {
                "enabled": True,
                "scale": 4,
                "common_factor": 2,
                "window": 8,
                "stride": 4,
                "max_shift_hr": 1.0,
                "ridge": 0.0001,
                "movement_penalty": 0.0001,
                "min_relative_gain": 0.02,
                "min_relative_gap": 0.005,
                "absolute_gain_floor": 1e-7,
            },
        },
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return config, config_path


def test_precheck_and_two_warmup_stages_run_on_cpu_fixture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, config_path = _fixture(tmp_path)
    report = validate_split(config, "train", None, tmp_path / "check", None)
    assert report["success_count"] == 1
    assert report["failure_count"] == 0
    assert Path(report["manifest"]).is_file()

    override_config = deepcopy(config)
    override_config["tri_input"]["data"]["train"]["raw_ms_dir"] = None
    override_config["tri_input"]["data"]["train"]["unmixing_dir"] = None
    override_report = validate_split(
        override_config,
        "train",
        None,
        tmp_path / "check_overrides",
        None,
        Path(config["tri_input"]["data"]["train"]["raw_ms_dir"]),
        Path(config["tri_input"]["data"]["train"]["unmixing_dir"]),
    )
    assert override_report["success_count"] == 1

    stage_a = tmp_path / "warmup_a"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train_tri_input_warmup.py",
            "--config",
            str(config_path),
            "--stage",
            "A",
            "--output_dir",
            str(stage_a),
            "--max_steps",
            "1",
            "--checkpointing_steps",
            "1",
            "--device",
            "cpu",
        ],
    )
    warmup_main()
    loaded_a = TriInputConditioner.from_pretrained(stage_a / "final")
    assert loaded_a.config["checker"]["enabled"] is False

    stage_a_resumed = tmp_path / "warmup_a_resumed"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train_tri_input_warmup.py",
            "--config",
            str(config_path),
            "--stage",
            "A",
            "--resume",
            str(stage_a / "final"),
            "--output_dir",
            str(stage_a_resumed),
            "--max_steps",
            "2",
            "--checkpointing_steps",
            "2",
            "--device",
            "cpu",
        ],
    )
    warmup_main()
    resumed_summary = json.loads(
        (stage_a_resumed / "warmup_summary.json").read_text(encoding="utf-8")
    )
    assert resumed_summary["global_step"] == 2

    stage_b = tmp_path / "warmup_b"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train_tri_input_warmup.py",
            "--config",
            str(config_path),
            "--stage",
            "B",
            "--init_conditioner",
            str(stage_a / "final"),
            "--output_dir",
            str(stage_b),
            "--max_steps",
            "1",
            "--checkpointing_steps",
            "1",
            "--device",
            "cpu",
        ],
    )
    warmup_main()
    loaded_b = TriInputConditioner.from_pretrained(stage_b / "final")
    assert loaded_b.config["checker"]["enabled"] is True
    summary = json.loads((stage_b / "warmup_summary.json").read_text(encoding="utf-8"))
    assert summary["global_step"] == 1
    assert summary["checker_enabled"] is True
    assert "checker_internal_gain" in summary["checker_diagnostics_mean"]
    assert isinstance(summary["checker_fallback_reasons"], dict)
