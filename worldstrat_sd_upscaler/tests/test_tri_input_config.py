from pathlib import Path

import yaml


def test_stage3_uses_stage1_rgb_pairs_and_initial_checkpoint() -> None:
    config = yaml.safe_load(Path("configs/stage3_tri_input.yaml").read_text(encoding="utf-8"))
    stage1 = yaml.safe_load(Path("configs/stage1_synthetic.yaml").read_text(encoding="utf-8"))
    tri = config["tri_input"]
    assert tri["enabled"] is True
    for key in ("data_root", "train_lr_subdir", "val_lr_subdir", "gt_subdir", "scale"):
        assert config[key] == stage1[key]
    assert config["test_lr_subdir"] == "LR_bicubic"
    assert config["init_lora_path"] == f"{stage1['output_dir']}/final"
    assert config["init_adapter_path"] == f"{stage1['output_dir']}/final"
    assert config["output_dir"] == "outputs/stage3_tri_input_synthetic"
    assert config["synthetic_replay_probability"] == 0.0
    assert config["phi_enabled"] is False
    assert tri["train_existing_lora"] is False
    assert tri["train_existing_condition_adapter"] is False
    assert tri["synthetic_aux_policy"] == "error"
    # Auxiliary paths/band calibration are user configuration, not RGB sources.
    # Do not require them to stay null after the user has supplied real values.
    assert tri["data"]["train"]["raw_ms_dir"] == "/data/zhengay/EDiffSR-main/data/new_star/train/lr"
    for split in ("train", "val", "test"):
        assert "raw_ms_dir" in tri["data"][split]
        assert "unmixing_dir" in tri["data"][split]
