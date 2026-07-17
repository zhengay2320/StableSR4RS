#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export DATA_ROOT="${DATA_ROOT:-/data/zhengay/EDiffSR-main/data/EDiffSR_worldstrat_rgb_x4_per_image}"

python -m compileall src
python -m pytest -q
python src/validate_dataset.py --data_root "${DATA_ROOT}" --split train \
  --lr_subdir LR_bicubic --max_samples 1 \
  --output_csv outputs/smoke/invalid_sample.csv
accelerate launch --num_processes 1 --num_machines 1 --mixed_precision fp16 --dynamo_backend no \
  src/train_lora_upscaler.py \
  --config configs/stage1_synthetic.yaml --data_root "${DATA_ROOT}" \
  --output_dir outputs/smoke/training \
  --max_train_steps 2 --train_batch_size 1 --num_workers 0
python src/infer_upscaler.py \
  --config configs/stage1_synthetic.yaml \
  --lora_path outputs/smoke/training/final \
  --adapter_path outputs/smoke/training/final/condition_adapter.safetensors \
  --input_dir "${DATA_ROOT}/test/LR_bicubic" --gt_dir "${DATA_ROOT}/test/GT" \
  --output_dir outputs/smoke/inference --limit 1 --num_inference_steps 2
python - <<'PY'
import os
from pathlib import Path
from PIL import Image
root = Path("outputs/smoke/inference")
sr_path = next((root / "sr_raw").iterdir())
lr_path = Path(os.environ["DATA_ROOT"]) / "test" / "LR_bicubic" / sr_path.name
with Image.open(sr_path) as sr, Image.open(lr_path) as lr:
    assert sr.size == (lr.width * 4, lr.height * 4), (sr.size, lr.size)
print(f"Smoke output geometry passed: {lr.size} -> {sr.size}")
PY
