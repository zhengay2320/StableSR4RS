#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
DATA_ROOT="${DATA_ROOT:-/data/zhengay/EDiffSR-main/data/EDiffSR_worldstrat_rgb_x4_per_image}"
ARTIFACT_PATH="${ARTIFACT_PATH:-outputs/stage2_cross_sensor/final}"
python src/infer_upscaler.py \
  --config configs/stage2_cross_sensor.yaml \
  --checkpoint_path "${ARTIFACT_PATH}" \
  --input_dir "${DATA_ROOT}/test/LR" --gt_dir "${DATA_ROOT}/test/GT" \
  --output_dir outputs/inference_stage2_real "$@"
