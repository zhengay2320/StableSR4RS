#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
DATA_ROOT="${DATA_ROOT:-/data/zhengay/EDiffSR-main/data/EDiffSR_worldstrat_rgb_x4_per_image}"
ARTIFACT_PATH="${ARTIFACT_PATH:-outputs/stage1_synthetic/final}"
python src/infer_upscaler.py \
  --config configs/stage1_synthetic.yaml \
  --checkpoint_path "${ARTIFACT_PATH}" \
  --input_dir "${DATA_ROOT}/test/LR_bicubic" --gt_dir "${DATA_ROOT}/test/GT" \
  --output_dir outputs/inference_stage1_synthetic "$@"
