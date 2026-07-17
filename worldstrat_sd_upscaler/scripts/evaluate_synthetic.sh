#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
DATA_ROOT="${DATA_ROOT:-/data/zhengay/EDiffSR-main/data/EDiffSR_worldstrat_rgb_x4_per_image}"
python src/evaluate.py \
  --sr_dir outputs/inference_stage1_synthetic/sr_projected \
  --gt_dir "${DATA_ROOT}/test/GT" --lr_dir "${DATA_ROOT}/test/LR_bicubic" \
  --output_dir outputs/evaluation_stage1_synthetic "$@"

