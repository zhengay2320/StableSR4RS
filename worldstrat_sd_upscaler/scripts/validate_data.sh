#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
DATA_ROOT="${DATA_ROOT:-/data/zhengay/EDiffSR-main/data/EDiffSR_worldstrat_rgb_x4_per_image}"

for split in train val test; do
  for lr_subdir in LR LR_bicubic; do
    python src/validate_dataset.py \
      --data_root "${DATA_ROOT}" --split "${split}" --lr_subdir "${lr_subdir}" \
      --output_csv "outputs/data_validation/${split}_${lr_subdir}_invalid.csv"
  done
done

