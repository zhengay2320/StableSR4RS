#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
accelerate launch --multi_gpu --num_processes 4 \
  src/train_lora_upscaler.py --config configs/stage1_synthetic.yaml "$@"

