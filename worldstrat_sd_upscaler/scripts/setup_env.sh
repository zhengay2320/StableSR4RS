#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

if ! command -v python >/dev/null 2>&1; then
  echo "ERROR: python was not found. Activate a Python 3.10 or 3.11 environment first." >&2
  exit 1
fi

PYTHON_VERSION="$(python -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
if [[ "${PYTHON_VERSION}" != "3.10" && "${PYTHON_VERSION}" != "3.11" ]]; then
  echo "ERROR: Python 3.10 or 3.11 is required; found ${PYTHON_VERSION}." >&2
  exit 1
fi

if ! python -c 'import torch; print(f"Found torch {torch.__version__}")' >/dev/null 2>&1; then
  cat >&2 <<'EOF'
ERROR: PyTorch is not installed. This script intentionally does not choose a CUDA build.
Install the torch/torchvision build matching this machine from https://pytorch.org/get-started/locally/
then rerun scripts/setup_env.sh.
EOF
  exit 1
fi
python -c 'import torch; print(f"Using torch {torch.__version__}; CUDA runtime={torch.version.cuda}")'
TORCH_VERSION="$(python -c 'import torch; print(torch.__version__)')"
CONSTRAINTS_FILE="$(mktemp)"
trap 'rm -f "${CONSTRAINTS_FILE}"' EXIT
printf 'torch==%s\n' "${TORCH_VERSION}" >"${CONSTRAINTS_FILE}"

if [[ ! -d third_party/diffusers/.git ]]; then
  git clone --branch v0.39.0 --depth 1 https://github.com/huggingface/diffusers.git third_party/diffusers
else
  EXPECTED_COMMIT="$(git -C third_party/diffusers rev-list -n 1 v0.39.0 2>/dev/null || true)"
  ACTUAL_COMMIT="$(git -C third_party/diffusers rev-parse HEAD 2>/dev/null || true)"
  if [[ -z "${EXPECTED_COMMIT}" || "${ACTUAL_COMMIT}" != "${EXPECTED_COMMIT}" ]]; then
    echo "ERROR: third_party/diffusers exists but HEAD is not the v0.39.0 commit." >&2
    exit 1
  fi
fi

python -m pip install -c "${CONSTRAINTS_FILE}" -e third_party/diffusers
python -m pip install -c "${CONSTRAINTS_FILE}" -r requirements.txt
python -c 'import diffusers; assert diffusers.__version__ == "0.39.0", diffusers.__version__; print("Verified diffusers 0.39.0")'
python -c 'import torch; print(f"Verified unchanged torch {torch.__version__}; CUDA runtime={torch.version.cuda}")'
echo "Environment ready. Optional memory packages are not installed automatically: bitsandbytes, xformers."
