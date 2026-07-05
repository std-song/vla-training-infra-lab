#!/usr/bin/env bash
set -euo pipefail

# Tested on AutoDL / SeeTaCloud RTX 3090 image:
# Python 3.10.8, PyTorch 2.1.2+cu118, CUDA toolkit 11.8.

PROJECT_ROOT="${PROJECT_ROOT:-/root/autodl-tmp/vla-infra}"
NANOTRON_DIR="${NANOTRON_DIR:-${PROJECT_ROOT}/nanotron}"

mkdir -p "${PROJECT_ROOT}/pip-cache"
export PIP_CACHE_DIR="${PROJECT_ROOT}/pip-cache"

python --version
python - <<'PY'
import torch
print(torch.__version__, torch.version.cuda, torch.cuda.is_available())
if torch.cuda.is_available():
    print(torch.cuda.get_device_name(0))
PY

python -m pip install -U "setuptools<70" wheel packaging ninja
python -m pip install -e "${NANOTRON_DIR}"
python -m pip install transformers==4.41.2 "huggingface_hub<0.24" accelerate sentencepiece protobuf pytest pytest-xdist datasets
python -m pip install --no-build-isolation git+https://github.com/fanshiqing/grouped_gemm@main
MAX_JOBS=4 python -m pip install --no-build-isolation --no-cache-dir --force-reinstall --no-deps "flash-attn==2.5.8"

python -c "import grouped_gemm.ops as ops; print('grouped_gemm ok')"
python -c "from flash_attn.ops.triton.layer_norm import layer_norm_fn; print('flash_attn layer_norm ok')"
