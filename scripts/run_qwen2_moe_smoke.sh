#!/usr/bin/env bash
set -euo pipefail

export CUDA_DEVICE_MAX_CONNECTIONS="${CUDA_DEVICE_MAX_CONNECTIONS:-1}"
export PYTHONPATH="${PYTHONPATH:-src}"

CONFIG_FILE="${1:-examples/smoke/config_qwen2_moe_smoke.yaml}"
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"

python - <<'PY'
import torch
print(f"torch={torch.__version__} cuda={torch.version.cuda} available={torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"gpu={torch.cuda.get_device_name(0)}")
PY

torchrun --nproc_per_node="${NPROC_PER_NODE}" run_train.py --config-file "${CONFIG_FILE}"
