#!/usr/bin/env bash
set -euo pipefail

export CUDA_DEVICE_MAX_CONNECTIONS="${CUDA_DEVICE_MAX_CONNECTIONS:-1}"
export PYTHONPATH="${PYTHONPATH:-src}"

CONFIG_FILE="${1:-examples/smoke/config_qwen2_moe_resume.yaml}"
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"

torchrun --nproc_per_node="${NPROC_PER_NODE}" run_train.py --config-file "${CONFIG_FILE}"
