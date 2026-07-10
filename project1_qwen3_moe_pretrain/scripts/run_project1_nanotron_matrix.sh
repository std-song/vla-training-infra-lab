#!/usr/bin/env bash
set -euo pipefail

# Run from the Nanotron repository root on AutoDL.
# The clean configs should be copied into examples/smoke/ first.

LOG_DIR="${LOG_DIR:-profiling_logs}"
mkdir -p "${LOG_DIR}"

run_job() {
  local name="$1"
  local nproc="$2"
  local config="$3"
  echo "===== ${name} ====="
  CUDA_DEVICE_MAX_CONNECTIONS=1 PYTHONPATH=src \
    torchrun --nproc_per_node="${nproc}" \
    run_train.py --config-file "${config}" \
    > "${LOG_DIR}/${name}.log" 2>&1
  tail -20 "${LOG_DIR}/${name}.log"
}

run_job qwen3_moe_style_100m_single20_clean 1 examples/smoke/config_qwen3_moe_style_100m_single20_clean.yaml
run_job qwen3_moe_style_100m_dp2_20_clean 2 examples/smoke/config_qwen3_moe_style_100m_dp2_20_clean.yaml
run_job qwen3_moe_style_100m_tp2_20_clean 2 examples/smoke/config_qwen3_moe_style_100m_tp2_20_clean.yaml
run_job qwen3_moe_style_100m_pp2_20_clean 2 examples/smoke/config_qwen3_moe_style_100m_pp2_20_clean.yaml
run_job qwen3_moe_style_100m_ep2_20_clean 2 examples/smoke/config_qwen3_moe_style_100m_ep2_20_clean.yaml
