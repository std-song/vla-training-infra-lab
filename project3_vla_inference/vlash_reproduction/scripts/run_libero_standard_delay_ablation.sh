#!/usr/bin/env bash
set -euo pipefail

PROJ=${PROJ:-/root/autodl-tmp/vla-infra-project3-pi05}
export PYTHONPATH="$PROJ/repos/vlash:$PROJ/vlash_reproduction/scripts"
export LIBERO_CONFIG_PATH="$PROJ/libero_runtime/config"
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export HF_HOME="$PROJ/hf_cache"
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6
export VLASH_PALIGEMMA_TOKENIZER_PATH="$PROJ/models/paligemma-3b-pt-224-tokenizer"

exec "$PROJ/venv/bin/python" \
  "$PROJ/vlash_reproduction/scripts/evaluate_libero_closed_loop.py" \
  --standard-policy "$PROJ/outputs/pi05_libero_standard_5000/checkpoints/005000/pretrained_model" \
  --standard-loader vlash \
  --out-dir "$PROJ/results/libero_standard_delay_ablation_5000" \
  --suites libero_spatial \
  --task-ids 3 \
  --episodes 10 \
  --delays 1 2 4 \
  --conditions standard_naive standard_skip \
  --replan-interval 10 \
  --seed 1000
