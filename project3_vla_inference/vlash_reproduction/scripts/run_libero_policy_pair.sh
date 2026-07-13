#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=/root/autodl-tmp/vla-infra-project3-pi05
cd "$PROJECT_ROOT"

export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6
export VLASH_PALIGEMMA_TOKENIZER_PATH="$PROJECT_ROOT/models/paligemma-3b-pt-224-tokenizer"

"$PROJECT_ROOT/venv/bin/vlash" train \
  "$PROJECT_ROOT/vlash_reproduction/rendered/pi05_libero_stale_5000.yaml"

"$PROJECT_ROOT/venv/bin/vlash" train \
  "$PROJECT_ROOT/vlash_reproduction/rendered/pi05_libero_learned_5000.yaml"
