#!/usr/bin/env bash
set -euo pipefail

python --version
nvcc --version || true
nvidia-smi
python - <<'PY'
import torch
print("torch", torch.__version__, "cuda", torch.version.cuda)
print("cuda_available", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu", torch.cuda.get_device_name(0))
PY
python -c "import grouped_gemm.ops as ops; print('grouped_gemm ok')"
python -c "from flash_attn.ops.triton.layer_norm import layer_norm_fn; print('flash_attn layer_norm ok')"
