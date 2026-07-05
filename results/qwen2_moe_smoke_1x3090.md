# Qwen2-MoE Smoke Training on 1x RTX 3090

## Goal

Validate the minimum end-to-end Qwen2-MoE training path before scaling to multi-GPU experiments.

This smoke run verifies:

- Qwen2-MoE model construction.
- Dummy CLM dataloader.
- Forward and backward pass.
- Optimizer step.
- BF16 training path.
- Checkpoint save.

## Environment

- Provider: AutoDL / SeeTaCloud
- GPU: NVIDIA GeForce RTX 3090, 24 GiB
- Python: 3.10.8
- PyTorch: 2.1.2+cu118
- CUDA toolkit: 11.8
- flash-attn: 2.5.8 target
- grouped_gemm: `fanshiqing/grouped_gemm@main`

## Commands

```bash
cd /root/autodl-tmp/vla-infra/nanotron
PYTHONPATH=src pytest -q tests/test_moe.py
CUDA_DEVICE_MAX_CONNECTIONS=1 PYTHONPATH=src torchrun --nproc_per_node=1 \
  run_train.py --config-file examples/smoke/config_qwen2_moe_smoke.yaml
```

## Observed Output

```text
tests/test_moe.py: 1 passed
iteration: 5 / 5
consumed_tokens: 1.28K
time_per_iteration_ms: 20.3
tokens_per_sec: 12.6K
tokens_per_sec_per_gpu: 12.6K
global_batch_size: 256
grad_norm: 1.64
lm_loss: 8.35
lr: 1e-05
model_tflops_per_gpu: 0.0843
Saving checkpoint at checkpoints/qwen2_moe_smoke/5
```

## Interpretation

The smoke run confirms that the Nanotron Qwen2-MoE path can execute an end-to-end training loop on a single RTX 3090. This gives a stable base for checkpoint resume, multi-GPU DP, and later EP experiments.

## Next Validation

```bash
find checkpoints/qwen2_moe_smoke/5 -maxdepth 2 -type f | sort
```

Then update the config:

```yaml
resume_checkpoint_path: checkpoints/qwen2_moe_smoke/5
train_steps: 7
```

Rerun the smoke command and check that training resumes from step 5 and advances to step 7.

