# Checkpoint Resume Validation

This document describes the second validation step for the Qwen2-MoE training path: restoring training state from a Nanotron checkpoint and advancing training.

## Goal

Verify that checkpointing captures enough state to continue training reliably:

- model weights
- optimizer state
- LR scheduler state
- random states
- consumed samples / step metadata

## Procedure

Start from the completed smoke run checkpoint:

```text
checkpoints/qwen2_moe_smoke/5
```

Copy the resume config into the Nanotron checkout:

```bash
cd /root/autodl-tmp/vla-infra/nanotron
mkdir -p examples/smoke
cp /root/autodl-tmp/vla-infra/vla-training-infra-lab/configs/qwen2_moe/config_qwen2_moe_resume.yaml \
  examples/smoke/config_qwen2_moe_resume.yaml
```

Run resume training:

```bash
CUDA_DEVICE_MAX_CONNECTIONS=1 PYTHONPATH=src torchrun --nproc_per_node=1 \
  run_train.py --config-file examples/smoke/config_qwen2_moe_resume.yaml
```

## Expected Evidence

The log should show that Nanotron loads from step 5 and advances to step 7. A successful result should include a final line similar to:

```text
iteration: 7 / 7
Saving checkpoint at checkpoints/qwen2_moe_smoke/7
```

## Result

Status: pending.
