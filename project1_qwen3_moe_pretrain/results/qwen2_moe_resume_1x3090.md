# Checkpoint Resume Validation

This document records the second validation step for the Qwen2-MoE training path: restoring training state from a Nanotron checkpoint and advancing training.

## Goal

Verify that checkpointing captures enough state to continue training reliably:

- model weights
- optimizer state
- LR scheduler state
- random states
- consumed samples / step metadata

## Procedure

Start from the completed smoke-run checkpoint:

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

## Result

Status: completed.

The run loaded the step-5 checkpoint and advanced training to step 7.

```text
iteration: 7 / 7 | consumed_tokens: 1.79K | time_per_iteration_ms: 19.8 | tokens_per_sec: 12.9K | tokens_per_sec_per_gpu: 12.9K | global_batch_size: 256 | grad_norm: 4.14M | lm_loss: 8.15 | lr: 0.0003 | model_tflops_per_gpu: 0.0863
Saving checkpoint at checkpoints/qwen2_moe_smoke/7
```

## Notes

The dummy CLM dataloader triggered a metadata sanity-check issue in upstream Nanotron because global consumed tokens can be nonzero while per-dataset folder counters are empty. This repository documents the local compatibility patch in [`../patches/nanotron_dummy_resume_metadata.patch`](../patches/nanotron_dummy_resume_metadata.patch). The patch keeps the strict check for real datasets while allowing dummy-data resume validation.
