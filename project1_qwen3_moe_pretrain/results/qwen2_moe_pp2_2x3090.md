# Qwen2-MoE PP=2 Distributed Smoke on 2xRTX 3090

Date: 2026-07-06

This report records pipeline-parallel validation for the Qwen2-MoE training-infrastructure project. It uses two RTX 3090 GPUs, enables Nanotron pipeline parallelism, and keeps data, tensor, and expert parallelism disabled.

## Goal

Validate the pipeline-parallel axis independently before 8-GPU composition:

- `torchrun --nproc_per_node=2`
- `dp=1, tp=1, pp=2, expert_parallel_size=1`
- Qwen2-MoE stage placement across two pipeline ranks
- 1F1B pipeline execution with two microbatches per step
- finite-loss training on the final pipeline stage
- PP checkpoint save
- PP checkpoint resume from step 100 to step 120

## Environment

| Item | Value |
| --- | --- |
| Cloud | AutoDL / SeeTaCloud cloned instance |
| GPUs | 2x NVIDIA GeForce RTX 3090, 24 GiB each |
| Driver | 570.124.04 |
| Python | 3.10.8 |
| PyTorch | 2.1.2+cu118 |
| CUDA build | 11.8 |
| Transformers | 4.41.2 |
| flash-attn | 2.5.8 |
| Nanotron | 0.4 checkout from `huggingface/nanotron` |

## Model and Parallelism

| Field | Value |
| --- | ---: |
| Total parameters | 75.5M |
| Local parameters on PP rank 0 | 54.5M |
| Local parameters on PP rank 1 | 21M |
| Layers | 4 |
| Hidden size | 512 |
| Sequence length | 512 |
| MoE experts | 8 |
| Router top-k | 1 |
| Shared expert | enabled |
| Attention | FlashAttention 2 |
| Expert MLP | GroupedGEMM |
| Precision | BF16 |
| DP | 1 |
| TP | 1 |
| PP | 2 |
| EP | 1 |
| PP engine | `1f1b` |
| Micro batch size | 1 |
| Batch accumulation per replica | 2 |
| Global batch tokens | 1.02K |

Stage placement observed from Nanotron logs:

```text
model.token_position_embeddings | PP: 0/2
model.decoder.0                 | PP: 0/2
model.decoder.1                 | PP: 0/2
model.decoder.2                 | PP: 0/2
model.decoder.3                 | PP: 1/2
model.final_layer_norm          | PP: 1/2
model.lm_head                   | PP: 1/2
loss                            | PP: 1/2
```

## Compatibility Patch

Config-only PP initially exposed two Qwen2 pipeline compatibility issues and one logging issue:

1. Non-owner pipeline ranks received `position_ids` as `TensorPointer`, but Qwen2 tried to call `.numel()` on it.
2. `cu_seqlens` needed to be represented as a `TensorPointer` when it is produced by an earlier pipeline stage.
3. The trainer attempted to log `loss_avg.item()` on PP ranks that do not own the loss.

The minimal patch is archived as [`../patches/nanotron_qwen2_moe_pp2_compat.patch`](../patches/nanotron_qwen2_moe_pp2_compat.patch). This is useful project evidence: PP validation required debugging the model/pipeline boundary, not merely changing a YAML field.

## 100-Step PP=2 Run

Config: [`../configs/qwen2_moe/config_qwen2_moe_pp2_100step.yaml`](../configs/qwen2_moe/config_qwen2_moe_pp2_100step.yaml)

Launch command:

```bash
CUDA_DEVICE_MAX_CONNECTIONS=1 PYTHONPATH=src torchrun --nproc_per_node=2 \
  run_train.py --config-file examples/smoke/config_qwen2_moe_pp2_100step.yaml
```

| Metric | Value |
| --- | ---: |
| Logged steps parsed | 10 |
| Steady window | steps 31..91 |
| Avg time / iteration | 90.46 ms |
| Avg total throughput | 11,357 tokens/s |
| Avg throughput / GPU | 5,674 tokens/s/GPU |
| Avg model TFLOPs / GPU | 0.786 |
| Final logged step | 91 / 100 |
| Final logged loss | 9.20 |
| Nanotron observed peak allocated | 1,124.03 MiB |
| Nanotron observed peak reserved | 1,176.00 MiB |
| GPU0 max sampled memory | 1,762 MiB |
| GPU1 max sampled memory | 1,314 MiB |
| GPU0 max sampled util | 88% |
| GPU1 max sampled util | 100% |
| GPU0 max sampled power | 225.29 W |
| GPU1 max sampled power | 252.70 W |
| Checkpoint size at step 100 | 1009 MiB |

Final logged training line from the 100-step run:

```text
iteration: 91 / 100 | consumed_tokens: 93.2K | time_per_iteration_ms: 90.4 | tokens_per_sec: 11.3K | tokens_per_sec_per_gpu: 5.66K | global_batch_size: 1.02K | grad_norm: 1.4M | lr: 3.3e-05 | lm_loss: 9.2 | model_tflops_per_gpu: 0.784
```

The run completed all 100 steps and saved `checkpoints/qwen2_moe_pp2_100step/100`.

## Checkpoint Resume: 100 -> 120

Config: [`../configs/qwen2_moe/config_qwen2_moe_pp2_resume_120step.yaml`](../configs/qwen2_moe/config_qwen2_moe_pp2_resume_120step.yaml)

Resume metadata confirmed that Nanotron loaded the checkpoint and resumed from the expected step:

```text
CheckpointMetadata(... tp=1, dp=1, consumed_train_samples=200, last_train_step=100, consumed_tokens_total=102400 ...)
start_iteration_step: 100 | consumed_tokens_total: 102400
```

Final resumed training line:

```text
iteration: 120 / 120 | consumed_tokens: 123K | time_per_iteration_ms: 96.9 | tokens_per_sec: 10.6K | tokens_per_sec_per_gpu: 5.28K | global_batch_size: 1.02K | grad_norm: 5.99M | lr: 0.00019 | lm_loss: 9.16 | model_tflops_per_gpu: 0.731
```

The resume run saved `checkpoints/qwen2_moe_pp2_100step/120`. The checkpoint directory containing steps 100 and 120 occupied about 2.0 GiB.

## Checkpoint Layout Evidence

The PP checkpoint includes separate optimizer, scheduler, and random states for both pipeline-parallel ranks:

```text
lr_scheduler/lr_scheduler_pp-0-of-2_tp-0-of-1_exp-0-of-1.pt
lr_scheduler/lr_scheduler_pp-1-of-2_tp-0-of-1_exp-0-of-1.pt
optimizer/optimizer_pp-0-of-2_tp-0-of-1_exp-0-of-1.pt
optimizer/optimizer_pp-1-of-2_tp-0-of-1_exp-0-of-1.pt
random/tp-0-of-1_dp-0-of-1_pp-0-of-2.pt
random/tp-0-of-1_dp-0-of-1_pp-1-of-2.pt
```

## Analysis

This run completes the third independent distributed axis after DP=2 and TP=2. PP=2 is not presented as a speedup result for this tiny model. The model is small, the split is imbalanced, and pipeline bubbles plus point-to-point communication dominate the performance profile. The important result is correctness: stage assignment is visible, both ranks execute, the final stage owns loss computation, training remains finite, checkpointing is PP-rank aware, and resume restores state correctly.

The debugging process is also a useful infrastructure signal. Enabling PP for Qwen2-MoE required handling `TensorPointer` values at the model boundary and avoiding loss logging on non-loss pipeline stages. These are exactly the kinds of edge cases that appear when moving from single-rank model code to distributed training systems.

## Limitations

- This is single-node PP only; no multi-node pipeline communication is covered.
- EP is still disabled in this run.
- Dummy data avoids dataloader and VLA data-format bottlenecks.
- The pipeline split is imbalanced: rank 0 owns embedding plus three decoder layers, while rank 1 owns one decoder layer, final norm, lm head, and loss.
- The repeated `Timer 'iteration_time' already running` warning should be cleaned up before making this a polished upstream contribution.
