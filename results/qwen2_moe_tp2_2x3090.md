# Qwen2-MoE TP=2 Distributed Smoke on 2xRTX 3090

Date: 2026-07-06

This report records the first tensor-parallel validation for the Qwen2-MoE training-infrastructure project. It reuses the 2xRTX 3090 AutoDL environment and enables Nanotron tensor parallelism while keeping data, pipeline, and expert parallelism disabled.

## Goal

Validate the tensor-parallel axis independently before composing it with DP, PP, or EP:

- `torchrun --nproc_per_node=2`
- `dp=1, tp=2, pp=1, expert_parallel_size=1`
- tensor-parallel rank-aware logs with `TP=0` and `TP=1`
- finite-loss training under sharded linear layers
- TP checkpoint save
- TP checkpoint resume from step 100 to step 120

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
| Total parameters reported by Nanotron | 126M |
| Local parameters per TP rank | 62.9M |
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
| TP | 2 |
| PP | 1 |
| EP | 1 |
| TP mode | `ALL_REDUCE` |
| Micro batch size | 1 |
| Global batch tokens | 512 |

Note: the parameter count differs from the non-TP baseline report because Nanotron reports both total and local sharded parameter views under tensor parallelism. The key evidence for this milestone is the per-rank sharding, finite training, and TP-aware checkpoint layout.

## 100-Step TP=2 Run

Config: [`../configs/qwen2_moe/config_qwen2_moe_tp2_100step.yaml`](../configs/qwen2_moe/config_qwen2_moe_tp2_100step.yaml)

Launch command:

```bash
CUDA_DEVICE_MAX_CONNECTIONS=1 PYTHONPATH=src torchrun --nproc_per_node=2 \
  run_train.py --config-file examples/smoke/config_qwen2_moe_tp2_100step.yaml
```

| Metric | Value |
| --- | ---: |
| Logged steps parsed | 10 |
| Steady window | steps 31..91 |
| Avg time / iteration | 49.67 ms |
| Avg total throughput | 10,311 tokens/s |
| Avg throughput / GPU | 5,154 tokens/s/GPU |
| Avg model TFLOPs / GPU | 0.714 |
| Final logged step | 91 / 100 |
| Final logged loss | 9.20 |
| Nanotron observed peak allocated | 1,223.85 MiB |
| Nanotron observed peak reserved | 1,314.00 MiB |
| GPU0 max sampled memory | 1,892 MiB |
| GPU1 max sampled memory | 1,892 MiB |
| GPU0 max sampled util | 90% |
| GPU1 max sampled util | 100% |
| GPU0 max sampled power | 237.97 W |
| GPU1 max sampled power | 256.88 W |
| Checkpoint directory size after steps 100 and 120 | 3.1 GiB |

Final logged training line from the 100-step run:

```text
iteration: 91 / 100 | consumed_tokens: 46.6K | time_per_iteration_ms: 49.8 | tokens_per_sec: 10.3K | tokens_per_sec_per_gpu: 5.14K | global_batch_size: 512 | grad_norm: 144K | lm_loss: 9.2 | lr: 3.3e-05 | model_tflops_per_gpu: 0.712
```

The run completed all 100 steps and saved `checkpoints/qwen2_moe_tp2_100step/100`.

## Checkpoint Resume: 100 -> 120

Config: [`../configs/qwen2_moe/config_qwen2_moe_tp2_resume_120step.yaml`](../configs/qwen2_moe/config_qwen2_moe_tp2_resume_120step.yaml)

Resume metadata confirmed that Nanotron loaded a TP-aware checkpoint:

```text
CheckpointMetadata(... tp=2, dp=1, consumed_train_samples=100, last_train_step=100, consumed_tokens_total=51200 ...)
```

The trainer resumed from the expected iteration:

```text
start_iteration_step: 100 | consumed_tokens_total: 51200
```

Final resumed training line:

```text
iteration: 120 / 120 | consumed_tokens: 61.4K | time_per_iteration_ms: 49.1 | tokens_per_sec: 10.4K | tokens_per_sec_per_gpu: 5.22K | global_batch_size: 512 | grad_norm: 1.34K | lm_loss: 9.2 | lr: 0.00019 | model_tflops_per_gpu: 0.722
```

The resume run saved `checkpoints/qwen2_moe_tp2_100step/120`.

## Checkpoint Layout Evidence

The TP checkpoint includes separate optimizer, scheduler, and random states for both tensor-parallel ranks:

```text
lr_scheduler/lr_scheduler_pp-0-of-1_tp-0-of-2_exp-0-of-1.pt
lr_scheduler/lr_scheduler_pp-0-of-1_tp-1-of-2_exp-0-of-1.pt
optimizer/optimizer_pp-0-of-1_tp-0-of-2_exp-0-of-1.pt
optimizer/optimizer_pp-0-of-1_tp-1-of-2_exp-0-of-1.pt
random/tp-0-of-2_dp-0-of-1_pp-0-of-1.pt
random/tp-1-of-2_dp-0-of-1_pp-0-of-1.pt
```

## Analysis

TP=2 is intentionally not expected to improve throughput for this small model on PCIe RTX 3090s. Compared with the single-GPU baseline v2, total throughput stays near 10.3K tokens/s while throughput per GPU drops to about 5.15K tokens/s/GPU. That is expected because tensor parallelism introduces collectives in the forward/backward path and the model is too small to amortize the communication overhead.

The useful result is functional rather than speedup-oriented: tensor-parallel model construction works, parameters are sharded across TP ranks, both GPUs are active, the loss remains finite, checkpoint artifacts are TP-rank aware, and resume restores TP metadata correctly.

This completes the second distributed axis after DP=2. The next immediate milestone is PP=2 on the same two-GPU machine, then an 8-GPU run that composes DP with TP/PP.

## Limitations

- This is single-node TP only; no multi-node tensor parallel communication is covered.
- EP is still disabled in this run.
- Dummy data avoids dataloader and VLA data-format bottlenecks.
- The model is small enough that TP communication overhead dominates the performance story.
- Step logging interval means the last logged metric before the 100-step checkpoint is step 91.
