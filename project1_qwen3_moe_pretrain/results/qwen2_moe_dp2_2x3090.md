# Qwen2-MoE DP=2 Distributed Smoke on 2xRTX 3090

Date: 2026-07-06

This report records the first multi-GPU distributed validation for the Qwen2-MoE training-infrastructure project. It reuses the 75.5M-parameter baseline v2 model and runs Nanotron with two local processes on two RTX 3090 GPUs.

## Goal

Validate the first distributed axis before attempting TP, PP, or EP:

- `torchrun --nproc_per_node=2`
- `dp=2, tp=1, pp=1, expert_parallel_size=1`
- rank-aware Nanotron logs with `DP=0` and `DP=1`
- data-parallel training stability
- DP checkpoint save
- DP checkpoint resume from step 200 to step 220

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
| Parameters per rank | 75.5M |
| Layers | 4 |
| Hidden size | 512 |
| Sequence length | 512 |
| MoE experts | 8 |
| Router top-k | 1 |
| Shared expert | enabled |
| Attention | FlashAttention 2 |
| Expert MLP | GroupedGEMM |
| Precision | BF16 |
| DP | 2 |
| TP | 1 |
| PP | 1 |
| EP | 1 |
| Micro batch size | 1 |
| Global batch tokens | 1.02K |

## 200-Step DP=2 Run

Config: [`../configs/qwen2_moe/config_qwen2_moe_dp2_200step.yaml`](../configs/qwen2_moe/config_qwen2_moe_dp2_200step.yaml)

Launch command:

```bash
CUDA_DEVICE_MAX_CONNECTIONS=1 PYTHONPATH=src torchrun --nproc_per_node=2 \
  run_train.py --config-file examples/smoke/config_qwen2_moe_dp2_200step.yaml
```

| Metric | Value |
| --- | ---: |
| Logged steps parsed | 20 |
| Steady window | steps 51..191 |
| Avg time / iteration | 56.97 ms |
| Avg total throughput | 17,987 tokens/s |
| Avg throughput / GPU | 8,989 tokens/s/GPU |
| Max total throughput | 18,300 tokens/s |
| Min total throughput | 17,700 tokens/s |
| Avg model TFLOPs / GPU | 1.244 |
| Final logged step | 191 / 200 |
| Final logged loss | 9.18 |
| Nanotron observed peak allocated | 1,607.50 MiB |
| Nanotron observed peak reserved | 1,750.00 MiB |
| GPU0 max sampled memory | 2,328 MiB |
| GPU1 max sampled memory | 2,312 MiB |
| GPU0 max sampled util | 89% |
| GPU1 max sampled util | 100% |
| GPU0 max sampled power | 248.47 W |
| GPU1 max sampled power | 245.95 W |
| Checkpoint size at step 200 | 1009 MiB |

Final logged training line:

```text
iteration: 191 / 200 | consumed_tokens: 196K | time_per_iteration_ms: 56.9 | tokens_per_sec: 18K | tokens_per_sec_per_gpu: 9K | global_batch_size: 1.02K | grad_norm: 201K | lm_loss: 9.18 | lr: 1.58e-05 | model_tflops_per_gpu: 1.25
```

The run completed all 200 steps and saved `checkpoints/qwen2_moe_dp2_200step/200`.

## Checkpoint Resume: 200 -> 220

Config: [`../configs/qwen2_moe/config_qwen2_moe_dp2_resume_220step.yaml`](../configs/qwen2_moe/config_qwen2_moe_dp2_resume_220step.yaml)

Resume metadata confirmed that Nanotron loaded a DP-aware checkpoint:

```text
CheckpointMetadata(... tp=1, dp=2, consumed_train_samples=400, last_train_step=200, consumed_tokens_total=204800 ...)
```

Final resumed training line:

```text
iteration: 220 / 220 | consumed_tokens: 225K | time_per_iteration_ms: 58.8 | tokens_per_sec: 17.4K | tokens_per_sec_per_gpu: 8.71K | global_batch_size: 1.02K | grad_norm: 125K | lm_loss: 9.16 | lr: 0.000262 | model_tflops_per_gpu: 1.21
```

The resume run saved `checkpoints/qwen2_moe_dp2_200step/220`. The checkpoint directory containing steps 200 and 220 occupied about 2.0 GiB.

## Checkpoint Layout Evidence

The DP checkpoint includes data-parallel random states for both DP ranks:

```text
random/tp-0-of-1_dp-0-of-2_pp-0-of-1.pt
random/tp-0-of-1_dp-1-of-2_pp-0-of-1.pt
```

The model, optimizer, scheduler, config, and metadata are stored once because TP=PP=EP=1 and Nanotron's current save path writes shared state for this configuration.

## Analysis

This is the first distributed training milestone in the project. Compared with the single-GPU baseline v2, total throughput increased from about 10.5K tokens/s to about 18.0K tokens/s. Per-GPU throughput dropped from about 10.5K tokens/s/GPU to about 9.0K tokens/s/GPU, which is expected because DP introduces gradient synchronization and multi-process overhead.

The important result is not perfect scaling; it is that the distributed training path is operational: two ranks launch, both GPUs are active, loss remains finite, checkpointing works, and resume restores DP metadata and advances training.

This is now a credible bridge from single-GPU correctness/profiling to multi-GPU infrastructure work. The next natural experiments are TP=2 and PP=2 on the same two-GPU machine.

## Limitations

- This is single-node DP only; no multi-node network or NCCL/RDMA tuning is covered.
- TP, PP, and EP are still disabled in this run.
- Dummy data avoids dataloader and VLA decoding bottlenecks.
- Step logging interval means the last logged metric before checkpoint is step 191 for the 200-step run.
