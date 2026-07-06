# Qwen2-MoE Baseline v2 and Recompute A/B on RTX 3090

Date: 2026-07-06

This report records the stronger single-GPU baseline after the initial tiny smoke/profile runs. The purpose is to make the experiment more representative for training-infrastructure discussion: larger model, longer sequence length, 500 training steps, checkpoint/resume, and activation recomputation comparison.

## Environment

| Item | Value |
| --- | --- |
| Cloud | AutoDL / SeeTaCloud container |
| GPU | 1x NVIDIA GeForce RTX 3090, 24 GiB |
| Driver | 595.58.03 |
| Python | 3.10.8 |
| PyTorch | 2.1.2+cu118 |
| CUDA toolkit | 11.8 |
| Transformers | 4.41.2 |
| flash-attn | 2.5.8 |
| Nanotron | 0.4 checkout from `huggingface/nanotron` |

## Model

| Field | Baseline v2 |
| --- | ---: |
| Parameters | 75.5M |
| Layers | 4 |
| Hidden size | 512 |
| Attention heads | 8 |
| Sequence length | 512 |
| Vocabulary size | 8192 |
| MoE experts | 8 |
| Router top-k | 1 |
| MoE layers | 0, 1, 2, 3 |
| Shared expert | enabled |
| Expert MLP | GroupedGEMM |
| Attention | FlashAttention 2 |
| Precision | BF16 |
| Parallelism | DP=1, TP=1, PP=1, EP=1 |
| Micro batch size | 1 |
| Global batch tokens | 512 |

## 500-Step Baseline

Config: [`../configs/qwen2_moe/config_qwen2_moe_baseline_v2_500step.yaml`](../configs/qwen2_moe/config_qwen2_moe_baseline_v2_500step.yaml)

| Metric | Value |
| --- | ---: |
| Parsed logged steps | 50 |
| Logged steady window | steps 51..491 |
| Avg time / iteration | 49.59 ms |
| Avg throughput | 10,544 tokens/s |
| Max throughput | 17,200 tokens/s |
| Min throughput | 8,860 tokens/s |
| Avg model TFLOPs / GPU | 1.460 |
| Last logged step | 491 / 500 |
| Last logged throughput | 9,430 tokens/s |
| Last logged LM loss | 9.16 |
| Nanotron peak allocated memory | 1,464.44 MiB |
| Nanotron peak reserved memory | 1,586.00 MiB |
| Max sampled `nvidia-smi` memory | 2,271 MiB |
| Max sampled GPU util | 45% |
| Max sampled power | 221.84 W |
| Max sampled temperature | 46 C |
| Checkpoint size at step 500 | 1009 MiB |

Note: `iteration_step_info_interval=10`, so the last printed training metric is step 491. The run completed all 500 steps and saved `checkpoints/qwen2_moe_baseline_v2_500step/500`.

## Checkpoint Resume: 500 -> 520

The step-500 checkpoint was loaded and training continued to step 520.

Evidence from checkpoint metadata:

```text
last_train_step=500
consumed_train_samples=500
consumed_tokens_total=256000
```

Final resumed training line:

```text
iteration: 520 / 520 | consumed_tokens: 266K | time_per_iteration_ms: 46 | tokens_per_sec: 11.1K | tokens_per_sec_per_gpu: 11.1K | global_batch_size: 512 | grad_norm: 35.9 | lm_loss: 9.16 | lr: 0.000293 | model_tflops_per_gpu: 1.54
```

The resume run saved a new checkpoint at `checkpoints/qwen2_moe_baseline_v2_500step/520`. The checkpoint directory containing steps 500 and 520 occupied about 2.0 GiB.

## Activation Recompute A/B

Config with recompute: [`../configs/qwen2_moe/config_qwen2_moe_baseline_v2_recompute_500step.yaml`](../configs/qwen2_moe/config_qwen2_moe_baseline_v2_recompute_500step.yaml)

| Metric | No recompute | Recompute |
| --- | ---: | ---: |
| Avg time / iteration | 49.59 ms | 62.09 ms |
| Avg throughput | 10,544 tokens/s | 8,274 tokens/s |
| Avg model TFLOPs / GPU | 1.460 | 1.145 |
| Nanotron peak allocated | 1,464.44 MiB | 1,470.00 MiB |
| Nanotron peak reserved | 1,586.00 MiB | 1,568.00 MiB |
| Max sampled GPU memory | 2,271 MiB | 2,253 MiB |
| Max sampled GPU util | 45% | 41% |
| Max sampled power | 221.84 W | 261.60 W |
| Checkpoint size | 1009 MiB | 1009 MiB |

Derived comparison:

| Metric | Change |
| --- | ---: |
| Throughput change | -21.5% |
| Step time change | +25.2% |
| Max sampled memory change | -0.8% |
| Peak reserved memory change | -1.1% |

## Analysis

The larger baseline is a better portfolio artifact than the initial tiny model. It validates the same core path while creating a meaningful optimizer/checkpoint footprint: 75.5M parameters, 1 GiB checkpoint, and roughly 2.2 GiB sampled GPU memory.

Activation recomputation did not materially reduce memory for this configuration. That is an important result rather than a failure: the current memory footprint is dominated by model states, optimizer states, CUDA context, and framework overhead more than by saved activations. Recompute adds extra forward work, so throughput drops by about 21.5% without a useful memory win at this scale.

The next recompute experiment should use a configuration where activation memory is more dominant, such as longer sequence length, more layers, or larger micro-batch size. For the immediate project roadmap, however, this run is already enough to demonstrate an A/B profiling methodology and a hardware-aware interpretation.

## What This Validates

- Qwen2-MoE BF16 training on RTX 3090
- FlashAttention path
- GroupedGEMM expert MLP path
- Router top-k=1 routing
- Shared expert path
- 500-step training stability
- Step-500 checkpoint save
- Step-500 to step-520 checkpoint resume
- Activation recomputation A/B comparison
- Coarse memory, throughput, power, and checkpoint-size analysis

## Limitations

- EP remains `1`; cross-rank expert dispatch is not validated yet.
- TP and PP remain `1`; no tensor/pipeline communication is validated by this report.
- Dummy data removes dataloader and real VLA sample decoding from the profile.
- `nvidia-smi` sampling at one-second intervals is coarse for short runs.
- GPU utilization is low because the model is still modest and per-step Python/logging overhead is visible.
