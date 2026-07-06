# Qwen2-MoE Single-GPU Baseline on RTX 3090

Date: 2026-07-06

This report records the first measurable baseline for the Qwen2-MoE training-infrastructure path. The goal is not model quality; the goal is to validate the Nanotron training path, checkpoint behavior, MoE kernels, and coarse GPU resource profile on a single RTX 3090 before scaling to multiple GPUs.

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

## Model and Training Setup

| Field | Value |
| --- | --- |
| Model family | Qwen2-style decoder with MoE layers |
| Parameters | 2.36M total |
| Hidden size | 128 |
| Layers | 2 |
| Attention heads | 4 |
| Sequence length | 128 |
| Vocabulary size | 4096 |
| MoE experts | 4 |
| Router top-k | 1 |
| Shared expert | enabled |
| Expert MLP | `grouped_gemm.ops.gmm` |
| Attention path | `flash_attention_2` |
| Precision | BF16 |
| Data | Nanotron dummy CLM generator |
| Global batch size | 256 tokens |
| Parallelism | DP=1, TP=1, PP=1, EP=1 |

## Validation Steps

1. Installed Nanotron runtime dependencies on the RTX 3090 instance.
2. Fixed environment compatibility for PyTorch 2.1.2 / CUDA 11.8.
3. Passed Nanotron MoE smoke test: `PYTHONPATH=src pytest -q tests/test_moe.py`.
4. Ran 5-step Qwen2-MoE smoke training and saved checkpoint at step 5.
5. Resumed checkpoint from step 5 and advanced to step 7.
6. Ran 20-step baseline profiling.
7. Ran 100-step baseline profiling with `nvidia-smi` sampling and checkpoint save at step 100.

## Smoke Result

The 5-step smoke run validated the end-to-end forward/backward/update/checkpoint path.

```text
iteration: 5 / 5 | consumed_tokens: 1.28K | time_per_iteration_ms: 20.3 | tokens_per_sec: 12.6K | tokens_per_sec_per_gpu: 12.6K | global_batch_size: 256 | grad_norm: 1.64 | lm_loss: 8.35 | lr: 1e-05 | model_tflops_per_gpu: 0.0843
Saving checkpoint at checkpoints/qwen2_moe_smoke/5
```

## Checkpoint Resume Result

The resume run loaded the step-5 checkpoint and advanced to step 7.

```text
iteration: 7 / 7 | consumed_tokens: 1.79K | time_per_iteration_ms: 19.8 | tokens_per_sec: 12.9K | tokens_per_sec_per_gpu: 12.9K | global_batch_size: 256 | grad_norm: 4.14M | lm_loss: 8.15 | lr: 0.0003 | model_tflops_per_gpu: 0.0863
Saving checkpoint at checkpoints/qwen2_moe_smoke/7
```

## 20-Step Baseline Result

The first profiling run kept per-step logging enabled and produced a checkpoint at step 20.

| Metric | Value |
| --- | ---: |
| Final step | 20 / 20 |
| Final time / iteration | 20.8 ms |
| Final throughput | 12.3K tokens/s |
| Final loss | 8.35 |
| Final model TFLOPs / GPU | 0.0822 |
| Nanotron peak allocated memory | 289.34 MiB |
| Nanotron peak reserved memory | 308 MiB |
| Max sampled `nvidia-smi` memory | 993 MiB |
| Max sampled GPU util | 91% |
| Max sampled power | 282.65 W |
| Checkpoint size | 32 MiB |

## 100-Step Baseline Result

The 100-step run provides a slightly more stable baseline, although the model is still intentionally tiny and the total runtime is only a few seconds.

| Metric | Value |
| --- | ---: |
| Parsed train steps | 100 |
| Steady window | steps 10..100 |
| Avg time / iteration | 27.09 ms |
| Avg throughput | 9,860 tokens/s |
| Max throughput | 15,400 tokens/s |
| Min throughput | 7,060 tokens/s |
| Final step | 100 / 100 |
| Final time / iteration | 35.30 ms |
| Final throughput | 7,250 tokens/s |
| Final loss | 8.34 |
| Final model TFLOPs / GPU | 0.0485 |
| Max sampled GPU memory | 993 MiB |
| Max sampled GPU util | 98% |
| Max sampled power | 254.60 W |
| Checkpoint size | 32 MiB |

Final train line:

```text
iteration: 100 / 100 | consumed_tokens: 25.6K | time_per_iteration_ms: 35.3 | tokens_per_sec: 7.25K | tokens_per_sec_per_gpu: 7.25K | global_batch_size: 256 | grad_norm: 50.1 | lm_loss: 8.34 | lr: 1e-05 | model_tflops_per_gpu: 0.0485
```

## Analysis

This milestone proves that the local Qwen2-MoE training path is functional on a commodity 24 GiB GPU. The run exercises router top-k routing, token permutation/unpermutation, grouped expert GEMM, shared expert computation, FlashAttention, BF16 training, optimizer update, checkpoint save, and checkpoint resume.

The measured GPU memory is far below the 24 GiB limit because the model is deliberately small. This is useful as a correctness baseline, but it is not yet a meaningful utilization benchmark. The gap between Nanotron's allocator peak and `nvidia-smi` memory is expected: `nvidia-smi` includes CUDA context, kernels, framework overhead, and reserved memory outside model tensors.

The 100-step throughput drops from the early 10K-15K tokens/s range to roughly 7K-8K tokens/s near the end. Because this workload is tiny and logs every iteration, the numbers are sensitive to Python scheduling, logging overhead, CUDA warmup, LR schedule/end-of-run effects, and the one-second granularity of `nvidia-smi`. The next profiling run should use a larger model, fewer log lines, and 500-1000 steps.

The current result should be presented as a baseline validation, not as a final performance claim.

## Reproduction

Inside the Nanotron checkout:

```bash
cd /root/autodl-tmp/vla-infra/nanotron
mkdir -p examples/smoke
cp /root/autodl-tmp/vla-infra/vla-training-infra-lab/configs/qwen2_moe/config_qwen2_moe_baseline_100step.yaml \
  examples/smoke/config_qwen2_moe_baseline_100step.yaml
bash /root/autodl-tmp/vla-infra/vla-training-infra-lab/scripts/run_qwen2_moe_profile_100step.sh \
  examples/smoke/config_qwen2_moe_baseline_100step.yaml \
  qwen2_moe_baseline_100step
```

## Resume/Checkpoint Evidence

The 100-step checkpoint contains model state, optimizer state, scheduler state, random state, config, and metadata.

```text
checkpoints/qwen2_moe_baseline_100step/100/checkpoint_metadata.json
checkpoints/qwen2_moe_baseline_100step/100/config.yaml
checkpoints/qwen2_moe_baseline_100step/100/lr_scheduler/lr_scheduler_pp-0-of-1_tp-0-of-1_exp-0-of-1.pt
checkpoints/qwen2_moe_baseline_100step/100/model_config.json
checkpoints/qwen2_moe_baseline_100step/100/optimizer/optimizer_config.json
checkpoints/qwen2_moe_baseline_100step/100/optimizer/optimizer_pp-0-of-1_tp-0-of-1_exp-0-of-1.pt
checkpoints/qwen2_moe_baseline_100step/100/random/tp-0-of-1_dp-0-of-1_pp-0-of-1.pt
checkpoints/qwen2_moe_baseline_100step/latest.txt
```

## Limitations

- EP is still `1`; this run does not validate cross-rank expert dispatch.
- TP and PP are still `1`; this run does not validate tensor or pipeline communication.
- Dummy data avoids dataloader bottlenecks and does not represent VLA data IO.
- The model is too small for stable GPU utilization measurements.
- RTX 3090 uses PCIe and consumer networking assumptions, so multi-node conclusions must be made carefully.
