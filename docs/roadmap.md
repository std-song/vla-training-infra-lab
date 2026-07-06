# Roadmap

This repository is organized as a staged VLA training-infrastructure portfolio.

## Stage 1: Qwen2-MoE Single-GPU Correctness

Status: completed.

- Install Nanotron dependencies on RTX 3090.
- Validate grouped_gemm and flash-attn imports.
- Run `tests/test_moe.py`.
- Run 5-step Qwen2-MoE dummy-data training.
- Save checkpoint at step 5.

## Stage 2: Checkpoint Resume

Status: completed.

Goal: prove training-state reliability.

Completed evidence:

- Resumed from `checkpoints/qwen2_moe_smoke/5`.
- Extended training from step 5 to step 7.
- Loaded optimizer and scheduler state.
- Saved checkpoint at `checkpoints/qwen2_moe_smoke/7`.

## Stage 3: Single-GPU Baseline Profiling

Status: completed.

Completed:

- 20-step tiny baseline profile.
- 100-step tiny baseline profile.
- 75.5M-parameter 500-step baseline profile.
- Step-500 to step-520 checkpoint resume.
- Activation recomputation A/B.
- Memory, throughput, utilization, power, and checkpoint-size analysis.

Latest baseline v2 result:

| Metric | Value |
| --- | ---: |
| Parameters | 75.5M |
| Avg throughput, logged steps >= 50 | 10,544 tokens/s |
| Avg step time, logged steps >= 50 | 49.59 ms |
| Max sampled GPU memory | 2,271 MiB |
| Checkpoint size | 1009 MiB |
| Recompute throughput change | -21.5% |
| Recompute memory change | -0.8% sampled memory |

## Stage 4: Multi-GPU Data Parallel

Status: completed for `dp=2`; larger DP scaling pending.

Goal: validate distributed launcher and gradient synchronization.

Suggested configs:

| Case | DP | TP | PP | EP | Purpose |
| --- | ---: | ---: | ---: | ---: | --- |
| dp_2 | 2 | 1 | 1 | 1 | distributed smoke, completed |
| dp_4 | 4 | 1 | 1 | 1 | throughput scaling |
| dp_8 | 8 | 1 | 1 | 1 | 8x3090 baseline |

## Stage 5: Tensor and Pipeline Parallel

Status: completed for `tp=2` and `pp=2`; 8-GPU composition pending.

Goal: validate dense-model distributed axes before introducing cross-rank experts.

| Case | GPUs | DP | TP | PP | EP | Purpose |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| tp_2 | 2 | 1 | 2 | 1 | 1 | tensor-parallel smoke, completed |
| pp_2 | 2 | 1 | 1 | 2 | 1 | pipeline-parallel smoke, completed |
| tp2_dp4 | 8 | 4 | 2 | 1 | 1 | TP+DP scaling |
| pp2_dp4 | 8 | 4 | 1 | 2 | 1 | PP+DP scaling |

## Stage 6: Expert Parallel Experiments

Goal: study MoE memory/communication tradeoffs on PCIe GPUs.

| Case | DP | TP | PP | EP | Purpose |
| --- | ---: | ---: | ---: | ---: | --- |
| ep_2 | 4 | 1 | 1 | 2 | expert placement |
| ep_4 | 2 | 1 | 1 | 4 | communication stress |
| ep_8 | 1 | 1 | 1 | 8 | all-to-all limit |

Caveat: do not claim true EP until cross-rank expert token dispatch and checkpoint layout are validated under `expert_parallel_size > 1`.

## Stage 7: SmolVLA Finetuning

Goal: connect the training-infra work to VLA/robotics data.

Tasks:

- Keep synthetic SmolVLA smoke test reproducible.
- Add LeRobot-style collator.
- Support image/language/state/action batch schema.
- Validate action-head or expert-only finetuning.
- Track dataloader throughput and GPU idle time.

## Stage 8: Inference Latency Optimization

Goal: profile VLA action generation latency.

Candidate optimizations:

- FlashAttention / SDPA comparison.
- KV-cache reuse for language prefix.
- image encoder feature caching.
- CUDA Graph for fixed-shape inference.
- `torch.compile` or Triton kernels for action modules.
- DFlash-style acceleration where token decoding dominates.
