# Roadmap

This repository is organized as a staged VLA training-infrastructure portfolio.

## Stage 1: Qwen2-MoE Single-GPU Smoke

Status: completed.

- Install Nanotron dependencies on RTX 3090.
- Validate grouped_gemm and flash-attn imports.
- Run `tests/test_moe.py`.
- Run 5-step Qwen2-MoE dummy-data training.
- Save checkpoint at step 5.

## Stage 2: Checkpoint Resume

Status: in progress.

Goal: prove training-state reliability.

Tasks:

- Resume from `checkpoints/qwen2_moe_smoke/5`.
- Extend `train_steps` from 5 to 7.
- Verify optimizer and scheduler state loading.
- Confirm loss/logging continuity.

## Stage 3: Multi-GPU Data Parallel

Goal: validate distributed launcher and gradient synchronization.

Suggested configs:

| Case | DP | TP | PP | EP | Purpose |
| --- | ---: | ---: | ---: | ---: | --- |
| dp_2 | 2 | 1 | 1 | 1 | distributed smoke |
| dp_4 | 4 | 1 | 1 | 1 | throughput scaling |
| dp_8 | 8 | 1 | 1 | 1 | 8x3090 baseline |

## Stage 4: Expert Parallel Experiments

Goal: study MoE memory/communication tradeoffs on PCIe GPUs.

| Case | DP | TP | PP | EP | Purpose |
| --- | ---: | ---: | ---: | ---: | --- |
| ep_2 | 4 | 1 | 1 | 2 | expert placement |
| ep_4 | 2 | 1 | 1 | 4 | communication stress |
| ep_8 | 1 | 1 | 1 | 8 | all-to-all limit |

Metrics:

- tokens/sec and tokens/sec/GPU
- peak allocated/reserved VRAM
- forward/backward/optimizer time
- all-reduce and all-to-all time when available
- checkpoint save/load time

## Stage 5: SmolVLA Finetuning

Goal: connect the training-infra work to VLA/robotics data.

Tasks:

- Keep synthetic SmolVLA smoke test reproducible.
- Add LeRobot-style collator.
- Support image/language/state/action batch schema.
- Validate action-head or expert-only finetuning.
- Track dataloader throughput and GPU idle time.

## Stage 6: Inference Latency Optimization

Goal: profile VLA action generation latency.

Candidate optimizations:

- FlashAttention / SDPA comparison.
- KV-cache reuse for language prefix.
- image encoder feature caching.
- CUDA Graph for fixed-shape inference.
- `torch.compile` or Triton kernels for action modules.
- speculative decoding or DFlash where token decoding dominates.

