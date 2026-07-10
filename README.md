# VLA Training Infrastructure Lab

This repository collects three independent infrastructure projects for VLA training and inference roles. The focus is not model quality benchmarking, but system work: distributed training behavior, multimodal data pipelines, profiling, checkpointing, serving, and control-loop latency analysis under limited GPU resources.

For resume and interview preparation, start with [`docs/xiaomi_vla_infra_application_pack.md`](docs/xiaomi_vla_infra_application_pack.md).

## Projects

| Project | Focus | Start here |
| --- | --- | --- |
| Project 1 | Nanotron-based Qwen3-MoE-style pretraining infra, DP/TP/PP/EP validation, MoE dispatch, checkpoint/resume, profiling | [`project1_qwen3_moe_pretrain/README.md`](project1_qwen3_moe_pretrain/README.md) |
| Project 2 | LeRobot/SmolVLA multimodal data pipeline, DDP fine-tuning baseline, Nanotron-style DP wrapper, dataloader and BF16 profiling | [`project2_smolvla_training/README.md`](project2_smolvla_training/README.md) |
| Project 3 | VLM/VLA inference infra: Qwen-VL serving, KV/cache and batching analysis, Pi0.5 action inference, async action queue simulation | [`project3_vla_inferenceence/README.md`](project3_vla_inferenceence/README.md) |

## Role Alignment

| VLA infra requirement | Repository coverage |
| --- | --- |
| PyTorch distributed training | Nanotron DP/TP/PP/EP experiments and SmolVLA DDP adapter |
| MoE training | Router top-k, expert token dispatch, GroupedGEMM, load-balancing hooks |
| Mixed precision and kernels | BF16, FlashAttention, Triton action post-processing, vLLM serving paths |
| Data pipeline | LeRobot video shards, parquet state/action, task text, action mask collation |
| Experiment tracking | Reproducible configs, launch scripts, parsed logs, CSV summaries, SVG figures |
| Serving and control loop | VLM concurrency profiling, visual token cost, KV budget simulation, Pi0.5 action chunk latency |

## Repository Layout

```text
project1_qwen3_moe_pretrain/   Qwen3-MoE-style pretraining infra and Nanotron experiments
project2_smolvla_training/     LeRobot/SmolVLA data and distributed fine-tuning work
project3_vla_inferenceence/        VLM/VLA inference profiling and async action serving
docs/                          Cross-project resume pack, roadmap, troubleshooting
```

Large generated artifacts, model weights, checkpoints, and local AutoDL credentials are intentionally ignored.

## Current Status

- Project 1: Qwen3-MoE-style 100M-scale Nanotron smoke/distributed validation completed on 1-2x RTX 3090, including PP resume fix and EP2 dispatch validation.
- Project 2: SmolVLA official DDP and custom Nanotron-style DP wrapper are documented, with dataloader worker and BF16 profiling results.
- Project 3: Qwen3-VL vLLM baseline, Qwen2.5-VL visual-token profiling, Pi0.5 action chunk inference, and VLASH-inspired async queue simulation are documented with figures.

The project folders are intentionally self-contained so each can be moved to its own GitHub repository later without dragging unrelated artifacts along.
