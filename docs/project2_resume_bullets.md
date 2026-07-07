# Project 2 Resume Bullets: SmolVLA / LeRobot Training Infrastructure

Use one of the following versions depending on resume space.

## Strong 3-Bullet Version

- Built a reproducible VLA training-infrastructure lab around LeRobot/SmolVLA on 2 GPUs, covering multi-camera video shard resolution, parquet state/action loading, multimodal batch collation, action-mask loss plumbing, DDP launch, checkpoint/resume, and throughput/memory profiling.
- Implemented a compact Nanotron-style DP wrapper with `torchrun`, NCCL DDP, `DistributedSampler`, all-reduced metrics, rank-0 checkpointing, and resume; compared it against the official LeRobot/SmolVLA Accelerate DDP baseline on the same `aloha_mobile_cabinet` subset.
- Profiled official SmolVLA DDP data pipeline and training knobs: `num_workers=1` improved throughput from 17.20 to 23.63 samples/s, BF16 improved the 50-step window from 23.63 to 24.61 samples/s, and disabling DDP unused-parameter search improved FP32 throughput to 26.35 samples/s while not helping BF16.

## Compact 2-Bullet Version

- Developed a LeRobot/SmolVLA distributed fine-tuning infra prototype with explicit VLA batch contract, multi-camera video loading, action-mask loss, DDP metric reduction, checkpoint/resume, and profiling; validated both official SmolVLA Accelerate DDP and a custom Nanotron-style DP wrapper on 2 GPUs.
- Analyzed throughput bottlenecks and training knobs for official SmolVLA DDP: dataloader workers, BF16 mixed precision, checkpoint behavior, and `find_unused_parameters`; improved FP32 throughput from 23.63 to 26.35 samples/s via DDP tuning and documented dtype-dependent tradeoffs.

## One-Line Version

Built and profiled a 2-GPU LeRobot/SmolVLA training-infra lab, comparing official Accelerate DDP with a custom Nanotron-style DP wrapper and analyzing multi-camera data loading, BF16, checkpoint/resume, and DDP tuning with concrete throughput/memory metrics.

## Honest Boundary For Interviews

This project is not a full native SmolVLA port into Nanotron `DistributedTrainer`. The custom path is a Nanotron-style DP prototype that validates the required infrastructure surfaces before a deeper Nanotron integration. The official LeRobot/SmolVLA DDP path is used as the real-model baseline.
