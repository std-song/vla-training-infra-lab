# Project 2: SmolVLA Training Infra

This project builds a LeRobot/SmolVLA fine-tuning infrastructure lab around multimodal data loading, distributed training wrappers, and practical profiling.

## What It Covers

- LeRobot dataset parsing for `aloha_mobile_cabinet`: three-camera video shards, parquet state/action, task text, episode/frame metadata.
- `VLABatch`-style collation with image/state/action/action-mask/task-text fields.
- Lightweight SmolVLA-compatible policy wrapper and Nanotron-style DDP wrapper for validating multimodal distributed training mechanics.
- Official SmolVLA Accelerate DDP baseline as the real-model reference path.
- Dataloader worker, BF16, DDP setting, checkpoint/resume, and memory profiling.

## Key Results

Official SmolVLA baseline on 2x RTX 4080 SUPER:

| Setting | Throughput |
| --- | ---: |
| DataLoader workers = 0 | 17.2 samples/s |
| DataLoader workers = 1 | 23.6 samples/s |
| BF16 | 24.6 samples/s |

The useful engineering result is the bottleneck boundary: for this small two-GPU setup, improving data loading and decode overlap matters more than tuning DDP flags.

![SmolVLA worker tuning](assets/figures/project2_official_smolvla_workers.svg)

![SmolVLA BF16 profile](assets/figures/project2_official_smolvla_bf16.svg)

## Reading Path

- Final report: [`results/project2_final_report.md`](results/project2_final_report.md)
- Data schema: [`docs/project2_lerobot_schema.md`](docs/project2_lerobot_schema.md)
- Adapter design: [`docs/project2_adapter_design.md`](docs/project2_adapter_design.md)
- Nanotron-style DP wrapper: [`results/project2_nanotron_style_dp.md`](results/project2_nanotron_style_dp.md)
- Official SmolVLA DDP: [`results/project2_official_smolvla_ddp.md`](results/project2_official_smolvla_ddp.md)
- Worker/BF16 profiling: [`results/project2_official_smolvla_worker_profile.md`](results/project2_official_smolvla_worker_profile.md), [`results/project2_official_smolvla_bf16_profile.md`](results/project2_official_smolvla_bf16_profile.md)
- Resume bullets: [`docs/project2_resume_bullets.md`](docs/project2_resume_bullets.md)

## Code Pointers

- Dataset/collation: [`smolvla_nanotron/data`](smolvla_nanotron/data)
- Model wrappers: [`smolvla_nanotron/models`](smolvla_nanotron/models)
- Training scripts: [`scripts`](scripts)
