# Project 2: Episode-Aware Sampling Profile

## Goal

Frame-wise distributed shuffling is simple, but it can cause ranks to reopen unrelated multi-camera video shards frequently. This experiment compares the original frame-wise `DistributedSampler` against `EpisodeAwareSampler`, which assigns each episode to one rank before shuffling frames within that rank.

This is a data-pipeline experiment using the compact SmolVLA-compatible policy wrapper. It measures end-to-end video decode, host-to-device transfer, DDP update, and data-loader waiting; it is **not** a claim about official SmolVLA policy quality.

## Matched Setup

| Item | Value |
| --- | --- |
| Hardware | 2x RTX 4080 SUPER, 32 GiB each |
| Runtime | PyTorch 2.8.0 + CUDA 12.8, LeRobot 0.6.0 |
| Dataset | `lerobot/aloha_mobile_cabinet` |
| Input | three 224x224 camera frames + state + task text + action target |
| DDP | NCCL, 2 ranks |
| Global batch | 4 (2 per rank) |
| Window | 200 steps |
| DataLoader | 2 workers/rank, pinned memory, persistent workers, prefetch factor 2 |

## Results

| Sampling policy | Samples/s | Data wait / step | Update / step | Peak allocated memory |
| --- | ---: | ---: | ---: | ---: |
| Frame-wise | 147.83 | 2.494 ms | 22.603 ms | 64.7 MiB/rank |
| Episode-aware | 156.77 | 0.742 ms | 22.217 ms | 64.7 MiB/rank |

Episode-aware sampling improves end-to-end throughput by **6.1%** and reduces rank-0 data waiting by **70.2%**. Update time is effectively unchanged, so the gain is consistent with better video-shard locality rather than a change in model compute.

## Load-Balance Trade-off

Episode ownership intentionally trades perfect frame-level balance for camera-shard locality. For the 85 ALOHA episodes:

| World size | Assigned samples per rank |
| ---: | --- |
| 2 | 64,500 / 63,000 |
| 4 | 33,000 / 31,500 / 31,500 / 31,500 |

At 2 ranks the imbalance is 2.35%, which is small relative to the observed data-wait reduction. This should still be profiled again for datasets with highly uneven episode lengths.

## Engineering Changes

- Added a pre-training dataset admission stage that generated a 127,500-sample manifest from actual LeRobot metadata and parquet payloads with zero structural issues.
- Added deterministic episode-aware rank planning, rank-disjointness checks, and a sampler state interface for future exact data-resume integration.
- Corrected the custom DDP loop to call `model(batch)` through the DDP wrapper before masked action-loss computation, rather than bypassing DDP with `model.module.loss(...)`.
- Added data-wait and update-time metrics to distinguish I/O improvements from GPU compute changes.

## Boundary

The sampler state interface is unit-tested, but exact mid-epoch sampler-state restoration is not yet wired into the custom trainer checkpoint payload. The current training checkpoint continues to restore model and optimizer state; data-sequence restoration is the next extension.
