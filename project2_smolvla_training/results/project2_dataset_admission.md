# Project 2: LeRobot Dataset Admission and Distributed Sampling

## Scope

Before launching distributed SmolVLA fine-tuning, this stage validates the training-data contract independently of GPU execution. The pipeline constructs a stable sample manifest from LeRobot metadata and parquet tables, checks training-blocking data faults, summarizes trajectory dynamics, and plans rank ownership at episode granularity.

The implementation is in:

- `smolvla_nanotron/data/manifest.py`
- `smolvla_nanotron/data/sampler.py`
- `scripts/audit_lerobot_dataset.py`
- `scripts/inspect_episode_sampler.py`

## Dataset

| Item | Value |
| --- | ---: |
| Dataset | `lerobot/aloha_mobile_cabinet` |
| Episodes | 85 |
| Frames / samples | 127,500 |
| Expected frequency | 50 Hz |
| Cameras | cam_high, cam_left_wrist, cam_right_wrist |

The local audit used the actual `meta/` and `data/` artifacts. Video payload decode is intentionally separated into the later high-throughput DataLoader stage.

## Admission Results

The metadata-only audit completed with **0 structural issues across 127,500 samples**.

| Check | Result |
| --- | --- |
| Required state/action features and shapes | pass |
| NaN / Inf state or action values | none |
| Timestamp validity and intra-episode monotonicity | pass |
| Task-index to task-text mapping | pass |
| Camera-shard reference construction | pass; payload existence deferred in metadata-only mode |

## Trajectory Dynamics

Adjacent frames are compared only within one episode, so episode transitions do not create artificial jumps.

| Metric | Median | P95 | P99 | Max |
| --- | ---: | ---: | ---: | ---: |
| Action delta L2 | 0.02052 | 0.09215 | 0.13954 | 0.23626 |
| State delta L2 | 0.01994 | 0.07859 | 0.09537 | 0.15349 |
| Frame interval | 20.0002 ms | 20.0005 ms | 20.0005 ms | 20.0005 ms |

There are 127,415 adjacent frame pairs, exactly excluding one boundary per episode. A robust-MAD action-jump review rule with multiplier 12 produced **0 review flags**. It is intentionally described as a statistical review signal rather than a robot-specific safety assertion.

## Episode-Aware Distributed Sampling

Unlike frame-wise random assignment, `EpisodeAwareSampler` assigns a complete episode to exactly one rank, then shuffles frame order within that rank. This avoids cross-rank duplicate opening of the same camera shards and provides a compact resume state: rank, world size, seed, epoch, and consumed sample count.

| World size | Per-rank assigned samples | Max imbalance |
| ---: | --- | ---: |
| 2 | 64,500 / 63,000 | 1,500 (2.35%) |
| 4 | 33,000 / 31,500 / 31,500 / 31,500 | 1,500 (4.71%) |

The bounded imbalance is the explicit trade-off for preserving episode and video-shard locality. For the equal-length ALOHA episodes used here, it remains small and avoids the more expensive random video-shard access pattern.

## Verification

CPU-only regression tests cover clean snapshots, missing shards, non-finite actions, manifest output, rank disjointness, deterministic ordering, and sampler resume. Latest local result: `4 passed`.

## Next Step

Integrate the sampler through a `torch.utils.data.Sampler` adapter in the custom DDP trainer, then measure whether locality improves video-decode throughput against the existing frame-wise `DistributedSampler` baseline.
