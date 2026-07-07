# Shard-Aware LeRobot DataLoader Profiling

Date: 2026-07-07

This report profiles the `meta/episodes` shard-aware video resolver introduced in Project 2. Unlike the earlier sampled adapter, this path resolves each sample through episode metadata before decoding camera frames.

## Environment

| Item | Value |
| --- | --- |
| Cloud | AutoDL / SeeTaCloud |
| GPU | NVIDIA vGPU-32GB |
| Python | 3.12.3 |
| PyTorch | 2.8.0+cu128 |
| CUDA | 12.8 |
| Dataset | `lerobot/aloha_mobile_cabinet` |
| Image size | 224 x 224 |
| Cameras | high, left wrist, right wrist |

## Command

The same profiling command was run on two regions:

- `start_index=0`: early dataset region backed by `file-000.mp4`
- `start_index=99000`: later dataset region backed by `file-001.mp4`

```bash
PYTHONPATH=. python scripts/profile_lerobot_dataloader.py \
  --cache-root /root/autodl-tmp/vla-infra-project2/hf \
  --batch-size 4 \
  --start-index <0-or-99000> \
  --limit 96 \
  --workers 0 1 2 4 \
  --warmup-batches 2 \
  --profile-batches 12 \
  --include-images \
  --image-size 224 \
  --device-transfer \
  --pin-memory \
  --persistent-workers
```

## Region A: `start_index=0`

| images | batch | workers | prefetch | pin | persistent | first_batch_s | steady_batches | fps | mean_batch_ms | p95_batch_ms | cuda_mib |
| --- | ---: | ---: | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| True | 4 | 0 |  | True | False | 3.107 | 12 | 10.42 | 383.9 | 595.9 | 0.6 |
| True | 4 | 1 | 2 | True | True | 1.904 | 12 | 86.79 | 46.1 | 51.7 | 0.6 |
| True | 4 | 2 | 2 | True | True | 2.011 | 12 | 180.57 | 22.2 | 34.4 | 0.6 |
| True | 4 | 4 | 2 | True | True | 2.057 | 12 | 352.58 | 11.3 | 45.6 | 0.6 |

## Region B: `start_index=99000`

| images | batch | workers | prefetch | pin | persistent | first_batch_s | steady_batches | fps | mean_batch_ms | p95_batch_ms | cuda_mib |
| --- | ---: | ---: | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| True | 4 | 0 |  | True | False | 2.613 | 12 | 11.45 | 349.3 | 404.2 | 0.6 |
| True | 4 | 1 | 2 | True | True | 1.520 | 12 | 83.92 | 47.7 | 57.6 | 0.6 |
| True | 4 | 2 | 2 | True | True | 1.623 | 12 | 187.85 | 21.3 | 51.1 | 0.6 |
| True | 4 | 4 | 2 | True | True | 1.772 | 12 | 228.03 | 17.5 | 82.4 | 0.6 |

## Interpretation

The shard-aware resolver preserves the same qualitative behavior as the sampled video path:

- single-process video decoding is slow: roughly `10-11 frames/s`
- worker parallelism gives large gains by overlapping TorchCodec decode, resize, collation, and device transfer
- `workers=2` is already strong and stable across both tested regions
- `workers=4` is best in the early region, but less stable in the later region, likely because seek/decode/cache behavior differs around `file-001.mp4`
- first-batch latency should remain a separate metric because it includes worker startup and decoder initialization

Compared with the earlier sampled `file-000.mp4` result, the shard-aware path has a small overhead from per-sample metadata resolution and `get_frame_played_at(timestamp)`, but it is much closer to a real LeRobot training data path.

## Practical Recommendation

For this small validation setup:

```text
low-dimensional state/action path: num_workers=0
three-camera video path: num_workers=2 or 4, persistent_workers=True, prefetch_factor=2
```

For production-scale VLA fine-tuning, the next experiment should profile:

- larger batch sizes
- more sampled episodes across video shard boundaries
- image size tradeoffs such as 224 vs 336
- CPU utilization and per-worker decode time
- model-side overlap once the SmolVLA-compatible wrapper is introduced
