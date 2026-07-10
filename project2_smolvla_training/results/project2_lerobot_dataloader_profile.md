# LeRobot DataLoader Worker Profiling

Date: 2026-07-07

This report profiles the Project 2 LeRobot VLA data pipeline on a cloned vGPU-32GB instance. The goal is to separate low-dimensional parquet loading from sampled multi-camera video decoding, then evaluate `DataLoader` worker and prefetch behavior.

## Environment

| Item | Value |
| --- | --- |
| Cloud | AutoDL / SeeTaCloud |
| GPU | NVIDIA vGPU-32GB |
| GPU memory | 32,760 MiB |
| Python | 3.12.3 |
| PyTorch | 2.8.0+cu128 |
| CUDA | 12.8 |
| Dataset | `lerobot/aloha_mobile_cabinet` |
| Video sample | three `videos/*/chunk-000/file-000.mp4` shards |
| Image resize | 224 x 224 |

## Low-Dimensional Parquet Path

Command summary:

```bash
PYTHONPATH=. python scripts/profile_lerobot_dataloader.py \
  --cache-root /root/autodl-tmp/vla-infra-project2/hf \
  --batch-size 32 \
  --limit 1024 \
  --workers 0 1 2 4 \
  --warmup-batches 2 \
  --profile-batches 8 \
  --device-transfer \
  --pin-memory \
  --persistent-workers
```

| images | batch | workers | prefetch | pin | persistent | first_batch_s | steady_batches | fps | mean_batch_ms | p95_batch_ms | cuda_mib |
| --- | ---: | ---: | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| False | 32 | 0 |  | True | False | 0.144 | 8 | 12480.89 | 2.6 | 2.6 | 0.0 |
| False | 32 | 1 | 2 | True | True | 0.052 | 8 | 3063.07 | 10.4 | 10.9 | 0.0 |
| False | 32 | 2 | 2 | True | True | 0.062 | 8 | 5749.07 | 5.6 | 6.8 | 0.0 |
| False | 32 | 4 | 2 | True | True | 0.066 | 8 | 8731.08 | 3.7 | 8.3 | 0.0 |

Interpretation: for parquet-only state/action loading, samples are tiny and cheap. `num_workers=0` gives the best steady throughput because multiprocessing serialization and IPC overhead dominate the actual data access cost.

## Sampled Three-Camera Video Path

Command summary:

```bash
PYTHONPATH=. python scripts/profile_lerobot_dataloader.py \
  --cache-root /root/autodl-tmp/vla-infra-project2/hf \
  --batch-size 4 \
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

| images | batch | workers | prefetch | pin | persistent | first_batch_s | steady_batches | fps | mean_batch_ms | p95_batch_ms | cuda_mib |
| --- | ---: | ---: | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| True | 4 | 0 |  | True | False | 3.094 | 12 | 13.35 | 299.7 | 394.0 | 0.6 |
| True | 4 | 1 | 2 | True | True | 1.958 | 12 | 111.80 | 35.8 | 40.9 | 0.6 |
| True | 4 | 2 | 2 | True | True | 2.043 | 12 | 192.14 | 20.8 | 44.5 | 0.6 |
| True | 4 | 4 | 2 | True | True | 2.033 | 12 | 395.74 | 10.1 | 41.1 | 0.6 |

Interpretation: sampled video decoding is dominated by CPU-side codec work, resize, and batch assembly. Increasing DataLoader workers gives a large throughput gain because decoding can be overlapped and prefetched. The first batch remains expensive because each worker must initialize TorchCodec decoders and open video shards.

## Practical Takeaways

- Low-dimensional VLA state/action batches do not need multiprocessing by default.
- Video-heavy VLA batches benefit strongly from worker parallelism and persistent workers.
- First-batch latency should be measured separately from steady-state throughput.
- Pinned memory matters only if the custom batch type implements `pin_memory()`; `VLABatch.pin_memory()` was added for this reason.
- The sampled-video adapter is useful for pipeline validation, but a production path still needs a full `meta/episodes` resolver for arbitrary shards.

## Next Engineering Step

Replace the sampled `file-000.mp4` resolver with a metadata-driven resolver:

```text
episode_index, frame_index -> video file index, local frame, timestamp -> decoded image tensors
```

After that, benchmark `num_workers`, `prefetch_factor`, `persistent_workers`, `pin_memory`, image size, and camera count under the full shard strategy.
