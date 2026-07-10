# LeRobot Shard-Aware Video Resolver

Date: 2026-07-07

This report records the Project 2 upgrade from a sampled `file-000.mp4` video adapter to a `meta/episodes` shard-aware resolver.

## Goal

The previous video adapter intentionally decoded only:

```text
videos/<camera>/chunk-000/file-000.mp4
```

That was enough to validate three-camera image batches, but not enough for a real LeRobot data pipeline. A production loader must resolve each frame through dataset metadata:

```text
episode_index + timestamp/frame_index
-> meta/episodes row
-> camera chunk_index + file_index + from_timestamp
-> videos/<camera>/chunk-xxx/file-yyy.mp4
-> TorchCodec frame decode
```

## Implementation

Added in `smolvla_nanotron/data/lerobot_parquet_dataset.py`:

- `VideoFrameRef`: normalized reference to one camera frame source.
- `VideoShardResolver`: maps `(camera, episode_index, timestamp)` to a video shard and video-local timestamp.
- `VideoFrameDecoder`: lazily opens TorchCodec decoders keyed by concrete video path.
- `LeRobotParquetDataset(start_index=...)`: supports slicing into later global frame regions for cross-shard validation.

Resolution logic:

```text
row = meta/episodes[episode_index]
prefix = videos/<camera>
chunk_index = row[prefix/chunk_index]
file_index = row[prefix/file_index]
video_timestamp = row[prefix/from_timestamp] + sample.timestamp
video_path = snapshot/videos/<camera>/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4
frame = VideoDecoder(video_path).get_frame_played_at(video_timestamp)
```

## Validation

The vGPU-32GB instance had all six sampled video files available:

```text
cam_high/file-000.mp4
cam_high/file-001.mp4
cam_left_wrist/file-000.mp4
cam_left_wrist/file-001.mp4
cam_right_wrist/file-000.mp4
cam_right_wrist/file-001.mp4
```

Two dry-runs were validated by the user:

```bash
PYTHONPATH=. python scripts/dry_run_lerobot_batch.py \
  --cache-root /root/autodl-tmp/vla-infra-project2/hf \
  --batch-size 2 \
  --start-index 0 \
  --limit 4 \
  --train-steps 1 \
  --include-images \
  --image-size 224
```

```bash
PYTHONPATH=. python scripts/dry_run_lerobot_batch.py \
  --cache-root /root/autodl-tmp/vla-infra-project2/hf \
  --batch-size 2 \
  --start-index 99000 \
  --limit 4 \
  --train-steps 1 \
  --include-images \
  --image-size 224
```

The first run validates the beginning of the dataset, backed by `file-000.mp4`. The second run validates a later global frame region backed by `file-001.mp4`.

Expected and observed batch contract:

```text
images[observation.images.cam_high]=(2, 3, 224, 224) torch.uint8
images[observation.images.cam_left_wrist]=(2, 3, 224, 224) torch.uint8
images[observation.images.cam_right_wrist]=(2, 3, 224, 224) torch.uint8
```

## Interpretation

This milestone moves the Project 2 data pipeline from a sampled demo to a metadata-driven LeRobot shard strategy. The adapter can now reason over episode metadata instead of assuming a fixed video file.

Remaining production work:

- benchmark the shard-aware resolver with multiple workers
- add robust boundary tests around episode/file transitions
- support datasets with multiple data chunks and more than two video files
- add a SmolVLA-compatible model wrapper on top of the validated batch contract
