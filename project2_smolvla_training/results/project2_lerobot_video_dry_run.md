# LeRobot Sampled Video Batch Dry-Run

Date: 2026-07-07

This result extends the Project 2 adapter from low-dimensional parquet batches to sampled RGB video decoding.

## Input Video Shards

Only three sampled video shards were downloaded, not the full dataset video payload:

```text
videos/observation.images.cam_high/chunk-000/file-000.mp4
videos/observation.images.cam_left_wrist/chunk-000/file-000.mp4
videos/observation.images.cam_right_wrist/chunk-000/file-000.mp4
```

The adapter uses TorchCodec to decode frames and optionally resizes them with PyTorch interpolation.

## Command

```bash
PYTHONPATH=. python scripts/dry_run_lerobot_batch.py \
  --cache-root /root/autodl-tmp/vla-infra-project2/hf \
  --batch-size 2 \
  --limit 8 \
  --train-steps 2 \
  --include-images \
  --image-size 224
```

## Batch Contract

Observed batch:

```text
VLABatch(
  state=(2, 14) torch.float32
  effort=(2, 14) torch.float32
  action=(2, 14) torch.float32
  action_mask=(2, 14) torch.bool
  episode_index=(2,) torch.int64
  frame_index=(2,) torch.int64
  timestamp=(2,) torch.float32
  done=(2,) torch.bool
  task_index=(2,) torch.int64
  task_text[0]='Open the top cabinet, store the pot inside it then close the cabinet.'
  images[observation.images.cam_high]=(2, 3, 224, 224) torch.uint8
  images[observation.images.cam_left_wrist]=(2, 3, 224, 224) torch.uint8
  images[observation.images.cam_right_wrist]=(2, 3, 224, 224) torch.uint8
)
```

## Timing

| Metric | Value |
| --- | ---: |
| first batch load time | 2.983 s |
| train steps | 2 |
| frames processed | 4 |
| elapsed train loop time | 0.367 s |
| train loop throughput | 10.89 frames/s |
| CUDA max allocated | 18.4 MiB |

For comparison, the low-dimensional parquet-only regression run loaded the first batch in `0.023 s` and processed the tiny policy loop at `67.23 frames/s` for the same environment. The large gap is expected: sampled video decoding introduces file open, codec initialization, seek, resize, and host-to-device movement overhead.

## Interpretation

This milestone validates that the VLA batch contract can carry both low-dimensional robot state/action and multi-camera RGB observations:

```text
state/action/task metadata + three RGB camera tensors -> trainable batch
```

The current image adapter is intentionally scoped to sampled `file-000.mp4` shards. The next production step is to use `meta/episodes` to resolve arbitrary episode ranges, video file indices, and per-frame timestamps, then benchmark dataloader workers and prefetching.
