# LeRobot Low-Dimensional Batch Dry-Run

Date: 2026-07-07

This result is the first milestone for Project 2: adapting LeRobot / SmolVLA-style data into a Nanotron-oriented VLA training pipeline.

The goal of this dry-run is deliberately narrow: validate the low-dimensional LeRobot parquet path before downloading or decoding large video payloads.

## Environment

| Item | Value |
| --- | --- |
| Cloud | AutoDL / SeeTaCloud |
| GPU | 1x NVIDIA GeForce RTX 3090 |
| Python | 3.12.3 |
| PyTorch | 2.8.0+cu128 |
| CUDA | 12.8 |
| LeRobot | 0.6.0 |
| TorchCodec | 0.6.0 |
| FFmpeg | 4.4.2 |

## Dataset

| Item | Value |
| --- | --- |
| repo_id | `lerobot/aloha_mobile_cabinet` |
| robot_type | `aloha` |
| episodes | 85 |
| frames | 127,500 |
| fps | 50 |
| task | Open the top cabinet, store the pot inside it then close the cabinet. |

The first stage reads only:

```text
meta/info.json
meta/tasks.parquet
meta/episodes/chunk-000/file-000.parquet
data/chunk-000/file-000.parquet
```

It does not require downloading the full video payload.

## Batch Contract Validated

```text
state:         float32 [B, 14]
effort:        float32 [B, 14]
action:        float32 [B, 14]
action_mask:   bool    [B, 14]
episode_index: int64   [B]
frame_index:   int64   [B]
timestamp:     float32 [B]
done:          bool    [B]
task_index:    int64   [B]
task_text:     list[str]
```

Observed first batch with `B=8`:

```text
VLABatch(
  state=(8, 14) torch.float32
  effort=(8, 14) torch.float32
  action=(8, 14) torch.float32
  action_mask=(8, 14) torch.bool
  episode_index=(8,) torch.int64
  frame_index=(8,) torch.int64
  timestamp=(8,) torch.float32
  done=(8,) torch.bool
  task_index=(8,) torch.int64
  task_text[0]='Open the top cabinet, store the pot inside it then close the cabinet.'
)
```

## Tiny Policy Smoke Train

Command:

```bash
PYTHONPATH=. python scripts/dry_run_lerobot_batch.py \
  --cache-root /root/autodl-tmp/vla-infra-project2/hf \
  --batch-size 8 \
  --limit 512 \
  --train-steps 5
```

Observed result:

```text
step=1 loss=0.362173
step=2 loss=0.447859
step=3 loss=0.430274
step=4 loss=0.395294
step=5 loss=0.365325

steps=5
frames=40
elapsed_sec=0.256
frames_per_sec=156.46
cuda_max_allocated_mib=17.6
```

## Interpretation

This is not a SmolVLA training claim yet. It validates the data plumbing layer needed before introducing the full multimodal model:

- local LeRobot cache discovery
- schema parsing from `info.json`
- task-text mapping from `tasks.parquet`
- frame-level low-dimensional sample loading from parquet
- collating into a VLA batch contract
- action-mask construction
- GPU forward/backward smoke training

Next steps:

1. Add image/video frame resolution and sampled TorchCodec decoding.
2. Measure dataloader throughput with and without video decoding.
3. Replace the tiny policy with a SmolVLA-compatible wrapper.
4. Add Nanotron-facing checkpoint/resume and DP smoke training.
