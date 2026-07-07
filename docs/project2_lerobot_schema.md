# LeRobot ALOHA Schema Notes

Dataset inspected: `lerobot/aloha_mobile_cabinet`

Environment:

- LeRobot 0.6.0
- Python 3.12.3
- PyTorch 2.8.0+cu128
- TorchCodec 0.6.0
- FFmpeg 4.4.2
- GPU: RTX 3090

## Dataset Summary

| Field | Value |
| --- | ---: |
| robot_type | aloha |
| total_episodes | 85 |
| total_frames | 127,500 |
| fps | 50 |
| total_tasks | 1 |
| task | Open the top cabinet, store the pot inside it then close the cabinet. |

## Feature Schema

| Feature | dtype | shape | Notes |
| --- | --- | ---: | --- |
| `observation.images.cam_high` | video | `[480, 640, 3]` | RGB camera, AV1 video |
| `observation.images.cam_left_wrist` | video | `[480, 640, 3]` | RGB wrist camera, AV1 video |
| `observation.images.cam_right_wrist` | video | `[480, 640, 3]` | RGB wrist camera, AV1 video |
| `observation.state` | float32 | `[14]` | ALOHA joint state |
| `observation.effort` | float32 | `[14]` | ALOHA joint effort |
| `action` | float32 | `[14]` | ALOHA target action |
| `episode_index` | int64 | `[1]` | episode id |
| `frame_index` | int64 | `[1]` | frame id inside episode |
| `timestamp` | float32 | `[1]` | seconds |
| `next.done` | bool | `[1]` | episode boundary flag |
| `index` | int64 | `[1]` | global frame id |
| `task_index` | int64 | `[1]` | task lookup id |

## Storage Layout

LeRobot v3 separates low-dimensional metadata from video payloads:

```text
meta/info.json                         dataset schema and feature metadata
meta/tasks.parquet                     task text to task_index mapping
meta/episodes/chunk-000/file-000.parquet episode ranges, stats, and video offsets
data/chunk-000/file-000.parquet        state/action/timestamp/task columns
videos/.../*.mp4                       RGB video frames
```

The low-dimensional parquet file inspected here has shape `(127500, 9)` and columns:

```text
observation.state
observation.effort
action
episode_index
frame_index
timestamp
next.done
index
task_index
```

This means the first adapter stage can validate VLA batch construction without downloading or decoding videos. Video decoding can be added as a second stage once the low-dimensional training path is stable.
