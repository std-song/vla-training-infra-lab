# LeRobot Data Admission Pipeline

This CPU-only stage validates a LeRobot snapshot before a SmolVLA training job starts. It does not decode video frames or require a GPU.

## What It Produces

- `manifest.jsonl`: one stable record per training sample, including episode/frame identity, task text, and camera-shard references.
- `quality_report.json`: sample count, valid-sample count, issue counts, and individual rejected samples.

## Checks

- required state/action fields and expected shapes;
- NaN or Inf in state/action tensors;
- finite, non-negative, monotonic timestamps inside an episode;
- task index to task-text mapping;
- per-camera video-shard existence derived from `meta/episodes`.
- per-episode frame interval, state delta, and action delta distributions.

## Local Development

From `project2_smolvla_training`:

```powershell
.\.venv\Scripts\python.exe -m pytest -q tests\test_manifest.py
```

The tests create a temporary LeRobot-like snapshot and inject a NaN action plus a missing video shard. They should pass without downloading any dataset.

## Audit a Real Dataset

Copy a LeRobot dataset snapshot, or its Hugging Face cache root, to the laptop. The snapshot needs `meta/`, `data/`, and `videos/` if video existence checks are enabled.

```powershell
.\.venv\Scripts\python.exe scripts\audit_lerobot_dataset.py `
  --dataset-root D:\datasets\aloha_mobile_cabinet `
  --output-dir artifacts\aloha_audit
```

For metadata-only development, omit video existence checks:

```powershell
.\.venv\Scripts\python.exe scripts\audit_lerobot_dataset.py `
  --dataset-root D:\datasets\aloha_mobile_cabinet `
  --output-dir artifacts\aloha_metadata_audit `
  --skip-video-existence-check
```

To flag unusually large adjacent action deltas for human review, enable the robust-MAD rule explicitly. This is a statistical quality signal, not a robot-specific physical safety assertion.

```powershell
.\.venv\Scripts\python.exe scripts\audit_lerobot_dataset.py `
  --dataset-root D:\datasets\aloha_mobile_cabinet `
  --output-dir artifacts\aloha_metadata_audit `
  --skip-video-existence-check `
  --action-jump-mad-multiplier 12
```

The `artifacts/` directory is intentionally ignored by Git. Commit the code, tests, and a curated aggregate report instead of raw dataset paths or sample manifests.

## Multi-GPU Planning

`EpisodeAwareSampler` assigns complete episodes to one rank, then shuffles frames inside that rank. This avoids duplicate video-shard opening across ranks and provides a compact sampler state for checkpoint/resume. Inspect ownership without CUDA:

```powershell
.\.venv\Scripts\python.exe scripts\inspect_episode_sampler.py `
  --manifest artifacts\aloha_metadata_audit\manifest.jsonl `
  --world-size 2 --seed 17 --epoch 0
```

The custom DDP wrapper can select it at training time with `--sampling-policy episode`. The default remains frame-wise `DistributedSampler`, which preserves the existing baseline for an apples-to-apples throughput comparison.
