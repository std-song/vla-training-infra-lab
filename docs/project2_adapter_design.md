# SmolVLA Nanotron Adapter Design

Project 2 connects LeRobot / SmolVLA-style VLA data to a Nanotron-oriented training workflow. The implementation is staged so that each training-infrastructure risk can be validated independently.

## Stage 1: Low-Dimensional Batch Dry-Run

Goal: prove that LeRobot parquet metadata can be turned into stable VLA batches without downloading video payloads.

Implemented components:

- `smolvla_nanotron.data.lerobot_parquet_dataset.LeRobotParquetDataset`
- `smolvla_nanotron.data.collator.collate_lerobot_lowdim`
- `smolvla_nanotron.models.tiny_policy.TinyLowDimVLAPolicy`
- `scripts/dry_run_lerobot_batch.py`

Batch contract:

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

The tiny policy is not intended to be SmolVLA. It is a cheap trainable target used to validate dataloader, collator, tensor dtype/shape, optimizer, loss, and profiling plumbing before introducing multimodal model complexity.

## Stage 2: Image Batch Adapter

Implemented as a sampled-video path after the low-dimensional path became stable:

1. Decode sampled `videos/*/chunk-000/file-000.mp4` shards with TorchCodec.
2. Resize sampled frames to a fixed square size for batch validation.
3. Add image tensors to the batch contract:

```text
images.cam_high:        float32 or uint8 [B, C, H, W]
images.cam_left_wrist:  float32 or uint8 [B, C, H, W]
images.cam_right_wrist: float32 or uint8 [B, C, H, W]
```

4. Measure first-batch decode time and training-loop throughput separately from low-dimensional parquet loading.

Current limitation: this adapter intentionally targets `file-000.mp4` for a small sampled validation. A full production adapter should resolve arbitrary episode/file shards from `meta/episodes`.


## Stage 2b: DataLoader Profiling

Implemented components:

- `scripts/profile_lerobot_dataloader.py`
- `VLABatch.pin_memory()` for custom-batch pinned memory support

The profiling separates parquet-only loading from sampled three-camera video decoding. On vGPU-32GB, parquet-only loading is fastest with `num_workers=0`, while sampled video decoding improves strongly with worker parallelism and prefetching.
## Stage 3: SmolVLA / Nanotron Integration

Once batch construction is stable, the Nanotron-facing integration should add:

- model wrapper with explicit `forward(batch) -> loss`
- checkpoint save/resume for model, optimizer, scheduler, and dataloader state
- DP smoke training on one node
- throughput and memory profiling with and without image decode

The important training-infra claim is not simply that LeRobot can train SmolVLA. It is that the data schema, shard strategy, batch contract, checkpointing, and profiling path are explicit and reproducible.
