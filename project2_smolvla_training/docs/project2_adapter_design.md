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

The sampled adapter has been upgraded to a `meta/episodes` shard-aware resolver that maps `episode_index + timestamp` to camera-specific `chunk_index`, `file_index`, and video-local timestamp.



## Stage 2b: Shard-Aware Video Resolver

Implemented components:

- `VideoShardResolver`
- `VideoFrameRef`
- `LeRobotParquetDataset(start_index=...)`

The resolver uses `meta/episodes` to map each sample to the correct camera video shard. Validation covered both `file-000.mp4` and `file-001.mp4` regions using `start_index=0` and `start_index=99000`.
## Stage 2c: DataLoader Profiling

Implemented components:

- `scripts/profile_lerobot_dataloader.py`
- `VLABatch.pin_memory()` for custom-batch pinned memory support

The profiling separates parquet-only loading, sampled three-camera video decoding, and the shard-aware resolver with `start_index=0` and `start_index=99000`. On vGPU-32GB, parquet-only loading is fastest with `num_workers=0`, while video decoding improves strongly with worker parallelism and prefetching.

## Stage 3: SmolVLA / Nanotron Integration

Once batch construction is stable, the Nanotron-facing integration should add:

- model wrapper with explicit `forward(batch) -> loss`
- checkpoint save/resume for model, optimizer, scheduler, and dataloader state
- DP smoke training on one node
- throughput and memory profiling with and without image decode


## Stage 3a: SmolVLA-Compatible Wrapper

Implemented components:

- `smolvla_nanotron.models.smolvla_compatible.SmolVLACompatiblePolicy`
- `scripts/train_smolvla_compatible.py`

This wrapper consumes multi-camera images, robot state, and task text, then predicts a 14-dimensional action target with masked MSE loss. It validates the end-to-end training path before introducing the official SmolVLA model: DataLoader, device transfer, model forward, loss, optimizer, checkpoint save, and checkpoint resume.
The important training-infra claim is not simply that LeRobot can train SmolVLA. It is that the data schema, shard strategy, batch contract, checkpointing, and profiling path are explicit and reproducible.

## Stage 3b: Nanotron-Style DP Trainer

Implemented components:

- `scripts/train_smolvla_compatible_dp.py`

This trainer adds the distributed surfaces that are needed before moving toward a Nanotron engine integration:

- `torchrun` rank/local-rank/world-size discovery
- NCCL process-group initialization
- `DistributedSampler` ownership of LeRobot parquet/video samples
- `DistributedDataParallel` wrapping
- all-reduced metrics
- rank-0 checkpoint save
- checkpoint resume on 2 GPUs

The current implementation intentionally keeps the lightweight SmolVLA-compatible wrapper so that distributed correctness can be measured without official SmolVLA model complexity. The next comparison step is to run the official LeRobot/SmolVLA DDP fine-tuning entrypoint on the same dataset subset and compare throughput, memory, checkpoint behavior, and DataLoader sensitivity.

## Stage 3c: Official LeRobot/SmolVLA DDP Baseline

Implemented validation:

- official `lerobot.scripts.lerobot_train` launched with `accelerate launch --num_processes=2 --multi_gpu`
- local LeRobot snapshot root to avoid Hub metadata lookup during dataset loading
- `HF_ENDPOINT=https://hf-mirror.com` for the SmolVLM config/processor cache
- official SmolVLA from scratch with `load_vlm_weights=false`, reduced VLM/expert layers, and default 512x512 image preprocessing
- checkpoint save at step 2 and official resume to step 3 through checkpoint `train_config.json`

This stage establishes the official baseline that the custom Nanotron-style DP trainer should be compared against. The comparison is infrastructure-level rather than model-quality-level: launcher complexity, data pipeline behavior, checkpoint layout, resume ergonomics, memory, and samples/s after warmup.
