# Project 2: Nanotron-Style DP Smoke Training

Date: 2026-07-07

This report validates a local Nanotron-style data-parallel training path for the SmolVLA-compatible policy wrapper. The goal is to separate VLA training-infra mechanics from official SmolVLA model complexity before comparing against the LeRobot/SmolVLA training entrypoint.

## Environment

| Item | Value |
| --- | --- |
| Host | AutoDL cloned 2-GPU instance |
| GPU | 2x NVIDIA GeForce RTX 4080 SUPER, 32 GiB each |
| Python | 3.12.3 |
| PyTorch | 2.8.0+cu128 |
| CUDA runtime | 12.8 |
| Distributed backend | NCCL |
| Dataset | `lerobot/aloha_mobile_cabinet` |
| Input | 3 camera videos + robot state + task text |
| Model | Lightweight SmolVLA-compatible multimodal policy wrapper |

## Implemented Training Surface

Script:

```text
scripts/train_smolvla_compatible_dp.py
```

The script implements:

- `torch.distributed.init_process_group` with NCCL on CUDA.
- rank/local-rank/world-size discovery from `torchrun`.
- `DistributedSampler` over the LeRobot parquet/video dataset.
- `DistributedDataParallel` model wrapping.
- rank-0 logging and checkpoint writing.
- all-reduced loss / grad-norm / prediction metrics.
- checkpoint resume from a rank-0 checkpoint.
- per-rank CUDA peak-memory reporting reduced by max.

This is called "Nanotron-style" because it mirrors the infrastructure surfaces required by Nanotron and other large-model trainers: explicit rank state, deterministic sampler ownership, rank-aware checkpointing, distributed metric reduction, and torchrun launch reproducibility. It is not yet a direct Nanotron trainer plugin.

## Batch Validation

Rank 0 observed the following batch contract:

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

## 1-GPU Baseline

Launch:

```bash
PYTHONPATH=. python scripts/train_smolvla_compatible_dp.py \
  --cache-root /root/autodl-tmp/vla-infra-project2/hf \
  --output-dir outputs/project2_smolvla_compatible_dp1_baseline \
  --batch-size 2 \
  --limit 64 \
  --train-steps 6 \
  --workers 2 \
  --prefetch-factor 2 \
  --persistent-workers \
  --pin-memory \
  --save-every 3
```

Result:

| Metric | Value |
| --- | ---: |
| world size | 1 |
| per-rank batch size | 2 |
| global batch size | 2 |
| final step | 6 |
| loss, step 1 -> step 6 | 0.460942 -> 0.230743 |
| elapsed | 0.832 s |
| steps/s | 7.209 |
| samples/s | 14.418 |
| max CUDA allocated | 62.1 MiB |
| checkpoint | `outputs/project2_smolvla_compatible_dp1_baseline/latest.pt` |

## 2-GPU DP Smoke

Launch:

```bash
PYTHONPATH=. torchrun --nproc_per_node=2 scripts/train_smolvla_compatible_dp.py \
  --cache-root /root/autodl-tmp/vla-infra-project2/hf \
  --output-dir outputs/project2_smolvla_compatible_dp2_smoke \
  --batch-size 2 \
  --limit 64 \
  --train-steps 6 \
  --workers 2 \
  --prefetch-factor 2 \
  --persistent-workers \
  --pin-memory \
  --save-every 3
```

Result:

| Metric | Value |
| --- | ---: |
| backend | NCCL |
| world size | 2 |
| per-rank batch size | 2 |
| global batch size | 4 |
| final step | 6 |
| loss, step 1 -> step 6 | 0.461880 -> 0.228570 |
| elapsed | 0.852 s |
| steps/s | 7.042 |
| samples/s | 28.170 |
| max CUDA allocated per rank | 64.4 MiB |
| checkpoint | `outputs/project2_smolvla_compatible_dp2_smoke/latest.pt` |

## Resume Validation

Resume launch:

```bash
PYTHONPATH=. torchrun --nproc_per_node=2 scripts/train_smolvla_compatible_dp.py \
  --cache-root /root/autodl-tmp/vla-infra-project2/hf \
  --output-dir outputs/project2_smolvla_compatible_dp2_smoke \
  --batch-size 2 \
  --limit 64 \
  --train-steps 8 \
  --workers 2 \
  --prefetch-factor 2 \
  --persistent-workers \
  --pin-memory \
  --resume-from outputs/project2_smolvla_compatible_dp2_smoke/latest.pt
```

Result:

| Metric | Value |
| --- | ---: |
| resumed step | 6 |
| final step | 8 |
| loss, step 7 -> step 8 | 0.133604 -> 0.047061 |
| max CUDA allocated per rank | 64.4 MiB |

The two-step resume run is not used for throughput comparison because process-group setup, DataLoader worker warmup, and first-batch decode overhead dominate such a short window.

## Initial Scaling Interpretation

For this tiny wrapper, DP scaling is almost linear in the 6-step smoke window:

| Run | Global batch | Samples/s | Speedup |
| --- | ---: | ---: | ---: |
| 1 GPU | 2 | 14.418 | 1.00x |
| 2 GPU DP | 4 | 28.170 | 1.95x |

This result should not be overclaimed. The model is intentionally small, gradients are small, and the measured loop excludes the first-batch decode. In a larger official SmolVLA run, scaling will be limited by:

- video decode and CPU-side preprocessing if DataLoader workers cannot keep both GPUs fed;
- NCCL all-reduce cost as model size grows;
- per-rank batch size, because small batches underutilize each GPU;
- startup/warmup overhead in short experiments;
- storage/cache locality for video shards.

The value of this run is correctness: the rank topology, sampler ownership, checkpoint/resume path, and multimodal batch contract are now validated on 2 GPUs.

## Next Comparison Target

The next project step is to run an official LeRobot/SmolVLA DDP fine-tuning baseline on the same machine and dataset subset, then compare it against this Nanotron-style DP path on:

- launch/config complexity;
- samples/s and GPU memory;
- dataloader worker sensitivity;
- checkpoint/resume ergonomics;
- where training time is spent: video decode, forward/backward, optimizer, or gradient sync.
