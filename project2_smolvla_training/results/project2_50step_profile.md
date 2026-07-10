# Project 2: 50-Step DP Profiling Comparison

Date: 2026-07-07

This report compares a 50-step official LeRobot/SmolVLA DDP run with the custom Nanotron-style DP wrapper on the same 2-GPU machine and dataset subset.

The goal is infrastructure comparison, not model-quality comparison. The official policy is a 226M-parameter SmolVLA configuration; the custom wrapper is intentionally lightweight and exists to validate the data/distributed/checkpoint surfaces.

## Data Readiness

The required data was already prepared before this run:

- dataset: `lerobot/aloha_mobile_cabinet`
- subset: `dataset.episodes=[0]`
- local snapshot:

```text
/root/autodl-tmp/vla-infra-project2/hf/hub/datasets--lerobot--aloha_mobile_cabinet/snapshots/7a752b39f7e69de7e38aee485a6bab07528a061a
```

Episode 0 uses the already-downloaded `file-000.mp4` shards for all three cameras:

```text
observation.images.cam_high/chunk-000/file-000.mp4
observation.images.cam_left_wrist/chunk-000/file-000.mp4
observation.images.cam_right_wrist/chunk-000/file-000.mp4
```

No additional video download was needed for the 50-step profile.

## Shared Environment

| Item | Value |
| --- | --- |
| GPU | 2x NVIDIA GeForce RTX 4080 SUPER, 32 GiB each |
| Python | 3.12.3 |
| PyTorch | 2.8.0+cu128 |
| CUDA runtime | 12.8 |
| Dataset subset | episode `[0]`, 1,500 frames |
| Workers | 1 per rank |
| Global batch size | 2 |

## Official LeRobot/SmolVLA DDP

Launch summary:

```bash
HF_HOME=/root/autodl-tmp/vla-infra-project2/hf \
HF_ENDPOINT=https://hf-mirror.com \
TRANSFORMERS_CACHE=/root/autodl-tmp/vla-infra-project2/hf/transformers \
python -m accelerate.commands.launch \
  --num_processes=2 \
  --multi_gpu \
  -m lerobot.scripts.lerobot_train \
  --policy.type=smolvla \
  --policy.repo_id=local/smolvla_official_ddp_50step \
  --policy.push_to_hub=false \
  --save_checkpoint_to_hub=false \
  --dataset.repo_id=lerobot/aloha_mobile_cabinet \
  --dataset.root=/root/autodl-tmp/vla-infra-project2/hf/hub/datasets--lerobot--aloha_mobile_cabinet/snapshots/7a752b39f7e69de7e38aee485a6bab07528a061a \
  --dataset.episodes='[0]' \
  --batch_size=1 \
  --steps=50 \
  --num_workers=1 \
  --log_freq=1 \
  --save_checkpoint=true \
  --save_freq=50 \
  --wandb.enable=false \
  --policy.freeze_vision_encoder=true \
  --policy.train_expert_only=true \
  --policy.load_vlm_weights=false \
  --policy.num_vlm_layers=2 \
  --policy.num_expert_layers=2 \
  --policy.chunk_size=10 \
  --policy.n_action_steps=10 \
  --policy.num_steps=2
```

Configuration:

| Metric | Value |
| --- | ---: |
| launcher | Accelerate DDP |
| per-rank batch size | 1 |
| effective batch size | 2 |
| total params | 226,429,216 |
| learnable params | 13,916,512 |
| image size | 512x512 |
| VLM layers | 2 |
| expert layers | 2 |

Profiling result:

| Window | Avg loss | Avg grad norm | Avg update s | Avg data s | Avg samples/s | Max GPU mem |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| steps >= 2 | 1.953 | 16.213 | 0.0828 | 0.0021 | 23.53 | 0.88 GiB |
| steps >= 5 | 1.921 | 15.468 | 0.0825 | 0.0021 | 23.63 | 0.88 GiB |
| steps >= 10 | 1.765 | 13.963 | 0.0825 | 0.0021 | 23.61 | 0.88 GiB |

Warmup:

| Step | Loss | Update s | Data s | Samples/s | GPU mem |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1.777 | 1.122 | 0.523 | 1 | 0.88 GiB |
| 50 | 0.974 | 0.085 | 0.002 | 23 | 0.88 GiB |

Checkpoint:

| Path | Size |
| --- | ---: |
| `outputs/project2_official_smolvla_ddp2_50step/checkpoints/000050` | 774 MiB |

## Nanotron-Style DP Wrapper

Launch summary:

```bash
PYTHONPATH=. torchrun --nproc_per_node=2 \
  scripts/train_smolvla_compatible_dp.py \
  --cache-root /root/autodl-tmp/vla-infra-project2/hf \
  --output-dir outputs/project2_smolvla_compatible_dp2_50step_512 \
  --batch-size 1 \
  --limit 128 \
  --train-steps 50 \
  --image-size 512 \
  --workers 1 \
  --prefetch-factor 2 \
  --persistent-workers \
  --pin-memory \
  --save-every 50
```

Configuration:

| Metric | Value |
| --- | ---: |
| launcher | `torchrun` + explicit DDP |
| per-rank batch size | 1 |
| global batch size | 2 |
| image size | 512x512 |
| dataset samples | 128-frame window |
| checkpoint format | plain PyTorch checkpoint |

Result:

| Metric | Value |
| --- | ---: |
| final step | 50 |
| loss, step 1 -> step 50 | 0.496270 -> 0.024558 |
| elapsed | 2.630 s |
| steps/s | 19.009 |
| samples/s | 38.018 |
| max CUDA allocated per rank | 110.4 MiB |
| checkpoint dir size | 12 MiB |
| checkpoint file size | 6.0 MiB each |

## Comparison

| Dimension | Official LeRobot/SmolVLA DDP | Nanotron-style DP wrapper |
| --- | ---: | ---: |
| GPUs | 2 | 2 |
| global batch size | 2 | 2 |
| image size | 512x512 | 512x512 |
| model scale | 226M total / 14M trainable | lightweight wrapper |
| warmup-excluded samples/s | 23.6 | 38.0 overall |
| GPU memory/rank | 0.88 GiB | 110.4 MiB |
| checkpoint size | 774 MiB | 12 MiB dir / 6 MiB per file |
| launcher | Accelerate | `torchrun` |
| checkpoint resume interface | `train_config.json + --resume=true` | `--resume-from latest.pt` |

## Interpretation

The official run is slower and heavier for expected reasons:

- it constructs a real SmolVLA stack with a SmolVLM backbone and action expert;
- it keeps official preprocessing, processor state, optimizer state, RNG state, and safetensors model packaging;
- it uses 512x512 image preprocessing and language/model processors;
- Accelerate sets `find_unused_parameters=True`, which emitted a warning and adds an extra autograd graph traversal in this configuration.

The custom DP wrapper is faster because it is deliberately smaller. Its value is not replacing official SmolVLA; its value is exposing and validating the training-infra mechanics in a compact, hackable form:

- rank topology and NCCL initialization;
- shard-aware LeRobot video loading;
- `DistributedSampler` behavior;
- rank-0 checkpointing;
- all-reduced metrics;
- fast iteration for dataloader and checkpoint experiments.

The two paths are now complementary:

- official DDP proves compatibility with LeRobot/SmolVLA's real training stack;
- Nanotron-style DP provides a controlled infra sandbox for experiments before porting ideas into a heavier trainer.

## Next Questions

The next useful experiments are:

- official DDP with `num_workers=0/2/4` to measure video pipeline sensitivity;
- official DDP with bf16 mixed precision if the current GPU/runtime supports it well;
- `find_unused_parameters=False` experiment by patching the official Accelerate DDP kwargs;
- longer run with checkpoints every 50-100 steps to measure checkpoint overhead.
