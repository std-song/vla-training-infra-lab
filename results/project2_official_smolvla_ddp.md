# Project 2: Official LeRobot/SmolVLA DDP Baseline

Date: 2026-07-07

This report validates the official LeRobot SmolVLA training entrypoint on the same 2-GPU instance used by the Nanotron-style DP wrapper. The goal is not model quality; it is to establish a reproducible official baseline for launch behavior, DDP mechanics, checkpoint/resume, and throughput interpretation.

## Environment

| Item | Value |
| --- | --- |
| Host | AutoDL cloned 2-GPU instance |
| GPU | 2x NVIDIA GeForce RTX 4080 SUPER, 32 GiB each |
| Python | 3.12.3 |
| PyTorch | 2.8.0+cu128 |
| CUDA runtime | 12.8 |
| LeRobot | 0.6.0 local editable checkout |
| Launcher | `accelerate launch --num_processes=2 --multi_gpu` |
| Dataset | `lerobot/aloha_mobile_cabinet`, episode `[0]` |
| Dataset root | local Hugging Face snapshot |
| Policy | official `policy.type=smolvla` |

## Launch Notes

The official training entrypoint uses Hugging Face Accelerate internally:

- `Accelerator`
- `DistributedDataParallelKwargs(find_unused_parameters=True)`
- Accelerate-sharded dataloader
- rank-main logging and checkpointing
- checkpoint resume through `train_config.json`

The first successful run used a local snapshot root:

```text
/root/autodl-tmp/vla-infra-project2/hf/hub/datasets--lerobot--aloha_mobile_cabinet/snapshots/7a752b39f7e69de7e38aee485a6bab07528a061a
```

Using a new empty `dataset.root` caused LeRobot to query the Hub for dataset version tags. On this AutoDL machine that failed with:

```text
httpx.ConnectError: [Errno 99] Cannot assign requested address
```

Pointing `dataset.root` at the existing snapshot avoided the metadata download path and kept dataset loading offline.

The official SmolVLA policy still needs the VLM config/processor from:

```text
HuggingFaceTB/SmolVLM2-500M-Video-Instruct
```

`HF_ENDPOINT=https://hf-mirror.com` was used to cache the small config/processor files. Full pretrained SmolVLA weights were not loaded; the run used:

```text
--policy.load_vlm_weights=false
```

## Failed Attempt Worth Recording

For comparability with the lightweight wrapper, an initial run tried:

```text
--policy.resize_imgs_with_padding='[224,224]'
```

That reached the DDP forward pass but failed in the SmolVLM connector:

```text
RuntimeError: shape '[1, 14, 3, 3072]' is invalid for input of size 150528
```

The fix was to keep the official SmolVLA default:

```text
resize_imgs_with_padding=[512,512]
```

This is a useful infra lesson: image size is not an arbitrary dataloader knob for this model; it must satisfy the VLM patch/grid and connector assumptions.

## Successful 2-GPU DDP Smoke

Launch:

```bash
HF_HOME=/root/autodl-tmp/vla-infra-project2/hf \
HF_ENDPOINT=https://hf-mirror.com \
TRANSFORMERS_CACHE=/root/autodl-tmp/vla-infra-project2/hf/transformers \
python -m accelerate.commands.launch \
  --num_processes=2 \
  --multi_gpu \
  -m lerobot.scripts.lerobot_train \
  --policy.type=smolvla \
  --policy.repo_id=local/smolvla_official_ddp \
  --policy.push_to_hub=false \
  --save_checkpoint_to_hub=false \
  --dataset.repo_id=lerobot/aloha_mobile_cabinet \
  --dataset.root=/root/autodl-tmp/vla-infra-project2/hf/hub/datasets--lerobot--aloha_mobile_cabinet/snapshots/7a752b39f7e69de7e38aee485a6bab07528a061a \
  --dataset.episodes='[0]' \
  --batch_size=1 \
  --steps=2 \
  --num_workers=1 \
  --log_freq=1 \
  --save_checkpoint=true \
  --save_freq=2 \
  --output_dir=/root/autodl-tmp/vla-infra-project2/vla-training-infra-lab/outputs/project2_official_smolvla_ddp2_smoke \
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

Run configuration from logs:

| Metric | Value |
| --- | ---: |
| world size | 2 |
| per-rank batch size | 1 |
| effective batch size | 2 |
| dataset frames | 1,500 |
| dataset episodes | 1 |
| learnable params | 13,916,512 |
| total params | 226,429,216 |
| image size | 512x512 |
| VLM layers | 2 |
| expert layers | 2 |

Training logs:

| Step | Loss | Grad norm | Update s | Data s | Samples/s | GPU mem |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1.776 | 22.964 | 0.991 | 0.615 | 1 | 0.88 GiB |
| 2 | 2.863 | 31.092 | 0.081 | 0.002 | 24 | 0.88 GiB |

The first step includes warmup effects. Step 2 is the more representative short-window datapoint, but a longer run is needed before making strong throughput claims.

DDP emitted:

```text
find_unused_parameters=True was specified in DDP constructor, but did not find any unused parameters
```

This means the official path pays an extra autograd graph traversal in this configuration. It is likely a conservative default for policies with conditional computation.

## Checkpoint / Resume

Checkpoint size:

| Checkpoint | Size |
| --- | ---: |
| `checkpoints/000002` | 774 MiB |
| `checkpoints/000003` | 774 MiB |

Checkpoint structure:

```text
pretrained_model/config.json
pretrained_model/model.safetensors
pretrained_model/policy_preprocessor.json
pretrained_model/policy_postprocessor.json
pretrained_model/train_config.json
training_state/optimizer_state.safetensors
training_state/rng_state.safetensors
training_state/scheduler_state.json
training_state/training_step.json
```

Official resume requires `--config_path` pointing at the checkpoint's `train_config.json`:

```bash
python -m accelerate.commands.launch \
  --num_processes=2 \
  --multi_gpu \
  -m lerobot.scripts.lerobot_train \
  --config_path=/root/autodl-tmp/vla-infra-project2/vla-training-infra-lab/outputs/project2_official_smolvla_ddp2_smoke/checkpoints/000002/pretrained_model/train_config.json \
  --resume=true \
  --steps=3 \
  --save_freq=3 \
  --wandb.enable=false
```

Resume result:

| Metric | Value |
| --- | ---: |
| resumed checkpoint | `000002` |
| resumed data order | epoch 0, sample 4 |
| final step | 3 |
| step 3 loss | 2.693 |
| step 3 samples/s | 1 |
| step 3 GPU mem | 0.88 GiB |

The resume step again includes setup and first-batch overhead, so it should not be used as a steady-state throughput measurement.

## Comparison Against Nanotron-Style DP Wrapper

| Dimension | Nanotron-style DP wrapper | Official LeRobot/SmolVLA DDP |
| --- | --- | --- |
| Model | lightweight multimodal wrapper | official SmolVLA policy |
| Total params | small, smoke-test scale | 226M with 2 VLM layers |
| Launcher | `torchrun` | `accelerate launch` |
| DP implementation | explicit `DistributedSampler` + DDP | Accelerate-prepared dataloader + DDP |
| Batch size | 2 per rank, global 4 | 1 per rank, global 2 |
| Image size | 224x224 | 512x512 |
| 2-GPU smoke throughput | 28.17 samples/s | step 2: 24 samples/s |
| Peak GPU memory | 64.4 MiB/rank | 0.88 GiB/rank |
| Checkpoint size | small PyTorch checkpoint | 774 MiB per checkpoint |
| Resume style | `--resume-from latest.pt` | `--config_path .../train_config.json --resume=true` |

These numbers are not an apples-to-apples model-performance benchmark because the official policy is much larger and uses a 512x512 image path. The useful comparison is infrastructure-level:

- official LeRobot has more complete processor/checkpoint packaging;
- the custom DP wrapper has clearer minimal rank mechanics and much smaller iteration overhead;
- official SmolVLA has stricter image-size/model-shape assumptions;
- official resume is richer but more coupled to its checkpoint directory layout;
- both paths now validate 2-GPU DDP correctness on the same dataset family.

## Next Measurement

The next useful run is a longer official DDP profile, for example 20-50 steps after warmup, while recording:

- average `samples/s` after step 5;
- max `gpu_mem_gb`;
- dataloading vs update time;
- effect of `num_workers=0/1/2/4`;
- bf16 mixed precision if supported by the rented GPU.
