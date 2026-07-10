# Project 2 Final Report: SmolVLA / LeRobot Distributed Fine-Tuning Infrastructure

Date: 2026-07-07

This project builds a compact but reproducible VLA training-infrastructure study around LeRobot/SmolVLA. The work intentionally separates two goals:

1. validate the official LeRobot/SmolVLA DDP path as the real-model baseline;
2. build a small Nanotron-style DP wrapper to expose and debug the same infrastructure surfaces with faster iteration.

This is not presented as a full native Nanotron port of SmolVLA. Nanotron is primarily built around text LLM pretraining contracts, while SmolVLA needs multimodal batches, action targets, action masks, video/image preprocessing, and different logging/checkpoint semantics. The current project therefore treats the custom path as a Nanotron-oriented prototype and compares it against the official SmolVLA DDP implementation.

## Why This Project Matters

VLA fine-tuning is not just a model call. A useful training stack must coordinate:

- LeRobot dataset metadata, parquet state/action tables, and multi-camera video shards;
- CPU video decode, image preprocessing, host-to-device transfer, and distributed samplers;
- multimodal model inputs: images, robot state, task text, action targets, and action masks;
- DDP launch, rank ownership, all-reduced metrics, checkpoint/resume, and precision policy;
- throughput and memory profiling that explains where time is spent.

The project follows that order: first make the data contract explicit, then validate a cheap training loop, then run the official SmolVLA baseline, then tune the real training path.

## Completed Scope

| Area | Completed work |
| --- | --- |
| Dataset | LeRobot `aloha_mobile_cabinet` local snapshot, episode filtering, parquet state/action loading, task text, metadata inspection |
| Video | three-camera video decode, shard-aware resolver via `meta/episodes`, 512x512 official-compatible preprocessing |
| Batch contract | `VLABatch` with state, effort, action, action_mask, episode/frame/timestamp, task text, and images |
| Custom DP | `torchrun` + explicit DDP, `DistributedSampler`, all-reduced metrics, rank-0 checkpoint, resume |
| Official baseline | LeRobot `policy.type=smolvla` with Accelerate DDP on 2 GPUs |
| Profiling | official vs custom 50-step comparison, DataLoader worker sweep, BF16 profile, DDP `find_unused_parameters` tuning |
| Reporting | reproducible commands, metrics tables, interpretation, and figures committed to GitHub |

## System Setup

| Item | Value |
| --- | --- |
| GPU | 2x NVIDIA GeForce RTX 4080 SUPER, 32 GiB each |
| Python | 3.12.3 |
| PyTorch | 2.8.0+cu128 |
| Dataset | `lerobot/aloha_mobile_cabinet`, episode `[0]` |
| Official launcher | `accelerate launch --num_processes=2 --multi_gpu` |
| Custom launcher | `torchrun --nproc_per_node=2` |
| Global batch size | 2 for matched official/custom profiling |
| Image size | 512x512 for matched official/custom profiling |

The official SmolVLA run used a reduced but real policy configuration:

| Metric | Value |
| --- | ---: |
| Total params | 226,429,216 |
| Learnable params | 13,916,512 |
| VLM layers | 2 |
| Expert layers | 2 |
| Vision encoder | frozen |
| Training mode | expert-only |
| Full pretrained VLM weights | not loaded |

## Custom Nanotron-Style DP Wrapper

The custom path is implemented in:

```text
scripts/train_smolvla_compatible_dp.py
smolvla_nanotron/data/collator.py
smolvla_nanotron/models/smolvla_compatible.py
```

It implements the training surfaces expected from a distributed trainer:

- explicit rank/local-rank/world-size discovery;
- NCCL process-group initialization;
- `DistributedSampler` ownership of the LeRobot dataset;
- multimodal `VLABatch` movement to GPU;
- lightweight image/state/task-text fusion policy;
- masked action MSE loss;
- gradient clipping and optimizer step;
- all-reduced loss, grad norm, prediction mean, and target mean;
- rank-0 checkpoint writing and resume from `latest.pt`;
- max per-rank CUDA memory reporting.

This path is intentionally not a full native Nanotron trainer plugin. It is called Nanotron-style because it mirrors the infrastructure boundaries that would be required before a true Nanotron integration: explicit rank state, deterministic sampler ownership, config-driven launch, checkpoint/resume, distributed metric reduction, and profiling.

## Official SmolVLA DDP Baseline

The official path validates the real LeRobot/SmolVLA training stack:

```text
accelerate launch --num_processes=2 --multi_gpu -m lerobot.scripts.lerobot_train
```

Important configuration choices:

- `--policy.type=smolvla`
- `--policy.freeze_vision_encoder=true`
- `--policy.train_expert_only=true`
- `--policy.load_vlm_weights=false`
- `--policy.num_vlm_layers=2`
- `--policy.num_expert_layers=2`
- `--dataset.episodes='[0]'`
- `--batch_size=1`
- `--steps=50`
- `--num_workers=1`

This path is slower and heavier than the custom wrapper because it constructs the real SmolVLA stack, uses official processors/checkpoint packaging, and carries more model and optimizer state. That is exactly why it is the right baseline.

## 50-Step Matched Comparison

| Dimension | Official LeRobot/SmolVLA DDP | Nanotron-style DP wrapper |
| --- | ---: | ---: |
| GPUs | 2 | 2 |
| Global batch size | 2 | 2 |
| Image size | 512x512 | 512x512 |
| Model scale | 226M total / 14M trainable | lightweight wrapper |
| Throughput | 23.63 samples/s, steps >= 5 | 38.02 samples/s overall |
| GPU memory/rank | 0.88 GiB | 110.4 MiB |
| Checkpoint size | 774 MiB | 12 MiB dir / 6 MiB per file |
| Launcher | Accelerate DDP | `torchrun` + explicit DDP |
| Resume | official train config + checkpoint dir | `--resume-from latest.pt` |

The custom wrapper is faster because it is much smaller. Its value is not beating official SmolVLA. Its value is giving a compact, inspectable loop for testing data contracts, DDP mechanics, checkpoint behavior, and profiling ideas before porting them to the official path.

## DataLoader Worker Analysis

Official SmolVLA DDP was profiled with `num_workers={0,1,2,4}`. Metrics are averaged over steps `>= 5`.

| num_workers | Avg samples/s | Avg update time | Avg data time | Max GPU mem |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 17.20 | 85.8 ms | 31.4 ms | 0.88 GiB |
| 1 | 23.63 | 82.5 ms | 2.1 ms | 0.88 GiB |
| 2 | 23.17 | 84.1 ms | 2.2 ms | 0.88 GiB |
| 4 | 22.67 | 86.0 ms | 2.5 ms | 0.88 GiB |

Conclusion: one worker per rank is enough to hide most video decode and preprocessing latency in this setup. More workers do not improve end-to-end throughput because the critical path moves to model update and DDP overhead.

![Official SmolVLA worker profile](../assets/figures/project2_official_smolvla_workers.svg)

## BF16 Mixed Precision

For this LeRobot version, `--policy.dtype=bfloat16` is not a valid SmolVLA config field. The working path is launcher-level Accelerate mixed precision:

```text
accelerate launch --mixed_precision=bf16 ...
```

| Precision | Window | Avg samples/s | Avg update time | Avg data time | Max GPU mem |
| --- | --- | ---: | ---: | ---: | ---: |
| fp32/no AMP | steps >= 5 | 23.63 | 82.5 ms | 2.1 ms | 0.88 GiB |
| bf16 | steps >= 5 | 24.61 | 80.1 ms | 2.0 ms | 0.88 GiB |
| fp32/no AMP | steps >= 40 | 23.27 | 83.5 ms | 2.1 ms | 0.88 GiB |
| bf16 | steps >= 40 | 27.45 | 71.3 ms | 1.9 ms | 0.88 GiB |

BF16 is stable in the 50-step smoke profile and gives a modest average improvement. The memory reading is unchanged because the official metric is coarse and the reduced training graph only has 14M trainable parameters.

![Official SmolVLA BF16 profile](../assets/figures/project2_official_smolvla_bf16.svg)

## DDP `find_unused_parameters` Tuning

The official LeRobot training entrypoint uses:

```python
DistributedDataParallelKwargs(find_unused_parameters=True)
```

A temporary remote patch changed this to `False`, ran 50-step profiles, and then restored the official file.

| Precision | DDP unused-parameter search | Avg samples/s | Avg update time | Avg data time | Max mem/rank |
| --- | --- | ---: | ---: | ---: | ---: |
| FP32 | enabled, official default | 23.63 | 82.5 ms | 2.1 ms | 0.88 GiB |
| FP32 | disabled | 26.35 | 74.5 ms | 1.8 ms | 0.88 GiB |
| BF16 | enabled | 24.61 | 80.1 ms | 2.0 ms | 0.88 GiB |
| BF16 | disabled | 24.04 | 81.3 ms | 2.1 ms | 0.88 GiB |

The FP32 run benefits by about 11.5%. The BF16 run does not. The correct infra conclusion is not "disable it everywhere"; it is "make it configurable and profile it under the target precision/model graph."

![Official SmolVLA DDP tuning](../assets/figures/project2_official_smolvla_ddp_tuning.svg)

## What We Can Claim

This project can honestly claim:

- built a reproducible LeRobot/SmolVLA distributed fine-tuning lab on 2 GPUs;
- validated multi-camera VLA data loading, shard resolution, multimodal collation, and action-mask loss plumbing;
- implemented a compact Nanotron-style DP wrapper with DDP, distributed sampler, all-reduced metrics, checkpoint/resume, and memory profiling;
- established an official LeRobot/SmolVLA DDP baseline on the same machine and dataset subset;
- analyzed DataLoader workers, BF16 mixed precision, and DDP unused-parameter detection with concrete throughput/memory measurements.

This project should not claim:

- full native SmolVLA integration into Nanotron `DistributedTrainer`;
- Nanotron TP/PP for SmolVLA;
- real large-scale policy quality improvement;
- full pretrained SmolVLA fine-tuning with complete VLM weights.

## Why Native Nanotron SmolVLA Is More Work

A direct Nanotron integration would require deeper changes because Nanotron's core path is LLM-oriented. The missing pieces include:

| Required area | Why it matters |
| --- | --- |
| VLA batch provider | Nanotron expects token-centric batches; SmolVLA needs images, state, text, actions, and masks |
| Loss adapter | LM loss must be replaced by masked action loss or SmolVLA's native action objective |
| Logging semantics | `tokens/s` and `lm_loss` should become samples/s, frames/s, action loss, data time, update time |
| Checkpoint adapter | official SmolVLA and Nanotron checkpoint layouts differ |
| Config schema | LeRobot dataset, cameras, action horizon, transforms, freezing policy, and precision must be represented |
| Model wrapper | official SmolVLA forward/loss must be exposed through a Nanotron-compatible training module |
| Future TP/PP | splitting vision/VLM/action expert modules needs explicit partition policy |

That is why the current project intentionally stops at a wrapper plus official baseline comparison. It keeps the claim strong and honest while still demonstrating the core training-infra skills.

## Interview Narrative

The story to tell in an interview:

1. I first made the VLA data path observable: parquet fields, video shards, task text, multi-camera tensors, and action masks.
2. I built a small Nanotron-style DDP loop to validate rank topology, sampler behavior, checkpoint/resume, and metric reduction without waiting on a heavy model.
3. I then ran the official LeRobot/SmolVLA DDP path on the same data to anchor the experiment in the real stack.
4. I profiled the official path and found that one dataloader worker per rank hides most decode latency, BF16 is stable with modest speedup, and `find_unused_parameters=False` helps FP32 but not BF16.
5. I kept the boundary clear: this is a distributed VLA training-infra prototype and benchmark, not a completed native Nanotron SmolVLA port.

## Key Result Summary

| Experiment | Main result |
| --- | --- |
| Official SmolVLA DDP | 23.63 samples/s, 0.88 GiB/rank, 774 MiB checkpoint |
| Custom Nanotron-style DP wrapper | 38.02 samples/s, 110.4 MiB/rank, 12 MiB checkpoint dir |
| DataLoader workers | `num_workers=1` improves official DDP from 17.20 to 23.63 samples/s |
| BF16 | steps >= 5 throughput improves from 23.63 to 24.61 samples/s |
| DDP tuning | FP32 `find_unused_parameters=False` improves 23.63 to 26.35 samples/s; BF16 does not benefit |

## Next Engineering Step

If this project continues, the best next step is not to rewrite all of Nanotron immediately. A realistic next phase is:

1. define a `NanotronSmolVLAConfig` schema;
2. implement a model/loss adapter that exposes official SmolVLA through a Nanotron-like `forward(batch) -> loss, metrics` contract;
3. add a VLA batch provider and logger with samples/s, frames/s, data time, update time, and action loss;
4. only then evaluate whether to patch Nanotron `DistributedTrainer` or keep a separate adapter trainer.
