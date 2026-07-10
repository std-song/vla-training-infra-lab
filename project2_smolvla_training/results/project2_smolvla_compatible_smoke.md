# SmolVLA-Compatible Wrapper Smoke Train

Date: 2026-07-07

This report records the first Project 2 model-wrapper milestone on top of the validated LeRobot VLA batch contract.

## Scope

This is not a claim of full official SmolVLA reproduction. The wrapper is a compact SmolVLA-compatible policy used to validate the training path before integrating a heavier pretrained model.

The wrapper consumes the same major modalities expected by a VLA fine-tuning path:

```text
multi-camera RGB images + robot state + task text -> action prediction -> masked MSE loss
```

## Model

Implemented in `smolvla_nanotron/models/smolvla_compatible.py`:

- `MultiCameraVisionEncoder`: shared CNN over sorted camera tensors
- `TaskTextHashEncoder`: dependency-free hashed task-text embedding
- `SmolVLACompatiblePolicy`: fuses vision, state, and task features into an action head

Input batch:

```text
images[observation.images.cam_high]        uint8 [B, 3, 224, 224]
images[observation.images.cam_left_wrist]  uint8 [B, 3, 224, 224]
images[observation.images.cam_right_wrist] uint8 [B, 3, 224, 224]
state                                      float32 [B, 14]
action                                     float32 [B, 14]
task_text                                  list[str]
action_mask                                bool [B, 14]
```

## Smoke Train

Command summary:

```bash
PYTHONPATH=. python scripts/train_smolvla_compatible.py \
  --cache-root /root/autodl-tmp/vla-infra-project2/hf \
  --output-dir /root/autodl-tmp/vla-infra-project2/outputs/smolvla_compatible_smoke_v2 \
  --batch-size 2 \
  --start-index 99000 \
  --limit 24 \
  --train-steps 4 \
  --image-size 224 \
  --workers 2 \
  --pin-memory \
  --persistent-workers \
  --save-every 2
```

Observed output:

```text
step=1 loss=0.345691 grad_norm=0.3624 pred_mean=-0.0093 target_mean=-0.0080
step=2 loss=0.316953 grad_norm=0.3497 pred_mean=-0.0128 target_mean=0.0010
step=3 loss=0.287255 grad_norm=0.3497 pred_mean=-0.0153 target_mean=0.0057
step=4 loss=0.253827 grad_norm=0.3605 pred_mean=-0.0173 target_mean=0.0056

steps=4
final_step=4
elapsed_sec=0.701
steps_per_sec=5.707
cuda_max_allocated_mib=62.1
checkpoint=/root/autodl-tmp/vla-infra-project2/outputs/smolvla_compatible_smoke_v2/latest.pt
```

## Resume Validation

Command summary:

```bash
PYTHONPATH=. python scripts/train_smolvla_compatible.py \
  --cache-root /root/autodl-tmp/vla-infra-project2/hf \
  --output-dir /root/autodl-tmp/vla-infra-project2/outputs/smolvla_compatible_smoke_v2_resume \
  --batch-size 2 \
  --start-index 99000 \
  --limit 24 \
  --train-steps 6 \
  --image-size 224 \
  --workers 2 \
  --pin-memory \
  --persistent-workers \
  --resume-from /root/autodl-tmp/vla-infra-project2/outputs/smolvla_compatible_smoke_v2/latest.pt
```

Observed output:

```text
resumed_from=/root/autodl-tmp/vla-infra-project2/outputs/smolvla_compatible_smoke_v2/latest.pt step=4
step=5 loss=0.215416 grad_norm=0.3817 pred_mean=-0.0188 target_mean=-0.0080
step=6 loss=0.163825 grad_norm=0.4015 pred_mean=-0.0220 target_mean=0.0010

steps=2
final_step=6
elapsed_sec=0.646
steps_per_sec=3.098
cuda_max_allocated_mib=62.1
checkpoint=/root/autodl-tmp/vla-infra-project2/outputs/smolvla_compatible_smoke_v2_resume/latest.pt
```

## Interpretation

This milestone validates the first end-to-end VLA model training loop in Project 2:

- shard-aware LeRobot video decoding
- multi-camera image batch collation
- task-text conditioning
- robot-state conditioning
- action prediction loss
- optimizer step
- checkpoint save
- checkpoint resume from step 4 to step 6

The wrapper is intentionally small so data-pipeline and training-loop correctness remain easy to inspect. The next step is to replace the lightweight encoders with a closer SmolVLA architecture or pretrained policy while preserving the same batch/checkpoint/profiling surface.
