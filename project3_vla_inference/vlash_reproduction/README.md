# VLASH Pi0.5 Reproduction

This directory contains only the project-specific configuration and replay adapters.
Training and asynchronous action scheduling use the upstream Apache-2.0 VLASH
implementation in `repos/vlash` on the experiment host.

## Experiment Matrix

1. `pi05_sync_lora_aloha.yaml`: standard Pi0.5 LoRA fine-tuning baseline.
2. `pi05_async_shared_lora_aloha.yaml`: VLASH future-state delay training with
   shared observation encoding.

The configs use `__PROJECT_ROOT__` as a deployment placeholder. On the AutoDL
host it is rendered as `/root/autodl-tmp/vla-infra-project3-pi05` before launch.

The initial five-step run validates the real upstream dataset, policy, LoRA,
checkpoint, and loss path. Longer matched runs are only launched after the
smoke run succeeds.

The AutoDL CUDA 11.8 host uses the upstream TorchCodec backend. Its Conda
runtime ships an older `libstdc++`, so launch commands preload the compatible
system `libstdc++.so.6`. This does not change VLASH data, policy, LoRA, or
shared-observation code.
