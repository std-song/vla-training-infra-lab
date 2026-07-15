# VLASH Pi0.5 Reproduction

This directory contains only the project-specific configuration and replay adapters.
Training and asynchronous action scheduling use the upstream Apache-2.0 VLASH
implementation in `repos/vlash` on the experiment host.

## Experiment Matrix

1. `pi05_sync_lora_aloha.yaml`: standard Pi0.5 LoRA fine-tuning baseline.
2. `pi05_async_shared_lora_aloha.yaml`: VLASH future-state delay training with
   shared observation encoding.
3. `pi05_libero_standard_5000.yaml`: no-delay LIBERO LoRA baseline.
4. `pi05_libero_stale_5000.yaml`: shared-observation LIBERO delay augmentation
   with stale/current state and offsets `0..2`.
5. `pi05_libero_learned_5000.yaml`: the same delay range with the learned GRU
   future-state predictor.
6. `results/libero_d4_delay_sweep/d4_training_overrides.yaml`: the compact
   override manifest used to extend both LIBERO delay variants to offsets
   `0..4` without duplicating the episode split and normalization statistics.

The configs use `__PROJECT_ROOT__` as a deployment placeholder. On the AutoDL
host it is rendered as `/root/autodl-tmp/vla-infra-project3-pi05` before launch.

The initial five-step run validates the real upstream dataset, policy, LoRA,
checkpoint, and loss path. Longer matched runs are only launched after the
smoke run succeeds.

## LIBERO Closed-Loop Extension

The reproduction was extended from held-out trajectory alignment to paired
LIBERO rollouts. `evaluate_libero_closed_loop.py` models inference delay in
10 Hz control ticks while the existing action queue continues to execute.
`standard_naive` installs a late chunk from index 0, whereas `standard_skip`
discards the first `d` stale actions. The learned-state path can be compared
against stale input with the same policy weights.

The detailed protocol, raw episode tables, paired bootstrap results, and limits
are in
[`../results/libero_standard_delay_ablation/README_CN.md`](../results/libero_standard_delay_ablation/README_CN.md).
The follow-up paired `d=0..4` sweep and same-weight future-state ablation are in
[`../results/libero_d4_delay_sweep/README_CN.md`](../results/libero_d4_delay_sweep/README_CN.md).

The AutoDL CUDA 11.8 host uses the upstream TorchCodec backend. Its Conda
runtime ships an older `libstdc++`, so launch commands preload the compatible
system `libstdc++.so.6`. This does not change VLASH data, policy, LoRA, or
shared-observation code.
