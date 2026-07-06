# VLA Training Infrastructure Lab

A practical training-infrastructure lab for Vision-Language-Action (VLA) model training experiments on resource-constrained RTX 3090 hardware.

This repository is organized as a portfolio project for VLA training infrastructure roles. The emphasis is training-system correctness, distributed behavior, memory and throughput measurement, checkpoint reliability, and practical debugging under limited hardware rather than benchmark model quality.

## Target Role Alignment

The project maps directly to common VLA training-infra requirements:

| Requirement | Project coverage |
| --- | --- |
| PyTorch distributed training | Nanotron-based Qwen2-MoE training path, planned DP/TP/PP/EP validation |
| MoE training | Router top-k, expert token permutation, GroupedGEMM expert MLP, shared expert |
| Mixed precision | BF16 training on RTX 3090 |
| Operator acceleration | FlashAttention and fused RMSNorm/rotary paths where available |
| Checkpoint/resume | Step-5 to step-7, step-500 to step-520 resume validation |
| Performance analysis | tokens/s, step time, memory, GPU utilization, power, checkpoint size |
| Data pipeline | planned VLA/LeRobot-style data schema and shard strategy |
| Experiment management | structured configs, scripts, troubleshooting notes, result reports |

## Current Status

Completed:

- AutoDL RTX 3090 environment validated with Python 3.10.8, PyTorch 2.1.2+cu118, CUDA toolkit 11.8.
- Nanotron dependencies installed, including `flash-attn==2.5.8` and `grouped_gemm`.
- Nanotron MoE test passed: `PYTHONPATH=src pytest -q tests/test_moe.py`.
- Qwen2-MoE single-GPU smoke run completed for 5 steps.
- Checkpoint resume validated from step 5 to step 7.
- Tiny 20-step and 100-step baseline profiling completed.
- Stronger 75.5M-parameter 500-step baseline profiling completed.
- Step-500 checkpoint resumed to step 520.
- Activation recomputation A/B completed on the 75.5M-parameter baseline v2.
- Compatibility patches documented for PyTorch 2.1.2 collect-env behavior and dummy-data resume metadata.

Latest baseline summary:

| Metric | Value |
| --- | ---: |
| GPU | 1x RTX 3090 24 GiB |
| Model size | 75.5M params |
| MoE | 8 experts, top-k=1, shared expert |
| Parallelism | DP=1, TP=1, PP=1, EP=1 |
| 500-step avg throughput, logged steps >= 50 | 10,544 tokens/s |
| 500-step avg step time, logged steps >= 50 | 49.59 ms |
| Max sampled GPU memory | 2,271 MiB |
| Checkpoint size | 1009 MiB |
| Resume validation | step 500 -> step 520 |
| Recompute A/B | -21.5% throughput, no useful memory win at this scale |

See the full reports: [`results/qwen2_moe_baseline_v2_1x3090.md`](results/qwen2_moe_baseline_v2_1x3090.md) and [`results/qwen2_moe_baseline_1x3090.md`](results/qwen2_moe_baseline_1x3090.md).

## Repository Layout

```text
configs/qwen2_moe/       Qwen2-MoE training configs
scripts/                 setup, launch, and profiling scripts
results/                 curated experiment reports
docs/                    design notes, roadmap, experiment matrix, troubleshooting
patches/                 compatibility patches for upstream Nanotron
```

## Quick Start

Clone Nanotron separately on the training machine:

```bash
cd /root/autodl-tmp/vla-infra
git clone https://github.com/huggingface/nanotron.git nanotron
cd nanotron
```

Install the environment following [`scripts/setup_autodl_3090.sh`](scripts/setup_autodl_3090.sh). Then copy a config into the Nanotron checkout:

```bash
mkdir -p examples/smoke
cp /root/autodl-tmp/vla-infra/vla-training-infra-lab/configs/qwen2_moe/config_qwen2_moe_baseline_v2_500step.yaml \
  examples/smoke/config_qwen2_moe_baseline_v2_500step.yaml
```

Run the baseline v2 training:

```bash
CUDA_DEVICE_MAX_CONNECTIONS=1 PYTHONPATH=src torchrun --nproc_per_node=1 \
  run_train.py --config-file examples/smoke/config_qwen2_moe_baseline_v2_500step.yaml
```

## What This Project Can Honestly Claim Today

This project currently validates the single-GPU Qwen2-MoE path: router top-k, expert dispatch inside one rank, GroupedGEMM, FlashAttention, BF16, checkpoint save/resume, 500-step stability, activation recomputation A/B, and coarse profiling.

It does not yet claim full 8-GPU TP/PP/EP training. That is the next milestone and should be validated step by step before appearing as a finished resume bullet.

## Next Step

The immediate next experiment should be 2-GPU distributed smoke validation:

1. Run `dp=2, tp=1, pp=1, ep=1` to validate launcher, rank logs, and gradient synchronization.
2. Run `dp=1, tp=2, pp=1, ep=1` to validate tensor-parallel sharding.
3. Run `dp=1, tp=1, pp=2, ep=1` to validate pipeline-parallel scheduling.
4. Only after these are stable, rent 8 GPUs for DP/TP/PP scaling and then inspect EP readiness.

The detailed plan is in [`docs/next_steps.md`](docs/next_steps.md) and [`docs/experiment_matrix.md`](docs/experiment_matrix.md).
