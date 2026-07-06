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
| Checkpoint/resume | Step-5 to step-7 resume validation, step-100 checkpoint artifacts |
| Performance analysis | tokens/s, step time, memory, GPU utilization, power sampling |
| Data pipeline | planned VLA/LeRobot-style data schema and shard strategy |
| Experiment management | structured configs, scripts, troubleshooting notes, result reports |

## Current Status

Completed:

- AutoDL RTX 3090 environment validated with Python 3.10.8, PyTorch 2.1.2+cu118, CUDA toolkit 11.8.
- Nanotron dependencies installed, including `flash-attn==2.5.8` and `grouped_gemm`.
- Nanotron MoE test passed: `PYTHONPATH=src pytest -q tests/test_moe.py`.
- Qwen2-MoE single-GPU smoke run completed for 5 steps.
- Checkpoint resume validated from step 5 to step 7.
- 20-step and 100-step single-GPU baseline profiling completed.
- Compatibility patches documented for PyTorch 2.1.2 collect-env behavior and dummy-data resume metadata.

Latest baseline summary:

| Metric | Value |
| --- | ---: |
| GPU | 1x RTX 3090 24 GiB |
| Model size | 2.36M params |
| MoE | 4 experts, top-k=1, shared expert |
| Parallelism | DP=1, TP=1, PP=1, EP=1 |
| 100-step avg throughput, steps >= 10 | 9,860 tokens/s |
| 100-step avg step time, steps >= 10 | 27.09 ms |
| Max sampled GPU memory | 993 MiB |
| Max sampled GPU util | 98% |
| Checkpoint size | 32 MiB |

See the full report: [`results/qwen2_moe_baseline_1x3090.md`](results/qwen2_moe_baseline_1x3090.md).

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
cp /root/autodl-tmp/vla-infra/vla-training-infra-lab/configs/qwen2_moe/config_qwen2_moe_baseline_100step.yaml \
  examples/smoke/config_qwen2_moe_baseline_100step.yaml
```

Run 100-step profiling:

```bash
bash /root/autodl-tmp/vla-infra/vla-training-infra-lab/scripts/run_qwen2_moe_profile_100step.sh \
  examples/smoke/config_qwen2_moe_baseline_100step.yaml \
  qwen2_moe_baseline_100step
```

## What This Project Can Honestly Claim Today

This project currently validates the single-GPU Qwen2-MoE path: router top-k, expert dispatch inside one rank, GroupedGEMM, FlashAttention, BF16, checkpoint save, checkpoint resume, and coarse profiling.

It does not yet claim full 8-GPU TP/PP/EP training. That is the next milestone and should be validated step by step before appearing as a finished resume bullet.

## Next Step

The immediate next experiment should be a stronger single-GPU baseline before renting 8 GPUs:

1. Increase model size enough to use several GiB of VRAM.
2. Run 500-1000 steps with `iteration_step_info_interval=10`.
3. Compare activation recomputation on/off.
4. Resume from the final checkpoint for another short run.
5. Then move to 2-GPU DP and 2-GPU TP smoke tests.

The detailed plan is in [`docs/next_steps.md`](docs/next_steps.md) and [`docs/experiment_matrix.md`](docs/experiment_matrix.md).
