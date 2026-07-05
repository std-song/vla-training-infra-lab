# VLA Training Infrastructure Lab

A practical training-infrastructure lab for Vision-Language-Action (VLA) model training experiments on resource-constrained RTX 3090 hardware.

The first milestone validates a Qwen2-MoE small-scale pretraining path with Nanotron. Later milestones extend the same project toward multi-GPU DP/EP experiments, SmolVLA finetuning, data-pipeline profiling, and low-latency VLA inference optimization.

## Why This Project

This project is designed for a VLA training infrastructure engineering portfolio. The emphasis is not benchmark model quality; it is training-system correctness, distributed behavior, memory and throughput measurement, checkpoint reliability, and practical debugging under limited hardware.

## Current Status

Completed:

- Set up AutoDL RTX 3090 environment with Python 3.10.8, PyTorch 2.1.2+cu118, and CUDA toolkit 11.8.
- Installed Nanotron dependencies, flash-attn, and grouped_gemm.
- Passed Nanotron MoE kernel smoke test: `tests/test_moe.py`.
- Ran single-GPU Qwen2-MoE dummy-data training for 5 steps.
- Saved checkpoint at `checkpoints/qwen2_moe_smoke/5`.

Observed smoke output:

```text
iteration: 5 / 5
consumed_tokens: 1.28K
time_per_iteration_ms: 20.3
tokens_per_sec: 12.6K
tokens_per_sec_per_gpu: 12.6K
global_batch_size: 256
grad_norm: 1.64
lm_loss: 8.35
lr: 1e-05
model_tflops_per_gpu: 0.0843
Saving checkpoint at checkpoints/qwen2_moe_smoke/5
```

## Repository Layout

```text
configs/qwen2_moe/       Qwen2-MoE training configs
scripts/                 setup and launch scripts
results/                 experiment records
docs/                    design notes, roadmap, troubleshooting
patches/                 compatibility patches for upstream Nanotron
```

## Quick Start

Clone Nanotron separately on the training machine:

```bash
cd /root/autodl-tmp/vla-infra
git clone https://github.com/huggingface/nanotron.git nanotron
cd nanotron
```

Install the environment following [`scripts/setup_autodl_3090.sh`](scripts/setup_autodl_3090.sh). Then copy the smoke config into Nanotron:

```bash
mkdir -p examples/smoke
cp /path/to/vla-training-infra-lab/configs/qwen2_moe/config_qwen2_moe_smoke.yaml \
  examples/smoke/config_qwen2_moe_smoke.yaml
```

Run the smoke training:

```bash
CUDA_DEVICE_MAX_CONNECTIONS=1 PYTHONPATH=src torchrun --nproc_per_node=1 \
  run_train.py --config-file examples/smoke/config_qwen2_moe_smoke.yaml
```

## Roadmap

1. Validate checkpoint resume from step 5 to step 7.
2. Run 2-GPU data parallel smoke training.
3. Run 4/8-GPU DP baseline.
4. Add expert-parallel experiments: `dp=4, ep=2`, `dp=2, ep=4`, `dp=1, ep=8`.
5. Add profiling reports for tokens/sec, VRAM, step time, communication, and checkpoint cost.
6. Extend the project with SmolVLA finetuning and VLA data-pipeline experiments.
