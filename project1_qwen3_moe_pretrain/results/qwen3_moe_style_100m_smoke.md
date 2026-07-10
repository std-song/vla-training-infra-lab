# Qwen3-MoE-style Nanotron Smoke Result

Date: 2026-07-09
Hardware: 1 x RTX 3090 24GB on AutoDL
Software: Python 3.10.8, PyTorch 2.1.2+cu118, FlashAttention 2.5.8, Nanotron 0.4

## Objective

Upgrade the project-1 MoE training lab from a Qwen2-MoE-only baseline toward a Qwen3-MoE-style training path while keeping the claim factual and reproducible.

This implementation keeps Nanotron's existing Qwen2 training stack and MoE layer as the base, then adds the Qwen3-relevant architectural and training features that are small enough to validate in the current lab:

- QK-Norm after Q/K projection and before RoPE.
- Bias-free QKV projection through `attention_bias: false`.
- Qwen3-style large RoPE base through `rope_theta: 1000000.0`.
- MoE top-2 routing with 8 local experts and no shared expert.
- Router auxiliary loss added back into the training loss.
- Optional global-batch router load-balancing statistics through the data-parallel process group.
- Checkpoint and resume validation.

## Model Config

The 100M smoke config is stored at:

- `configs/qwen3_moe_style/config_qwen3_moe_style_100m_smoke.yaml`
- `configs/qwen3_moe_style/config_qwen3_moe_style_100m_resume.yaml`

Key settings:

| Field | Value |
| --- | --- |
| Parameters | 108M |
| Layers | 4 |
| Hidden size | 576 |
| Attention heads | 8 |
| KV heads | 4 |
| Vocabulary size | 16384 |
| Sequence length | 128 |
| Experts | 8 |
| Active experts | top-2 |
| Shared expert | disabled |
| Router aux loss | enabled, coef=0.01 |
| Router load balance | global_batch |
| Precision | BF16 |
| Attention backend | FlashAttention-2 |

## Validation

Command:

```bash
cd /root/autodl-tmp/vla-infra/nanotron
CUDA_DEVICE_MAX_CONNECTIONS=1 PYTHONPATH=src /root/miniconda3/bin/torchrun --nproc_per_node=1 \
  run_train.py --config-file examples/smoke/config_qwen3_moe_style_100m_smoke.yaml
```

Result:

- Model built successfully: 108M parameters.
- 5 training steps completed.
- Checkpoint saved to `checkpoints/qwen3_moe_style_100m_smoke/5`.
- Warm-step throughput after startup: around 5.8K to 6.6K tokens/s on one RTX 3090.
- Peak reserved memory during warm steps: about 2.2 GiB.

Resume command:

```bash
cd /root/autodl-tmp/vla-infra/nanotron
CUDA_DEVICE_MAX_CONNECTIONS=1 PYTHONPATH=src /root/miniconda3/bin/torchrun --nproc_per_node=1 \
  run_train.py --config-file examples/smoke/config_qwen3_moe_style_100m_resume.yaml
```

Resume result:

- Loaded checkpoint metadata from step 5.
- Continued to step 7.
- Checkpoint saved to `checkpoints/qwen3_moe_style_100m_smoke/7`.

## Code Patch

The Nanotron patch is stored at:

- `patches/qwen3_moe_style_nanotron.patch`

The patch touches:

- `src/nanotron/config/models_config.py`
- `src/nanotron/models/qwen.py`
- `src/nanotron/nn/moe.py`

## Current Claim Boundary

This should be described as a Qwen3-MoE-style Nanotron adaptation, not as a full official Qwen3-MoE reproduction.

Validated:

- QK-Norm placement.
- Bias-free QKV setting.
- Top-k MoE routing path.
- Router auxiliary loss wiring.
- Global-batch load-balance statistics path in single-rank mode.
- BF16 FlashAttention training smoke.
- Checkpoint/resume.

Not yet validated:

- Multi-node or 8-GPU scaling.
- Expert-parallel All-to-All overlap.
- Non-contiguous token dispatch memory coalescing optimization.
- Full Qwen3 tokenizer/data recipe.

## Resume Wording

Recommended factual wording:

> ?? Nanotron ?? Qwen3-MoE-style ??????????? Qwen2-MoE ??????? QK-Norm???? QKV?Top-2 MoE router ????? global-batch ????????? 108M ?? smoke ???? RTX 3090 ??? BF16/FlashAttention ???checkpoint/resume ? MoE ?????
