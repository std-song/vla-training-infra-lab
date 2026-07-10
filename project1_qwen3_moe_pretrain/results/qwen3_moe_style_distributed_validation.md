# Qwen3-MoE-style Distributed Validation

Date: 2026-07-09
Hardware: 2 x RTX 3090 24GB on AutoDL
Software: Python 3.10.8, PyTorch 2.1.2+cu118, FlashAttention 2.5.8, Nanotron 0.4

## Goal

Validate the Qwen3-MoE-style Nanotron adaptation beyond single-GPU smoke tests. The focus is not long pretraining quality, but distributed training correctness and failure-boundary analysis:

- DP2: verify data-parallel training, global-batch router load-balancing statistics, and checkpoint/resume.
- TP2: verify QK-Norm, packed QKV, FlashAttention, and tensor-parallel parameter partitioning.
- EP2: push expert parallelism far enough to identify the concrete dispatch/GEMM blocker.

## Configs

- `configs/qwen3_moe_style/config_qwen3_moe_style_100m_dp2_smoke.yaml`
- `configs/qwen3_moe_style/config_qwen3_moe_style_100m_dp2_resume.yaml`
- `configs/qwen3_moe_style/config_qwen3_moe_style_100m_tp2_smoke.yaml`
- `configs/qwen3_moe_style/config_qwen3_moe_style_100m_ep2_smoke.yaml`

The model keeps the same Qwen3-MoE-style settings as the 1-GPU validation:

- QK-Norm enabled.
- QKV bias disabled.
- BF16 + FlashAttention-2.
- 8 experts, top-2 routing.
- Shared expert disabled.
- Router auxiliary loss enabled.
- Global-batch router load-balancing path enabled.

## DP2 Result

Command:

```bash
CUDA_DEVICE_MAX_CONNECTIONS=1 PYTHONPATH=src /root/miniconda3/bin/torchrun --nproc_per_node=2 \
  run_train.py --config-file examples/smoke/config_qwen3_moe_style_100m_dp2_smoke.yaml
```

Result:

- Passed 5 training steps on 2 x RTX 3090.
- Model parameters: 108M per replica.
- `parallelism.dp=2` confirmed in Nanotron config log.
- Global batch increased from single-GPU 256 tokens/step to DP2 512 tokens/step.
- Checkpoint saved at `checkpoints/qwen3_moe_style_100m_dp2_smoke/5`.
- Warm-step throughput: around 6.37K to 6.58K tokens/s total, or about 3.2K tokens/s/GPU.
- Peak reserved memory during warm steps: about 2.5 GiB/GPU.

Resume command:

```bash
CUDA_DEVICE_MAX_CONNECTIONS=1 PYTHONPATH=src /root/miniconda3/bin/torchrun --nproc_per_node=2 \
  run_train.py --config-file examples/smoke/config_qwen3_moe_style_100m_dp2_resume.yaml
```

Resume result:

- Loaded checkpoint metadata with `tp=1, dp=2`.
- Continued from step 5 to step 7.
- Saved checkpoint at `checkpoints/qwen3_moe_style_100m_dp2_resume/7`.

Interpretation:

The Qwen3-MoE-style changes are compatible with Nanotron DP. The global-batch router load-balancing statistics now execute across a real data-parallel process group rather than only single rank.

DP2 does not improve throughput over the single-GPU smoke because the experiment is intentionally tiny: 108M parameters, sequence length 128, micro-batch 2. In this regime, gradient synchronization, router statistics all-reduce, process-group overhead, and short-kernel launch overhead dominate the extra compute.

## TP2 Result

Command:

```bash
CUDA_DEVICE_MAX_CONNECTIONS=1 PYTHONPATH=src /root/miniconda3/bin/torchrun --nproc_per_node=2 \
  run_train.py --config-file examples/smoke/config_qwen3_moe_style_100m_tp2_smoke.yaml
```

Result:

- Passed 5 training steps on 2 x RTX 3090.
- `parallelism.tp=2` confirmed in Nanotron config log.
- Local parameters: 96.4M per rank, logged total 193M under TP accounting.
- Checkpoint saved at `checkpoints/qwen3_moe_style_100m_tp2_smoke/5`.
- Warm-step throughput: around 5.02K to 6.45K tokens/s total, or about 2.5K to 3.2K tokens/s/GPU.
- Peak reserved memory during warm steps: about 2.0 GiB/GPU.

Interpretation:

The Qwen3-style attention changes are compatible with Nanotron tensor parallelism: packed QKV projection, QK-Norm, RoPE, FlashAttention, and sharded linear layers all execute successfully.

TP2 does not improve throughput in this small setting because tensor parallelism adds collectives around sharded linear layers, while the per-rank compute is too small to amortize communication.

## EP2 Result And Blocker

Update: this section records the first EP2 failure boundary from 2026-07-09. It has since been superseded by the All-to-All dispatcher implementation documented in:

- [`qwen3_moe_style_ep_alltoall_dispatch.md`](qwen3_moe_style_ep_alltoall_dispatch.md)
- [`qwen3_moe_style_ep_token_scaling.md`](qwen3_moe_style_ep_token_scaling.md)

The original blocker was useful because it identified the exact missing piece: local expert token-count alignment and contiguous expert buffers before GroupedGEMM.

Initial pure EP2 failed before model execution because Nanotron's `ParallelContext` had inconsistent expert-parallel world-size handling:

- The initialization assertion checked only `TP*CP*DP*PP == WORLD_SIZE`.
- The process-group reshape later used `(EP, PP, DP, CP, TP)` and therefore required EP to be included.

A minimal local patch changed the world-size assertion to include `expert_parallel_size`, allowing `dp=1, ep=2` to initialize and enter the MoE path.

After that fix, EP2 reached the expert computation path and failed in GroupedGEMM:

```text
RuntimeError: Expected batch_sizes.size(0) == num_experts to be true, but got false.
```

Stack location:

- `src/nanotron/nn/moe.py`, `Qwen2MoELayer._core_forward`
- `self.experts(dispatched_inputs, num_tokens_per_expert)`
- `grouped_gemm.ops.gmm(...)`

Interpretation:

This is the concrete EP blocker. After expert-parallel dispatch, `num_tokens_per_expert` is still not aligned with the local expert shard expected by GroupedGEMM. The likely fix is to split or remap the global expert token counts into local expert counts after all-to-all dispatch, then ensure dispatched token buffers are contiguous before calling GroupedGEMM.

This gives a clear next engineering direction:

- Fix EP process-group semantics in `ParallelContext`.
- Make router dispatch produce local-expert token counts per rank.
- Coalesce non-contiguous dispatched token buffers before GroupedGEMM.
- Then consider overlapping all-to-all dispatch with local expert compute.

## Patch

The combined Nanotron adaptation and distributed validation patch is stored at:

- `patches/qwen3_moe_style_dist_nanotron.patch`

## Resume Wording

Recommended factual wording:

> 基于 Nanotron 实现 Qwen3-MoE-style 训练适配，在 Qwen2-MoE 训练链路上补充 QK-Norm、无偏置 QKV、Top-2 路由辅助损失与全局批次负载均衡统计；构建 108M 参数验证模型，在 1/2 x RTX 3090 上完成 BF16/FlashAttention 训练、DP2 checkpoint/resume 与 TP2 张量并行验证，并定位 EP2 在 expert token dispatch 到 GroupedGEMM 本地专家计数对齐处的 blocker。
