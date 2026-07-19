# Qwen3-MoE-style EP All-to-All Dispatch Validation

> This document preserves the first forward-path validation. A later audit
> found that payload collectives did not preserve autograd and replicated
> parameters diverged. The corrected implementation and results are documented
> in `qwen3_moe_ep_correctness_revalidation.md`.

Date: 2026-07-09
Hardware: 2 x RTX 3090 24GB on AutoDL
Software: Python 3.10.8, PyTorch 2.1.2+cu118, FlashAttention 2.5.8, Nanotron 0.4

## Goal

Move the Qwen3-MoE-style EP path from a correctness-first local-expert all-reduce implementation toward real token dispatch. The target is a minimal but trainable all-to-all expert-parallel path:

- Route tokens with top-k router.
- Assign each routed token copy to the rank owning its target expert.
- Exchange token hidden states, token ids, expert ids, and routing weights through EP-group all-to-all.
- Coalesce received token buffers by local expert id.
- Run GroupedGEMM on local expert shards.
- Send expert outputs back to token-owner ranks.
- Scatter-add weighted expert outputs by original token id.
- Replicate final token outputs for the existing non-EP layers.

## Implementation Notes

The implementation is stored in:

- `patches/qwen3_moe_style_ep_alltoall_dispatch.patch`

Main changes:

- `ParallelContext` world-size validation now includes `expert_parallel_size`.
- `Qwen2Model -> Qwen2DecoderLayer -> Qwen2MoELayer` now passes `ep_pg` into the MoE layer.
- `Qwen2MoELayer._core_forward_ep` implements token-owner to expert-owner dispatch.
- Dispatch metadata includes token id, local expert id, and routing weight.
- Received token buffers are sorted by local expert id before GroupedGEMM, giving contiguous expert-wise buffers.
- Empty local-expert cases keep a zero-valued dependency on expert weights, so Nanotron's gradient accumulator sees zero gradients instead of missing gradients.
- `run_train.py` explicitly destroys the distributed process group after training, avoiding a shutdown-time NCCL cleanup segfault observed after all-to-all collectives.

## Validation

Smoke command:

```bash
CUDA_DEVICE_MAX_CONNECTIONS=1 PYTHONPATH=src /root/miniconda3/bin/torchrun --nproc_per_node=2 \
  run_train.py --config-file examples/smoke/config_qwen3_moe_style_100m_ep2_smoke.yaml
```

Result:

- EP2 all-to-all dispatch completed 5 training steps.
- Checkpoint saved at `checkpoints/qwen3_moe_style_100m_ep2_smoke/5`.
- Warm-step throughput: about 6.1K to 7.2K tokens/s total, about 3.0K to 3.6K tokens/s/GPU.
- Peak reserved memory during warm steps: about 1.3 GiB/GPU.
- Exit code: 0 after explicit distributed cleanup.

Resume command:

```bash
CUDA_DEVICE_MAX_CONNECTIONS=1 PYTHONPATH=src /root/miniconda3/bin/torchrun --nproc_per_node=2 \
  run_train.py --config-file examples/smoke/config_qwen3_moe_style_100m_ep2_resume.yaml
```

Resume result:

- Loaded checkpoint from `checkpoints/qwen3_moe_style_100m_ep2_smoke/5`.
- Continued from step 5 to step 7.
- Checkpoint saved at `checkpoints/qwen3_moe_style_100m_ep2_resume/7`.
- Exit code: 0.

## 2026-07-10 Revalidation

The EP2 All-to-All dispatcher was revalidated on a freshly rented 2 x RTX 3090 AutoDL host.

Command shape:

```bash
NANO_QWEN_MOE_EP_PROFILE=1 NANO_QWEN_MOE_EP_PROFILE_CALLS=8 \
CUDA_DEVICE_MAX_CONNECTIONS=1 PYTHONPATH=src /root/miniconda3/bin/torchrun --nproc_per_node=2 \
  run_train.py --config-file examples/smoke/config_qwen3_moe_style_100m_ep2_validation_0710.yaml
```

Result:

- EP2 completed 5 training steps.
- Checkpoint saved at `checkpoints/qwen3_moe_style_100m_ep2_validation_0710/5`.
- Warm-step average throughput, excluding step 1: **6,162.5 tokens/s total**, about **3,081 tokens/s/GPU**.
- Warm-step throughput values: 5.72K, 6.69K, 5.51K, 6.73K tokens/s.
- Peak reserved memory: **1,302 MiB/GPU**.

Warm EP dispatcher profile, averaged over calls after the first cold call:

| Segment | Avg latency |
| --- | ---: |
| full EP dispatcher | 2.847 ms |
| route pack + count exchange | 0.667 ms |
| token hidden/state dispatch all-to-all | 0.186 ms |
| expert buffer coalesce | 0.225 ms |
| GroupedGEMM expert compute | 1.024 ms |
| return all-to-all | 0.226 ms |
| scatter-add restore | 0.064 ms |
| final replication all-reduce | 0.242 ms |

Interpretation:

- The correctness path now performs real cross-rank token movement rather than local-only expert execution.
- The dominant warm segment at this tiny shape is still expert compute plus routing/count overhead; the raw hidden-state all-to-all itself is small because each layer only moves hundreds of token copies.
- The final all-reduce is intentionally retained as a compatibility boundary so the surrounding non-EP Qwen stack can remain replicated. Removing that boundary is the next meaningful systems optimization.

## Performance Interpretation

This is still a small-shape validation, not a throughput-optimized EP benchmark. Sequence length is 128 and micro-batch size is 2, so communication latency, routing metadata exchange, token sorting, and launch overhead dominate. The point of this stage is correctness and system integration rather than speedup.

Compared with the previous correctness-first EP path, this version performs real token movement between token-owner and expert-owner ranks. It still keeps one replication boundary after MoE output so that the surrounding non-EP Qwen stack can remain unchanged. A more optimized version would remove or move this replication point by making subsequent layers sequence-sharded or by fusing the return dispatch with the next layer boundary.

## Remaining Optimization Directions

- Replace blocking all-to-all calls with async collectives.
- Overlap dispatch of the next token block with GroupedGEMM of the current block.
- Use CUDA streams to separate metadata movement, hidden-state movement, and local expert compute.
- Pack metadata more compactly and avoid multiple all-to-all calls for token id, expert id, and routing weight.
- Remove the final replication all-reduce by carrying sequence-sharded activations through later layers or gathering only at required boundaries.

## Resume Wording

Recommended factual wording:

> 基于 Nanotron 实现 Qwen3-MoE-style 训练适配，补充 QK-Norm、无偏置 QKV、Top-2 MoE 路由辅助损失与全局批次负载均衡统计；在 108M 参数模型上验证 BF16/FlashAttention、DP2 checkpoint/resume 与 TP2 张量并行，并实现 EP2 token all-to-all dispatch：按 expert owner 交换 token hidden/route metadata，按本地专家重排并合并连续 buffer 后调用 GroupedGEMM，完成 2 x RTX 3090 上 5-step smoke 与 checkpoint/resume。
