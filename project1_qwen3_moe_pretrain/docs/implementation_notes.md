# Implementation Notes

This project is built on top of Nanotron. The goal is not to rewrite a training framework, but to adapt and validate the pieces that matter for a MoE pretraining infra path.

## Qwen3-MoE-style Model Adaptation

The compact validation model keeps Nanotron's Qwen2-MoE training path and adds Qwen3-style behavior needed for the project:

- QK-Norm-style attention normalization.
- QKV projection without bias.
- Top-k MoE routing with router auxiliary loss.
- Global-batch router load-balancing statistics.
- BF16 + FlashAttention-2 + packed QKV path.

Main patch:

- [`../patches/qwen3_moe_style_nanotron.patch`](../patches/qwen3_moe_style_nanotron.patch)

## Pipeline Parallel Fix

The PP2 validation exposed a real ownership bug: only the last pipeline stage owns the language-model loss. Non-final stages must not construct logging fields from `loss_avg.item()`.

The fix keeps metric creation stage-aware so non-final stages can continue the pipeline without touching a loss tensor they do not own.

Main patch:

- [`../patches/nanotron_qwen2_moe_pp2_compat.patch`](../patches/nanotron_qwen2_moe_pp2_compat.patch)

## Expert Parallel Dispatch

The initial EP2 run reached a GroupedGEMM local expert-count mismatch. The implemented path makes EP execution explicit:

```text
router top-k
  -> assign routed token copies to expert-owner ranks
  -> all-to-all hidden states and route metadata
  -> sort received tokens by local expert id
  -> coalesce contiguous local expert buffers
  -> GroupedGEMM local expert compute
  -> all-to-all outputs back to token-owner ranks
  -> scatter-add by original token id
  -> replicated output boundary
```

The replicated output boundary is intentionally retained so the surrounding Qwen stack remains unchanged. It is correct for this integration, but it is also the next optimization target for a larger EP design.

Main patch:

- [`../patches/qwen3_moe_style_ep_alltoall_dispatch.patch`](../patches/qwen3_moe_style_ep_alltoall_dispatch.patch)

## Profiling Utilities

The project includes lightweight outer-engineering pieces so the experiments are reproducible:

- Mini corpus manifest and packing scripts.
- Nanotron launch scripts.
- Log parser for throughput, memory, checkpoint status, and final loss.
- Curated CSV summaries and SVG figures.

Representative result files:

- [`../results/qwen3_moe_style_ep_token_scaling.md`](../results/qwen3_moe_style_ep_token_scaling.md)
- [`../results/qwen3_moe_style_4gpu_mixed_parallel.md`](../results/qwen3_moe_style_4gpu_mixed_parallel.md)

## Current Boundary

The project demonstrates a compact, validated training-infra path. It does not claim production-scale multi-node optimization. The next meaningful improvements would be:

- Pack EP route metadata into fewer collectives.
- Remove or delay the final replication all-reduce.
- Prototype CUDA-stream overlap between token dispatch and local expert compute.
- Re-run EP2+DP2 on larger sequence lengths or multiple nodes when hardware is available.
