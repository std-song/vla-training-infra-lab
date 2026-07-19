# Project 1 Completion Checklist

This document keeps the project claims honest. It separates finished work from useful follow-up work.

## Completed

- Built a Qwen3-MoE-style Nanotron training path on top of the existing Qwen2-MoE implementation.
- Added Qwen3-style attention changes for the compact validation model: QK-Norm-style normalization, no QKV bias, and router load-balancing hooks.
- Validated single-GPU BF16 + FlashAttention-2 + GroupedGEMM training and checkpoint save.
- Validated DP2, TP2, PP2, and EP2 on 2 x RTX 3090.
- Fixed PP logging/loss ownership issue so non-final pipeline stages do not access loss values they do not own.
- Implemented EP2 real token All-to-All dispatch:
  - route token copies by expert owner,
  - exchange hidden states and route metadata,
  - coalesce local expert buffers,
  - run GroupedGEMM on local experts,
  - return outputs to token-owner ranks,
  - scatter-add outputs by original token id.
- Added differentiable payload All-to-All, EP token-shard backward reduction, and replicated shared-gradient averaging.
- Passed analytical communication-gradient tests and a 20-step audit with all 31 replicated parameters bitwise aligned.
- Re-ran corrected EP2+DP2 for 100 steps: 36.07K tokens/s and 2,008 MiB peak reserved memory per GPU.
- Fixed LR scheduler resume ordering and validated continuous resume from step 100 to 102.
- Completed a 4-GPU mixed-parallel comparison for DP4, TP2+DP2, and EP2+DP2 at the same 4096 tokens/step.
- Built outer pretraining artifacts: mini corpus manifest, tokenizer/packing scripts, launch matrix, log parser, CSV summaries, and figures.

## Current Boundaries

- The model is a 108M validation model, not a useful pretrained foundation model.
- The EP path keeps a final all-reduce after MoE output so the surrounding Qwen stack can remain replicated. This is correct for integration, but not the final large-scale EP design.
- Current EP profiling is still a small-shape benchmark. It proves the systems path and fixed-cost behavior, not multi-node expert-parallel efficiency.
- The implementation has not yet added deep communication-compute overlap with separate CUDA streams. The current work should be described as token dispatch plus buffer coalescing, with overlap listed as next-stage optimization.

## Best Next Experiments

1. EP metadata packing
   - Pack token id, expert id, and route weight into fewer collectives.
   - Measure route/count and dispatch latency before and after.

2. Remove or move final replication all-reduce
   - Carry sharded activations through a longer boundary where possible.
   - Gather only where the non-EP layer contract requires it.

3. Larger-token EP benchmark
   - Increase sequence length or micro-batch size until all-to-all bandwidth becomes visible.
   - Compare mbs8 against at least one larger shape, subject to 24GB memory.

4. 4-GPU EP/DP composition
   - Re-run EP2+DP2 after the EP dispatcher fix.
   - Compare pure DP4 against EP2+DP2 for memory and throughput.

5. Optional CUDA-stream overlap prototype
   - Split token buffers into chunks.
   - Dispatch the next chunk asynchronously while GroupedGEMM computes the current chunk.
   - Report whether the added scheduling complexity helps at this small model size.

## Interview-Safe Boundary

Safe wording:

> I implemented and validated real expert-parallel token dispatch and local expert buffer coalescing, then profiled why EP throughput depends strongly on routed tokens per expert. I did not claim full Megatron-scale EP optimization; the next boundary is metadata packing, final all-reduce removal, and deeper communication-compute overlap.
