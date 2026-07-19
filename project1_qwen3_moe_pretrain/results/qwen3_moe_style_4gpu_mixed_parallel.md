# Qwen3-MoE-style 4-GPU Mixed Parallel Profiling

Date: 2026-07-10; EP2+DP2 corrected and re-run on 2026-07-19
Hardware: 4 x RTX 3090 24GB on AutoDL
Software: Python 3.10.8, PyTorch 2.1.2+cu118, Nanotron 0.4

## Goal

The earlier 2-GPU runs validated each parallel strategy independently. This run checks a more useful training-infra question:

> Under the same global token budget per step, how do DP4, TP2+DP2, and EP2+DP2 trade throughput, memory, and communication overhead?

All three runs use the same Qwen3-MoE-style 108M validation model and about 4096 tokens per training step.

## Result

The table reports 100-step runs. The stable average uses steps >= 50.

| Strategy | GPUs | Tokens/step | Stable tokens/s | Tokens/s/GPU | Avg step ms | Peak reserved MiB/GPU | Final loss | Checkpoint |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| DP4 | 4 | 4096 | 27.73K | 6.93K | 147.7 | 2642 | 9.79 | yes |
| TP2+DP2 | 4 | 4096 | 37.23K | 9.31K | 110.0 | 2572 | 9.81 | yes |
| EP2+DP2 | 4 | 4096 | 36.07K | 9.02K | 113.7 | 2008 | 9.80 | yes |

![4-GPU mixed parallel](../assets/figures/qwen3_moe_4gpu_mixed_parallel.svg)

## Interpretation

DP4 is the simplest baseline, but it replicates all experts on every GPU. At this small model size, it pays gradient synchronization and router-statistics synchronization cost while doing less useful per-rank expert work than the larger-token EP2+DP2 setup.

TP2+DP2 improves throughput over DP4 because it increases the per-DP-replica micro-batch while sharding dense tensor-parallel layers. It still pays tensor-parallel collectives through the model, so it is not the best choice for this MoE-heavy small validation setting.

EP2+DP2 shards experts across two ranks inside each data-parallel replica. After
the dispatcher backward path and replicated-gradient synchronization were fixed,
it reaches 36.07K tokens/s and 1.96 GiB/GPU. This is 30.1% faster than DP4 and
uses about 24% less peak reserved memory, while remaining slightly slower than
TP2+DP2 for this small model and PCIe topology.

The earlier 48.51K result was a real wall-clock measurement but is withdrawn as
a training result: payload All-to-All did not preserve autograd and replicated
EP parameters were not synchronized. The missing backward and synchronization
work made that path artificially fast.

## What This Proves

This is not a claim that EP2+DP2 is universally faster than DP4 or TP2+DP2. It proves a narrower and more useful systems point:

- The Nanotron Qwen3-MoE-style path supports real mixed parallel execution on 4 GPUs.
- EP all-to-all dispatch remains trainable when combined with DP.
- Expert sharding changes the memory/throughput tradeoff in a measurable way.
- The best strategy depends on token granularity, model shape, and communication overhead, not just GPU count.

## Interview-Safe Summary

> On 4 x RTX 3090, I compared DP4, TP2+DP2, and corrected EP2+DP2 under the same 4096-token step size. EP2+DP2 reached 36.1K tokens/s with 1.96 GiB/GPU peak memory, versus DP4 at 27.7K and 2.64 GiB/GPU. I also built parameter-consistency and autograd tests that caught and removed an earlier artificially fast EP result.
