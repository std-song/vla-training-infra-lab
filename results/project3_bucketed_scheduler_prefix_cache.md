# Project 3 Stage 6: Bucketed Scheduling and Prefix Cache Simulation

Date: 2026-07-08

This stage adds VLA-specific request scheduling on top of the previous serving prototype. The goal is to model the part of VLA serving that is easy to miss in language-only systems: requests can have very different visual-token lengths.

Measured Qwen2.5-VL profiles used by the simulator:

| Input | Input tokens | Visual marker tokens |
| --- | ---: | ---: |
| 1x224 | 105 | 66 |
| 3x224 | 237 | 198 |
| 1x448 | 297 | 258 |
| 3x448 | 813 | 774 |

The simulator compares three scheduling policies:

| Policy | Behavior |
| --- | --- |
| FCFS | batch requests in arrival order |
| shape bucket | prefer requests with the same image count / image size |
| token-budget bucket | prefer shorter visual prefixes under a max batch token budget |

It also adds a prefix-cache model for repeated task/image prefixes in robot control rollouts. A prefix hit reduces modeled prefill cost to 18% of cold prefill. This is intentionally a simulator, not a claim that Qwen2.5-VL KV prefix injection is implemented.

## Results

Configuration:

| Item | Value |
| --- | ---: |
| Requests | 256 |
| Mean arrival interval | 70 ms |
| Max batch requests | 16 |
| Max batch prompt tokens | 4096 |
| Prefix pool | 32 |

| Policy | Prefix cache | Throughput | P95 latency | Padding waste | Prefix hit rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| FCFS | no | 4.56 req/s | 35,373.7 ms | 33.3% | 0.00 |
| FCFS | yes | 4.93 req/s | 31,566.1 ms | 33.3% | 0.88 |
| shape bucket | no | 6.01 req/s | 23,569.1 ms | 5.7% | 0.00 |
| shape bucket | yes | 6.45 req/s | 20,690.8 ms | 3.9% | 0.88 |
| token-budget bucket | no | 5.51 req/s | 28,050.6 ms | 9.5% | 0.00 |
| token-budget bucket | yes | 5.86 req/s | 26,042.7 ms | 12.0% | 0.88 |

![Bucketed scheduler throughput](../assets/figures/project3_bucketed_scheduler_throughput.svg)

![Bucketed scheduler P95](../assets/figures/project3_bucketed_scheduler_p95.svg)

![Bucketed scheduler padding waste](../assets/figures/project3_bucketed_scheduler_padding_waste.svg)

## Interpretation

1. **Shape-aware batching is the most useful policy in this synthetic VLA workload.** It raises throughput from 4.56 to 6.01 req/s without prefix cache, and reduces padding waste from 33.3% to 5.7%.
2. **Prefix cache helps, but does not replace scheduling.** FCFS with prefix cache reaches 4.93 req/s, still below shape-aware batching without cache.
3. **Token-budget-only batching is not always best.** Sorting by short prefixes lowers padding waste relative to FCFS, but can hurt fairness and does not group same camera layouts as cleanly as shape buckets.
4. **The production implication is clear:** VLA serving should bucket by visual-token shape first, then apply prefix/cache and token-budget guards inside each bucket.

## Boundary

The prefix-cache part is a workload simulator. It models repeated task/image prefixes and reduced prefill cost, but does not inject cached Qwen2.5-VL `past_key_values` into the model. The real implementation would need model-specific multimodal RoPE/cache handling and careful cache invalidation when images change.
