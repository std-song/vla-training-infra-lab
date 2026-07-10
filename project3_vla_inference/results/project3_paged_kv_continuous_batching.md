# Project 3 Stage 5: Paged KV and Continuous Batching Simulator

Date: 2026-07-08

This stage adds a lightweight vLLM-style simulator for VLA serving workloads. It does not implement CUDA PagedAttention kernels. Instead, it implements the scheduler and memory-management ideas that matter for interview discussion:

- paged KV block allocation;
- append/free lifecycle for autoregressive decode;
- continuous batching over a stream of VLA requests;
- static full-reservation vs paged append allocation;
- guarded paged admission to avoid decode-time KV starvation;
- KV budget sweep under visual-token-heavy workloads.

The simulator uses the measured Qwen2.5-VL profiles from `qwen25vl_visual_tokens_dynamic_pixels_sdpa_bf16.csv` as workload input. The workload mixes single-camera and three-camera requests, with visual-token lengths from 66 to 774 and decode lengths sampled from 16/32/64.

## Default Run

Configuration:

| Item | Value |
| --- | ---: |
| Requests | 128 |
| Mean arrival interval | 90 ms |
| Max active requests | 16 |
| KV budget | 512 MiB |
| Block size | 16 tokens |

| Scenario | Throughput | Mean latency | P95 latency | Peak KV | Speedup |
| --- | ---: | ---: | ---: | ---: | ---: |
| serial no batch | 1.40 req/s | 40,901.6 ms | 78,171.5 ms | 30.9 MiB | 1.00x |
| continuous static KV | 10.44 req/s | 1,904.7 ms | 3,220.3 ms | 341.4 MiB | 7.45x |
| continuous paged KV | 10.44 req/s | 1,904.7 ms | 3,220.3 ms | 330.8 MiB | 7.45x |
| continuous paged KV guarded | 10.44 req/s | 1,904.7 ms | 3,220.3 ms | 330.8 MiB | 7.45x |

![Paged KV throughput](../assets/figures/project3_paged_kv_throughput.svg)

![Paged KV P95 latency](../assets/figures/project3_paged_kv_p95_latency.svg)

Main observation: continuous batching is the dominant throughput win in this workload, improving throughput by 7.45x and cutting P95 latency from 78.2 s to 3.2 s. Paged KV slightly reduces peak KV memory at this budget.

## KV Budget Sweep

| Budget | Scenario | Throughput | Mean latency | P95 latency | Peak KV | Speedup |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 256 MiB | continuous static KV | 9.73 req/s | 2,335.7 ms | 3,834.1 ms | 253.1 MiB | 6.94x |
| 256 MiB | continuous paged KV | 7.42 req/s | 4,345.1 ms | 7,849.6 ms | 255.9 MiB | 5.29x |
| 256 MiB | continuous paged KV guarded | 9.76 req/s | 2,190.4 ms | 3,747.2 ms | 255.9 MiB | 6.96x |
| 320 MiB | continuous static KV | 10.29 req/s | 1,959.3 ms | 3,368.6 ms | 316.1 MiB | 7.34x |
| 320 MiB | continuous paged KV | 10.51 req/s | 1,906.4 ms | 3,221.5 ms | 319.5 MiB | 7.50x |
| 320 MiB | continuous paged KV guarded | 10.51 req/s | 1,906.4 ms | 3,221.5 ms | 319.5 MiB | 7.50x |
| 384 MiB | continuous static KV | 10.44 req/s | 1,904.7 ms | 3,220.3 ms | 341.4 MiB | 7.45x |
| 384 MiB | continuous paged KV guarded | 10.44 req/s | 1,904.7 ms | 3,220.3 ms | 330.8 MiB | 7.45x |

![Paged KV budget sweep throughput](../assets/figures/project3_paged_kv_budget_sweep_throughput.svg)

![Paged KV budget sweep memory](../assets/figures/project3_paged_kv_budget_sweep_memory.svg)

Main observation: naive paged allocation can over-admit prompt KV under a tight budget and stall later when decode needs more blocks. The guarded paged scheduler fixes this by keeping a decode-block watermark. This is the important serving-infra lesson: PagedAttention-style memory management must be paired with admission control and token-budget scheduling.

## What This Adds

Before this stage, Project 3 had real Qwen2.5-VL visual-token profiling and a measured microbatch serving prototype. This stage adds the missing scheduler/KV-memory layer:

- how visual tokens translate into KV cache pressure;
- how block size affects allocation and fragmentation;
- why continuous batching improves request throughput;
- why paged KV needs a decode-token budget guard;
- where a production vLLM engine would add CUDA PagedAttention kernels, block tables, prefix cache, and async scheduling.

## Boundary

This simulator is intentionally not a full vLLM implementation. It does not run a CUDA PagedAttention kernel and does not inject paged KV blocks into Qwen2.5-VL forward. It is a systems prototype for understanding and measuring scheduling and memory behavior under VLA-style visual-token workloads.
