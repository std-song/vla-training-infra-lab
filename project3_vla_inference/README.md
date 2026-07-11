# Project 3: VLM/VLA Inference Infra

This project turns the original inference benchmark into a three-layer VLM/VLA inference-infra lab: VLM serving, real VLA action inference, and asynchronous action serving for robot control loops.

## What It Covers

## VLASH Reproduction Addendum

The final VLA path reproduces upstream VLASH Pi0.5 LoRA training on all 85
ALOHA episodes for 1,000 steps with future-state delay offsets 0..8 and shared
observation encoding. Final replay traces use the step-1,000 checkpoint through
the upstream `VLASHAsyncManager`. The report separates verified queue behavior
from claims that require a physical robot I/O loop; see
[`results/vlash_final/final_vlash_report.md`](results/vlash_final/final_vlash_report.md).
For the exact GPU, software, data, timing scope, and how to interpret each
plot, see [`results/vlash_final/experiment_protocol.md`](results/vlash_final/experiment_protocol.md).

![VLASH Pi0.5 training](assets/figures/vlash_pi05_training.svg)

- Qwen3-VL vLLM serving baseline with concurrency, latency, throughput, and memory measurements.
- Qwen2.5-VL visual-token profiling for single-camera and multi-camera inputs.
- KV-cache, paged-cache, prefix/cache reuse, and bucketed continuous-batching simulations.
- Pi0.5/LeRobot real VLA action inference with action chunk latency and `select_action` queue amortization.
- Historical control-loop simulator used during early exploration. It is kept as a simulator rather than presented as the final VLASH result.
- Triton fused action post-processing benchmark.

## Key Results

Qwen3-VL-4B vLLM baseline on a 32GiB GPU:

| Image size | Concurrency | Throughput |
| --- | ---: | ---: |
| 224px | 8 | 10.08 req/s |
| 448px | 8 | 8.73 req/s |

Qwen2.5-VL visual-token cost:

| Input | Visual marker tokens | Prefill latency |
| --- | ---: | ---: |
| Single camera | 66 | 40.3 ms |
| Three cameras | 774 | 166.4 ms |

Pi0.5 action inference:

| Metric | Value |
| --- | ---: |
| Action chunk | `(1, 50, 7)` |
| Warm chunk latency | 87.7 ms |
| Queue pop latency | 3.47 ms |
| Peak memory | about 7.3 GiB |

The earlier 30 Hz control-loop numbers are simulation-only results. They are not
used as claims about the final Pi0.5/VLASH reproduction. For measured training,
offline action-queue traces, exact experimental conditions, and limits of the
offline setup, read the [final VLASH report](results/vlash_final/final_vlash_report.md).

![Qwen3-VL vLLM throughput](assets/figures/project3_qwen3vl_vllm_throughput.svg)

![VLASH-inspired async trace](assets/figures/project3_vlash_async_trace.svg)

## Reading Path

- Final report: [`results/project3_final_report.md`](results/project3_final_report.md)
- Upgrade plan: [`docs/project3_vlm_vla_upgrade_plan.md`](docs/project3_vlm_vla_upgrade_plan.md)
- Quickstart: [`docs/project3_quickstart.md`](docs/project3_quickstart.md)
- Qwen3-VL vLLM serving: [`results/project3_qwen3vl_vllm_serving.md`](results/project3_qwen3vl_vllm_serving.md)
- Pi0.5 action inference: [`results/project3_pi05_vla_action_inference.md`](results/project3_pi05_vla_action_inference.md)
- Async action queue: [`results/project3_vlash_async_control_loop.md`](results/project3_vlash_async_control_loop.md)
- Resume bullets: [`docs/project3_resume_bullets.md`](docs/project3_resume_bullets.md)

## Code Pointers

- Benchmarks: [`benchmarks`](benchmarks)
- Simulators: [`simulators`](simulators)
- Figure scripts: [`scripts`](scripts)
