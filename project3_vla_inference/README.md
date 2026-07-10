# Project 3: VLM/VLA Inference Infra

This project turns the original inference benchmark into a three-layer VLM/VLA inference-infra lab: VLM serving, real VLA action inference, and asynchronous action serving for robot control loops.

## What It Covers

- Qwen3-VL vLLM serving baseline with concurrency, latency, throughput, and memory measurements.
- Qwen2.5-VL visual-token profiling for single-camera and multi-camera inputs.
- KV-cache, paged-cache, prefix/cache reuse, and bucketed continuous-batching simulations.
- Pi0.5/LeRobot real VLA action inference with action chunk latency and `select_action` queue amortization.
- VLASH-inspired asynchronous action queue simulator for 30Hz control-loop reaction latency and state staleness.
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

Async control-loop simulation:

| Setting | Effect |
| --- | --- |
| Future-state refill at 30Hz | Reaction latency reduced from 266.7 ms to 166.7 ms |
| Action quantization ratio = 2 | Simulated control-side action overhead reduced by about 50% |

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
