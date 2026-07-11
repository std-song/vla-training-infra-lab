# Project 3 Resume Bullets: Multimodal VLM/VLA Inference Acceleration

## Resume Title

**Multimodal VLM/VLA Inference Acceleration and Edge-Deployment Analysis**

## Current Completed Scope

- Built a multimodal serving baseline covering Qwen3-VL-4B through vLLM and Qwen2.5-VL-3B through Hugging Face. The benchmarks separate image preprocessing, multimodal prefill, estimated TTFT, decode TPOT, visual-token count, request throughput, GPU memory, and KV-cache footprint.
- Profiled single-camera and three-camera inputs on an RTX 4080 SUPER. From `1x224` to `3x448`, visual marker tokens increased from 66 to 774, multimodal prefill increased from 40.3 ms to 166.4 ms, and GPU memory increased from about 7.2 GiB to 7.6 GiB.
- Implemented visual input cache, same-shape microbatching, and KV footprint accounting. For 8 three-camera requests, `3x224, decode=32` improved from 1.58 req/s to 8.82 req/s, reaching 5.62x throughput; `3x448` improved from 1.29 req/s to 4.13 req/s, reaching 3.21x.
- Implemented a PagedAttention-style KV block manager and continuous-batching simulator. On a 128-request workload, throughput improved from 1.40 req/s to 10.44 req/s, reaching 7.45x. Guarded paged admission was used to analyze decode-block starvation under tight KV budgets.
- Implemented a shape-aware scheduler and prefix-cache simulator. Shape-aware batching reduced padding waste from 33.3% to 5.7% and improved throughput from 4.56 req/s to 6.01 req/s; adding modeled prefix cache reached 6.45 req/s.
- Implemented a Triton fused action post-processing kernel that combines action denormalization, clamp, and mask select into one kernel. The benchmark reached 1.43x median speedup and up to 14.24x on larger action tensor shapes.

## Upgraded Target Scope

- Added a Qwen3-VL-4B vLLM serving track on a 32 GiB GPU. The default vLLM path reached 10.08 req/s for 224px images and 8.73 req/s for 448px images at concurrency 8, with about 21.3 GiB peak memory. Compared with eager mode, default vLLM improved concurrent throughput by 18-41% across most tested shapes.
- Added a Pi0.5 / LeRobot real VLA action-inference track. On a 32 GiB vGPU, `predict_action_chunk` produced `(1, 50, 7)` actions with 87.7 ms warm latency and 7.3 GiB peak memory; `select_action` showed chunk-queue behavior with full model calls around 92.8 ms and queue pops around 3.47 ms.
- Reproduced upstream VLASH Pi0.5 LoRA fine-tuning with shared-observation delay offsets 0..8 on real ALOHA multi-camera data. Replayed the step-1,000 checkpoint through `VLASHAsyncManager` to separate costly action-chunk refills from sub-millisecond action-queue pops; clearly scoped the remaining hardware-I/O validation gap.
- Keep the existing Qwen2.5-VL serving prototype, scheduler simulator, KV-memory analysis, Pi0.5 action queue benchmark, and Triton action kernel as supporting systems experiments, while explicitly avoiding claims about robot task success or policy quality.

## Target Short Version

Built a multimodal VLM/VLA inference-acceleration lab covering three layers: Qwen3-VL/Qwen2.5-VL VLM serving, Pi0.5 real VLA action inference, and VLASH-inspired async control-loop serving. The project profiles visual tokens, multimodal prefill/decode, KV-cache memory, batching, concurrency, Pi0.5 action chunk latency, action queue amortization, state staleness, future-state queue refill, action quantization, and action post-processing. It also implements shape-aware batching, KV-memory accounting, continuous-batching simulation, and a Triton fused action post-processing kernel to compare VLM serving bottlenecks with VLA control-loop bottlenecks.

## Interview Talking Points

- Why VLM serving and VLA action inference have different bottlenecks.
- How visual-token count affects prefill, TTFT, KV footprint, memory, and batch capacity.
- Why Qwen3-VL is a better path for vLLM serving analysis, while Pi0.5 is a better path for real VLA policy inference.
- Why Pi0.5 action chunk inference changes the serving problem from per-token decode to queue refill, state staleness, and control-loop scheduling.
- Why visual input cache alone helps less than microbatching in the measured Qwen2.5-VL path.
- Why shape-aware batching can beat naive token-budget batching for multimodal VLA-like workloads.
- How VLASH-style async inference, future-state awareness, and action quantization fit into a VLA serving stack.
- Which parts are real model experiments, which parts are simulators, and which claims should not be overstated.
