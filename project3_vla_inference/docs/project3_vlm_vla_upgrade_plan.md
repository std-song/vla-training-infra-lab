# Project 3 Upgrade Plan: Multimodal VLM/VLA Inference Acceleration

Date: 2026-07-08

Project 3 is being repositioned as a realistic multimodal inference-infrastructure project instead of a self-made VLA-style demo. The core idea is to use two real model families:

- **VLM serving track:** Qwen3-VL / Qwen2.5-VL through vLLM or Hugging Face, measuring visual-token cost, prefill/decode, batching, quantization, KV memory, latency, throughput, and concurrency.
- **VLA action-inference track:** Pi0.5 through LeRobot / LIBERO, measuring real robot-policy action latency, denoising-loop behavior, prefix KV cache, control Hz, memory, and action-output consistency.

The existing Qwen2.5-VL profiling, serving prototype, paged-KV simulator, shape-aware scheduler, and Triton action post-processing kernel remain useful, but they become supporting systems experiments rather than the only Project 3 story.

## Why This Upgrade

The previous Qwen2.5-VL path has real image inputs and visual tokens, but it is not a complete robot policy. It is useful for multimodal serving analysis, but weaker as a VLA claim. Pi0.5 adds a real VLA policy and a real action-generation loop, while Qwen3-VL adds a cleaner vLLM serving path for general VLM inference.

This gives the project a stronger interview narrative:

1. VLM serving bottlenecks: visual tokens, multimodal prefill, KV memory, batching, quantization, and concurrency.
2. VLA policy bottlenecks: action latency, denoising steps, prefix reuse, control-loop Hz, and environment-step overhead.
3. Systems bridge: shape-aware scheduling, continuous batching, action post-processing kernels, and edge-deployment constraints.

## Target Scope

| Track | Model / stack | Main measurements | Status |
| --- | --- | --- | --- |
| VLM serving | Qwen3-VL-4B + vLLM | latency, output tokens/s, req/s, memory, concurrency curve, eager/default comparison | Done |
| VLM profiling | Qwen2.5-VL-3B + HF | visual tokens, multimodal prefill, TTFT estimate, microbatching, KV footprint | Done |
| VLA action inference | Pi0.5 + LeRobot/LIBERO | action latency, control Hz, prefix KV cache on/off, action similarity, memory | Next |
| Scheduler analysis | measured VLM/VLA profiles | continuous batching, paged-KV-style block accounting, shape-aware batching, prefix-cache simulation | Done |
| Kernel optimization | action post-processing | Triton fused denorm + clamp + mask select | Done |
| Edge deployment side track | TensorRT / TensorRT-LLM | visual projector export or small decoder side test | Optional |

## What We Should Not Overclaim

- Do not claim a full vLLM engine unless we actually implement async scheduling, PagedAttention kernels, block tables, and model integration.
- Do not claim Qwen2.5-VL is a real VLA policy; it is a VLM serving benchmark with VLA-style prompts and action post-processing.
- Do not claim Pi0.5 prefix KV cache is our original idea if it is already present in the current LeRobot implementation. We can benchmark, analyze, and extend the profiling story.
- Do not claim TensorRT deployment unless we run a real engine and report model support limits.

## Next Experiments

### 1. Qwen3-VL vLLM Baseline

Status: done for BF16 eager/default serving on a 32 GiB GPU. Quantization ablations remain optional.

Goal: establish a general VLM serving baseline with real vLLM metrics.

Minimum output:

- BF16 single-request latency.
- concurrency curve, e.g. concurrency 1/2/4/8.
- TTFT, TPOT, req/s, tokens/s, GPU memory.
- notes on visual-token length and image resolution.

Optional output:

- AWQ/GPTQ/SmoothQuant comparison if model export and vLLM compatibility are stable.
- OCRBench subset or a small local image-question set.

### 2. Pi0.5 Action-Inference Baseline

Goal: add a true VLA policy path.

Minimum output:

- LIBERO or synthetic action inference smoke.
- action latency and control Hz.
- prefix KV cache on/off comparison if supported by the installed LeRobot version.
- action output difference metrics, such as MAE/cosine similarity.
- GPU memory and environment-step overhead when available.

### 3. Integrated Report Refresh

Goal: rewrite Project 3 final report around the two-track story.

Report sections:

1. Qwen3-VL / Qwen2.5-VL VLM serving results.
2. Pi0.5 VLA action-inference results.
3. Scheduler and KV-memory analysis.
4. Triton action post-processing.
5. VLM vs VLA bottleneck comparison.
6. Resume bullets and interview talking points.

## Resume Positioning

Recommended project title:

**Multimodal VLM/VLA Inference Acceleration and Edge-Deployment Analysis**

Recommended one-paragraph summary:

Built a multimodal inference-optimization lab for robotics workloads, covering Qwen3-VL/Qwen2.5-VL VLM serving and Pi0.5 real VLA action inference. The project profiles visual tokens, prefill/decode, KV-cache memory, batching, quantization, action latency, prefix KV cache, control Hz, and action-output consistency. It also implements shape-aware batching, KV-memory accounting, and a Triton action post-processing kernel to compare VLM serving bottlenecks with VLA control-loop bottlenecks.
