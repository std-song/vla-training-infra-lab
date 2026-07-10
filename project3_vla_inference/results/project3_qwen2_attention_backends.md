# Project 3 Stage 3: Qwen2 Attention Backend Comparison

Date: 2026-07-07

This report compares three Hugging Face / PyTorch attention backends for the Qwen2-0.5B VLA-style inference benchmark:

- `sdpa`: PyTorch scaled-dot-product attention;
- `eager`: unfused eager attention path;
- `flash_attention_2`: FlashAttention 2 through Transformers.

The goal is to avoid a common oversimplification: installing FlashAttention does not automatically make every inference path faster. Backend choice depends on model size, sequence shape, prefill vs decode phase, framework integration overhead, and whether the benchmark is dominated by attention compute or per-step serving overhead.

## Setup

| Item | Value |
| --- | --- |
| GPU | NVIDIA GeForce RTX 4080 SUPER, 32 GiB |
| PyTorch | 2.8.0+cu128 |
| flash-attn | 2.8.3 |
| Model | `Qwen/Qwen2-0.5B-Instruct` |
| dtype | BF16 |
| Cache | Hugging Face `past_key_values` |
| Script | `project3_vla_inference/benchmarks/bench_qwen2_prefill_decode.py` |

## Sweep

The comparison uses a selected subset of the Stage 1 sweep:

| Dimension | Values |
| --- | --- |
| batch size | 1, 4 |
| prompt length | 128, 1024 tokens |
| decode length | 32, 128 tokens |
| repeats | 3 |

The SDPA rows come from the Stage 1 baseline. Eager and FlashAttention 2 were run with the same script and selected shapes.

## Key Results

### Long-Prompt Prefill

For `batch=4`, `prompt_len=1024`, `decode_len=128`, prefill latency is:

| Backend | Prefill | Relative to SDPA |
| --- | ---: | ---: |
| SDPA | 58.4 ms | 1.00x |
| eager | 128.7 ms | 2.20x slower |
| FlashAttention 2 | 60.2 ms | 1.03x slower |

This is the expected direction: fused attention paths matter more during long-context prefill. Eager attention is clearly worse for this shape, while SDPA and FlashAttention 2 are close.

### Cached Decode

For the same `batch=4`, `prompt_len=1024`, `decode_len=128` shape, cached decode is:

| Backend | TPOT | Decode tokens/s |
| --- | ---: | ---: |
| SDPA | 12.28 ms | 325.6 |
| eager | 12.76 ms | 313.5 |
| FlashAttention 2 | 22.90 ms | 174.7 |

![Project 3 attention backend comparison](../assets/figures/project3_qwen2_attention_backends.svg)

## Interpretation

The measured behavior is nuanced:

1. **SDPA is the best default for this Qwen2-0.5B inference path.** It is fastest or near-fastest across the selected shapes and avoids extra FlashAttention integration overhead.
2. **Eager attention is a useful negative baseline.** It is especially weak for long-prompt prefill: 128.7 ms vs 58.4 ms for SDPA at `batch=4`, `prompt_len=1024`.
3. **FlashAttention 2 helps the long-prefill path relative to eager, but not cached decode here.** In this Transformers Qwen2 path, FlashAttention 2 decode TPOT is around 22 ms/token, much slower than SDPA's 12 ms/token. This likely reflects small-model, one-token decode, framework dispatch, cache layout, and backend integration overhead rather than FlashAttention being universally slow.

For VLA inference, this distinction matters. The visual/task context prefill and the action-token decode loop have different bottlenecks. A backend that is strong for long prefill may not be ideal for per-token decode in a small model. Production systems such as vLLM solve this with specialized decode kernels, KV-cache layout control, and batching schedulers rather than relying only on a generic model-level attention flag.

## Resume-Worthy Claim

Benchmarked Qwen2-0.5B VLA-style inference under PyTorch SDPA, eager attention, and FlashAttention 2 on RTX 4080 SUPER. Found SDPA to be the best default for this Hugging Face cached-decode path; FlashAttention 2 matched SDPA on long-prompt prefill but underperformed on one-token cached decode, motivating phase-specific backend and kernel selection rather than one-size-fits-all attention acceleration.

## Next Experiments

The next step is to attach a simplified VLA action head and benchmark action post-processing. A small Triton fused kernel can combine:

```text
action = pred * std + mean
action = clamp(action, low, high)
action = where(mask, action, previous_action)
```

This will create a compact, VLA-specific kernel benchmark instead of staying only at language-model attention.
