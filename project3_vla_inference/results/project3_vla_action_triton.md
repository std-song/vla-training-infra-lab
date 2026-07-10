# Project 3 Stage 4: VLA Action Head and Triton Fused Post-Processing

Date: 2026-07-08

This report moves Project 3 from language-model inference profiling into a VLA-specific action-output path. The benchmark attaches a simplified action head to a Qwen2-style hidden state and compares standard PyTorch action post-processing against a custom Triton fused kernel.

The action post-processing pattern is common in robot policies:

```text
action = pred * std + mean
action = clamp(action, low, high)
action = where(mask, action, previous_action)
```

In PyTorch, this becomes several elementwise ops and intermediate tensors. The Triton version fuses denormalization, clamp, and mask selection into one kernel.

## Setup

| Item | Value |
| --- | --- |
| GPU | NVIDIA GeForce RTX 4080 SUPER, 32 GiB |
| PyTorch | 2.8.0+cu128 |
| Triton | available through the PyTorch 2.8 environment |
| dtype | BF16 |
| hidden dim | 896, matching Qwen2-0.5B hidden size |
| action head | `Linear -> SiLU -> Linear` |
| post-process baseline | PyTorch elementwise ops |
| fused kernel | custom Triton JIT kernel |
| script | `project3_vla_inference/benchmarks/bench_vla_action_head_triton.py` |
| result CSV | `project3_vla_inference/results/vla_action_head_triton_bf16.csv` |

## Sweep

| Dimension | Values |
| --- | --- |
| batch size | 1, 4, 16, 64, 256 |
| action horizon | 10, 32, 64 |
| action dim | 14, 64 |
| repeats | 200 |
| warmup | 50 |

## Results

Overall average Triton speedup across the sweep is **1.44x**. The best observed speedup is **1.81x** at `batch=16`, `horizon=32`, `action_dim=14`.

| Batch | Horizon | Action dim | Elements | Action head | PyTorch post | Triton post | Speedup |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 10 | 14 | 140 | 0.030 ms | 30.37 us | 34.18 us | 0.89x |
| 4 | 10 | 14 | 560 | 0.036 ms | 30.88 us | 21.54 us | 1.43x |
| 16 | 32 | 14 | 7,168 | 0.039 ms | 37.38 us | 20.60 us | 1.81x |
| 64 | 32 | 14 | 28,672 | 0.040 ms | 36.92 us | 20.71 us | 1.78x |
| 256 | 64 | 64 | 1,048,576 | 0.068 ms | 30.66 us | 21.28 us | 1.44x |

![Project 3 Triton action post-processing](../assets/figures/project3_vla_action_triton.svg)

## Correctness

The max absolute error is around `7.81e-03` for most BF16 shapes, which is consistent with BF16 precision and acceptable for this post-processing benchmark. The smallest shape reports `1.95e-03`.

## Interpretation

The result is deliberately modest and practical. This post-processing op is tiny compared with Qwen2 decode, so the main benefit is not massive FLOP reduction. The benefit is reducing multiple elementwise launches and intermediate tensors into one fused operation.

The shape dependence is important:

1. **Tiny shapes can lose.** At `batch=1`, `horizon=10`, `action_dim=14`, Triton is slower (`0.89x`) because fixed launch and JIT/kernel overhead dominate only 140 elements.
2. **Normal VLA batches benefit.** For `batch=4`, `horizon=10`, `action_dim=14`, Triton is already `1.43x` faster.
3. **Mid-size shapes benefit most in this benchmark.** `batch=16/64`, `horizon=32`, `action_dim=14` reaches `1.78-1.81x`.
4. **The action head itself is very cheap at this scale.** The simplified MLP is around `0.03-0.07 ms`, so post-processing fusion matters mainly when running many environments, high-frequency control, or chaining multiple small action-space transforms.

For VLA serving, this is the right level of kernel work to start with: small, inspectable, easy to verify, and clearly tied to the action output path. It also demonstrates the engineering habit that matters in production: fused kernels should be benchmarked across shape regimes rather than assumed to help everywhere.

## Resume-Worthy Claim

Implemented a simplified VLA action-output benchmark and a custom Triton fused post-processing kernel for action denormalization, clamp, and mask selection. Compared PyTorch elementwise post-processing against the fused kernel across batch, horizon, and action-dim shapes on RTX 4080 SUPER, observing an average 1.44x speedup and up to 1.81x while documenting small-shape overhead and BF16 correctness.

## Next Step

Project 3 now has all core components: prefill/decode profiling, KV cache analysis, attention backend comparison, and a VLA-specific Triton kernel. The next step is a final integrated Project 3 report and resume-ready bullets.
