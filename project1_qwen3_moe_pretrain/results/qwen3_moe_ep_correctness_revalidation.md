# Qwen3-MoE EP Correctness Revalidation

Date: 2026-07-19  
Hardware: 4 x RTX 3090 24GB, PCIe topology without NVLink  
Software: Python 3.10.8, PyTorch 2.1.2+cu118, Nanotron 0.4

## Why This Was Re-run

The first EP2+DP2 benchmark reported 48.51K tokens/s. Arithmetic and log
parsing were correct, but the speedup was unusually large for a 108M model on
PCIe-connected 3090s. A parameter audit found that replicated non-expert
parameters diverged after training.

After 20 steps, 16 of 31 replicated parameters differed across EP ranks and
the maximum absolute difference reached 0.00390625. The forward pass ran, but
ordinary `all_to_all_single` payload exchanges cut autograd, and shared
parameters had no EP replica-gradient synchronization.

## Fix

The corrected dispatcher adds:

1. Differentiable All-to-All whose backward performs the inverse exchange.
2. Token-owner sharding whose backward all-reduces the full hidden/router
   gradient onto every replicated EP rank.
3. Replicated output reduction with model-parallel identity backward.
4. Averaged EP gradient synchronization for non-expert replicated parameters.
5. No-op TP/EP tied-parameter groups are skipped when their size is one.

The expert payload is still coalesced into contiguous local-expert buffers
before GroupedGEMM. Integer routing metadata remains outside autograd.

## Correctness Tests

- Analytical 2-GPU tests passed for All-to-All backward, token-shard backward,
  and replicated-output backward.
- An EP2-vs-dense-reference test gathered all eight experts and compared the
  complete layer. Forward output, hidden gradient, and router gradient matched
  exactly; expert-weight gradient maximum relative error was 2.96e-4.
- Nanotron's per-step tied-gradient checks passed for 20 steps.
- All 31 replicated parameters remained bitwise aligned after 20 steps:
  `max_abs_diff=0`.
- The same 20-step test also passed with sanity checks disabled, ruling out a
  debug-synchronization dependency.

## Corrected 4-GPU Result

All strategies use 4096 logical tokens per step. Stable statistics use steps
50 through 100 inclusive.

| Strategy | Tokens/s | Avg step ms | Peak reserved MiB/GPU |
| --- | ---: | ---: | ---: |
| DP4 | 27.73K | 147.7 | 2642 |
| TP2+DP2 | 37.23K | 110.0 | 2572 |
| EP2+DP2 corrected | 36.07K | 113.69 | 2008 |

Corrected EP2+DP2 is 30.1% faster than DP4 and uses about 24% less peak
reserved memory. It is slightly slower than TP2+DP2 for this small model and
hardware topology. The old 48.51K value is withdrawn because it omitted real
backward and synchronization work.

## Checkpoint/Resume Finding

The corrected 100-step checkpoint exposed a separate scheduler bug. Building
LambdaLR after optimizer restoration captured the checkpoint's current LR as
the base LR, and `_initial_step()` advanced the schedule during load. The
construction order was fixed and the extra step removed. Resume from step 100
to 102 now keeps LR continuous: approximately 1.01e-5 at step 101 and 1.00e-5
at step 102.
