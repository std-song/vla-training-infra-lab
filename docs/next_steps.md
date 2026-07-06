# Next Steps

The project now has single-GPU correctness/profiling, a 75.5M-parameter baseline, checkpoint resume, activation recomputation A/B, completed 2-GPU DP smoke with resume, completed 2-GPU TP smoke with resume, and completed 2-GPU PP smoke with resume.

## Step 1: Package 8-GPU DP/TP/PP Runs

The next experiments should compose the validated axes without enabling expert parallelism yet:

| Case | GPUs | DP | TP | PP | EP | Purpose |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| dp_8 | 8 | 8 | 1 | 1 | 1 | pure DP scaling baseline |
| tp2_dp4 | 8 | 4 | 2 | 1 | 1 | DP + TP composition |
| pp2_dp4 | 8 | 4 | 1 | 2 | 1 | DP + PP composition |
| tp2_pp2_dp2 | 8 | 2 | 2 | 2 | 1 | combined dense-model parallelism |

## Step 2: Clean Up PP Compatibility

Before presenting the PP work as upstream-quality, clean up or document these issues:

- Qwen2-MoE PP needs `TensorPointer` handling for `position_ids` and `cu_seqlens`.
- Trainer logging should skip `lm_loss` on non-loss pipeline stages.
- The repeated `Timer 'iteration_time' already running` warning should be investigated.
- The current 4-layer model has an imbalanced PP split; a deeper model would show cleaner pipeline behavior.

## Step 3: Inspect Expert Parallel Readiness

Do not claim EP until this is verified in code and by experiment.

Checklist:

- `expert_parallel_size > 1` process groups are constructed consistently
- global expert ids map to local expert ids correctly
- token dispatch performs real cross-rank all-to-all when experts are placed on different ranks
- returned tokens are restored to original order
- router auxiliary loss remains correct across ranks
- checkpoint naming and loading include expert-parallel rank correctly

If upstream Nanotron does not fully support this path for Qwen2-MoE, this becomes a valuable project contribution: implement and test EP rather than merely running a config.

## Resume Bullet Draft After Current Milestone

Implemented a Nanotron-based Qwen2-MoE training-infra baseline on RTX 3090, validating BF16 training, FlashAttention, GroupedGEMM MoE expert MLP, router top-k dispatch, checkpoint save/resume, and profiling of tokens/s, GPU memory, utilization, power, and checkpoint artifacts. Completed single-GPU smoke/resume, 75.5M-parameter 500-step baseline, activation recomputation A/B analysis, 2-GPU DP checkpoint/resume, 2-GPU TP checkpoint/resume, and 2-GPU PP checkpoint/resume; fixed Qwen2-MoE pipeline compatibility issues around `TensorPointer` propagation and loss-stage logging before moving to 8-GPU DP/TP/PP composition.
