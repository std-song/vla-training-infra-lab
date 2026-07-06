# Next Steps

The project now has single-GPU correctness/profiling, a 75.5M-parameter baseline, checkpoint resume, activation recomputation A/B, and a completed 2-GPU DP smoke with resume. The next work should validate TP=2 and PP=2 on the same 2-GPU machine before renting 8 GPUs.

## Step 1: Tensor and Pipeline Parallel Smoke

Run two independent 2-GPU smoke cases now that DP=2 is complete:

| Case | DP | TP | PP | EP | Purpose |
| --- | ---: | ---: | ---: | ---: | --- |
| tp_2 | 1 | 2 | 1 | 1 | tensor-parallel sharding |
| pp_2 | 1 | 1 | 2 | 1 | pipeline schedule and stage checkpointing |

Each case should first run 20-100 steps, then save and resume from checkpoint.

## Step 2: 8-GPU DP/TP/PP Scaling Prep

Before renting 8 GPUs, package the working 2-GPU configs and scripts. The first 8-GPU targets should still avoid EP:

1. `dp=8, tp=1, pp=1, ep=1`
2. `dp=4, tp=2, pp=1, ep=1`
3. `dp=4, tp=1, pp=2, ep=1`

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

## Step 4: 8-GPU Target Run

After 2-GPU DP/TP/PP are green, rent an 8x3090 instance and run:

1. `dp=8, tp=1, pp=1, ep=1`
2. `dp=4, tp=2, pp=1, ep=1`
3. `dp=4, tp=1, pp=2, ep=1`
4. only then attempt EP compositions

## Resume Bullet Draft After Current Milestone

Implemented a Nanotron-based Qwen2-MoE training-infra baseline on RTX 3090, validating BF16 training, FlashAttention, GroupedGEMM MoE expert MLP, router top-k dispatch, checkpoint save/resume, and profiling of tokens/s, GPU memory, utilization, power, and checkpoint artifacts. Completed single-GPU smoke/resume, 100-step tiny baseline, 75.5M-parameter 500-step baseline, step-500 to step-520 resume, and activation recomputation A/B analysis; next stage expands to 2-GPU DP/TP/PP validation before 8-GPU scaling.


