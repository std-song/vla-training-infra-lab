# Next Steps

The project now has single-GPU correctness/profiling, a 75.5M-parameter baseline, checkpoint resume, activation recomputation A/B, completed 2-GPU DP smoke with resume, and completed 2-GPU TP smoke with resume. The next work should validate PP=2 on the same 2-GPU machine before renting 8 GPUs.

## Step 1: Pipeline Parallel Smoke

Run one independent 2-GPU smoke case now that DP=2 and TP=2 are complete:

| Case | DP | TP | PP | EP | Purpose |
| --- | ---: | ---: | ---: | ---: | --- |
| pp_2 | 1 | 1 | 2 | 1 | pipeline schedule and stage checkpointing |

The case should first run 20-100 steps, then save and resume from checkpoint.

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

Implemented a Nanotron-based Qwen2-MoE training-infra baseline on RTX 3090, validating BF16 training, FlashAttention, GroupedGEMM MoE expert MLP, router top-k dispatch, checkpoint save/resume, and profiling of tokens/s, GPU memory, utilization, power, and checkpoint artifacts. Completed single-GPU smoke/resume, 75.5M-parameter 500-step baseline, activation recomputation A/B analysis, 2-GPU DP checkpoint/resume, and 2-GPU TP checkpoint/resume; next stage validates PP=2 before 8-GPU DP/TP/PP scaling.
