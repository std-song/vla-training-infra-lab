# Next Steps

The project has a valid single-GPU correctness baseline. The next work should make the result more robust before spending money on an 8-GPU rental.

## Step 1: Stronger Single-GPU Baseline

Goal: produce a more stable profile that uses meaningful RTX 3090 memory.

Suggested changes:

- increase `hidden_size` from 128 to 512 or 768
- increase `intermediate_size` proportionally
- increase MoE intermediate size to 1024 or 2048
- keep `num_hidden_layers` at 4 first, then try 8
- keep `sequence_length=512` if memory allows
- set `train_steps=500` or `1000`
- set `logging.iteration_step_info_interval=10`
- keep `dp=tp=pp=ep=1`

Success criteria:

- memory reaches several GiB but stays under 24 GiB
- loss remains finite
- checkpoint saves successfully
- resume from final checkpoint advances training
- average tokens/s is computed over a long enough steady window

## Step 2: Activation Recompute A/B

Run the stronger baseline twice:

| Case | recompute_layer | Expected result |
| --- | --- | --- |
| no_recompute | false | faster, higher activation memory |
| recompute | true | lower memory, slower step time |

Record memory reduction and throughput penalty. This is a strong training-infra talking point because it connects implementation detail to hardware limits.

## Step 3: 2-GPU Distributed Smoke

Before renting 8 GPUs, validate 2 GPUs.

Recommended order:

1. `dp=2, tp=1, pp=1, ep=1`
2. `dp=1, tp=2, pp=1, ep=1`
3. `dp=1, tp=1, pp=2, ep=1`

Each case should prove:

- all ranks launch
- loss is finite
- checkpoint is saved
- per-rank logs are interpretable
- tokens/s/GPU is reported

## Step 4: Inspect Expert Parallel Readiness

Do not claim EP until this is verified in code and by experiment.

Checklist:

- `expert_parallel_size > 1` process groups are constructed consistently
- global expert ids map to local expert ids correctly
- token dispatch performs real cross-rank all-to-all when experts are placed on different ranks
- returned tokens are restored to original order
- router auxiliary loss remains correct across ranks
- checkpoint naming and loading include expert-parallel rank correctly

If upstream Nanotron does not fully support this path for Qwen2-MoE, this becomes a valuable project contribution: implement and test EP rather than merely running a config.

## Step 5: 8-GPU Target Run

After 2-GPU DP/TP/PP are green, rent an 8x3090 instance and run:

1. `dp=8, tp=1, pp=1, ep=1`
2. `dp=4, tp=2, pp=1, ep=1`
3. `dp=4, tp=1, pp=2, ep=1`
4. only then attempt EP compositions

## Resume Bullet Draft After Current Milestone

Implemented a Nanotron-based Qwen2-MoE training-infra baseline on RTX 3090, validating BF16 training, FlashAttention, GroupedGEMM MoE expert MLP, router top-k dispatch, checkpoint save/resume, and profiling of tokens/s, GPU memory, utilization, and checkpoint artifacts. Completed single-GPU smoke, resume, 20-step, and 100-step baseline experiments; next stage expands to stable large-model profiling and DP/TP/PP multi-GPU validation.
