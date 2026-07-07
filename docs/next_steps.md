# Next Steps

The first Qwen2-MoE training-infrastructure project is complete as a 4xRTX 3090 portfolio artifact.

Completed scope:

- single-GPU baseline profiling and checkpoint/resume
- activation recomputation A/B
- 2-GPU DP, TP, and PP validation with checkpoint/resume
- 4-GPU DP4, TP2+DP2, and PP2+DP2 composition runs
- Qwen2-MoE pipeline compatibility patches
- EP2+DP2 readiness attempt with the next blocker localized
- GitHub reports, configs, patches, and SVG figures

## Remaining Engineering Work

True expert parallelism is the main unfinished engineering item. The next implementation plan is:

1. Make `ParallelContext` consistently account for `expert_parallel_size` in world-size validation and rank reshaping.
2. Define global-to-local expert id mapping for Qwen2-MoE.
3. Dispatch routed tokens across `ep_pg` with all-to-all or a verified all-gather path.
4. Convert global `num_tokens_per_expert` into local expert counts before GroupedGEMM.
5. Restore token order after expert computation.
6. Validate router auxiliary loss and checkpoint naming/loading under `expert_parallel_size > 1`.

## Resume Bullet Draft

Built a Nanotron-based Qwen2-MoE distributed training lab on RTX 3090 GPUs, validating BF16 training, FlashAttention, GroupedGEMM MoE expert MLP, router top-k dispatch, checkpoint save/resume, activation recomputation analysis, and throughput/memory profiling. Completed single-GPU, 2-GPU DP/TP/PP, and 4-GPU DP4 / TP2+DP2 / PP2+DP2 runs; debugged Qwen2-MoE pipeline `TensorPointer` handling and loss-stage logging; analyzed EP2+DP2 readiness and localized the next blocker to local expert token accounting before GroupedGEMM.
