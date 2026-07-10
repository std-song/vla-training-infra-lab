# Next Steps

The repository is now split into three self-contained projects:

1. `project1_qwen3_moe_pretrain`: Nanotron/Qwen3-MoE-style pretraining infra.
2. `project2_smolvla_training`: LeRobot/SmolVLA training infra.
3. `project3_vla_inferenceence`: VLM/VLA inference infra.

## Project 1

Completed:

- Qwen3-MoE-style 100M-scale single/DP2/TP2/PP2/EP2 smoke and profiling runs.
- PP resume and non-final pipeline-stage logging fixes.
- EP2 local expert dispatch validation.
- Outer pretraining pipeline: corpus manifest, packed shard generation, launch matrix, log parser, and SVG figures.

Remaining engineering direction:

- Implement cross-rank expert All-to-All dispatch.
- Merge non-contiguous token buffers before expert compute.
- Overlap EP communication with local expert computation.
- Validate auxiliary-loss aggregation and checkpoint layout under `expert_parallel_size > 1`.

## Project 2

Completed:

- LeRobot schema discovery and VLA batch collation.
- Nanotron-style DDP wrapper for multimodal batch validation.
- Official SmolVLA Accelerate DDP baseline.
- DataLoader worker, BF16, DDP tuning, checkpoint/resume, and memory profiling.

Remaining engineering direction:

- Deeper Nanotron Trainer integration for multimodal loss surfaces.
- Broader dataset coverage beyond `aloha_mobile_cabinet`.
- More realistic action metrics or simulator rollouts.

## Project 3

Completed:

- Qwen3-VL vLLM serving baseline.
- Qwen2.5-VL visual-token and prefill analysis.
- Paged-KV / continuous-batching / prefix-cache simulations.
- Pi0.5 action chunk profiling.
- VLASH-inspired async action queue simulator.

Remaining engineering direction:

- Replace simulator-only scheduler components with real serving integration.
- Add real rollout or robot-simulator validation for action queue quality.
- Explore quantization and CUDA Graph paths on supported hardware.
