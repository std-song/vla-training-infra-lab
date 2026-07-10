# Roadmap

The repository is organized as three independent project folders. Each folder is designed to be readable and portable enough to become its own GitHub repository later.

## Project 1: Qwen3-MoE-style Pretraining Infra

Completed:

- Qwen3-MoE-style 100M-scale Nanotron configs and patches.
- Single/DP2/TP2/PP2/EP2 smoke and profiling.
- PP resume and pipeline-stage logging fixes.
- Outer pretraining pipeline: corpus manifest, tokenizer fallback, sequence packing, launch matrix, log parser, figures.
- Earlier 4-GPU Qwen2-MoE DP/TP/PP composition results retained as scaling evidence.

Next:

- Implement true cross-rank EP All-to-All dispatch.
- Add token buffer compaction and communication/compute overlap.
- Validate EP checkpoint layout and auxiliary-loss aggregation.

## Project 2: SmolVLA Training Infra

Completed:

- LeRobot dataset schema parsing and VLA batch collation.
- SmolVLA-compatible wrapper and Nanotron-style DDP wrapper.
- Official SmolVLA Accelerate DDP baseline.
- DataLoader worker, BF16, and DDP tuning reports.

Next:

- Extend from wrapper validation toward deeper Nanotron Trainer integration.
- Add more datasets and simulator/rollout-facing metrics.

## Project 3: VLM/VLA Inference Infra

Completed:

- Qwen3-VL vLLM serving baseline.
- Qwen2.5-VL visual-token and prefill analysis.
- Paged-KV, prefix-cache, and batching simulations.
- Pi0.5 action chunk profiling.
- VLASH-inspired asynchronous action queue simulator.
- Triton fused action post-processing benchmark.

Next:

- Connect scheduler simulations to an actual serving engine.
- Validate action queue behavior with dataset observations or simulator rollouts.
- Explore quantization/CUDA Graph/TensorRT paths on compatible hardware.
