# Experiment Matrix

This matrix summarizes the completed and planned validation points across the three independent projects.

## Project 1: Qwen3-MoE-style Pretraining

| ID | GPUs | DP | TP | PP | EP | Goal | Status |
| --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| qwen3_single | 1 | 1 | 1 | 1 | 1 | single-GPU smoke/profile | done |
| qwen3_dp2 | 2 | 2 | 1 | 1 | 1 | data-parallel validation and resume | done |
| qwen3_tp2 | 2 | 1 | 2 | 1 | 1 | tensor-parallel validation | done |
| qwen3_pp2 | 2 | 1 | 1 | 2 | 1 | pipeline-parallel validation and resume | done |
| qwen3_ep2 | 2 | 1 | 1 | 1 | 2 | local expert dispatch validation | done |
| qwen2_dp4 | 4 | 4 | 1 | 1 | 1 | early 4-GPU DP scaling baseline | done |
| qwen2_tp2_dp2 | 4 | 2 | 2 | 1 | 1 | early 4-GPU TP+DP composition | done |
| qwen2_pp2_dp2 | 4 | 2 | 1 | 2 | 1 | early 4-GPU PP+DP composition | done |

Metrics:

- loss, gradient norm, tokens/s, tokens/s/GPU
- iteration time, peak allocated/reserved CUDA memory
- checkpoint save/resume correctness
- communication notes for DP all-reduce, TP collectives, PP bubbles, EP dispatch

Next EP work:

- cross-rank expert All-to-All dispatch
- global-to-local expert id mapping
- token buffer compaction before GroupedGEMM
- communication/compute overlap
- auxiliary-loss and checkpoint validation under `expert_parallel_size > 1`

## Project 2: SmolVLA Training

| ID | GPUs | Goal | Status |
| --- | ---: | --- | --- |
| lerobot_schema | 0-1 | validate video/parquet/task schema | done |
| smolvla_wrapper | 1 | validate compatible VLA batch and masked action loss | done |
| nanotron_style_dp | 2 | validate DDP wrapper, sampler, metrics, checkpoint | done |
| official_ddp | 2 | official SmolVLA Accelerate baseline | done |
| worker_tuning | 2 | DataLoader worker throughput comparison | done |
| bf16_tuning | 2 | mixed-precision throughput comparison | done |

Metrics:

- samples/s, update time, dataloader time
- GPU memory, rank metrics, checkpoint/resume correctness
- video decode and CPU/GPU overlap notes

## Project 3: VLM/VLA Inference

| ID | GPU | Goal | Status |
| --- | --- | --- | --- |
| qwen3vl_vllm | 32GiB | vLLM concurrency baseline | done |
| qwen25vl_visual | 32GiB | visual-token and prefill profiling | done |
| paged_kv_sim | CPU/GPU optional | paged KV / batching simulator | done |
| pi05_action | 32GiB | real VLA action chunk profiling | done |
| async_queue | CPU | VLASH-inspired control-loop simulator | done |
| triton_action | GPU | fused action post-processing benchmark | done |

Metrics:

- request latency, request throughput, output tokens/s
- visual marker tokens, prefill time, KV footprint
- action chunk latency, queue pop latency, state staleness
- simulated control-loop reaction latency and action overhead
