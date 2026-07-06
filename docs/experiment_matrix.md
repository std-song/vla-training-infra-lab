# Experiment Matrix

This matrix defines the staged validation path from a single RTX 3090 smoke test to the intended 8x3090 distributed Qwen2-MoE project.

## Principle

Do not jump directly to `DP/TP/PP/EP=8-way complexity`. Each axis should be validated independently first, then composed.

## Completed Experiments

| ID | GPUs | DP | TP | PP | EP | Goal | Status |
| --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| moe_unit | 1 | 1 | 1 | 1 | 1 | Nanotron MoE unit test | done |
| smoke_5 | 1 | 1 | 1 | 1 | 1 | end-to-end train/checkpoint smoke | done |
| resume_7 | 1 | 1 | 1 | 1 | 1 | checkpoint resume from step 5 | done |
| baseline_20 | 1 | 1 | 1 | 1 | 1 | first profiling run | done |
| baseline_100 | 1 | 1 | 1 | 1 | 1 | longer tiny baseline profile | done |
| baseline_v2_500 | 1 | 1 | 1 | 1 | 1 | 75.5M-parameter 500-step profile | done |
| baseline_v2_resume_520 | 1 | 1 | 1 | 1 | 1 | resume baseline v2 from step 500 to 520 | done |
| recompute_ab | 1 | 1 | 1 | 1 | 1 | activation recomputation tradeoff | done |

## Near-Term Experiments

| ID | GPUs | DP | TP | PP | EP | Goal | Success criteria |
| --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| dp_2 | 2 | 2 | 1 | 1 | 1 | distributed launcher and DDP sync | both ranks train, checkpoint saves, tokens/s/GPU reported |
| tp_2 | 2 | 1 | 2 | 1 | 1 | tensor-parallel path | model shards correctly, loss finite, checkpoint saves |
| pp_2 | 2 | 1 | 1 | 2 | 1 | pipeline-parallel path | both stages active, 1F1B schedule works |

## 8-GPU Target Experiments

| ID | GPUs | DP | TP | PP | EP | Goal |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| dp_8 | 8 | 8 | 1 | 1 | 1 | pure data-parallel scaling baseline |
| tp2_dp4 | 8 | 4 | 2 | 1 | 1 | tensor parallel plus data parallel |
| pp2_dp4 | 8 | 4 | 1 | 2 | 1 | pipeline parallel plus data parallel |
| ep2_dp4 | 8 | 4 | 1 | 1 | 2 | first expert-parallel composition |
| tp2_pp2_dp2 | 8 | 2 | 2 | 2 | 1 | combined dense-model parallelism |
| tp2_pp2_ep2 | 8 | 1 | 2 | 2 | 2 | final MoE distributed stress case, if EP is implemented correctly |

## Metrics To Record

- final loss and finite-gradient check
- tokens/s and tokens/s/GPU
- time per iteration: average, p50, p95 if available
- peak allocated and reserved CUDA memory
- `nvidia-smi` sampled memory, utilization, power, and temperature
- checkpoint save and resume time
- model parameter count and per-rank parameter count
- communication notes: all-reduce for DP/TP, pipeline bubbles for PP, all-to-all for EP

## Known EP Caveat

The current Nanotron Qwen2-MoE path validates expert routing and grouped expert compute within one rank. Before claiming true expert parallelism, the project must verify cross-rank expert token dispatch, process-group semantics, local/global expert id mapping, all-to-all correctness, and checkpoint layout under `expert_parallel_size > 1`.
