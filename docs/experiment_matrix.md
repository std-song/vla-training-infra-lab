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
| dp2_200 | 2 | 2 | 1 | 1 | 1 | distributed DP training and checkpoint | done |
| dp2_resume_220 | 2 | 2 | 1 | 1 | 1 | resume DP checkpoint from step 200 to 220 | done |
| tp2_100 | 2 | 1 | 2 | 1 | 1 | tensor-parallel training and checkpoint | done |
| tp2_resume_120 | 2 | 1 | 2 | 1 | 1 | resume TP checkpoint from step 100 to 120 | done |
| pp2_100 | 2 | 1 | 1 | 2 | 1 | pipeline-parallel training and checkpoint | done |
| pp2_resume_120 | 2 | 1 | 1 | 2 | 1 | resume PP checkpoint from step 100 to 120 | done |
| dp4_4gpu | 4 | 4 | 1 | 1 | 1 | 4-GPU DP composition | done |
| tp2_dp2_4gpu | 4 | 2 | 2 | 1 | 1 | 4-GPU DP+TP composition | done |
| pp2_dp2_4gpu | 4 | 2 | 1 | 2 | 1 | 4-GPU DP+PP composition | done |
| ep2_dp2_4gpu | 4 | 2 | 1 | 1 | 2 | EP readiness attempt | blocked at local expert accounting |

## Near-Term Experiments

The 2-GPU DP, TP, and PP axes plus 4-GPU DP/TP/PP compositions are complete. The near-term work is true EP implementation, not more config-only scaling.

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
