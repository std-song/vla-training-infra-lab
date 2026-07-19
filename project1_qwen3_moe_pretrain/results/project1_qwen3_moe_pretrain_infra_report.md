# Project 1: Qwen3-MoE-style Pretraining Infra on Nanotron

## Goal

Build a small but complete pretraining infrastructure lab around Nanotron and a Qwen3-MoE-style model. The project separates two layers:

- Training-core layer: Nanotron model/distributed runtime with DP, TP, PP, EP, FlashAttention-2, GroupedGEMM expert MLP, router top-k, global-batch load balancing, activation recomputation, checkpoint/resume.
- Outer-engineering layer: corpus manifest, tokenizer/packing, fixed-length shard generation, profiling matrix scripts, log parser, CSV summary, and figures.

The outer-engineering layer is intentionally modeled after public Qwen3 pretraining repos that organize data preparation, packed datasets, distributed launch scripts, plots, and evaluation utilities as first-class project artifacts.

## Local Mini Data Pipeline

The mini pipeline is designed to be runnable without downloading large corpora:

```bash
python scripts/project1_prepare_mini_corpus.py \
  --config configs/data/mini_qwen3_pretrain_mix.yaml \
  --out-dir data/project1/processed

python scripts/project1_make_packed_dataset.py \
  --input data/project1/processed/mini_mixed_corpus.jsonl \
  --manifest data/project1/processed/manifest.json \
  --out-dir data/project1/packed/mini_s128 \
  --seq-len 128 \
  --shard-tokens 4096
```

The tokenizer path defaults to `Qwen/Qwen3-0.6B` and uses `local_files_only=True`. If the tokenizer is not present, the script falls back to a deterministic byte tokenizer so CI and offline smoke tests still validate the pipeline.

## Nanotron Profiling Matrix

The distributed matrix is run on AutoDL from the Nanotron repository root:

```bash
bash scripts/run_project1_nanotron_matrix.sh
```

The log parser converts raw Nanotron logs into a compact CSV:

```bash
python scripts/project1_parse_nanotron_logs.py \
  --log-dir profiling_logs \
  --out results/project1_qwen3_moe_style_clean_summary.csv \
  single=qwen3_moe_style_100m_single20_clean.log \
  dp2=qwen3_moe_style_100m_dp2_20_clean.log \
  tp2=qwen3_moe_style_100m_tp2_20_clean.log \
  pp2=qwen3_moe_style_100m_pp2_20_clean.log \
  ep2=qwen3_moe_style_100m_ep2_20_clean.log
```

## Clean 20-step Result

Environment: AutoDL, 2 x RTX 3090 for distributed runs, Python 3.10, PyTorch 2.1.2 + CUDA 11.8.

Warm average excludes the first four initialization-heavy steps.

| Strategy | Warm avg tokens/s | Tokens/s/GPU | Avg step ms | Peak reserved MiB | Checkpoint |
|---|---:|---:|---:|---:|---|
| single | 4455.6 | 4455.6 | 57.7 | 2242 | yes |
| dp2 | 6948.8 | 3473.8 | 73.9 | 2448 | yes |
| tp2 | 4201.3 | 2100.6 | 61.3 | 2016 | yes |
| pp2 | 4561.3 | 2280.0 | 56.7 | 2040 | yes |
| ep2 | 3248.8 | 1624.4 | 79.8 | 1302 | yes |

## Engineering Notes

- DP2 gives the highest total throughput but only reaches about 1.56x single-GPU throughput because this small model is dominated by launch, optimizer, logging, and synchronization overheads.
- TP2 reduces memory relative to single-GPU training but introduces tensor collective overhead at every transformer block.
- PP2 required a real runtime fix: only the last pipeline stage owns the actual loss, so non-final stages must not construct `lm_loss = loss_avg.item()` during logging. After this fix, PP2 passed 20-step training, checkpoint save, and a separate resume run.
- EP2 has the lowest peak reserved memory because experts are sharded. The current path now performs real token All-to-All dispatch: token-owner ranks send routed token copies to expert-owner ranks, received tokens are coalesced by local expert id before GroupedGEMM, and results are returned to token owners before a compatibility all-reduce.
- Single-rank torchrun needed a shutdown fix: skip unnecessary distributed barrier when world size is 1 to avoid post-checkpoint teardown SIGSEGV on the AutoDL image.

## EP2 All-to-All Revalidation

On 2026-07-10, EP2 was revalidated on a freshly rented 2 x RTX 3090 host with `NANO_QWEN_MOE_EP_PROFILE=1`.

| Metric | Value |
| --- | ---: |
| train steps | 5 |
| warm avg throughput | 6,162.5 tokens/s |
| warm avg throughput/GPU | 3,081 tokens/s/GPU |
| peak reserved memory | 1,302 MiB/GPU |
| warm EP dispatcher avg | 2.847 ms |
| route pack + count exchange | 0.667 ms |
| dispatch all-to-all | 0.186 ms |
| expert buffer coalesce | 0.225 ms |
| GroupedGEMM expert compute | 1.024 ms |
| return all-to-all | 0.226 ms |
| final replication all-reduce | 0.242 ms |

This run validated the first forward dispatch path only. The later gradient and
parameter audit found that its payload collectives cut autograd, so the
following timing breakdown is retained as debugging history rather than a
correct end-to-end training result.

## EP Token Scaling

The following sweep predates the differentiable dispatcher fix. It is useful
for understanding forward dispatch shape sensitivity, but its throughput is
not used in the final performance comparison.

| Micro-batch | Tokens/step | Warm tokens/s | Tokens/s/GPU | Avg step ms | Step>=10 tokens/s | Peak reserved MiB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| mbs2 | 256 | 5,658.8 | 2,828.8 | 45.9 | 5,325.5 | 1,302 |
| mbs4 | 512 | 10,840.6 | 5,423.8 | 47.7 | 10,231.8 | 1,316 |
| mbs8 | 1024 | 22,962.5 | 11,473.8 | 45.1 | 21,609.1 | 1,396 |
| mbs16 | 2048 | 45,737.5 | 22,868.8 | 45.5 | 42,254.5 | 1,818 |

The profiler run shows why larger token counts matter:

| Micro-batch | Routed tokens/layer | Median dispatcher ms | Route + count ms | Dispatch A2A ms | Coalesce ms | GroupedGEMM ms | Return A2A ms | Final all-reduce ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| mbs2 | 256 | 2.144 | 0.667 | 0.186 | 0.225 | 1.024 | 0.226 | 0.242 |
| mbs4 | 512 | 2.073 | 0.616 | 0.169 | 0.208 | 0.420 | 0.146 | 0.245 |
| mbs8 | 1024 | 2.191 | 0.622 | 0.182 | 0.208 | 1.118 | 0.920 | 0.323 |
| mbs16 | 2048 | 3.542 | 1.638 | 0.290 | 0.208 | 0.418 | 0.462 | 0.483 |

These historical values omit differentiable payload return and replicated
shared-gradient synchronization. They must not be compared with DP/TP as
correct training throughput. The corrected 4-GPU result is reported below.

Full details: [`qwen3_moe_style_ep_token_scaling.md`](qwen3_moe_style_ep_token_scaling.md).

## 4-GPU Mixed Parallel

The final Project 1 profiling pass compares three 4-GPU strategies at the same 4096 tokens/step. Each run uses 100 training steps, and the table reports the stable average from steps >= 50.

| Strategy | Tokens/step | Stable tokens/s | Tokens/s/GPU | Avg step ms | Peak reserved MiB/GPU | Final loss |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| DP4 | 4096 | 27.73K | 6.93K | 147.7 | 2642 | 9.79 |
| TP2+DP2 | 4096 | 37.23K | 9.31K | 110.0 | 2572 | 9.81 |
| EP2+DP2 | 4096 | 36.07K | 9.02K | 113.7 | 2008 | 9.80 |

The corrected EP2+DP2 path is 30.1% faster than DP4 and uses about 24% less
peak reserved memory, but is slightly slower than TP2+DP2. The original 48.51K
measurement is superseded because the payload collectives cut autograd and EP
replicas did not synchronize shared gradients. The corrected result includes
inverse All-to-All in backward and averaged shared-parameter gradients.

Full details: [`qwen3_moe_style_4gpu_mixed_parallel.md`](qwen3_moe_style_4gpu_mixed_parallel.md).

## Resume Validation

PP2 checkpoint/resume was validated by saving at step 20 and resuming to step
22. Corrected EP2+DP2 resume was validated from step 100 to 102. This test also
found and fixed scheduler construction order and an extra `_initial_step()` on
load: LR now continues from about `1e-5` instead of resetting to `3e-4`.

## What This Project Claims

This project does not claim to train a useful Qwen3 model. It claims to build and validate a compact pretraining infrastructure path:

```text
mini corpus -> tokenized packed shards -> Nanotron Qwen3-MoE-style training -> checkpoint/resume -> parallel profiling -> report/figures
```

That scope is appropriate for limited rented GPUs while still exercising the training-infra concerns relevant to larger VLA/LLM systems.
