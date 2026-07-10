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

This validates EP correctness and exposes the next optimization boundary: remove or move the final replication all-reduce, then overlap token dispatch with local expert compute.

## Resume Validation

PP2 checkpoint/resume was validated by saving at step 20 and resuming to step 22. This confirms that pipeline-stage model shards, optimizer state, scheduler state, and random state can be restored well enough for continued training.

## What This Project Claims

This project does not claim to train a useful Qwen3 model. It claims to build and validate a compact pretraining infrastructure path:

```text
mini corpus -> tokenized packed shards -> Nanotron Qwen3-MoE-style training -> checkpoint/resume -> parallel profiling -> report/figures
```

That scope is appropriate for limited rented GPUs while still exercising the training-infra concerns relevant to larger VLA/LLM systems.
