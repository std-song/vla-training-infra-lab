# Project 1: Qwen3-MoE-style Pretraining Infra

This project builds a small but complete Nanotron pretraining-infra lab around a Qwen3-MoE-style model. It is designed to show distributed training understanding rather than train a useful foundation model.

## What It Covers

- Qwen3-MoE-style model adaptation on Nanotron, including QK-Norm-style attention changes and global-batch load-balancing hooks.
- BF16 training, FlashAttention, GroupedGEMM expert MLP, router top-k, activation recomputation, checkpoint/resume.
- Distributed strategies: single GPU, DP2, TP2, PP2, EP2, and earlier 4-GPU DP/TP/PP composition on Qwen2-MoE.
- Outer pretraining engineering: corpus manifest, tokenizer-aware packing, packed shard generation, launch matrix, log parsing, and figure generation.

## Key Results

Clean 20-step Qwen3-MoE-style profiling on AutoDL RTX 3090:

| Strategy | Warm tokens/s | Peak reserved memory |
| --- | ---: | ---: |
| single | 4,455.6 | 2,242 MiB |
| DP2 | 6,948.8 | 2,448 MiB |
| TP2 | 4,201.3 | 2,016 MiB |
| PP2 | 4,561.3 | 2,040 MiB |
| EP2 | 3,248.8 | 1,302 MiB |

Earlier 75.5M Qwen2-MoE 4-GPU profiling:

| Strategy | Total throughput |
| --- | ---: |
| single | 10.5K tokens/s |
| DP4 | 22.7K tokens/s |
| TP2+DP2 | 20.1K tokens/s |
| PP2+DP2 | 20.5K tokens/s |

![Qwen3-MoE throughput](assets/project1/qwen3_moe_throughput.svg)

![Qwen3-MoE memory](assets/project1/qwen3_moe_memory.svg)

## Reading Path

- Final report: [`results/project1_qwen3_moe_pretrain_infra_report.md`](results/project1_qwen3_moe_pretrain_infra_report.md)
- Qwen3 distributed validation: [`results/qwen3_moe_style_distributed_validation.md`](results/qwen3_moe_style_distributed_validation.md)
- EP All-to-All dispatch validation: [`results/qwen3_moe_style_ep_alltoall_dispatch.md`](results/qwen3_moe_style_ep_alltoall_dispatch.md)
- 4-GPU composition: [`results/qwen2_moe_4gpu_composition.md`](results/qwen2_moe_4gpu_composition.md)
- Scaling analysis: [`docs/scaling_analysis.md`](docs/scaling_analysis.md)
- Resume bullets: [`docs/project1_resume_bullets.md`](docs/project1_resume_bullets.md)

## EP2 All-to-All Validation

The EP2 path now includes a correctness-oriented token dispatcher:

```text
router top-k -> expert-owner dispatch -> local expert buffer coalesce -> GroupedGEMM -> return dispatch -> scatter-add -> replicated output
```

On 2026-07-10 it completed a fresh 5-step validation on 2 x RTX 3090 with checkpoint save. Warm throughput averaged 6,162.5 tokens/s total, peak reserved memory was 1,302 MiB/GPU, and the warm EP dispatcher averaged 2.847 ms per MoE layer call.

## Reproduction Skeleton

```bash
cd /root/autodl-tmp/vla-infra/nanotron
CUDA_DEVICE_MAX_CONNECTIONS=1 PYTHONPATH=src torchrun --nproc_per_node=2 \
  run_train.py --config-file examples/smoke/config_qwen3_moe_style_100m_dp2_20step.yaml
```

Use [`scripts/run_project1_nanotron_matrix.sh`](scripts/run_project1_nanotron_matrix.sh) for the single/DP/TP/PP/EP launch matrix.
