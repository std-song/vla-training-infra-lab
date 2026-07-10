# Qwen2-MoE Training Design

## Purpose

The Qwen2-MoE experiment is the first project module in the VLA training-infra lab. It is intentionally small at first: the goal is to validate the infrastructure path before scaling model size or GPU count.

## Training Stack

- Framework: Nanotron
- Model family: Qwen2-style decoder model
- MoE implementation: Nanotron `Qwen2MoELayer`
- Expert MLP kernel: `grouped_gemm.ops.gmm`
- Attention: flash-attn path
- Precision: BF16
- Initial data source: dummy CLM generator

## Smoke Model

The smoke model is intentionally tiny:

- layers: 2
- hidden size: 128
- attention heads: 4
- vocabulary size: 4096
- experts: 4
- top-k: 1
- MoE layers: layer 0 and layer 1
- sequence length: 128
- micro batch size: 2
- training steps: 5

## Why Start With EP=1

The first goal is local correctness. With `expert_parallel_size=1`, the MoE layer still exercises router, token permutation, grouped expert MLP, unpermutation, backward, optimizer, and checkpointing without introducing cross-rank dispatch. Once this path is stable, EP can be increased to study communication behavior.

## Scaling Plan

1. `dp=1, ep=1`: single-GPU correctness.
2. `dp=2, ep=1`: distributed data parallel smoke.
3. `dp=8, ep=1`: DP throughput baseline.
4. `dp=4, ep=2`: first expert-parallel experiment.
5. `dp=2, ep=4`: stronger EP communication stress.
6. `dp=1, ep=8`: maximum EP on 8x3090.

## Expected Bottlenecks on RTX 3090

RTX 3090 systems are usually PCIe-based and lack the communication characteristics of production IB/RDMA clusters. Expected bottlenecks include:

- all-to-all cost for expert dispatch
- uneven token-to-expert routing
- dataloader wait time once real data is introduced
- activation memory at longer sequence lengths
- checkpoint IO if saving too frequently

This makes 3090 a useful resource-constrained testbed: it exposes practical training-infra tradeoffs clearly.
