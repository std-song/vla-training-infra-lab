# Scaling and Parallelism Analysis

This note explains the performance behavior observed in the Qwen2-MoE RTX 3090 experiments. The goal is not to present peak benchmark numbers, but to reason about why the measured throughput, memory, and scaling efficiency look the way they do.

## Metrics Used

Throughput is measured from Nanotron's logged `tokens_per_sec` over steady logged steps. For the main comparisons:

| Case | GPUs | DP | TP | PP | EP | Avg tokens/s | Avg tokens/s/GPU | Peak sampled memory |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1GPU baseline v2 | 1 | 1 | 1 | 1 | 1 | 10,544 | 10,544 | 2,271 MiB |
| DP2 | 2 | 2 | 1 | 1 | 1 | 17,987 | 8,989 | 2,328 MiB |
| DP4 | 4 | 4 | 1 | 1 | 1 | 22,686 | 5,671 | 2,318 MiB |
| TP2 | 2 | 1 | 2 | 1 | 1 | 10,311 | 5,154 | 1,892 MiB |
| PP2 | 2 | 1 | 1 | 2 | 1 | 11,357 | 5,674 | 1,866 MiB |
| TP2+DP2 | 4 | 2 | 2 | 1 | 1 | 20,071 | 5,019 | 2,050 MiB |
| PP2+DP2 | 4 | 2 | 1 | 2 | 1 | 20,500 | 5,127 | 1,866 MiB |

The 4-GPU composition runs used approximately the same global batch size: `2.05K` tokens per iteration. Earlier 1-GPU and 2-GPU experiments used smaller global batches, so exact comparisons should be read as system behavior evidence rather than a formal scaling benchmark.

## Why DP Scaling Is Not Linear

Ideal data-parallel scaling would multiply total throughput by the GPU count while keeping per-GPU throughput constant. Using the 1GPU baseline as reference:

| Case | Measured total tokens/s | Ideal linear tokens/s | Scaling efficiency |
| --- | ---: | ---: | ---: |
| DP2 | 17,987 | 21,088 | 85.3% |
| DP4 | 22,686 | 42,176 | 53.8% |

DP2 is reasonably efficient, but DP4 is much less linear. This is expected on small models and PCIe RTX 3090 hardware.

Main reasons:

1. Gradient synchronization grows with more DP ranks.

   DP replicates the full model on every GPU and all-reduces gradients after backward. The Qwen2-MoE model used here is only 75.5M parameters, so compute per step is small. As GPU count increases, gradient all-reduce latency and bandwidth cost become a larger fraction of the iteration.

2. The model is too small to hide communication.

   Large models have enough compute to amortize communication. Here, hidden size is 512 and the model has only 4 layers. The GPU kernels finish quickly, so communication and Python/framework overhead become visible.

3. PCIe 3090 communication is not datacenter interconnect.

   RTX 3090 instances usually lack NVLink/RDMA-style training interconnect. NCCL collectives over PCIe are much less forgiving when synchronization becomes frequent.

4. Per-rank batch is small.

   Micro batch size is 1. Small per-rank work means each process launches many small kernels and synchronizes often. This lowers hardware occupancy and makes launch/communication overhead more prominent.

5. Dummy data removes dataloader cost, so communication is exposed.

   This is useful for training-infra profiling: the bottleneck is not hidden behind CPU dataloading. It also means the observed scaling reflects model/distributed overhead more directly.

Interpretation: DP4 is correct and useful, but not throughput-linear. It validates launcher, DDP synchronization, rank-aware checkpointing, and 4-GPU operation. It does not claim that this tiny model saturates 4 GPUs efficiently.

## TP Behavior

TP2 alone produced `10,311 tokens/s`, almost the same total throughput as the 1GPU baseline, but split over two GPUs. TP2+DP2 produced `20,071 tokens/s`, below DP4's `22,686 tokens/s`.

This is expected for this model size.

TP reduces per-rank parameter and activation memory, but it introduces communication inside the forward/backward path. Tensor-parallel linear layers need collective communication around sharded projections. With a small hidden size and only 4 layers, the reduced matrix multiplication does not compensate for the extra collectives.

Observed memory behavior supports this:

| Case | Peak sampled memory | Memory change |
| --- | ---: | ---: |
| 1GPU baseline | 2,271 MiB | reference |
| TP2 | 1,892 MiB | -16.7% |
| DP4 | 2,318 MiB | reference |
| TP2+DP2 | 2,050 MiB | -11.6% vs DP4 |

TP is therefore doing what it should: it reduces per-rank memory and validates tensor-sharded checkpoint layout. It is not expected to speed up such a small model on PCIe GPUs.

When TP becomes valuable:

- the model no longer fits on one GPU
- hidden size and FFN size are large enough to make tensor-sharded matmuls substantial
- interconnect is fast enough to amortize collectives
- TP is combined with DP for larger global training scale

## PP Behavior

PP2 alone produced `11,357 tokens/s`, and PP2+DP2 produced `20,500 tokens/s`. PP2+DP2 is slightly below DP4 but close to TP2+DP2.

Pipeline parallelism primarily saves memory by placing different layers on different ranks. It does not automatically improve throughput. In this experiment, PP has three important costs:

1. Pipeline bubbles.

   PP needs multiple microbatches to keep stages busy. With `pp=2` and `batch_accumulation_per_replica=2`, only two microbatches are available. A rough bubble fraction for pipeline scheduling is `(pp - 1) / (microbatches + pp - 1)`, which is about `1 / 3` here. That means a meaningful fraction of time is pipeline fill/drain rather than full utilization.

2. Stage imbalance.

   The observed stage assignment was imbalanced:

   ```text
   PP rank 0: embedding + decoder layers 0, 1, 2 = 54.5M local parameters
   PP rank 1: decoder layer 3 + final norm + lm head + loss = 21M local parameters
   ```

   Rank 0 has more model work; rank 1 waits. For a 4-layer toy model, there are not enough layers to create a balanced pipeline split.

3. Point-to-point activation communication.

   PP sends activations forward and gradients backward between stages. On PCIe hardware, this cost is visible.

Observed memory behavior is the main benefit:

| Case | Peak sampled memory | Memory change |
| --- | ---: | ---: |
| DP4 | 2,318 MiB | reference |
| PP2+DP2 | 1,866 MiB | -19.5% |

PP is therefore successful as a memory-sharding validation, not a speedup mechanism for this small model.

When PP becomes valuable:

- the model has many layers and can be split evenly
- activation memory dominates
- enough microbatches are available to reduce bubbles
- PP is composed with DP and possibly TP for larger models

## DP vs TP vs PP: What Each Result Proves

| Mode | What it proves | Main cost | Main benefit in this project |
| --- | --- | --- | --- |
| DP | multi-rank launch, gradient all-reduce, replicated optimizer/checkpoint state | gradient synchronization | highest simple throughput |
| TP | tensor-sharded model path and TP checkpoint layout | collectives inside forward/backward | lower per-rank model memory |
| PP | pipeline stage placement, p2p activation/gradient flow, PP checkpoint layout | bubbles, imbalance, p2p communication | lower per-rank memory and model-stage execution |
| EP | not complete yet | local/global expert accounting and token dispatch | readiness gap localized |

A strong resume interpretation is: DP is the efficient baseline, TP/PP are capacity mechanisms. They are not expected to beat DP on a tiny model, but they are necessary for models that exceed one GPU's memory.

## Expert Parallel Analysis

The natural 4-GPU EP target is:

```text
ep=2, dp=2, tp=1, pp=1
```

This is conceptually correct because:

```text
EP * DP * TP * PP * CP = 2 * 2 * 1 * 1 * 1 = 4
```

The current Nanotron checkout did not support this path directly.

First blocker:

```text
ParallelContext world-size assertion omitted expert_parallel_size.
```

After a minimal local fix, initialization advanced and the run reached MoE execution. The next blocker was:

```text
RuntimeError: Expected batch_sizes.size(0) == num_experts to be true, but got false.
```

Interpretation: with `expert_parallel_size=2`, each rank owns fewer local experts, but the router/grouped-GEMM path still receives token counts shaped for global experts. True EP needs:

- global expert id to local expert id mapping
- token dispatch across `ep_pg`
- local `num_tokens_per_expert` before GroupedGEMM
- output combine and original token order restore
- router auxiliary loss aggregation across EP ranks
- EP-aware optimizer/checkpoint save/load validation

This is why the project honestly claims EP readiness analysis, not completed EP training.

## Bottom Line

The measured numbers are internally consistent:

- DP gives the best simple throughput but scales sublinearly due to gradient all-reduce and small-model overhead.
- TP lowers memory but does not improve throughput because tensor collectives dominate small matmuls.
- PP lowers memory but suffers from bubbles and an imbalanced 4-layer split.
- 4-GPU DP/TP/PP compositions are validated and checkpointed.
- EP is the next real engineering contribution rather than a config-only experiment.
