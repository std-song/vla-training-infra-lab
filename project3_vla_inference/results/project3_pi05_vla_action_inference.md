# Project 3 Stage: Pi0.5 Real VLA Action Inference

Date: 2026-07-08

This stage adds a real VLA policy path to Project 3. Unlike the Qwen3-VL/Qwen2.5-VL stages, which measure VLM serving, this benchmark loads a LeRobot Pi0.5 checkpoint and executes the robot action inference path:

```text
camera tensors + language tokens + robot state -> PaliGemma vision/language prefix -> Gemma action expert -> denoising action chunk -> action queue
```

The benchmark uses `lerobot/pi05_libero_finetuned_v044` with synthetic zero observations and real PaliGemma-tokenized task text. This is a compute-path and systems benchmark, not a robot-success or policy-quality benchmark.

## Environment

| Item | Value |
| --- | --- |
| GPU | NVIDIA vGPU-32GB |
| Python | 3.10.8 |
| PyTorch | 2.7.1+cu118 |
| CUDA runtime | 11.8 |
| LeRobot | 0.4.1 |
| Transformers | 4.53.3 |
| Tokenizers | 0.21.4 |
| Policy | Pi0.5 / LeRobot |
| Checkpoint | `lerobot/pi05_libero_finetuned_v044` |
| Weight file | `model.safetensors`, 7.47 GiB |

## Dependency and Loader Notes

The environment required exact dependency control: LeRobot 0.4.1 needed Python 3.10, PyTorch 2.7.x, `huggingface_hub<0.36`, `tokenizers<0.22`, and a Transformers build exposing Gemma internals used by Pi0.5. The checkpoint download also required falling back from HuggingFace `hf_transfer`/Xet to `aria2c` multi-connection resume.

Checkpoint loading uses `PI05Policy.from_pretrained(..., strict=False)`. With the correct PI05 policy class, all 812 tensors from the checkpoint are shape-matched and loaded; the loader still reports one missing tied/shared language embedding key. This is much cleaner than the earlier PI0Policy exploration and is the result used for the final benchmark.


## Task Text Adapter

LeRobot Pi0.5 does not consume raw `task` strings inside `PI05Policy.predict_action_chunk`. The processor first builds a prompt of the form:

```text
Task: <task>, State: <256-bin discretized normalized state>;
Action:
```

and then tokenizes it with `google/paligemma-3b-pt-224`. That tokenizer repository is gated on HuggingFace; after downloading it locally, the final benchmark uses real PaliGemma-tokenized task text. The benchmark scripts support `--tokenizer-name-or-path`; the final run uses the locally downloaded gated PaliGemma tokenizer and the task text `open the cabinet`.
## Action Chunk Benchmark

The benchmark calls `policy.predict_action_chunk(batch)` directly. This avoids `select_action` queue effects and measures the full action chunk inference path.

| Batch | Action shape | Cold start | Warm avg | Warm p50 | Warm min | Warm max | Chunk | Action dim | Amortized | Peak memory |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | `(1, 50, 7)` | 32,026.6 ms | 87.7 ms | 87.8 ms | 86.4 ms | 89.2 ms | 50 | 7 | 1.75 ms/action | 7,278 MiB |

The first call includes cold-start overhead such as model initialization side effects, Triton matmul autotuning, and CUDA kernel compilation. The steady-state full action chunk latency is about 80-85 ms.

## Action Queue Benchmark

Pi0.5 generates a chunk of 50 future actions. `select_action` runs the full model only when the internal queue is empty; otherwise it pops the next action from the queue.

| Metric | Value |
| --- | ---: |
| Full model calls in 60 control steps | 2 |
| Full model call average | 92.8 ms |
| Queue pops in 60 control steps | 58 |
| Queue pop average | 3.47 ms |
| Queue pop p50 | 3.49 ms |
| Peak memory | 7,142 MiB |

![Pi0.5 queue latency](../assets/figures/project3_pi05_queue_latency.svg)

## Interpretation

1. Pi0.5 action inference is chunk-based. The model produces `(batch, 50, 7)` actions, then `select_action` amortizes that cost across control steps.
2. The full action chunk path costs about 87-89 ms after warmup, while queue pop overhead is about 3.47 ms.
3. If a 50-step action chunk is consumed sequentially, the model-side amortized latency is about 1.8 ms/action. This does not mean the closed-loop robot runs at that rate, because real deployment must include observation acquisition, preprocessing, safety checks, actuator communication, and stale-action handling.
4. Cold-start latency is large and should be hidden by warmup in a serving/control runtime.
5. The remaining checkpoint key mismatches make this a systems benchmark of the Pi0.5 compute path rather than a claim of fully validated LIBERO policy quality.

## Why This Matters for VLA Inference Infra

This stage makes the project more defensible as VLA infrastructure work. The Qwen3-VL and Qwen2.5-VL stages cover VLM serving, visual-token cost, batching, KV memory, and scheduler behavior. The Pi0.5 stage adds the real robot-policy side: action chunk generation, action queue amortization, cold-start handling, and control-loop latency accounting.

## Reproduction

The curated result files are:

- `project3_vla_inference/results/pi05_policy_action_benchmark_tokenized.csv`
- `project3_vla_inference/results/pi05_policy_queue_benchmark_tokenized.csv`
- `project3_vla_inference/results/pi05_policy_queue_benchmark_tokenized_summary.csv`
- `project3_vla_inference/results/pi05_first_smoke_result.txt`

The model weights are intentionally not committed.







