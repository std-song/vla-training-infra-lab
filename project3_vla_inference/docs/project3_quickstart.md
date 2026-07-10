# Project 3 Quick Start: Multimodal VLM/VLA Inference Acceleration

Project 3 is a three-layer multimodal inference project. The Qwen3-VL/Qwen2.5-VL commands reproduce the VLM serving layer, the Pi0.5 commands reproduce the real VLA action-inference compute-path benchmark, and the VLASH-inspired simulator connects measured Pi0.5 latency to async control-loop scheduling.

## Track A: Qwen3-VL vLLM baseline

Validated on AutoDL with Python 3.12, PyTorch 2.8.0 + CUDA 12.8, vLLM 0.24.0, and a 32 GiB GPU.

Important setup notes:

- Do not run vLLM smoke tests from `python - <<'PY'` when `VLLM_WORKER_MULTIPROC_METHOD=spawn`; use real `.py` files because worker startup reloads the main script path.
- If an old `flash-attn` wheel fails with an undefined PyTorch/CUDA symbol, uninstall it and let vLLM use its own backend path.
- vLLM runs model execution in worker processes, so parent-process `torch.cuda.max_memory_allocated()` is not reliable. Use `nvidia-smi` sampling for peak memory.

```bash
export MODEL_DIR=/root/autodl-tmp/vla-infra-project3/modelscope/models/Qwen--Qwen3-VL-4B-Instruct/snapshots/master
export VLLM_USE_FLASHINFER_SAMPLER=0
export VLLM_WORKER_MULTIPROC_METHOD=spawn

python project3_vla_inference/benchmarks/bench_qwen3vl_vllm.py \
  --model-dir "$MODEL_DIR" \
  --image-sizes 224,448 \
  --concurrency 1,2,4,8 \
  --max-new-tokens 64 \
  --warmup 3 \
  --repeat 5 \
  --out project3_vla_inference/results/qwen3vl_vllm_default_concurrency.csv

python project3_vla_inference/benchmarks/bench_qwen3vl_vllm.py \
  --model-dir "$MODEL_DIR" \
  --image-sizes 224,448 \
  --concurrency 1,2,4,8 \
  --max-new-tokens 64 \
  --warmup 3 \
  --repeat 5 \
  --enforce-eager \
  --out project3_vla_inference/results/qwen3vl_vllm_eager_concurrency.csv
```

Memory sampling example:

```bash
nvidia-smi \
  --query-gpu=timestamp,index,name,utilization.gpu,memory.used,memory.total,power.draw,temperature.gpu \
  --format=csv \
  -lms 200 > qwen3vl_vllm_nvidia_smi.csv &
SMI_PID=$!
# run benchmark here
kill $SMI_PID
```

Regenerate figures:

```bash
python scripts/make_project3_qwen3vl_vllm_figures.py
```

## Track B: Pi0.5 / LeRobot action inference

Validated on AutoDL with Python 3.10.8, PyTorch 2.7.1+cu118, CUDA 11.8, LeRobot 0.4.1, and a 32 GiB vGPU.

Important setup notes:

- The Pi0.5 dependency stack is sensitive. Keep `huggingface_hub<0.36`, `tokenizers<0.22`, and a Transformers build compatible with LeRobot Pi0.5/Gemma internals.
- `lerobot/pi05_libero_finetuned_v044` is a 7.47 GiB checkpoint. On AutoDL, `aria2c` with multi-connection resume was more reliable than `hf_transfer`/Xet.
- The LeRobot 0.4.1 loader and this checkpoint have key-name differences. The benchmark uses `strict=False` and reports the remaining mismatch as a loader-compatibility boundary.
- Without `--tokenizer-name-or-path`, the benchmark uses dummy language tokens. With the local PaliGemma tokenizer path, it builds the PI05 prompt from real task text and discretized state. In both cases, the synthetic zero images/state mean this measures action-inference compute path and queue behavior, not robot task success.

```bash
export MODEL_DIR=/root/autodl-tmp/vla-infra-project3-pi05/models/pi05_libero_finetuned_v044

python project3_vla_inference/benchmarks/bench_pi05_action_chunk.py \
  --model-dir "$MODEL_DIR" \
  --batch-sizes 1 \
  --warmup 3 \
  --repeat 10 \
  --out project3_vla_inference/results/pi05_action_benchmark_strict_false.csv

python project3_vla_inference/benchmarks/bench_pi05_queue.py \
  --model-dir "$MODEL_DIR" \
  --steps 60 \
  --warmup-chunks 4 \
  --out project3_vla_inference/results/pi05_queue_benchmark_warm_strict_false.csv \
  --summary-out project3_vla_inference/results/pi05_queue_benchmark_warm_strict_false_summary.csv
```

The first benchmark measures full `predict_action_chunk` latency for `(1, 50, 7)` action chunks. The second benchmark measures `select_action` queue amortization: full model calls happen when the action queue is empty, while intermediate control steps pop from the queue.

Optional real task text tokenization requires access to the gated PaliGemma tokenizer repo. First accept the model terms for `google/paligemma-3b-pt-224` on HuggingFace, then create a token with access to public gated repositories. On the AutoDL host:

```bash
read -s HF_TOKEN
export TOKENIZER_DIR=/root/autodl-tmp/vla-infra-project3-pi05/models/paligemma-3b-pt-224-tokenizer

python - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="google/paligemma-3b-pt-224",
    local_dir="/root/autodl-tmp/vla-infra-project3-pi05/models/paligemma-3b-pt-224-tokenizer",
    allow_patterns=["tokenizer*", "special_tokens_map.json", "tokenizer_config.json", "config.json"],
    token=True,
)
PY
```

Then rerun the benchmark with real PI05 prompt construction:

```bash
python project3_vla_inference/benchmarks/bench_pi05_action_chunk.py \
  --model-dir "$MODEL_DIR" \
  --tokenizer-name-or-path "$TOKENIZER_DIR" \
  --task "open the cabinet" \
  --out project3_vla_inference/results/pi05_policy_action_benchmark_tokenized.csv
```

## Track C: VLASH-inspired async control-loop simulator

This layer does not require a GPU. It uses the measured Pi0.5 warm action-chunk latency and queue-pop latency to compare synchronous chunk execution, naive async queue refill, future-state-aware async refill, and future-state async with action quantization.

```bash
python project3_vla_inference/simulators/vlash_async_control_loop.py \
  --policy-latency-ms 87.65 \
  --queue-pop-ms 3.467 \
  --chunk-size 50 \
  --control-hz 30 \
  --out-summary project3_vla_inference/results/vlash_async_control_loop_summary.csv \
  --out-trace project3_vla_inference/results/vlash_async_control_loop_trace.csv

python scripts/make_project3_vlash_figures.py
```

The output reports mean tracking error, reaction latency after a target change, control-loop stall ratio, state staleness, and modeled control overhead. This is a simulator for inference-infra analysis, not a real robot success benchmark.
## Environment

Validated on AutoDL with:

- Python 3.12.3
- PyTorch 2.8.0+cu128
- CUDA runtime 12.8
- flash-attn 2.8.3
- RTX 4080 SUPER 32 GiB

## Qwen2.5-VL model

```bash
export MODEL_DIR=/root/autodl-tmp/vla-infra-project3/modelscope/models/Qwen--Qwen2.5-VL-3B-Instruct/snapshots/master
```

## Visual-token profiling

```bash
python project3_vla_inference/benchmarks/bench_qwen25vl_visual_tokens.py \
  --model-dir "$MODEL_DIR" \
  --image-sizes 224,448 \
  --image-counts 1,3 \
  --decode-lengths 16,64 \
  --repeat 3 \
  --warmup-decode 4 \
  --min-pixels 3136 \
  --max-pixels 802816 \
  --dtype bf16 \
  --attn-implementation sdpa \
  --out project3_vla_inference/results/qwen25vl_visual_tokens_dynamic_pixels_sdpa_bf16.csv
```

## Serving prototype

Smoke:

```bash
python project3_vla_inference/benchmarks/bench_qwen25vl_serving_prototype.py \
  --model-dir "$MODEL_DIR" \
  --request-count 2 \
  --image-count 1 \
  --image-size 224 \
  --decode-len 8 \
  --repeat 1 \
  --min-pixels 3136 \
  --max-pixels 802816 \
  --dtype bf16 \
  --attn-implementation sdpa \
  --out project3_vla_inference/results/qwen25vl_serving_prototype_smoke.csv
```

8-request three-camera benchmark:

```bash
python project3_vla_inference/benchmarks/bench_qwen25vl_serving_prototype.py \
  --model-dir "$MODEL_DIR" \
  --request-count 8 \
  --image-count 3 \
  --image-size 224 \
  --decode-len 32 \
  --repeat 3 \
  --min-pixels 3136 \
  --max-pixels 802816 \
  --dtype bf16 \
  --attn-implementation sdpa \
  --out project3_vla_inference/results/qwen25vl_serving_prototype_8req_3x224_d32.csv

python project3_vla_inference/benchmarks/bench_qwen25vl_serving_prototype.py \
  --model-dir "$MODEL_DIR" \
  --request-count 8 \
  --image-count 3 \
  --image-size 448 \
  --decode-len 32 \
  --repeat 3 \
  --min-pixels 3136 \
  --max-pixels 802816 \
  --dtype bf16 \
  --attn-implementation sdpa \
  --out project3_vla_inference/results/qwen25vl_serving_prototype_8req_3x448_d32.csv
```

The serving prototype compares:

- cold serial requests;
- visual-input-cache serial generation;
- visual-input-cache plus same-shape microbatching.

It reports requests/s, per-request latency, speedup, peak memory, and estimated KV cache footprint.


## Paged KV and continuous batching simulator

```bash
python project3_vla_inference/simulators/paged_kv_continuous_batching.py \
  --request-count 128 \
  --mean-arrival-ms 90 \
  --kv-budget-mib 512 \
  --max-active 16 \
  --block-size 16 \
  --out project3_vla_inference/results/qwen25vl_paged_kv_continuous_batching.csv

python project3_vla_inference/simulators/paged_kv_continuous_batching.py \
  --request-count 128 \
  --mean-arrival-ms 90 \
  --kv-budget-mib 256 \
  --max-active 16 \
  --block-size 16 \
  --out project3_vla_inference/results/qwen25vl_paged_kv_budget256.csv
```

The simulator compares serial execution, continuous batching with static KV reservation, naive paged KV, and guarded paged KV admission.


## Bucketed scheduler and prefix-cache simulator

```bash
python project3_vla_inference/simulators/bucketed_prefix_cache.py \
  --request-count 256 \
  --mean-arrival-ms 70 \
  --max-batch-tokens 4096 \
  --max-batch-requests 16 \
  --prefix-pool 32 \
  --out project3_vla_inference/results/qwen25vl_bucketed_prefix_cache_sim.csv
```

The simulator compares FCFS, shape-aware buckets, token-budget buckets, and prefix-cache hits for repeated VLA visual/task prefixes.

## Qwen3 language-backbone subtest

```bash
export QWEN3_DIR=/root/autodl-tmp/vla-infra-project3/modelscope/models/Qwen--Qwen3-0.6B/snapshots/master

python project3_vla_inference/benchmarks/bench_causallm_prefill_decode.py \
  --model-dir "$QWEN3_DIR" \
  --batch-sizes 1,2,4 \
  --prompt-lengths 128,512,1024 \
  --decode-lengths 32,128 \
  --repeat 3 \
  --dtype bf16 \
  --attn-implementation sdpa \
  --out project3_vla_inference/results/qwen3_prefill_decode_sdpa_bf16.csv
```

## Triton action post-processing

```bash
python project3_vla_inference/benchmarks/bench_vla_action_head_triton.py \
  --hidden-dim 1024 \
  --out project3_vla_inference/results/qwen3_vla_action_triton_hidden1024.csv
```

## Figures

```bash
python scripts/make_project3_qwen25vl_figures.py
python scripts/make_project3_serving_figures.py
python scripts/make_project3_paged_kv_figures.py
python scripts/make_project3_bucketed_scheduler_figures.py
python scripts/make_project3_qwen3_figures.py
python scripts/make_project3_vlash_figures.py
```
