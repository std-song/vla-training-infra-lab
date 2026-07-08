# Project 3 Quick Start: Qwen2.5-VL VLA-Style Serving Prototype

Project 3 studies VLA-style serving with real image inputs through Qwen2.5-VL-3B. It includes visual-token profiling, a lightweight serving prototype with visual input cache and same-shape microbatching, a Qwen3 language-backbone subtest for KV-cache and attention backend behavior, and a Triton fused action post-processing kernel.

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
python project3_vla_infer/benchmarks/bench_qwen25vl_visual_tokens.py \
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
  --out project3_vla_infer/results/qwen25vl_visual_tokens_dynamic_pixels_sdpa_bf16.csv
```

## Serving prototype

Smoke:

```bash
python project3_vla_infer/benchmarks/bench_qwen25vl_serving_prototype.py \
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
  --out project3_vla_infer/results/qwen25vl_serving_prototype_smoke.csv
```

8-request three-camera benchmark:

```bash
python project3_vla_infer/benchmarks/bench_qwen25vl_serving_prototype.py \
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
  --out project3_vla_infer/results/qwen25vl_serving_prototype_8req_3x224_d32.csv

python project3_vla_infer/benchmarks/bench_qwen25vl_serving_prototype.py \
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
  --out project3_vla_infer/results/qwen25vl_serving_prototype_8req_3x448_d32.csv
```

The serving prototype compares:

- cold serial requests;
- visual-input-cache serial generation;
- visual-input-cache plus same-shape microbatching.

It reports requests/s, per-request latency, speedup, peak memory, and estimated KV cache footprint.

## Qwen3 language-backbone subtest

```bash
export QWEN3_DIR=/root/autodl-tmp/vla-infra-project3/modelscope/models/Qwen--Qwen3-0.6B/snapshots/master

python project3_vla_infer/benchmarks/bench_causallm_prefill_decode.py \
  --model-dir "$QWEN3_DIR" \
  --batch-sizes 1,2,4 \
  --prompt-lengths 128,512,1024 \
  --decode-lengths 32,128 \
  --repeat 3 \
  --dtype bf16 \
  --attn-implementation sdpa \
  --out project3_vla_infer/results/qwen3_prefill_decode_sdpa_bf16.csv
```

## Triton action post-processing

```bash
python project3_vla_infer/benchmarks/bench_vla_action_head_triton.py \
  --hidden-dim 1024 \
  --out project3_vla_infer/results/qwen3_vla_action_triton_hidden1024.csv
```

## Figures

```bash
python scripts/make_project3_qwen25vl_figures.py
python scripts/make_project3_serving_figures.py
python scripts/make_project3_qwen3_figures.py
```
