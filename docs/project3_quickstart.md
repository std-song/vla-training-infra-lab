# Project 3 Quick Start: Qwen2.5-VL VLA-Style Inference Profiling

Project 3 studies VLA-style inference with real image inputs through Qwen2.5-VL-3B, plus a Qwen3 language-backbone subtest for KV-cache and attention backend behavior.

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

## Visual-token smoke

```bash
python project3_vla_infer/benchmarks/bench_qwen25vl_visual_tokens.py \
  --model-dir "$MODEL_DIR" \
  --image-sizes 224 \
  --image-counts 1 \
  --decode-lengths 4 \
  --repeat 1 \
  --warmup-decode 1 \
  --dtype bf16 \
  --attn-implementation sdpa \
  --out project3_vla_infer/results/qwen25vl_visual_tokens_smoke.csv
```

## Dynamic visual-token profiling

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

The VLM script measures:

- image preprocessing time;
- CPU-to-GPU tensor transfer time;
- multimodal prefill latency;
- estimated TTFT;
- estimated decode TPOT from `generate(max_new_tokens) - prefill`;
- input token count and visual marker token count;
- CUDA peak memory.

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
python scripts/make_project3_qwen3_figures.py
```
