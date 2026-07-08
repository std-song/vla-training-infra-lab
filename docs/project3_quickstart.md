# Project 3 Quick Start: Qwen3 VLA Inference Profiling

Project 3 studies VLA-style inference with Qwen3-0.6B, KV-cache profiling, attention backend comparison, and a Triton fused action post-processing kernel.

## Environment

Validated on AutoDL with:

- Python 3.12.3
- PyTorch 2.8.0+cu128
- CUDA runtime 12.8
- flash-attn 2.8.3
- RTX 4080 SUPER 32 GiB

## Model

```bash
export MODEL_DIR=/root/autodl-tmp/vla-infra-project3/modelscope/models/Qwen--Qwen3-0.6B/snapshots/master
```

## Smoke

```bash
python project3_vla_infer/benchmarks/bench_causallm_prefill_decode.py \
  --model-dir "$MODEL_DIR" \
  --batch-sizes 1 \
  --prompt-lengths 128 \
  --decode-lengths 16 \
  --repeat 1 \
  --dtype bf16 \
  --attn-implementation sdpa \
  --out project3_vla_infer/results/qwen3_prefill_decode_smoke.csv
```

## Full baseline

```bash
python project3_vla_infer/benchmarks/bench_causallm_prefill_decode.py \
  --model-dir "$MODEL_DIR" \
  --batch-sizes 1,2,4 \
  --prompt-lengths 128,512,1024 \
  --decode-lengths 32,128 \
  --repeat 3 \
  --dtype bf16 \
  --attn-implementation sdpa \
  --out project3_vla_infer/results/qwen3_prefill_decode_sdpa_bf16.csv
```

## KV cache comparison

```bash
python project3_vla_infer/benchmarks/bench_causallm_kv_cache.py \
  --model-dir "$MODEL_DIR" \
  --batch-sizes 1,2,4 \
  --prompt-lengths 128,512 \
  --decode-lengths 16,32,64 \
  --repeat 3 \
  --dtype bf16 \
  --attn-implementation sdpa \
  --out project3_vla_infer/results/qwen3_kv_cache_compare_sdpa_bf16.csv
```

## Attention backend comparison

```bash
python project3_vla_infer/benchmarks/bench_causallm_prefill_decode.py \
  --model-dir "$MODEL_DIR" \
  --batch-sizes 1,4 \
  --prompt-lengths 128,1024 \
  --decode-lengths 32,128 \
  --repeat 3 \
  --dtype bf16 \
  --attn-implementation eager \
  --out project3_vla_infer/results/qwen3_prefill_decode_eager_bf16_selected.csv

python project3_vla_infer/benchmarks/bench_causallm_prefill_decode.py \
  --model-dir "$MODEL_DIR" \
  --batch-sizes 1,4 \
  --prompt-lengths 128,1024 \
  --decode-lengths 32,128 \
  --repeat 3 \
  --dtype bf16 \
  --attn-implementation flash_attention_2 \
  --out project3_vla_infer/results/qwen3_prefill_decode_flashattn2_bf16_selected.csv
```

## Triton action post-processing

Qwen3-0.6B uses hidden size 1024.

```bash
python project3_vla_infer/benchmarks/bench_vla_action_head_triton.py \
  --hidden-dim 1024 \
  --out project3_vla_infer/results/qwen3_vla_action_triton_hidden1024.csv
```

## Figures

```bash
python scripts/make_project3_qwen3_figures.py
```

