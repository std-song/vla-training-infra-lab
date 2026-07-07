# Project 3 Quick Start: Qwen2 VLA Inference Baseline

Project 3 studies VLA-style inference acceleration with a Qwen2 language backbone, KV cache profiling, attention implementation comparison, and a later Triton fused action-head kernel.

## Stage 1: Prefill / Decode Baseline

On the GPU machine, clone or pull this repository, then run:

```bash
cd /root/autodl-tmp/vla-infra-project3

git clone https://github.com/std-song/vla-training-infra-lab.git || true
cd vla-training-infra-lab

git pull

export MODEL_DIR=/root/autodl-tmp/vla-infra-project3/modelscope/models/qwen--Qwen2-0.5B-Instruct/snapshots/master
```

Quick smoke test:

```bash
python project3_vla_infer/benchmarks/bench_qwen2_prefill_decode.py \
  --model-dir "$MODEL_DIR" \
  --batch-sizes 1 \
  --prompt-lengths 128 \
  --decode-lengths 16 \
  --repeat 1 \
  --out project3_vla_infer/results/qwen2_prefill_decode_smoke.csv
```

Full first baseline:

```bash
python project3_vla_infer/benchmarks/bench_qwen2_prefill_decode.py \
  --model-dir "$MODEL_DIR" \
  --batch-sizes 1,2,4 \
  --prompt-lengths 128,512,1024 \
  --decode-lengths 32,128 \
  --repeat 3 \
  --dtype bf16 \
  --attn-implementation sdpa \
  --out project3_vla_infer/results/qwen2_prefill_decode_sdpa_bf16.csv
```

The output CSV records:

- `prefill_ms`
- `decode_ms`
- `ttft_est_ms`
- `tpot_ms`
- `decode_tokens_per_s`
- `max_memory_mib`

## Notes

This script intentionally uses Hugging Face Transformers + PyTorch SDPA first. FlashAttention and Triton kernels will be added after this baseline is stable.
