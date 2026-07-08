# Project 3 Resume Bullets: Qwen2 VLA Inference Acceleration

Use one of the following versions depending on resume space.

## Strong 3-Bullet Version

- Built a Qwen2-0.5B based VLA inference acceleration lab on RTX 4080 SUPER, separating prefill/decode phases and reporting serving metrics including estimated TTFT, TPOT, decode tokens/s, KV-cache memory, and shape-dependent throughput under BF16.
- Benchmarked no-cache vs KV-cache decode and SDPA/eager/FlashAttention2 attention backends; observed KV-cache speedups up to 2.55x only for larger prompt/batch shapes and found SDPA to be the best default for this Hugging Face cached-decode path.
- Implemented a Triton fused VLA action post-processing kernel for action denormalization, clamp, and mask selection; achieved 1.44x average speedup and up to 1.81x over PyTorch elementwise ops while documenting small-shape overhead and BF16 correctness.

## Compact 2-Bullet Version

- Developed a Qwen2-based VLA-style inference benchmark with vLLM-inspired prefill/decode, TTFT/TPOT, KV-cache, attention backend, throughput, and memory profiling on RTX 4080 SUPER.
- Added a simplified VLA action head and Triton fused action post-processing kernel, achieving up to 1.81x speedup over PyTorch for action denorm/clamp/mask while documenting shape-dependent tradeoffs.

## One-Line Version

Built a Qwen2-based VLA inference acceleration lab covering prefill/decode profiling, KV-cache analysis, SDPA/eager/FlashAttention2 comparison, and a Triton fused action post-processing kernel with up to 1.81x speedup.

## Interview Boundary

This is not a production vLLM fork or full SmolVLA serving engine. It is a controlled VLA inference-infra lab that demonstrates measurement discipline, backend tradeoff analysis, and a working VLA-specific Triton kernel.
