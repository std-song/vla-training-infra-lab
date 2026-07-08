# Project 3 Resume Bullets: Qwen3 VLA Inference Profiling

## 中文简历版本

**Qwen3 VLA 推理链路分析与 Triton 算子优化**

- 基于 Qwen3-0.6B 构建 VLA-style 推理 profiling 框架，在 RTX 4080 SUPER 上拆分 prompt prefill 与逐 token decode，统计 TTFT、TPOT、tokens/s 和显存，分析 batch、上下文长度和生成长度对延迟/显存的影响。
- 对比无缓存重算与 KV cache 解码，发现 KV cache 并非小 shape 下必然加速；在 `batch=4, prompt=512, decode=64` 下达到 2.40x，而在小 batch/短 prompt 下可能因缓存管理开销变慢。
- 对比 SDPA、eager attention 与 FlashAttention 2 后端：在 `batch=4, prompt=1024, decode=128` 下 SDPA 达到 220.9 tokens/s，eager 长 prefill 明显变慢，FlashAttention 2 在该 Hugging Face cached-decode 路径下 decode 较慢，形成按阶段选择 attention backend 的分析结论。
- 实现 Triton 融合动作后处理 kernel，将动作反归一化、clamp 和 mask select 合并为单个算子；在 Qwen3 hidden size 1024 的 action head benchmark 中取得 1.43x median speedup，大 action tensor shape 下最高 14.24x。

## 更短版本

基于 Qwen3-0.6B 构建 VLA-style 推理 profiling 与 kernel 优化实验，拆分 prefill/decode 并统计 TTFT、TPOT、tokens/s、KV cache 和显存；在 RTX 4080 SUPER 上验证 KV cache 最高 2.40x 加速但小 shape 可能退化，分析 SDPA/eager/FlashAttention2 的阶段性性能差异，并实现 Triton 融合动作后处理 kernel，median 1.43x、最高 14.24x 加速。

## 面试展开点

- 为什么 KV cache 在小 batch/短 prompt 下可能比 no-cache 更慢？
- 为什么 FlashAttention 2 长 prefill 接近 SDPA，但 cached decode 反而慢？
- VLA action 后处理为什么值得单独做 kernel fusion？什么 shape 下不值得？
- 这个项目和完整 vLLM/PagedAttention 的边界在哪里？下一步如何扩展到 paged KV cache、continuous batching 或真实 SmolVLA action head？
