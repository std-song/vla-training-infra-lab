# Project 3 Resume Bullets: VLM/VLA Serving Prototype

## 中文简历版本

**Qwen2.5-VL VLA-style 推理服务原型与 Triton 动作后处理优化**

- 基于 Qwen2.5-VL-3B 构建轻量 VLA-style serving prototype，接入真实图像输入与 visual tokens，实现 visual input cache、同形状 micro-batching scheduler、KV cache footprint accounting，并拆分 image preprocessing、multimodal prefill、decode、TTFT/TPOT 和显存指标。
- 在 RTX 4080 SUPER 上对单相机/三相机、224/448 分辨率输入做 profiling；从 `1x224` 到 `3x448`，visual marker tokens 从 66 增至 774，multimodal prefill 从 40.3 ms 增至 166.4 ms，显存从约 7.2 GiB 增至 7.6 GiB。
- 对 8 个三相机 VLA-style 请求验证 serving 侧加速：`3x224, decode=32` 下 microbatching 将吞吐从 1.58 req/s 提升到 8.82 req/s，达到 5.62x；`3x448` 下从 1.29 req/s 提升到 4.13 req/s，达到 3.21x，并估算 KV footprint 从 75.7 MiB 增至 237.7 MiB。
- 保留 Qwen3-0.6B language-backbone 子实验，分析 KV cache、SDPA/eager/FlashAttention2 在不同 batch/prompt/decode shape 下的阶段性性能边界；KV cache 在 `batch=4, prompt=512, decode=64` 下达到 2.40x，但小 shape 可能退化。
- 实现 Triton 融合动作后处理 kernel，将动作反归一化、clamp 和 mask select 合并为单个算子；在 action post-processing benchmark 中取得 1.43x median speedup，大 action tensor shape 下最高 14.24x。

## 更短版本

基于 Qwen2.5-VL-3B 构建轻量 VLA-style 推理服务原型，接入真实图像输入和 visual tokens，实现 visual input cache、同形状 micro-batching、KV cache footprint accounting 与 Triton 动作后处理 kernel；在 RTX 4080 SUPER 上对 8 个三相机请求验证 microbatching 最高 5.62x 吞吐提升，并分析视觉 token 对 prefill、TTFT、显存和 KV cache 的影响。

## 面试展开点

- 为什么完整 vLLM 很复杂？这个 prototype 覆盖了哪些核心思想，哪些没有覆盖？
- 为什么 visual input cache 单独收益小，而 microbatching 收益大？
- visual token 数量如何影响 prefill、TTFT、KV footprint 和 batch capacity？
- 为什么 Qwen2.5-VL 的 true prefill KV cache 注入比纯 CausalLM 难？
- 如果继续扩展，如何做 PagedAttention、prefix cache、continuous batching 和异步 request scheduler？
