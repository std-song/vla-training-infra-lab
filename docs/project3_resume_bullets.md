# Project 3 Resume Bullets: VLM/VLA-Style Inference Profiling

## 中文简历版本

**Qwen2.5-VL 多模态推理链路分析与 Triton 动作后处理优化**

- 基于 Qwen2.5-VL-3B 构建 VLA-style 多模态推理 profiling 框架，接入真实图像输入与 visual tokens，拆分 image preprocessing、multimodal prefill、decode、TTFT、TPOT 和显存指标。
- 在 RTX 4080 SUPER 上对单相机/三相机、224/448 分辨率输入做 profiling；从 `1x224` 到 `3x448`，visual marker tokens 从 66 增至 774，multimodal prefill 从 40.3 ms 增至 166.4 ms，显存从约 7.2 GiB 增至 7.6 GiB。
- 保留 Qwen3-0.6B language-backbone 子实验，分析 KV cache、SDPA/eager/FlashAttention2 在不同 batch/prompt/decode shape 下的阶段性性能边界；KV cache 在 `batch=4, prompt=512, decode=64` 下达到 2.40x，但小 shape 可能退化。
- 实现 Triton 融合动作后处理 kernel，将动作反归一化、clamp 和 mask select 合并为单个算子；在 action post-processing benchmark 中取得 1.43x median speedup，大 action tensor shape 下最高 14.24x。

## 更短版本

基于 Qwen2.5-VL-3B 构建 VLA-style 多模态推理 profiling 实验，接入真实图像输入和 visual tokens，拆分 image preprocessing、multimodal prefill、decode、TTFT/TPOT 和显存；在 RTX 4080 SUPER 上分析单/三相机输入下视觉 token 对 prefill 的影响，并结合 Qwen3 language decode 子实验分析 KV cache/attention backend，另实现 Triton 融合动作后处理 kernel，median 1.43x、最高 14.24x 加速。

## 面试展开点

- 为什么 Qwen3 language-only profiling 不能单独称为 VLA？Qwen2.5-VL 补上了哪些链路？
- visual token 数量为什么主要影响 TTFT 和 prefill，而不是每个 decode token 的 TPOT？
- 三相机输入在 serving 系统里会带来哪些 batching、KV cache 和显存管理问题？
- 为什么 Qwen2.5-VL 的 cached generation 不能直接复用纯 CausalLM 的手写 decode loop？
- VLA action post-processing 哪些 shape 值得做 Triton fusion，哪些 shape 不值得？
