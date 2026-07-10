# Project 1 Resume Bullets

## 中文简历版本

**基于 Nanotron 的 Qwen3-MoE 混合并行预训练系统**

- 基于 Nanotron 搭建 Qwen3-MoE-style 小规模预训练系统，覆盖 BF16、FlashAttention-2、GroupedGEMM 专家计算、Router Top-k、全局批次负载均衡、激活重计算与 checkpoint/resume。
- 适配并验证 DP、TP、PP、EP 多种并行策略；修复 PP 非最终流水段误访问 loss 指标导致的跨 rank 卡死问题，并完成 PP checkpoint/resume 验证。
- 实现 EP2 token all-to-all dispatch：按 expert owner 交换 token hidden states 与路由元数据，按本地专家重排并合并连续 buffer 后调用 GroupedGEMM，再将专家输出返回 token owner。
- 在 2 x RTX 3090 上完成 108M 参数 Qwen3-MoE-style 训练矩阵 profiling：single 4.46K tokens/s、DP2 6.95K、TP2 4.20K、PP2 4.56K；EP2 在真实 token dispatch 下峰值显存约 1.3GB/GPU。
- 对 EP2 做 token 粒度扩展分析：tokens/step 从 256 增至 2048 时，吞吐从 5.66K 提升至 45.7K tokens/s，峰值显存仍低于 1.9GB/GPU，定位 metadata collectives、return all-to-all 与 final all-reduce 为下一步优化边界。
- 在 4 x RTX 3090 上完成同等 4096 tokens/step 的混合并行对比：DP4 27.7K tokens/s、TP2+DP2 37.2K、EP2+DP2 48.5K；EP2+DP2 峰值显存 1.94GB/GPU，验证专家切分在足够 routed tokens 下可同时改善吞吐和显存。
- 补齐预训练外层工程链路：语料 manifest、tokenizer/packing、固定长度 shard、实验矩阵脚本、Nanotron 日志解析、吞吐/显存图表与可复现实验报告。

## 面试展开重点

- DP2 为什么不是 2x：小模型、小 batch 场景下，kernel launch、optimizer、logging、gradient sync 和 router 统计同步开销占比较高。
- TP/PP 为什么不一定提速：TP 引入逐层 collective；PP 需要足够层数和 micro-batch 深度才能摊薄流水线 bubble。
- EP 为什么显存最低但小 shape 下吞吐敏感：专家权重被切分，但 token dispatch、metadata exchange、buffer coalesce 和 all-to-all 成为主要固定开销。
- PP bug 的本质：loss/metric ownership 只属于最后一个流水段，非最终 stage 不能构造 `loss_avg.item()`。
- EP 后续优化方向：压缩 metadata collective、用 CUDA stream 和 async collective 做更深的通信计算重叠、移除或后移 final replication all-reduce。

## 更保守的一句话版本

基于 Nanotron 构建 Qwen3-MoE-style 预训练 infra，完成 DP/TP/PP/EP 训练验证、checkpoint/resume、日志解析与 profiling；实现 EP2 token all-to-all dispatch 和 expert buffer coalesce，并在 4 x RTX 3090 上完成 DP4/TP2+DP2/EP2+DP2 对比，EP2+DP2 达到 48.5K tokens/s、峰值显存 1.94GB/GPU。
