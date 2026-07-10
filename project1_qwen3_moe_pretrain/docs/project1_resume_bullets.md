# Project 1 Resume Bullets

## 中文简历版本

**基于 Nanotron 的 Qwen3-MoE-style 混合并行预训练系统**

- 基于 Nanotron 搭建 Qwen3-MoE-style 小规模预训练系统，覆盖 BF16、FlashAttention-2、GroupedGEMM 专家计算、Router Top-k、全局批次负载均衡、激活重计算与 checkpoint/resume。
- 适配并验证 DP、TP、PP、EP 多种并行策略；修复 PP 下非最后流水段误访问 loss 指标导致的跨 rank 卡死问题，并完成 PP checkpoint/resume 验证。
- 实现 EP all-to-all token dispatch 路径，完成专家 token 分发、专家侧连续 buffer 组织、GroupedGEMM 计算与结果回传；分析小模型下 expert dispatch 通信开销与显存收益。
- 补齐预训练外层工程链路：多语料 manifest、tokenizer/packing、固定长度 shard、实验矩阵脚本、Nanotron 日志解析、吞吐/显存图表与复现实验报告。
- 在 AutoDL RTX 3090 环境完成 clean profiling：single 4.46K tokens/s，DP2 6.95K，TP2 4.20K，PP2 4.56K，EP2 3.25K；EP2 峰值显存约 1.3GB，为各策略最低。

## 面试展开重点

- 为什么 DP2 不是 2x：小模型场景下 launch、optimizer、logging、同步开销占比高。
- 为什么 TP/PP 不一定提速：TP 引入逐层 collective，PP 需要足够深度和 micro-batch 才能摊薄 bubble。
- 为什么 EP 显存最低但吞吐最低：专家权重被切分，但 token dispatch 和 all-to-all 成为主导。
- PP bug 的本质：loss/metric ownership 只属于最后流水段，非最后 stage 不能构造 `loss_avg.item()`。
- 外层工程价值：把实验从“单次 benchmark”升级成可复现的预训练 infra path。
