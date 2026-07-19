# 项目一简历表述

## 推荐版本

**基于 Nanotron 的 Qwen3-MoE 小型混合并行预训练系统**

- 基于 Nanotron 搭建 108M 参数 Qwen3-MoE-style 预训练基座，覆盖 BF16、FlashAttention-2、GroupedGEMM、Top-2 路由、全局批次负载均衡、激活重计算及 checkpoint/resume，并验证 DP、TP、PP、EP 及组合并行。
- 实现可反向传播的专家 Token All-to-All：按专家归属交换隐藏状态与路由权重，按本地专家重排为连续缓冲区后执行 GroupedGEMM，再将结果返回 Token 所属节点并聚合；补充 Token 分片梯度恢复及 EP 共享参数平均归并。
- 建立通信原语解析梯度测试、EP2 与未切分 8 专家参考等价测试及跨 rank 参数一致性审计，定位并修复普通 All-to-All 截断反向图、TP=1 无效 tied 标记遮蔽 EP 同步等问题；前向/输入梯度/Router 梯度完全一致，20 步训练后 31 个共享参数保持严格一致。
- 在 4×RTX 3090、同等 4096 tokens/step 下，修复后 EP2+DP2 达到 36.1K tokens/s、峰值显存 1.96GiB/GPU，相比 DP4 的 27.7K、2.64GiB/GPU，吞吐提升 30.1%、显存降低约 24%。
- 修复 checkpoint 恢复时学习率调度器构建顺序和重复步进问题，完成 EP2+DP2 从第 100 步恢复至第 102 步，学习率由 1.01e-5 连续衰减至 1.00e-5。

## 一句话版本

基于 Nanotron 实现 Qwen3-MoE 的 DP/TP/PP/EP 混合并行训练，完成可反向传播的专家 All-to-All、连续专家缓冲区和共享梯度同步；在 4×RTX 3090 上使 EP2+DP2 相比 DP4 吞吐提升 30.1%、峰值显存降低约 24%，并通过参数一致性及 checkpoint/resume 测试。

## 面试说明

- 不再使用旧的 48.5K tokens/s。该数值是实际 wall-clock 结果，但对应实现截断了通信反向图并缺少共享梯度同步。
- 修复后的可信结果为 36.07K tokens/s，略低于 TP2+DP2 的 37.23K，但高于 DP4 的 27.73K。
- 项目重点不是证明 EP 永远最快，而是展示从异常性能数据出发，通过梯度和参数审计定位训练语义缺陷，并完成修复、回归和公平复测。
