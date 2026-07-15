# 项目三实验地图：每个实验做什么，如何连接

## 总目标

项目三要回答的不是“某个模型好不好”，而是一个 VLA 推理系统从多相机观测进入，到动作
真正进入控制循环时，哪些部分消耗时间和显存，以及如何正确评估优化是否有效。

为避免把不同类型的结果混为一谈，实验分成三个层次：

| 层次 | 是否构成最终项目结论 | 核心对象 |
| --- | --- | --- |
| A. 真实 VLA 主线 | 是 | Pi0.5 + 上游 VLASH + ALOHA 数据 |
| B. 多模态服务辅助实验 | 是，但只支撑输入/服务分析 | Qwen2.5-VL、Qwen3-VL + vLLM |
| C. 方法探索与模拟 | 否，只保留方法和边界 | Qwen2/3、缓存/调度模拟、Triton 微基准 |

## A. 最终主线：Pi0.5 + VLASH

| 顺序 | 实验 | 输入与设置 | 输出 | 可以得出的结论 | 不能得出的结论 |
| ---: | --- | --- | --- | --- | --- |
| A1 | Pi0.5 动作推理基准 | Pi0.5 checkpoint、图像/state/task；产生 `(1,50,7)` 动作块 | warm action chunk 约 87.7 ms、显存约 7.3 GiB | VLA 推理不是文本逐 token decode，而是动作块生成与消费 | 真实任务成功率 |
| A2 | VLASH LoRA 微调 | ALOHA 85 episode、三相机、offset 0..8、shared observation、1,000 step | loss 0.413 到 0.059；可加载 checkpoint | 上游训练、数据、LoRA 与 checkpoint 链路跑通且稳定 | 已充分收敛或泛化能力 |
| A3 | 上游 manager 离线回放 | A2 checkpoint、96 个记录轨迹 tick；sync/async/q2 | 动作块重填次数、队列取动作、发送节奏 | 动作队列、future-state 路径、量化发送分支真正执行 | 实体机器人端时延或加速比例 |
| A4 | LIBERO 仿真闭环 | Standard/Stale/Learned 三组 5,000-step LoRA；task 3、10 个配对初态；固定延迟消融 | 成功率、完成步数、完整策略时延、动作交接 L2、队列欠载、bootstrap 区间 | 完整策略调用约 350 ms；`d=4` 动作跳步相对朴素延迟成功率 10% 到 50%，区间 `[+10,+70]` 个百分点 | 实体机器人成功率；未来状态必然提升闭环性能 |
| A5 | D4 延迟训练与扫描 | Stale-D4/Learned-D4，5,000-step LoRA，训练和评测覆盖 `d=0..4`；10 个配对初态 | 三条延迟曲线、同权重 future/stale state 消融、20,000 次 bootstrap | Learned-D4 从同步 50% 降至 `d=4` 的 30%；预测状态使交接 L2 改善 1.5%--5.6%，但没有成功率收益 | Learned-D4 相对 Stale-D4 的差异全部来自状态预测器；小样本趋势具备统计显著性 |

**必须先读：** [最终报告](../results/vlash_final/final_vlash_report.md) 和
[实验条件](../results/vlash_final/experiment_protocol.md)。
Future-state 与真实控制延迟的对齐边界见
[未来状态对齐说明](../results/vlash_final/future_state_alignment.md)。
LIBERO 闭环条件、配对结果与统计边界见
[闭环时延实验](../results/libero_standard_delay_ablation/README_CN.md) 和
[D4 延迟闭环扫描](../results/libero_d4_delay_sweep/README_CN.md)。

### 为什么会同时看到 87.7 ms 和几十秒

这两个数字来自**不同实验条件**，不能横向相减或比较加速比：

- `87.7 ms` 来自 A1 的早期 Pi0.5 warmed microbenchmark：固定合成观测、已完成
  warmup，只测 `predict_action_chunk` 的稳定模型侧路径。
- 最终 VLASH 回放中的几十秒级点来自 A3：每组独立进程首次进入上游 manager，且记录的是
  完整 `get_action()` 路径中的动作块初始化/重填。96 tick 内只有 2-3 个这样的点，其他
  tick 是约 0.05 ms 的队列取动作。

所以最终报告用对数图展示 A3 的“重填 vs 队列”两类路径，而不拿 A1 的 warm 微基准去
宣称 VLASH 的端到端时延。要得到可对比的稳态策略前向指标，需要在同一 checkpoint、
同一输入和统一 warmup 后重新测量。

## B. 辅助实验：为什么保留 Qwen-VL

### B1. Qwen2.5-VL 多相机输入成本

| 内容 | 说明 |
| --- | --- |
| 动机 | VLA 的观测通常来自多相机；需要先量化视觉 token 和 prefill 会如何增长 |
| 模型 | Qwen2.5-VL-3B，不是 Pi0.5，也不是控制策略 |
| 对比 | 单相机与三相机；不同图像大小 |
| 结果 | visual marker token 66 到 774；prefill 40.3 ms 到 166.4 ms |
| 与主线的关系 | 解释多相机 VLA 输入为什么会将系统瓶颈推向视觉编码和 prefill |
| 边界 | 不能据此推断 Pi0.5 的绝对 latency 或 robot success |

报告：[Qwen2.5-VL visual token study](../results/project3_qwen25vl_visual_tokens.md)。

### B2. Qwen3-VL + vLLM 并发服务

| 内容 | 说明 |
| --- | --- |
| 动机 | 了解一个成熟 VLM serving engine 如何处理多请求并发、显存和吞吐 |
| 模型与引擎 | Qwen3-VL-4B-Instruct + vLLM 0.24.0 |
| 测量 | single-image prompt，224/448px，concurrency 1/2/4/8，BF16 |
| 结果 | concurrency 8：224px 为 10.08 req/s，448px 为 8.73 req/s，峰值约 21.3 GiB |
| 与主线的关系 | 提供“若将视觉语言前端做成服务”时的并发与显存参照 |
| 边界 | 不等同 Pi0.5 action inference，也不说明 VLASH 更快 |

报告：[Qwen3-VL vLLM serving](../results/project3_qwen3vl_vllm_serving.md)。

## C. 历史探索：不要与实测主线混用

| 实验 | 当初目的 | 当前定位 |
| --- | --- | --- |
| Qwen2 / Qwen3 text prefill-decode、KV cache、attention backend | 建立 prefill、decode、KV cache 的分析模板 | 方法原型；不进入主线指标 |
| Paged KV、continuous batching、prefix cache、shape-aware scheduler | 探索缓存和批处理设计空间 | 纯模拟；只能讨论假设下的趋势 |
| Triton fused action post-processing | 练习小算子融合与 benchmark | 独立微基准；不能称 VLA 端到端加速 |
| 30 Hz future-state / quantization simulator | 早期描述 control-loop 机制 | 已被真实 VLASH replay 取代；不再用模拟数值作为最终结果 |

历史报告：[project3_final_report.md](../results/project3_final_report.md)。该文件顶部已标明
其中调度器/30 Hz 数值是模拟数据。

## 结果怎么对外讲

面试或简历中按以下优先级讲：

1. **先讲 Pi0.5 + VLASH：** 完整复现了真实 VLA 的训练、动作块和异步管理器调用，能够
   清楚区分模型重填与控制 tick 的队列路径。
2. **再讲 Qwen2.5-VL：** 多相机会显著放大视觉 token 与 prefill，是 VLA 输入侧的系统
   瓶颈依据。
3. **最后讲 Qwen3-VL + vLLM：** 用成熟引擎验证并发服务下的吞吐/显存边界。
4. **主动交代边界：** 早期缓存调度与 30 Hz 数字来自模拟；新增 LIBERO 结果是真实策略
   仿真闭环，但仍没有实体机器人 I/O，因而不外推为实机加速或实机任务成功率。

这套顺序将项目定位为“真实 VLA 路径复现 + 多模态推理系统分析”，而非零散 benchmark
集合。
