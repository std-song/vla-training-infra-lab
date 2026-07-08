# Xiaomi Robotics VLA Training Infra Application Pack

面向岗位：VLA 训练 Infra 算法工程师

这个文档把仓库中的三个项目整理成投递和面试材料。原则是：少说“我复现了某某模型”，多说“我围绕 VLA 训练/推理系统，验证了哪些工程接口、测了哪些瓶颈、知道哪些边界”。

## 一、简历项目描述

### 项目一：Nanotron Qwen2-MoE 分布式训练系统与并行策略分析

- 基于 Nanotron 构建 Qwen2-MoE 小规模预训练流程，在 1/2/4 张 RTX 3090 上验证 BF16、FlashAttention、GroupedGEMM expert MLP、router top-k、activation recomputation、checkpoint/resume，以及 DP、TP、PP 和组合并行配置。
- 完成 75.5M MoE 模型的稳定性和性能 profiling：单卡 10.5K tokens/s，DP4 22.7K tokens/s，TP2+DP2 20.1K tokens/s，PP2+DP2 20.5K tokens/s；分析 DP 非线性扩展、TP 通信开销、PP bubble 与 pipeline stage imbalance。
- 对 EP2+DP2 做 readiness 验证，定位 blocker 到 MoE local expert token dispatch / expert id 映射 / GroupedGEMM token count 适配，明确区分“已验证 DP/TP/PP 训练链路”和“EP 仍需实现 token dispatch”的工程边界。

### 项目二：LeRobot/SmolVLA 多模态数据 Pipeline 与分布式微调 Adapter

- 围绕 LeRobot/SmolVLA 构建 VLA fine-tuning infra 原型，完成 `aloha_mobile_cabinet` 三相机视频 shard 解析、parquet state/action 加载、task text 映射、action mask batch collation、device transfer、checkpoint/resume 和吞吐/显存 profiling。
- 实现 Nanotron-style DP training wrapper，覆盖 `torchrun`、NCCL DDP、`DistributedSampler`、rank-aware checkpoint、all-reduced metrics 与 CUDA memory profiling，并与官方 LeRobot/SmolVLA Accelerate DDP baseline 对照，作为真实模型路径参照。
- 在 2 张 RTX 4080 SUPER 上对官方 SmolVLA DDP 做端到端 profiling：DataLoader `num_workers=1` 将吞吐从 17.20 提升到 23.63 samples/s，BF16 提升到 24.61 samples/s；分析视频解码、CPU/GPU overlap、DDP unused parameter search 与 GPU update time 的瓶颈边界。

### 项目三：多模态 VLM/VLA 推理链路 Profiling 与异步 Action Serving 分析

- 构建三层 VLM/VLA 推理 infra lab：Qwen3-VL/Qwen2.5-VL 视觉语言 serving、Pi0.5/LeRobot 真实 VLA action inference、VLASH-inspired 异步 action queue/control-loop simulator，拆分 visual token、multimodal prefill、decode、KV cache、batching、action queue 和 state staleness 指标。
- 在 32GiB GPU 上完成 Qwen3-VL-4B vLLM serving baseline，concurrency=8 时 224px prompt 达到 10.08 req/s、448px prompt 达到 8.73 req/s；同时基于 Qwen2.5-VL 分析单/三相机输入下 visual marker tokens 从 66 增至 774、prefill 从 40.3ms 增至 166.4ms。
- 对 Pi0.5 `lerobot/pi05_libero_finetuned_v044` 做 action chunk profiling，测得 `(1,50,7)` action chunk warm latency 87.7ms、queue pop 3.47ms、峰值显存约 7.3GiB；进一步实现 VLASH-inspired async simulator，在 30Hz 控制循环下 future-state refill 将反应延迟从 266.7ms 降到 166.7ms，action quantization ratio=2 将模拟控制侧 action overhead 从 1237.7ms 降到 618.9ms。

## 二、简历压缩版

如果简历空间紧张，可以写成三条：

- **Nanotron Qwen2-MoE 分布式训练 Infra**：在 1/2/4 张 RTX 3090 上构建 Qwen2-MoE 小规模预训练流程，验证 BF16、FlashAttention、GroupedGEMM、router top-k、activation recomputation、checkpoint/resume、DP/TP/PP 及组合并行；完成 75.5M MoE scaling profiling，DP4 达到 22.7K tokens/s，并分析 DP scaling、TP 通信、PP bubble 与 EP token dispatch blocker。
- **LeRobot/SmolVLA 多模态训练 Pipeline**：构建 VLA fine-tuning infra 原型，覆盖三相机视频 shard、parquet state/action、task text、action mask loss、DDP metric reduction、checkpoint/resume；对比官方 SmolVLA Accelerate DDP 与 Nanotron-style DP wrapper，并将 DataLoader worker/BF16/DDP tuning 的吞吐从 17.20 提升到 24.61 samples/s。
- **VLM/VLA 推理 Serving 与 Action Queue 分析**：构建 Qwen3-VL/Qwen2.5-VL serving、Pi0.5 action inference 与 VLASH-inspired async control-loop 三层实验；Qwen3-VL vLLM concurrency=8 达到 10.08 req/s，Pi0.5 action chunk warm latency 87.7ms，future-state async refill 将 30Hz 控制循环反应延迟从 266.7ms 降到 166.7ms。

## 三、3-5 分钟项目讲述稿

### 开场

我准备这组三个项目的目标不是追求模型效果，而是对齐 VLA 训练 infra 岗位里最核心的几类问题：分布式训练、MoE 并行、多模态数据 pipeline、真实 VLA 微调路径、推理 serving 和控制循环里的 action queue。

第一个项目偏 LLM/MoE 训练系统，回答“我是否真的理解 DP/TP/PP/MoE/checkpoint 的训练链路”。第二个项目偏 VLA 数据和训练 adapter，回答“我是否理解 LeRobot/SmolVLA 这类多模态机器人数据如何进入训练”。第三个项目偏推理 infra，回答“VLA 推理为什么不只是一个 VLM forward，而还包括 visual tokens、KV/batching、action chunk、queue 和控制循环延迟”。

### 项目一怎么讲

项目一我基于 Nanotron 跑 Qwen2-MoE 小规模预训练。模型规模不大，约 75.5M 参数，但训练链路覆盖了 BF16、FlashAttention、GroupedGEMM expert MLP、router top-k、checkpoint/resume，以及 DP、TP、PP 和 4 卡组合配置。

我最关注的不是单个吞吐数字，而是不同并行策略为什么表现不同。比如单卡大约 10.5K tokens/s，DP4 是 22.7K tokens/s，没有线性扩展。原因是模型太小、per-rank batch 小、PCIe 3090 通信弱，梯度 all-reduce 和 Python/framework overhead 很快暴露出来。TP2+DP2 和 PP2+DP2 都比 DP4 慢一点，但它们的价值是容量切分：TP 降低单卡参数/激活压力，PP 降低单 rank memory，但会引入 tensor collective、pipeline bubble 和 stage imbalance。

EP 我没有包装成“完成”，而是诚实地做了 readiness 分析。EP2+DP2 的合理配置是可以启动到 MoE 路径的，但 blocker 出现在 local expert token dispatch：router 给的是 global expert id，GroupedGEMM 需要 local expert token counts，需要补 global-to-local expert mapping、EP group token exchange、aux loss aggregation 和 checkpoint 适配。这一点反而能说明我知道 MoE 并行不是改一个配置项。

### 项目二怎么讲

项目二是把问题从文本 MoE 训练切到 VLA fine-tuning。Nanotron 本身主要围绕纯文本 LLM，所以我没有硬说“完整把 SmolVLA 接进 Nanotron Trainer”，而是做了两个路径。

第一条是真实 baseline：官方 LeRobot/SmolVLA Accelerate DDP，用真实模型链路做 50-step profiling。第二条是我自己的 Nanotron-style DP wrapper，重点验证 VLA batch contract 和 distributed training surfaces，比如 `DistributedSampler`、rank-aware checkpoint、all-reduced metrics、action mask loss 和 memory profiling。

数据侧我做了三相机视频 shard 解析、parquet state/action 加载、task text 映射和 batch collation。性能侧看到 DataLoader `num_workers=1` 可以把吞吐从 17.20 提升到 23.63 samples/s，BF16 到 24.61 samples/s。这个项目的核心收获是：VLA 训练瓶颈不只在 GPU，还经常在视频解码、多相机 sample 拼接、CPU/GPU overlap 和 batch contract 设计。

### 项目三怎么讲

项目三是推理 infra。我一开始用 Qwen2/Qwen3 小模型做 decode profiling，但后来意识到这不够 VLA，所以补成三层。

第一层是 VLM serving：用 Qwen3-VL/Qwen2.5-VL 测 visual token、prefill、decode、KV footprint 和 batching。比如从单张 224 到三张 448，visual marker tokens 从 66 到 774，prefill 从 40.3ms 到 166.4ms。这说明 VLA/VLM 的请求调度必须关注视觉 token shape，不是只有文本长度。

第二层是真实 VLA policy inference：用 LeRobot Pi0.5 checkpoint，跑出 `(1,50,7)` action chunk，warm latency 87.7ms，queue pop 3.47ms。这个结果说明 Pi0.5 不是每个控制 tick 都全模型 forward，而是 full model 生成 action chunk，后续 tick 从 queue 里 pop action。

第三层是 VLASH-inspired async control-loop simulator。我用实测 Pi0.5 latency 作为输入，模拟 30Hz 控制循环，比较 naive async、future-state refill 和 action quantization。结果是 future-state refill 把反应延迟从 266.7ms 降到 166.7ms，action quantization ratio=2 把模拟 action overhead 减半。这个项目最后想表达的是：VLA 推理加速不是只看 tokens/s，而要看 visual prefill、batching、action chunk、queue staleness 和控制循环是否会阻塞。

### 收尾

这三个项目都有边界：模型规模小，SmolVLA 不是完整 Nanotron 原生 Trainer port，推理部分也不是生产级 PagedAttention/VLASH 实现。但我刻意把边界写清楚，因为 infra 岗位更看重能不能定位系统瓶颈、解释 tradeoff、知道下一步工程该补哪里。

## 四、技术追问准备

### 分布式训练

**Q：为什么 DP4 没有接近 4 倍扩展？**

A：模型只有 75.5M，单步计算量小，per-rank microbatch 也小，通信和框架 overhead 占比高。DP 每步都要 gradient all-reduce，3090/PCIe 环境没有高速互联，所以 DP2 还能到 85% 左右效率，DP4 会明显下降。这个结果说明 DP 链路正确，但小模型不适合用来证明线性 scaling。

**Q：TP/PP 为什么没有比 DP 更快？**

A：TP/PP 在这个实验里主要是 capacity parallelism，不是 speedup。TP 会减少每 rank matmul 大小，但引入 forward/backward 内部 collective；小 hidden size 下通信吃掉收益。PP 会降低每 rank memory，但 4 层模型 stage 很难均衡，而且 microbatch 少，pipeline bubble 明显。

**Q：EP blocker 本质是什么？**

A：EP 需要把 router 的 global expert id 映射到每个 EP rank 的 local expert id，并把 token dispatch 到对应 expert owner。GroupedGEMM 接收的是 local expert token counts，而不是 global expert counts。当前 blocker 正是在 local expert accounting / token dispatch 这一层，不是启动参数问题。

### VLA 训练数据

**Q：SmolVLA 为什么不能直接套 Nanotron？**

A：Nanotron 原生抽象偏 causal LM 文本训练，VLA 需要多模态 batch：图像/视频、状态、动作、action mask、task text、episode metadata。loss 也不是纯 LM loss，动作维度和 mask 需要单独处理。因此短期更合理的是先做 adapter/wrapper，验证数据和 distributed surfaces，再考虑深度接入 Nanotron Trainer。

**Q：项目二有什么自己的工作，不只是跑官方脚本？**

A：官方 DDP 是真实模型 baseline。我自己的工作主要在数据接入和 infra 验证：视频 shard resolver、parquet state/action 加载、VLA batch contract、action mask loss plumbing、Nanotron-style DDP wrapper、rank checkpoint/all-reduced metrics，以及 worker/BF16/DDP tuning 的瓶颈分析。

**Q：为什么 DataLoader worker 能提升吞吐？**

A：VLA batch 需要视频解码、多相机帧读取、state/action 拼接和 task text 处理。如果 `num_workers=0`，CPU 数据准备和 GPU update 串行，GPU 等数据。适当增加 worker 后 CPU 和 GPU 有 overlap，所以吞吐从 17.20 到 23.63 samples/s。

### 推理 Serving

**Q：为什么 Qwen3-VL/Qwen2.5-VL 能支持 VLA 推理分析？**

A：Qwen3-VL/Qwen2.5-VL 不是 robot policy，但能真实反映视觉 token、multimodal prefill、KV footprint、batching 对 serving 的影响。Pi0.5 则补上真实 VLA action head/action chunk 路径。两者组合起来分别覆盖 VLM serving 和 VLA policy inference。

**Q：VLA 推理和普通 LLM decode 最大区别是什么？**

A：普通 LLM serving 主要围绕 prefill/decode/KV cache/tokens/s。VLA 推理还要考虑多相机 visual token、action horizon、action queue、控制频率、state staleness 和安全边界。Pi0.5 这类 chunk policy 不是每 tick full forward，而是生成 action chunk 后排队执行。

**Q：你实现了真正的 PagedAttention 或 VLASH 吗？**

A：没有，当前是 PagedAttention-style block manager / scheduler simulator 和 VLASH-inspired async control-loop simulator。它们用于分析机制和瓶颈，不是生产 CUDA kernel 或真实机器人 deployment。我会明确区分 measured model benchmark、simulator 和未完成的生产实现。

**Q：为什么 visual input cache 收益不如 microbatching？**

A：visual input cache 主要减少 CPU preprocessing 和 CPU-to-GPU transfer，但 generation/prefill 仍然占主要时间。same-shape microbatching 可以把多个请求合成一次模型执行，提高 GPU 利用率，所以收益更大。

## 五、GitHub 阅读路径

面试官如果只有 3 分钟，建议看：

1. [`README.md`](../README.md)：先看整体项目定位、三项目覆盖面和关键图。
2. [`docs/scaling_analysis.md`](scaling_analysis.md)：看项目一对 DP/TP/PP/EP 的分析能力。
3. [`results/project2_final_report.md`](../results/project2_final_report.md)：看 VLA 数据 pipeline 和 SmolVLA DDP profiling。
4. [`results/project3_final_report.md`](../results/project3_final_report.md)：看 VLM/VLA 推理三层结构和最终数据。

如果面试官愿意深入：

- 项目一：[`results/qwen2_moe_4gpu_composition.md`](../results/qwen2_moe_4gpu_composition.md)
- 项目二：[`docs/project2_adapter_design.md`](project2_adapter_design.md)、[`results/project2_official_smolvla_bf16_profile.md`](../results/project2_official_smolvla_bf16_profile.md)
- 项目三：[`results/project3_qwen3vl_vllm_serving.md`](../results/project3_qwen3vl_vllm_serving.md)、[`results/project3_pi05_vla_action_inference.md`](../results/project3_pi05_vla_action_inference.md)、[`results/project3_vlash_async_control_loop.md`](../results/project3_vlash_async_control_loop.md)

## 六、面试时不要过度声称的点

- 不说“完成了大规模训练”，说“在受限硬件上验证训练 infra 关键链路并分析扩展瓶颈”。
- 不说“完成了 Nanotron 原生 SmolVLA 训练”，说“实现 Nanotron-style DP wrapper，并用官方 SmolVLA DDP 作为真实模型 baseline”。
- 不说“实现了生产级 vLLM/PagedAttention/VLASH”，说“实现 serving/scheduler simulator，并用 Qwen3-VL/Pi0.5 做实测基线，分析下一步 production gap”。
- 不说“验证了机器人任务成功率”，说“验证了 action inference compute path、queue latency 和 control-loop scheduling 指标”。

## 七、最终推荐投递版本

如果只能放两个项目，我建议保留项目一和项目二，因为岗位 title 是 VLA 训练 infra。项目三可以放在“补充项目/开源项目”里，作为推理和部署理解的加分项。

如果简历允许三个项目，顺序建议：

1. Nanotron Qwen2-MoE 分布式训练系统与并行策略分析
2. LeRobot/SmolVLA 多模态数据 Pipeline 与分布式微调 Adapter
3. 多模态 VLM/VLA 推理 Serving 与 Action Queue 分析

这个顺序最贴小米岗位描述：先训练框架，再 VLA 数据/微调，再推理加速和机器人控制侧理解。

