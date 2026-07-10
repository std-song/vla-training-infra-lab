# 小米 Robotics VLA 训练 Infra 投递材料

面向岗位：VLA 训练 Infra 算法工程师

这份材料把仓库中的三个项目整理成投递和面试叙事。原则是少说“复现了某模型”，多说“围绕训练/推理系统，验证了哪些工程接口、定位了哪些瓶颈、知道下一步该怎么补齐”。

## 简历项目建议

### 项目一：基于 Nanotron 的 Qwen3-MoE-style 混合并行预训练系统

- 基于 Nanotron 搭建 Qwen3-MoE-style 小规模预训练链路，覆盖 BF16、FlashAttention、GroupedGEMM 专家计算、Router Top-k、激活值重计算、断点保存与恢复，并在 1-2 张 RTX 3090 上验证 DP、TP、PP、EP 配置。
- 补齐外层预训练工程：多语料 manifest、离线 tokenizer fallback、定长 packing、packed shard、启动矩阵、日志解析、吞吐/显存图表，使项目不只停留在单个训练脚本。
- 完成 100M 级 Qwen3-MoE-style profiling：single 4.46K tokens/s，DP2 6.95K，TP2 4.20K，PP2 4.56K，EP2 3.25K；同时保留早期 75.5M Qwen2-MoE 4 卡数据：DP4 22.7K，TP2+DP2 20.1K，PP2+DP2 20.5K tokens/s。
- 定位并修复 PP resume / 非最终 stage loss logging 问题；EP2 已跑通到本地专家 dispatch 验证，后续优化方向是 All-to-All dispatch、token buffer 合并和通信计算重叠。

阅读入口：[`project1_qwen3_moe_pretrain/README.md`](../project1_qwen3_moe_pretrain/README.md)

### 项目二：LeRobot/SmolVLA 多模态数据管线与分布式训练 Adapter

- 面向 VLA fine-tuning 构建 LeRobot 数据接入和训练 adapter，解析三相机视频 shard、parquet state/action、task text、episode/frame metadata，并实现 image/state/action/action_mask/task_text 的 batch collation。
- 实现 Nanotron-style DDP training wrapper，用于验证多模态 batch 在分布式训练中的 DistributedSampler、rank-aware checkpoint/resume、all-reduced metrics 与 CUDA memory profiling。
- 建立 official SmolVLA Accelerate DDP baseline 作为真实模型参照；在 2 张 RTX 4080 SUPER 上完成 50-step profiling：DDP baseline 23.6 samples/s，BF16 24.6 samples/s，DataLoader worker 从 0 到 1 将吞吐从 17.2 提升到 23.6 samples/s。
- 分析视频解码、CPU/GPU overlap、GPU update time 与 DDP tuning 的瓶颈边界。这里不声称“完整移植 SmolVLA 到 Nanotron 原生 Trainer”，而是明确 claim 为多模态数据/训练 wrapper 与官方基线对照。

阅读入口：[`project2_smolvla_training/README.md`](../project2_smolvla_training/README.md)

### 项目三：VLM/VLA 推理链路 Profiling 与异步 Action Serving

- 构建三层 VLM/VLA 推理 infra lab：Qwen3-VL/Qwen2.5-VL 视觉语言 serving、Pi0.5/LeRobot 真实 VLA action inference、VLASH-inspired 异步 action queue/control-loop simulator。
- 在 32GiB GPU 上完成 Qwen3-VL-4B vLLM serving baseline：concurrency=8 时 224px 输入达到 10.08 req/s，448px 输入达到 8.73 req/s；基于 Qwen2.5-VL 分析单/三相机输入下 visual marker tokens 从 66 增至 774，prefill 从 40.3ms 增至 166.4ms。
- 对 Pi0.5 `lerobot/pi05_libero_finetuned_v044` 做真实 action chunk profiling，测得 `(1,50,7)` action chunk warm latency 87.7ms、queue pop 3.47ms、峰值显存约 7.3GiB。
- 实现异步 action queue/control-loop simulator：在 30Hz 控制循环下，future-state refill 将模拟反应延迟从 266.7ms 降到 166.7ms，action quantization ratio=2 将模拟控制侧 action overhead 降低约 50%。

阅读入口：[`project3_vla_inferenceence/README.md`](../project3_vla_inferenceence/README.md)

## 面试讲述主线

开场可以这样说：

> 我准备这三个项目不是为了证明模型效果，而是为了对齐 VLA 训练 infra 岗位的核心能力：分布式训练、MoE 并行、多模态数据 pipeline、真实 VLA 微调路径、推理 serving 和控制循环中的 action queue。项目一回答我是否理解 DP/TP/PP/EP/checkpoint 的训练链路；项目二回答多模态机器人数据如何进入训练；项目三回答 VLA 推理为什么不只是一次 VLM forward，还包括 visual token、KV/batching、action chunk、queue 和控制循环延迟。

## 技术追问准备

**为什么 DP4 没有接近 4 倍扩展？**

模型规模小、per-rank batch 小，3090/PCIe 互联弱，通信和框架开销占比高。DP 能验证梯度同步和 checkpoint 正确性，但这个规模不适合证明线性 scaling。

**TP/PP 为什么不一定更快？**

TP/PP 在这里主要是容量切分和并行链路验证。TP 引入 tensor collective，PP 引入 stage imbalance 和 bubble；小模型、小 microbatch 下，通信和调度开销容易吃掉收益。

**EP 当前到底完成到什么程度？**

EP2 已验证到本地专家 dispatch 路径，能说明 expert placement、global/local expert id、local token count 这些接口问题。尚未声称生产级跨 rank expert All-to-All overlap；这是后续优化方向。

**项目二是不是只跑了官方脚本？**

不是。官方 SmolVLA DDP 是真实模型基线；自己的工作在数据接入、batch contract、action mask、Nanotron-style DDP wrapper、rank checkpoint、all-reduced metrics，以及 worker/BF16/DDP profiling 的瓶颈分析。

**项目三为什么能算 VLA 推理 infra？**

Qwen-VL 层负责分析视觉 token、prefill、KV cache、batching 这些 serving 机制；Pi0.5 层补上真实 VLA action chunk；异步队列层分析 action chunk 如何进入 30Hz 控制循环。三层合在一起，比单纯跑文本 decode 更接近 VLA 推理系统。

## 不要过度声称

- 不说“完成大规模训练”，说“在受限硬件上验证关键训练链路并分析扩展瓶颈”。
- 不说“完整将 SmolVLA 移植到 Nanotron 原生 Trainer”，说“实现 Nanotron-style DP wrapper，并用官方 SmolVLA DDP 作为真实模型 baseline”。
- 不说“实现生产级 vLLM/PagedAttention/VLASH”，说“实现 serving/scheduler simulator，并用 Qwen3-VL/Pi0.5 做实测基线，分析 production gap”。
- 不说“证明机器人任务成功率”，说“验证 action inference compute path、queue latency 和 control-loop scheduling 指标”。

## GitHub 阅读路径

1. [`README.md`](../README.md)：看三项目总览和角色匹配。
2. [`project1_qwen3_moe_pretrain/README.md`](../project1_qwen3_moe_pretrain/README.md)：看分布式训练和 MoE 项目。
3. [`project2_smolvla_training/README.md`](../project2_smolvla_training/README.md)：看 VLA 数据管线和训练 adapter。
4. [`project3_vla_inferenceence/README.md`](../project3_vla_inferenceence/README.md)：看推理 serving 和异步 action queue。

如果只能放两个项目，优先保留项目一和项目二；项目三作为推理和部署理解的加分项。
