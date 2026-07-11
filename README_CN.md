# VLA 训练与推理基础设施实验室

这是为 VLA 训练/推理基础设施岗位准备的三个相互独立的工程项目。重点是训练系统、
多模态数据链路、可复核性能分析与推理控制路径，而不是单纯比较模型精度。

英文说明见 [README.md](README.md)。

| 项目 | 主题 | 中文入口 |
| --- | --- | --- |
| 项目一 | Nanotron Qwen3-MoE 小型混合并行预训练：DP/TP/PP/EP、专家分发、恢复与性能分析 | [项目一](project1_qwen3_moe_pretrain/README.md) |
| 项目二 | LeRobot/SmolVLA 多模态数据管线、官方 DDP 基线与训练适配 | [项目二](project2_smolvla_training/README.md) |
| 项目三 | Pi0.5 + VLASH 推理主线；Qwen-VL 多模态 serving 作为辅助系统分析 | [项目三中文入口](project3_vla_inference/README_CN.md) |

## 项目三如何阅读

项目三最容易因为模型较多而混乱。它的主线始终是 Pi0.5 + 上游 VLASH；Qwen2.5-VL
只用于量化多相机视觉输入的成本，Qwen3-VL + vLLM 只用于量化通用 VLM 并发服务。

从 [项目三中文入口](project3_vla_inference/README_CN.md) 开始，随后阅读：

1. [VLASH 最终复现结果](project3_vla_inference/results/vlash_final/final_vlash_report.md)
2. [实验条件与测量边界](project3_vla_inference/results/vlash_final/experiment_protocol.md)
3. [完整实验地图](project3_vla_inference/docs/experiment_map_cn.md)

大模型权重、数据集、checkpoint、AutoDL 凭据和未整理原始日志不会上传到仓库。
