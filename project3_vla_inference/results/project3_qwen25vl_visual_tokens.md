# 辅助实验：Qwen2.5-VL 多相机输入成本

## 这不是 VLA 策略实验

本实验的作用是量化一个通用视觉语言模型在多相机输入下的视觉 token 与 prefill 成本，
为 Pi0.5/VLASH 主线解释“多相机观测为什么会成为推理系统瓶颈”。它不加载 Pi0.5，
不生成机器人动作，也不用于评价 VLASH。

## 实验条件

| 项目 | 设置 |
| --- | --- |
| 模型 | `Qwen/Qwen2.5-VL-3B-Instruct` |
| GPU | RTX 4080 SUPER，32 GiB |
| 软件 | Python 3.12.3、PyTorch 2.8.0+cu128、BF16、SDPA |
| 输入 | 合成 cabinet 图像 + 固定任务文本；1 或 3 张图像 |
| 分辨率 | 224 或 448；动态像素范围 `3136..802816` |
| 输出长度 | 16 或 64 token |
| 计时 | 预处理、H2D transfer、模型 prefill、decode 分开记录；单进程、非 vLLM serving |

## 关键对比

为避免把不同分辨率或 decode 长度混到一起，以下表格固定 decode=16，选取最能说明
多相机视觉前缀变化的两组：

| 输入 | 视觉 marker token | prefill | 显存峰值 | 说明 |
| --- | ---: | ---: | ---: | --- |
| 单相机，224px | 66 | 38.3 ms | 7.05 GiB | 较小视觉前缀 |
| 三相机，448px | 774 | 166.4 ms | 7.43 GiB | 更长视觉前缀与更高 prefill 成本 |

视觉 marker token 从 66 增至 774，约为 11.7 倍；prefill 从 38.3 ms 增至 166.4 ms，
约为 4.3 倍。显存增加相对温和，表明这组单请求测试的主要变化首先体现在视觉编码和
prefill 时间，而不是峰值显存。

![Qwen2.5-VL visual tokens](../assets/figures/project3_qwen25vl_visual_tokens.svg)

![Qwen2.5-VL prefill latency](../assets/figures/project3_qwen25vl_prefill_latency.svg)

## 与 VLASH 主线的关系

Pi0.5/VLASH 是本项目的真实 VLA 策略链路；这个 Qwen2.5-VL 实验只提供输入侧系统
证据：当 VLA 从单相机扩展到多视角输入时，视觉 token 的增长会显著拉长多模态前缀
计算。因此，实际 VLA serving 需要考虑相机选择、图像分辨率、视觉 token 预算和
prefill batching。

## 边界

- 合成图像用于控制输入形状，不代表 ALOHA 图像内容分布。
- 单进程 Hugging Face 计时不等同 vLLM 并发服务；并发服务见
  [Qwen3-VL vLLM 辅助实验](project3_qwen3vl_vllm_serving.md)。
- 这些数字不能直接套用到 Pi0.5，也不能推断机器人任务成功率。

原始数据：
[`qwen25vl_visual_tokens_dynamic_pixels_sdpa_bf16.csv`](qwen25vl_visual_tokens_dynamic_pixels_sdpa_bf16.csv)。
