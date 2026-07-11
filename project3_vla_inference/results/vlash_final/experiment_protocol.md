# VLASH Pi0.5 实验条件与结果阅读说明

本文件定义 `final_vlash_report.md` 中每个数字的来源、测量范围和不应做出的推断。

## 运行环境

| 类别 | 条件 |
| --- | --- |
| GPU | NVIDIA vGPU-32GB，32,760 MiB 显存，单卡 |
| Python / CUDA | Python 3.10.8；PyTorch 2.7.1+cu118；CUDA runtime 11.8 |
| 框架 | LeRobot 0.4.1；上游 VLASH；Pi0.5 基座 `lerobot/pi05_base` |
| 数据 | `lerobot/aloha_mobile_cabinet`，85 个 episode、127,500 frame、三路相机视频 |
| 训练数据后端 | 上游 TorchCodec 视频解码路径；单个 DataLoader worker |
| 硬件边界 | 没有实体 ALOHA 机器人、机械臂控制器、串口/以太网命令链路，也没有实时 30 Hz 控制时钟 |

## 训练设置

| 项目 | 值 |
| --- | --- |
| 训练配置 | 上游 Pi0.5 LoRA + VLASH shared observation 配置 |
| 总 / 可训练参数 | 3,772,830,752 / 153,940,128 |
| 训练步数 | 1,000；batch size 1；单卡 |
| 延迟训练 | `max_delay_steps=8`；每个样本构造 offset 0..8 并共享视觉/语言观测编码 |
| future-state 来源 | 上游默认：非零 offset 使用前一动作作为未来状态代理 |
| checkpoint | 第 1,000 步；策略约 7.48 GiB，优化器约 1.09 GiB |

### 训练图的读法

`vlash_pi05_training.svg` 的 loss 来自每 20 step 一次的上游训练日志：step 20 为
0.413、step 200 为 0.101、step 600 为 0.062、step 1000 为 0.059。它用于检查训练
是否稳定、是否能保存恢复，**不是**任务成功率或泛化能力评测。

日志中的 `updt_s` 是一次训练更新的模型侧时间；稳定区间约 0.63-0.78 s。`data_s`
约 1 ms，说明在本设置中数据 worker/cache 后，训练时的主瓶颈不再是当前 batch 的数据
准备。这个结论不应外推到所有分辨率、数据集或更大 worker 数。

## 离线回放设置

| 项目 | 值 |
| --- | --- |
| 加载模型 | 同一个第 1,000 步 checkpoint |
| 输入 | 从 ALOHA 录像逐帧取得的三相机观测、state 与任务文本 |
| 回放长度 | 每组连续 96 control tick；每个设置在独立进程运行 |
| 运行入口 | 上游 `VLASHAsyncManager.get_action()` |
| 记录字段 | `latency_ms`、`fetch_observation`、`chunk_index`、`queue_active`、`sent_action` |
| 计时范围 | 只包围 `get_action()`；不包含录像读取/解码、磁盘 I/O、机器人通信、执行器执行 |
| 输出动作 | Pi0.5 每次完整前向生成 `(1, 50, 7)`：50 步、7 维动作块 |

三组设置为：同步（`overlap=0`）、VLASH 异步（`overlap=4`）和异步加动作量化比 2。
这里的“异步”严格指上游 manager 的未来状态/预取调度配置。当前公开 manager 在
`get_action()` 内完成策略生成，回放环境没有独立硬件 I/O 可重叠，因此它并不是一个
可以在本机离线测试中直接证明端到端吞吐提升的后台 CUDA worker。

## 图和表的读法

### `vlash_pi05_replay_latency.svg`

该图为对数纵轴。每组 96 个 tick 中，只有 2-3 次需要重新生成或重填动作块，其余
93-94 次是从已生成的 50 步动作块取动作。

| 指标 | 同步 | 异步 | 异步 + 量化比 2 |
| --- | ---: | ---: | ---: |
| 动作块调用 | 2 | 3 | 3 |
| 队列取动作 | 94 | 93 | 93 |
| 队列取动作中位数 | 0.048 ms | 0.045 ms | 0.046 ms |
| `get_action` 均值 | 1802.987 ms | 1821.105 ms | 2815.374 ms |

均值被极少数冷启动/动作块调用支配，不能被当作稳定控制 tick 的延迟。图的价值是说明
动作块生成和队列取动作属于两个量级不同的路径，而非声称异步的原始模型前向更快。

### `vlash_pi05_command_cadence.svg`

同步和异步组都在 96/96 tick 标记 `sent_action`；量化比 2 的组为 48/96。该数据验证
上游管理器的动作发送节奏分支被执行。离线 adapter 没有调用真实 `robot.send_action()`，
所以不能把 48 次标记直接换算成“机器人传输开销减少 50%”。

## 可以得出的结论

1. 上游 VLASH 训练、共享观测、多延迟数据构造、Pi0.5 LoRA checkpoint 和异步管理器
   的策略级调用均已在真实多相机数据上跑通。
2. 动作块机制把绝大多数控制 tick 从完整 Pi0.5 前向转化为内存队列操作；本回放中队列
   操作的中位数约为 0.05 ms。
3. 当前实验没有实体机器人 I/O，不能把离线均值延迟、发送标记或模拟器数值写成实体
   机器人端的加速比例。

## 下一步的严格验证

在真实机器人或带有可控通信延迟的仿真环境中，固定任务、控制频率、初始状态和动作
尺度，比较同步/异步/量化三组的：control-tick stall、action age、命令传输耗时、轨迹
误差和 rollout 成功率。只有在该条件下，才能评价 VLASH 的异步调度和量化是否带来真实
控制收益。
