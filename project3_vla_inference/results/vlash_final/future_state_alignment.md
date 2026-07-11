# VLASH 未来状态、延迟范围与本次复现边界

## 结论先行

`max_delay_steps=8` 表示策略在训练时见过“首个动作会在当前观测之后 0 到 8 个
**控制 tick** 才开始执行”的条件。它不是动作块长度，也不是“策略只能执行 8 步”。

因此，若真实系统从采样观测到首个动作生效的总延迟为 4 个 tick，推理时使用 `d=4`
的未来状态条件与训练分布一致；若该总延迟长期大于 8 个 tick，则用于首个动作的未来
状态条件已超出训练范围，存在明显的分布外风险，不能指望 VLASH 自动补偿。

## 训练时到底学了什么

对每一个延迟 offset `d`，VLASH 的训练意图是：让策略在“动作不是立刻执行，而是
将在 `d` 个 tick 后生效”的场景下，仍输出与这个**延迟后的决策时刻**一致的动作块。
上游数据集的 action query 为：

```text
[a_(t+d), a_(t+d+1), ..., a_(t+d+49)]
```

也就是说，`d` 不是只改变 state 条件；监督的 50 步动作目标也同步右移 `d` 个 tick。

本次配置为：

```text
max_delay_steps = 8
shared_observation = true
use_state_ground_truth = false   # 上游默认路径
```

- `shared_observation=true`：同一观测会构造 `d=0..8` 的全部延迟分支，并复用视觉/语言
  观测编码，不是每次随机只训练一个 offset。
- `use_state_ground_truth=false`：非零 offset 的 future-state 条件并非由一个独立的
  state-prediction network 产生；上游实现使用前一时刻动作作为未来状态代理。
- 如果启用 `use_state_ground_truth=true`，才是将记录轨迹中的未来状态直接作为该条件。

所以“以**预测未来状态**为输入、以未来实际动作为标签”这句话不够准确。更准确的说法是：
**训练构造了延迟对齐的 future-state 条件和后续动作监督；默认实现以历史动作代理未来
状态，而不是额外学习一个未来状态预测器。** 上游实现见 [VLASH 源码](https://github.com/mit-han-lab/vlash)。

## `overlap=4` 在 50 步动作块中的准确位置

Pi0.5 的 `n_action_steps=50`。上游 `VLASHAsyncManager` 的触发条件是：

```text
chunk_index == n_action_steps - overlap_steps
```

对应的实现可见上游 [`run.py`](https://github.com/mit-han-lab/vlash/blob/main/vlash/run.py)：
`should_launch_next_inference()`、`launch_next_inference()` 和 `run_loop()` 分别定义
触发点、末动作 future-state 代理，以及量化比对 effective overlap 的缩放。

因此我们的普通异步组中：

```text
overlap_steps = 4
触发位置 = 50 - 4 = 46
```

即 manager 正在执行当前动作块的第 46 个索引位置时（还剩索引 46、47、48、49 共 4 个
动作），开始为下一块请求新观测并计算下一动作块。它把当前块的**最后一个动作**作为
future-state 代理，因此新块意图上从当前时刻约 4 个 tick 后开始接管。这正好对应训练的
`d=4` 分支。

量化比为 2 的回放组并不仍是 4。上游 `run_loop` 会计算：

```text
effective_overlap_steps = inference_overlap_steps * action_quant_ratio
                         = 4 * 2 = 8
```

我们自己的 replay adapter 沿用了这个乘法，因此 q2 组传入 manager 的是 8：它在当前
块还剩 8 个内部动作 step 时触发下一块。8 仍在训练范围 `0..8` 内，但已位于边界。

## 推理时怎样对应到 `d=4`

设控制周期为 `T`，在时刻 `t` 采集观测；真实首个动作执行前的总延迟应按下式估计：

```text
delay_ticks = ceil((图像/状态采集 + 预处理 + 策略前向 + 排队 + 命令传输 + 执行器生效) / T)
```

若这个值为 4，则应在推理时为“约 `t+4` 才生效的首个动作”构造相应 future-state
条件。这与训练时 `d=4` 分支匹配。注意 `d` 是离散控制步数：在 30 Hz 下 4 tick 约为
133 ms；在其他频率下对应的物理时间会不同。

动作块的后续 49 个动作天然发生在首个动作之后，因此“动作块会执行 50 步”不意味着
训练必须有 `max_delay_steps=50`。`max_delay_steps` 约束的是**开始执行前的未来状态
偏移**，不是动作块的预测长度。

## 超出 `d=8` 会怎样

若实际总延迟为 9、12 或更多 tick：

1. 策略仍可能输出动作，但其 future-state 条件已经超出本次训练覆盖的 `0..8`。
2. 这不是“完全没有作用”，而是没有训练分布保证；状态代理误差和 action age 都可能
   快速积累。
3. 更稳妥的工程策略是监控实际 `delay_ticks`，超过阈值时触发同步重规划/丢弃陈旧块，或
   将 `max_delay_steps` 扩展到经测量的 P95/P99 延迟范围后重新训练。

## 本次复现是否正确

**软件复现是正确的，但真实时间对齐尚未被验证。**

本次已经做到：

- 训练配置确实覆盖 `d=0..8`，并使用 upstream shared-observation 路径。
- 离线回放确实调用 upstream `VLASHAsyncManager` 的同步、异步和量化配置。
- 普通异步组的 `overlap=4` 位于训练延迟范围 `0..8` 内；q2 组的有效 overlap 为 8，
  也仍在范围内但恰好位于边界。

本次还没有做到：

- 回放没有 30 Hz 真实时钟、实体机器人通信或执行器反馈，不能测出真实的
  `delay_ticks`。
- 公开 manager 在 `get_action()` 内完成策略生成；本地离线 adapter 没有独立 I/O 与其
  重叠。因此没有验证“在真实 `t+4` 时使用预测状态并执行首个动作”的闭环语义。

要完成严格验证，需要在真实机器人或带有显式虚拟时钟的仿真环境里，逐 tick 记录采样
时刻、inference finish、enqueue、send、actuator apply；再检查实际首个动作延迟是否始终
落在训练范围，并比较 `d=0/4/8` 与超过范围时的 action age、轨迹误差和任务成功率。
