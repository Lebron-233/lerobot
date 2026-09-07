# L7 阶段结果：冻结的新域预测器已在真实异步控制中运行

日期：2026-09-07。前置结果为 [L4–L6](LEISAAC_SO101_L4_L6_RESULTS.md)，
本轮执行协议见 [L7 runtime plan](LEISAAC_SO101_L7_RUNTIME_PLAN.md)。
操作通过本机 DevSpace 完成；未依赖另一个 Pro 审阅。

## 结论

**新 SO101 predictor 不再只是离线模型：它已经进入原 production worker、
delay planner 和 scheduled queue，并在真实 SO101 物理环境中完成三次有界运行，
其中一次连续 1,800 步／60 秒。** 三次 predicted 测量共 3,000 个实际动作、
110 次 predictor 调用、110 次真实接管，没有控制完整时隙丢失、队列欠载、
推理 deadline miss、cap exceedance 或 stale result。

**整体六次工程试验不是全部通过。** 四次 600 步试验通过；随后 predicted
长试验完成 1,800 步并到达任务 timeout；同 seed 的 identity 长试验在完成
1,266 步后因腕目标超限终止。六次合计 **5,466 个已完成物理控制步**。
所有失败和慢 tick 均保留，未裁剪动作、放宽限位或重跑失败臂。

本轮通过的是新 predictor 的**有界在线可实施性**。没有获得官方完整任务成功，
没有证明任务成功率提高，也没有证明无限期、任意负载下的硬实时保证。

## 1. 实际实现变化

### 1.1 新候选绑定，不冒充旧 SO100

`leisaac_so101_predicted.py` 只对原 engine 的候选验证方法做专门化。
其余 worker、startup probe、队列、延迟规划、d0、residual application、late
whole-discard 和 epoch 逻辑仍使用同一个生产实现。旧公共 SO100 路径保留原验证。

绑定并实际加载的对象为：

- `wsagi/SmolVLA-PickOrange@c8c3318dba152b0ba671ff07b4314418d5aa4b4a`，
  其原生 front/wrist schema、自带 state/action statistics 和同一 pre/post 实例；
- L6 `best.pt`，epoch 3，训练 source
  `40e00e1654779d5e6bfec8f2f06ab57a6c697ff1`，**273,824 参数**；
- 当前 model-ready normalized state（32 维）、队列中 postprocessor **之前**的
  normalized 6 维已承诺动作、native 图像 tokens 和 delay。没有输入未来 state。

构造函数拒绝别的 epoch、训练来源、域、相机配置或 processor 实例。新类没有调用
旧 SO100 binding 去伪造身份。checkpoint 保存的 `enabled=false` 是训练配置字段；
在线调用由 `context_mode=predicted` 决定，逐请求真实调用计数见下文。

SmolVLA、VLM、action expert 与 L6 predictor 均冻结，本轮没有重新拟合、选模、
调阈值或重开 L6 的 test trajectories，risk thresholds 继续为 null。

### 1.2 找到并修正适配入口的节拍问题

上一轮 221-tick failure 中，平均工作只有 **24.999 ms**，但开始时间迟到从早期
数毫秒持续累积到 **29.271 ms**，最后一个 **50.016 ms** tick 才越过完整时隙门。
源码解释了这一形状：丢时隙检查使用固定 origin，而原 `CycleTimer.wait()` 在每个
周期重新锚定实际开始时间；短工作 tick 不能偿还前面普通迟到留下的累计偏移。

现在该 adapter 调用原 timer 的可选 absolute deadline，统一使用
`origin + (tick + 1) / 30`。未使用此参数的原调用者保持原行为。每 tick 仍恰好
一次 notify、get、env.step，不快进、跳过动作或删除帧；开始迟到 ≥1/30 秒和
完成时间越过原 2/30 秒窗口的两个拒绝条件均保留。

这是特定固定时隙 adapter 与通用 timer 的接线修正，不是改变物理时间、降低 FPS
或把之前失败改判通过。旧 L5 失败保持不变。

## 2. 执行配置与全部尝试

CPU PhysX + GPU RTX／模型，标准相机和 nominal reset anchor；原资产、零初始姿态、
双路 RGB 640×480／30 Hz、physics dt=1/60、decimation=2、control=30 Hz。
q=.9／margin=1／guard=2／max_late_steps=2／horizon=8；没有 risk gating。
启动和最终 reset-state bootstrap 在测量之前完成并单独计时，没有手工往队列填动作。

短试验实际 source：`09b9fb3f992b9102778683819ea66153ecb150e8`。
长试验实际 source：`fdaf88715f755ce40111010f982c6e5058cbf1b8`，其运行代码与
前者一致，只新增报告工具和预先冻结的长试验协议。

| 顺序 | Env seed / policy seed | Mode | 已完成动作 / 上限 | Planned / 实际接管 | 工程与任务结果 |
| ---: | --- | --- | ---: | ---: | --- |
| 1 | 20260930 / 2001 | identity | 600 / 600 | 22 / 22 | 有界工程 PASS；步数截断，success=null |
| 2 | 同上 | predicted | 600 / 600 | 22 / 22 | 有界工程 PASS；步数截断，success=null |
| 3 | 20261001 / 2002 | predicted | 600 / 600 | 22 / 22 | 有界工程 PASS；步数截断，success=null |
| 4 | 同上 | identity | 600 / 600 | 22 / 22 | 有界工程 PASS；步数截断，success=null |
| 5 | 20261002 / 2003 | predicted | 1,800 / 1,800 | 66 / 66 | 有界工程 PASS；timeout=true，success=false |
| 6 | 同上 | identity | 1,266 / 1,800 | 47 / 47 | technical_failure；success=null |

第六次记录有 1,267 个 attempted ticks，最后一个 `dispatch=not_sent`。
其 `wrist_flex=95.5175628662°` 超过原 +95° 限位，在发送前被拒绝；不能把尝试数
写成已执行动作数，不能把它当作正常任务 timeout 或改写为经过限幅的完成运行。
该失败臂没有重跑。

工程总计 201 个 planned requests／201 个 queue takeover events。全部六个进程
的请求终态完整且唯一，所有日志中的 action index、simulation step、两相机帧号
都与对应的 attempted tick 顺序一致。最后的非法动作只被消费／检查，没有推进物理。

## 3. 为什么这不是同步推理加计时

两次短 predicted 运行各有 **22** 次测量 predictor 调用，长运行有 **66** 次，
合计 **110**，不包含额外三个 startup probe 调用。每个 d>0 的测量 planned
request 都恰好调用 predictor 一次，delay 分布为 d7 **108** 次、d8 **2** 次。

通过原始 request/get/tick 时间戳交叉核对，至少 **552** 个完整的旧队列动作
在对应预测请求尚未完成时已经 get 并完成真实物理 step。计数还要求动作索引小于
该请求的 takeover_index；跨越推理完成边界的物理 step 没有算入这个保守计数。
其中长 predicted 运行贡献 **331** 步。

这些记录建立了实际链条：当前观测发起请求 → 旧动作在推理期间推动环境 →
预测 residual 输入冻结 action expert → 新 chunk 先 stage，再在指定索引接管。
没有停止物理时钟等待每次推理，也没有用同步 select_action 的耗时替代异步证据。
接口 fixture 还直接验证了非零 residual 进入新 chunk、normalized prefix 与真实接管。

## 4. 实际并发时间表现

以下是所有保留 tick 的主机端实际工作耗时，未剔除最慢 tick；P90 使用 linear quantile。

| Mode / seed | Mean work (ms) | P90 work (ms) | Max work (ms) | 最大开始迟到 (ms) | Work 超 33.333 ms 次数 |
| --- | ---: | ---: | ---: | ---: | ---: |
| identity / 20260930 | 25.233 | 30.374 | 42.212 | 10.034 | 14 / 600 |
| predicted / 20260930 | 24.965 | 30.072 | **65.719** | **32.399** | 18 / 600 |
| predicted / 20261001 | 25.384 | 30.724 | 48.136 | 14.964 | 15 / 600 |
| identity / 20261001 | 25.627 | 30.996 | 49.152 | 25.262 | 21 / 600 |
| predicted / 20261002 | **24.614** | **30.091** | 51.900 | 18.840 | 44 / 1,800 |
| identity / 20261002 | 24.981 | 29.770 | 46.077 | 12.759 | 31 / 1,266 完成步 |

长 predicted 从第一 tick 开始到最后一个 step 返回为 **59.991977942 秒**，
推进了 60 秒模拟控制时间。对应测量 planned chunk total P90 为 **180.030869 ms**，
66 个真实接管全完成，欠载／推理晚到／cap exceedance／stale result 均为零。

三次 predicted 共 110 个测量 forward 的 host wall：mean **2.528109 ms**、
P90 **3.089282 ms**、max **4.496785 ms**。这是模拟器与 policy 并发时的观测值；
不把它和上一轮无模拟器的独立 forward 计时混成同一个 benchmark。

最慢短试验 tick 很接近原完整丢槽阈值，因此本结果不是 zero-jitter、最坏情况
保证或任意时长稳定性。六次试验中没有发生 timing lost-slot，但第六次仍因动作
可行性而工程失败，整体 `all_bounded_runtime_gates_pass=false`。

## 5. 限制与科学解释

同 seed 的初始关节状态、物体位置、相机位姿可匹配，但渲染 RGB 并不逐位相同。
第一对初始 front/wrist 的 uint8 pixel MAE 分别为 **0.880971／0.986121**，
bootstrap 动作在任何 predicted 接管前已经不同。因此，不能将整个后续轨迹差异、
identity 超限而 predicted 未超限，归因为 predictor 的安全性或因果任务收益。

本轮不是成功率 benchmark。四次短运行是预定步数截断；长 predicted 正常到时
但未触发官方 success；长 identity 是技术终止。没有隐藏不利臂、合并异质终态或
宣称机器人成功率上升。L6 的离线 latent 改善与异质 action-oracle 信号保持原结论。

下一研究重点是固定 base-policy 的动作可行性／任务能力，以及在新独立数据上检验
动作相关的 predictor 收益。不能用已经打开的 L6 test 或本轮开发轨迹选择模型后
仍称它们为盲测；不能直接裁剪动作而忽略 normalized committed prefix 的语义变化。

## 6. 产物、测试与可复现入口

本地 artifact root：`/home/rp/Workspace/SmolVLA_RTC/artifacts/`。
四次短运行目录为 `m54l7_09b9fb3f_{identity|predicted}_seed{20260930|20261001}_v1`；
两次长运行目录为 `m54l7_fdaf8871_{identity|predicted}_seed20261002_v1`。
各目录保存运行前 manifest、result、完整 events/ticks、simulator.log 和稀疏 RGB 观测。
完整未舍入汇总为 `m54l7_full_runtime_summary_v1.json`；仓库机器摘要见
[LEISAAC_SO101_L7_SUMMARY.json](LEISAAC_SO101_L7_SUMMARY.json)。

本轮 **209 个受影响的 engine/timer/interface tests 通过**，lint/format 通过。
它们覆盖旧 frozen 拒绝、旧 d0/late 规则、新候选隔离、非零预测接管与绝对节拍。
报告工具另外对原始结果检查请求终态完整唯一、索引和时隙。没有重跑旧科学实验。

运行入口继续是 `examples/advanced/predictive_async/eval_leisaac_so101.py`；
新预测路径通过 `--mode predicted --matched-snapshot <WSAGI exact snapshot>
--predictor ../artifacts/m54l6_predictor_training_v1/best.pt --sim-device cpu
--camera-backend standard` 显式选择，其他参数以每次 manifest 和预注册协议为准。
结果复核入口为 `summarize_so101_runtime.py`，它只读取本轮产物，不触碰 L6 test cache。

全部六次执行均已结束，worker 已 join、metrics 已关闭、模拟器正常退出。
热路径 telemetry 保留在内存，控制停止后落盘。没有留下自动重试或后台训练任务。

**阶段裁决：新 SO101 predictor 的生产异步有界运行已验证，最长 60 秒；
全项目任务级有效性／安全性尚未通过，identity 长臂超限失败完整保留。**
