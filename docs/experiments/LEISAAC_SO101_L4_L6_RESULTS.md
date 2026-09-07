# L4–L6 阶段结果：新 SO101 predictor 的真实轨迹训练与一次性盲测

日期：2026-09-07。用户已授权本项目相关实现与实验；本阶段通过 DevSpace
在本机执行，正式计划、修复和结果持续记录于 GitHub Issue #1。

## 结论

**新域预测器的离线 forecasting pilot 取得正向结果；尚未证明任务成功率收益，
也没有通过新候选的持续实时闭环门槛。** 这三种结论必须分开。

| 问题 | 实际证据 | 本阶段判断 |
| --- | --- | --- |
| 能否预测真实 SO101 轨迹中的未来视觉 latent？ | 两条未参与拟合／选择的测试轨迹，8,928 个因果样本；SmoothL1 下降 **2.2759%**，两个 episode 的全部八个 delay 均改善 | 本次 L6 离线 pilot 的预注册主条件通过 |
| 更好的 latent 是否让冻结 action expert 更接近未来视觉参考？ | 18 个预注册离线案例，平均动作 L1 偏差下降 **9.7897%**；但只有 **9/18** 案例改善，一个 episode 的动作平均偏差恶化 | 有聚合正向信号，但并不稳定或普遍 |
| 新预测器是否足够轻量？ | **273,824 参数**；独立 CUDA 完成计时，predictor P90 **1.6739 ms**，完整 RGB policy P90 **142.4536 ms**，比值 **1.1750%** | 单独计算开销较小；不是并发控制 GO |
| 新候选是否已经解决真实任务？ | 九次不同开发条件的同步运行均未触发官方完整任务成功 | 仍未解决 |
| production async 是否已经接入真实环境？ | 一次运行执行 **221** 个真实控制步，八个 planned requests、零欠载、零 inference deadline misses；随后触发控制 lost-slot | 实际接线与请求机制已运行，但该实时试验仍为 FAIL |

机器可读摘要：[LEISAAC_SO101_L6_SUMMARY.json](LEISAAC_SO101_L6_SUMMARY.json)。
原始全部 cases、逐步记录、训练日志、权重与失败产物在下述本地目录保留。

## 1. 从旧候选到独立任务候选

旧 SO100 candidate 的首次动作超限失败、旧 M3/B4/M5 结果保持原状，没有把它们
改写为新部署成功。新候选使用 LeIsaac 官方的 motor-range 坐标，而不是旧候选的
物理角度；保留各 checkpoint 自带的 state/action statistics 和 processors。
转换发生在独立适配边界，不伪装成旧 frozen candidate。

本阶段实际加载并评估的三个公开候选为：

| 独立候选 | 精确 revision |
| --- | --- |
| `edge-inference/smolvla-so101-pick-orange` main | `71cf4a9d35ce317f6706efe1a9f9d4cbb2b8fb4d` |
| 同仓库 single-rank | `7f19c683128ed07c31240ea2b29fe61afcb1755b` |
| `wsagi/SmolVLA-PickOrange` | `c8c3318dba152b0ba671ff07b4314418d5aa4b4a` |

真实训练元数据的任务文字为 `Grab orange and place into plate`。新预测器采用第三个
候选，完整 task checkpoint、VLM 和 action expert 均冻结，仅训练新增小模块。
该选择用于新域预测与工程接线，不宣称它赢得了任务成功率比较。

### L4：九次真实任务开发运行

下表全部使用环境 seed 20260911、policy seed 1801。不同条件各自记录，不能视为
九个独立同分布 benchmark trial，也不能把它们合并成总体成功率估计。

| 本地 artifact 目录 | 改变／条件 | 已完成动作 | 真实终态 |
| --- | --- | ---: | --- |
| `m54l4_55af2294_edge_sync_capability_v1` | main，25 秒；早期未启用 native AMP 的实现版本 | 750 | timeout，success=false |
| `m54l4_edge_sync_60s_progress_v1` | main，独立 60 秒条件；仍为早期无 AMP 版本 | 1,800 | timeout，success=false |
| `m54l4_edge_sync_native_amp_v1` | main，按 checkpoint 的 use_amp=true 修正 | 1,800 | timeout，success=false |
| `m54l4_single_rank_sync_native_amp_v1` | single-rank，60 秒，CPU PhysX | 1,800 | timeout，success=false |
| `m54l4_single_rank_gpu_physics_v1` | 同候选／seed，改为 GPU PhysX，明确非实时诊断 | 1,800 | timeout，success=false |
| `m54l4_wsagi_native_120s_v1` | WSAGI 自身配置，独立 120 秒条件 | 3,600 | timeout，success=false |
| `m54l4_single_rank_feedback25_v1` | 生成 50 步，但执行 25 步后重新观测 | 1,800 | timeout，success=false |
| `m54l4_single_rank_native60_v1` | 原生 60 Hz 控制／30 Hz 相机，仅同步时间基诊断 | 3,600 | timeout，success=false |
| `m54l4_single_rank_rest_start_v1` | 原生 rest-pose 中心作为显式 prepared-start 条件 | 1,800 | timeout，success=false |

合计 **18,750 个已完成动作**，没有裁剪动作或放宽限位。早期 AMP 漏用已修正并保留
原始实现版本的结果；没有静默把它们改称 native-precision reproduction。

为排查整体加载／归一化错误，还在原训练 episode 0 的帧 0/150/300 做过固定噪声
teacher-forcing 检查，main 候选的第一步 motor-action MAE 分别为
2.2333 / 0.5866 / 0.5419。它仅是训练帧上的接口 sanity check，不是泛化或任务分数。

## 2. L5：真实 async 接线与重置修复

第一次标准相机 identity 实验在两个 hold-target 控制步后发生 lost-slot。随后修复了
两个实际问题：最终 reset 后先由真实 production worker 生成当前 episode 的初始
chunk，再开始控制计时；以及标准相机的重复 reset 位姿累积。没有手工插入队列动作，
没有把推理晚到或控制丢时隙从记录中删除。

### 相机重置的实测差异

同一环境内，原 tiled backend 的三次同 seed reset 返回相同相机位姿和物体位置，
因此没有凭源码猜测修改 tiled 路径。标准 `Camera` 则实测 front 位移依次累积
**0.0163003865 m、0.0326007730 m**，物体与 wrist 不变。

针对标准路径保存 nominal front 位姿，在每次显式 reset 前恢复，然后仍应用原有
随机化。修复后的三次真实 reset 返回完全一致的相机位置、四元数和物体位置，测得
位移为零。相机 world-pose 读回只保留在 reset 包，控制期仍保持完整图像／帧号／状态。

该局部可重复性结果不保证整个 PhysX／渲染／策略轨迹逐位确定性。

### 221 步 engineering 运行

`m54l5_wsagi_primed_identity_v2` 绑定
`5ab75cc705064df7b3e797238ead8cd48b91a1b2`：CPU PhysX、GPU RTX、标准相机、
30 Hz、环境 seed 20260912／policy seed 1802，预定最多 600 个测量 tick。

实际 221 tick 后触发 `lost_control_slot_during_tick`，结果保持
`technical_failure`，success=null。production stats 为：12 个请求，3 个 bootstrap，
8 个 planned requests，cap exceeded=0、deadline misses=0、underflows=0。
另一个请求属于 startup probe。最终 reset 后的初始 chunk priming 用时 137.4398 ms，
位于控制计时之外并单独记录。sink 已关闭，模拟器正常退出。

这证明真实环境与 production async 请求／队列已经发生交互，不证明持续实时通过。
新候选的 `predicted` production 路径仍未启用；旧 frozen-validator 没有被绕过。

## 3. L6：独立真实轨迹预测实验

完整预注册及执行修订：[L6 predictor plan](LEISAAC_SO101_PREDICTOR_PILOT_PLAN.md)。
首次预注册记录为 [Issue comment 5564781743](https://github.com/Lebron-233/lerobot/issues/1#issuecomment-5564781743)，
模型冻结后、打开测试前的记录为 [5564875274](https://github.com/Lebron-233/lerobot/issues/1#issuecomment-5564875274)。

采用七个事先指定的新环境 seed 20260920–20260926，policy seed 1900–1906。
最终缓存每个 episode 有 600 个真实执行动作和 601 个观测，共 4,200 个缓存动作／
4,207 个观测。状态为 checkpoint 自己的 normalized model-ready 32 维；每帧
视觉 token 为 `[2,64,960]`，保持原始 bfloat16，没有重读旧 B4/test cache。

| Split | Episode | 因果 forecasting pairs |
| --- | --- | ---: |
| Train | 0–3 | 17,856 |
| Validation | 4 | 4,464 |
| Held-out test | 5–6 | 8,928 |

每个样本只读取当前 Z、当前 state、delay 和当前已经承诺的 normalized 动作前缀。
跨越下一次观测生成的新 action chunk 的样本一律排除。目标是实际 `Z[t+d]`，d=1…8；
没有未来 state 或未来观测生成的动作泄漏到输入。原始 RGB 与当前-token override
在第一观测的同噪声动作对照中最大差异为 **0**。

### 收集故障留存

collection v1 已保存前四条训练轨迹，但验证 seed 的腕部目标 96.4203° 超出 95°
上限，拒绝执行。旧收集器未在异常退出时保存该验证轨迹前缀；修正为发送前校验并
记录 `technical_action_limit` 后，用新 namespace 引用原四个缓存，仅重启未持久化
的验证 seed 并继续原定两个 test seed。原失败没有覆盖。

同 seed 续采完成了 600 步，说明不能声称两次模拟轨迹逐位一致。没有换 seed、
裁剪动作或用测试表现选择缓存。4,200 是最终缓存中的动作数，不是包括未持久化失败
前缀在内的所有历史物理调用数；后者不能从缺失的旧前缀日志精确恢复。

### 训练与冻结

复用 `LightweightFutureLatentPredictor`，从随机初始化训练20轮，batch32，AdamW
lr0.001／weight_decay0.0001，gradient clip1，SmoothL1+0.1cosine loss。
rank64、action hidden128、state hidden64、delay embedding32、fusion128、
depthwise1d mixer、两相机、horizon8，risk head 关闭。

仅按 validation SmoothL1 选模型，选中 **epoch3**，参数 **273,824**。
其 validation SmoothL1 为28.51263448→27.83304339，下降2.3835%。
测试仅在该选择冻结后执行一次；之后没有调整模型或利用测试产生 fallback 阈值。

训练／测试 source：`40e00e1654779d5e6bfec8f2f06ab57a6c697ff1`。

## 4. Held-out latent 结果

合并 test SmoothL1：**28.02637873 → 27.38851977，下降2.2759%**。
Cosine distance：**0.047321408 → 0.044978314，下降4.9514%**。
Episode5／6 的 aggregate SmoothL1 分别下降 **1.9781%／2.5557%**。

| d / 30 Hz 下的预测步数 | 样本数 | Identity SmoothL1 | Predicted SmoothL1 | 相对下降 |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 1,200 | 19.96090436 | 19.83886621 | 0.6114% |
| 2 | 1,176 | 24.28059295 | 24.00442179 | 1.1374% |
| 3 | 1,152 | 26.74482589 | 26.29890126 | 1.6673% |
| 4 | 1,128 | 28.56139223 | 27.95277348 | 2.1309% |
| 5 | 1,104 | 29.99317357 | 29.23659984 | 2.5225% |
| 6 | 1,080 | 31.09674275 | 30.21274108 | 2.8427% |
| 7 | 1,056 | 32.14939304 | 31.14515769 | 3.1237% |
| 8 | 1,032 | 32.98299621 | 31.84646235 | 3.4458% |

这不是把测试时表现差的 delay 删除后的结果：两个 episode 的16个 episode×delay
单元均为正向。它仍是两条相关时间序列上的小规模 pilot，不是8,928个独立试验。

## 5. Frozen action expert 对照：聚合有效，但不均匀

预先固定每个 test episode 的 t=0/200/400 与 d=1/4/8，共18例。当前 state 与语言
固定、flow noise 配对。比较当前 tokens／预测 future tokens 所生成的50步动作，
相对于实际 future tokens 所生成动作的 normalized-action L1。未来 state 没有使用。

| 对象 | Identity action L1 | Predicted action L1 | 相对下降 | 改善案例 |
| --- | ---: | ---: | ---: | ---: |
| Episode5 | 0.223816744 | 0.176886307 | **20.9682%** | 6/9 |
| Episode6 | 0.159233198 | 0.168664366 | **−5.9229%** | 3/9 |
| 合并 | 0.191524971 | 0.172775337 | **9.7897%** | **9/18** |

全部案例如下，数值仅为表格显示四舍五入；本地 test report 保留原精度。

| Episode | t | d | Identity L1 | Predicted L1 | 方向 |
| ---: | ---: | ---: | ---: | ---: | --- |
| 5 | 0 | 1 | 0.516360 | 0.381788 | 改善 |
| 5 | 0 | 4 | 0.565732 | 0.444946 | 改善 |
| 5 | 0 | 8 | 0.494629 | 0.343813 | 改善 |
| 5 | 200 | 1 | 0.040623 | 0.044317 | 恶化 |
| 5 | 200 | 4 | 0.038784 | 0.045192 | 恶化 |
| 5 | 200 | 8 | 0.097696 | 0.092244 | 改善 |
| 5 | 400 | 1 | 0.024721 | 0.029771 | 恶化 |
| 5 | 400 | 4 | 0.111195 | 0.098693 | 改善 |
| 5 | 400 | 8 | 0.124610 | 0.111214 | 改善 |
| 6 | 0 | 1 | 0.294515 | 0.320869 | 恶化 |
| 6 | 0 | 4 | 0.154777 | 0.176598 | 恶化 |
| 6 | 0 | 8 | 0.274381 | 0.367894 | 恶化 |
| 6 | 200 | 1 | 0.034662 | 0.045153 | 恶化 |
| 6 | 200 | 4 | 0.152392 | 0.124681 | 改善 |
| 6 | 200 | 8 | 0.204431 | 0.161438 | 改善 |
| 6 | 400 | 1 | 0.018559 | 0.021915 | 恶化 |
| 6 | 400 | 4 | 0.124963 | 0.130641 | 恶化 |
| 6 | 400 | 8 | 0.174418 | 0.168789 | 改善 |

聚合正向变化明显受 Episode5 初始高误差案例影响。**latent 误差普遍下降不等于
动作误差普遍下降。** 这里的 oracle 只是离线未来视觉参考，不是示范动作 ground truth，
更不是实际任务成功；不能写成“机器人成功率提升9.79%”。

## 6. 独立计算开销

固定同一真实观测／当前 state，通过完整 RGB policy 得到已生成 normalized prefix。
两条路径分别50次 warmup、200次 measured，CUDA 完成后的 host wall 时间；没有
模拟器并发，测量期不写文件。Predictor使用d8、完整8行前缀。

| 路径 | Mean | P50 | P90 | Max |
| --- | ---: | ---: | ---: | ---: |
| 完整 normal-RGB policy | 135.7275 ms | 133.7617 ms | **142.4536 ms** | 218.2642 ms |
| 新 predictor forward | 1.3073 ms | 1.1554 ms | **1.6739 ms** | 2.5394 ms |

P90比值为 **1.1750294151%**。这是新增模块的独立 forward 开销，不包括模拟器、
并发争用和整个预测接管管线，也不是重新批准旧 M5 的效率／实时合同。
Benchmark source：`8bb16e8a12e2cdf1705a0e1c9b31e030d85f5401`。

## 7. 产物与复现入口

本地根目录：`/home/rp/Workspace/SmolVLA_RTC/artifacts/`。

| Artifact | 内容 |
| --- | --- |
| `m54l5_wsagi_primed_identity_v2` | 221步真实 async 请求／控制／技术失败 |
| `m54l5_reset_drift_standard_v1` | 标准相机同seed位姿漂移的原始证据 |
| `m54l5_reset_standard_fixed_v1` | 修复后同seed位姿一致的原始证据 |
| `m54l6_causal_collection_v1` | 原四条训练缓存与验证采集中断；保持不变 |
| `m54l6_causal_collection_v2` | 续采 manifest、validation/test缓存、全部episode来源映射 |
| `m54l6_predictor_training_v1/training.json` | 全20轮训练与验证历史 |
| `m54l6_predictor_training_v1/selection.json` | 测试前冻结的epoch3选择 |
| `m54l6_predictor_training_v1/best.pt` | 新预测器权重、配置、训练source与数据manifest |
| `m54l6_predictor_training_v1/test_report.json` | 一次性盲测的完整 per-episode/delay/case结果 |
| `m54l6_standalone_timing_v1/result.json` | 全部200+200条原始耗时和摘要 |

源码入口在 `examples/advanced/predictive_async/`：
`collect_so101_predictor_pilot.py`、`train_so101_predictor_pilot.py`、
`probe_so101_predictor_actions.py`、`benchmark_so101_predictor_pilot.py`。
测试阶段拒绝覆盖已有 `test_report.json`；旧模型／数据／失败产物未删除。

## 8. 限制与阶段裁决

L6 的正向结论限于这两个未参与选择的 simulator episode、当前固定策略生成的轨迹、
现有视觉表示和短延迟。轨迹是在采集上限处截断，不是成功示范；不能据此宣称
跨任务、跨机器人、统计显著性或真实机器人收益。

本阶段没有取得官方完整 PickOrange 成功，也没有完成新预测器 production 部署或
`identity vs predicted` 的任务级闭环比较。真实并发控制仍有 lost-slot 失败，风险
thresholds继续为null。标准相机局部reset修复不等于整个模拟器逐位确定。

**阶段裁决：L6离线预测pilot按预注册主条件PASS；动作一致性是异质的正向聚合信号；
独立计算开销较小；任务级收益与持续实时控制仍未获GO。** 后续应面向动作相关的
预测质量与并发控制稳定性设计新的独立验证，而不是用这两个已打开的测试episode调参。
