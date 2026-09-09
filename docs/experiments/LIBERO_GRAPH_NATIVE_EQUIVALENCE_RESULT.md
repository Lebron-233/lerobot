# LIBERO 十步 Graph 原生等价性结果

> 2026-09-09 验收勘误：本报告保留首次A/B/C执行记录。旧C的裁剪预期与此前正式裁决冲突，
> `contract_gap`不表示已确认的生产队列缺陷。严格whole-discard验收及新独立结果见
> [勘误版协议](SMOLVLA_ASYNC_TIMING_REPLAY_PLAN.md)与[新C结果](SMOLVLA_ASYNC_STRICT_DEADLINE_RESULT.md)。

2026-09-09。**A 通过，B 原生工程等价性通过；C 固定时序回放完成，轻微晚到裁剪合同未满足。**

A/B 执行源码：`5d45353ff98fc448bc514b284c1a19609405585b`。C 执行源码：`290c1a3dbaa90d4e3d900c4c4eb83aa6840b2ce8`。
[预注册评论 5594614809](https://github.com/Lebron-233/lerobot/issues/1#issuecomment-5594614809)
已在首个真实模型请求前发布并回读一致；[固定协议](LIBERO_GRAPH_NATIVE_EQUIVALENCE_PLAN.md)、
[40 条 manifest](LIBERO_GRAPH_NATIVE_EQUIVALENCE_MANIFEST.json) 与实现已先提交推送。
完整数字、每对分位数、逐 episode timing 和冷准备明细见[结果 JSON](LIBERO_GRAPH_NATIVE_EQUIVALENCE_RESULT.json)。

## 原生闭环结果

40/40 条全新 episode 完成，20/20 对逐步 exact，技术失败 0，not_run 0。
两条件各 3,159 个测量动作，合计 6,318 measured、400 settling；40 个环境关闭，未知 native 返回 0。
所有初始及后续原始双相机、raw state/四元数、normalized state、实际 noise、完整 `[1,50,32]`、
normalized/post 动作及 native 成功/终止完全一致。首个分歧为 **无**。
两条件均成功 18/20，另 2/20 达到 280 步 TimeLimit，首次成功和终止动作号逐对相同。
原始 sampler 已恢复，图已释放；worker 与 supervisor 均退出 0，整轮 wall 为 1,218.314 s。
`native_graph_equivalence_passed=true`。

| pair | task | state | 两条件结果 | 各自动作数 | eager 均值 ms | graph 均值 ms | exact |
|---:|---:|---:|---|---:|---:|---:|---|
| 0 | 0 | 41 | 成功 | 143 | 251.633 | 89.722 | 是 |
| 1 | 0 | 49 | 成功 | 166 | 247.909 | 80.296 | 是 |
| 2 | 1 | 41 | 超时 | 280 | 249.674 | 86.987 | 是 |
| 3 | 1 | 49 | 成功 | 132 | 247.588 | 82.656 | 是 |
| 4 | 2 | 41 | 成功 | 113 | 251.376 | 93.595 | 是 |
| 5 | 2 | 49 | 成功 | 141 | 249.659 | 84.114 | 是 |
| 6 | 3 | 41 | 成功 | 130 | 246.845 | 88.108 | 是 |
| 7 | 3 | 49 | 成功 | 132 | 243.935 | 82.642 | 是 |
| 8 | 4 | 41 | 成功 | 144 | 243.975 | 88.363 | 是 |
| 9 | 4 | 49 | 超时 | 280 | 246.055 | 81.219 | 是 |
| 10 | 5 | 41 | 成功 | 166 | 244.466 | 83.912 | 是 |
| 11 | 5 | 49 | 成功 | 148 | 240.304 | 76.959 | 是 |
| 12 | 6 | 41 | 成功 | 157 | 253.163 | 86.792 | 是 |
| 13 | 6 | 49 | 成功 | 158 | 244.977 | 83.281 | 是 |
| 14 | 7 | 41 | 成功 | 127 | 245.510 | 84.182 | 是 |
| 15 | 7 | 49 | 成功 | 147 | 237.992 | 75.527 | 是 |
| 16 | 8 | 41 | 成功 | 149 | 239.044 | 82.983 | 是 |
| 17 | 8 | 49 | 成功 | 185 | 239.461 | 75.988 | 是 |
| 18 | 9 | 41 | 成功 | 136 | 239.838 | 83.772 | 是 |
| 19 | 9 | 49 | 成功 | 125 | 239.714 | 75.786 | 是 |

## 端到端时序与冷准备

主工程延迟从观测返回到 CPU 动作可供 env.step 使用，覆盖新视觉、noise、输入刷新、输出独立副本、
post 和 GPU 完成，也包含这段实际间隔内的观测存档。环境推进和日志成本另有记录。
经验分位数使用排序后的 ceil(q*n)。以下包括所有首次请求和图准备成本，未删除慢请求。

| 模式 | 请求数 | 均值 ms | P50 ms | P95 ms | P99 ms | 最大 ms | >50 ms |
|---|---:|---:|---:|---:|---:|---:|---:|
| eager | 3159 | 245.304 | 242.633 | 262.919 | 277.034 | 660.751 | 3159 |
| graph | 3159 | 83.215 | 78.717 | 86.736 | 90.316 | 1330.974 | 3159 |

逐请求均值比为 **2.948×**。按 episode 等权后的 eager/graph 均值分别为 **245.156 / 83.344 ms**，
平均配对差为 **−161.812 ms**；每对的完整分位数和配对比值已保存。graph 的全部 3,159 次请求仍 >50 ms。
selector→CPU 动作均值为 229.750 / 67.640 ms；预处理均值 0.738 / 0.731 ms；
每请求归集的观测/预测存档日志均值 15.029 / 14.977 ms。完整 episode wall、环境 step、首请求与余下请求见 JSON。

模型 strict load **4.338 s**。十个任务的完整 graph preparation 共 **11.399 s**，
其中先行完整 eager setup 共 **2.320 s**，其余为视觉编码、warmup、capture 和完成同步。
首任务完整 preparation **1.174 s**（其中先行 eager setup **0.222 s**）。
本轮首个 eager / graph 控制请求分别 **660.751 / 1,258.040 ms**。
这些费用已分别记录，并且 preparation 也包含在对应 graph 首请求和上表总统计中，没有只报 capture 后稳态。

## A：连续随机序列与输出所有权

task0–9/state41 的 10 个既有初始观测，各连续请求两次：20/20 对 noise、全 chunk、normalized/post 动作 exact。
两条序列各仅在开头设 seed1009001。另 task0→1→0 的 3 对也 exact，capture_id 为 11、12、13。
46 次控制请求对应 46 次原方法噪声采样；保留的 request0 输出在 request1 后不变。
13 次 capture 准备共 14.774 s；sampler 恢复、graph 释放、worker exit0。没有原生动作。

## C：实际延迟驱动的 CPU 时序结果

按固定 manifest 顺序完成 40/40 条 trace、6,318/6,318 个请求。复用原 engine 的 request/planner/
worker method 和 ScheduledActionQueue；20 Hz 虚拟 tick 在模型在途或没有动作时仍推进。
观察与 next index 一致、已承诺 prefix 未变、最多一个在途、绝对动作索引连续且不重复。
planner 在计划后才得到当前延迟，只使用已完成历史。

| 延迟来源 | wall ticks | 消费的索引标记 | 启动无动作 ticks | ready 后 underflow | chunk deadline miss |
|---|---:|---:|---:|---:|---:|
| eager | 82498 | 82179 | 319 | 0 | 4 |
| graph | 71607 | 71257 | 350 | 0 | 0 |

合计 **669 个启动期无动作 tick** 明确可见；没有暂停时钟或重复旧动作填补。
固定边界中，提前 staging/精确接管、prefix 私有副本、严重迟到丢弃、reset/task epoch 失效均符合预期。
**首个未满足合同：晚 1 步时，队列返回 `deadline_miss`，下一动作仍为旧块标记 4；
裁前缀合同预期新块标记 101。** `max_late_steps=2` 当前只保存/校验，未传给队列。
保留现有整块丢弃承诺，C 返回 exit2 / `contract_gap`，不把这个用例计为裁剪通过。
下一轮具体接口和科学比较见 [SMOLVLA_ASYNC_NEXT_REVIEW.md](SMOLVLA_ASYNC_NEXT_REVIEW.md)。

## 默认路径与定向测试

production SmolVLA policy、sync/RTC factory 和原 queue 语义均未修改。graph 仅由新实验 helper 显式安装。
原 eager 十次 Python 投影审计保留；graph 使用独立 capture 证据和每请求真实 replay_count=1。
A 相关 59 项 CPU/fake 合同测试通过；最后相关 14 项重跑通过。
C 的 30 项现有队列/线程/默认配置检查通过，新驱动两项修正后通过。
[完整 A 命令及初次收集错误](LIBERO_GRAPH_NATIVE_EQUIVALENCE_TESTS.md)、
[C 命令与初次驱动属性错误](SMOLVLA_ASYNC_TIMING_REPLAY_TESTS.md) 均保留；没有安装或升级包。

## 已知问题与适用范围

C 轻微迟到的接受/裁剪会改变当前已承诺的接管规则，需要下一轮明确动作行的时间含义和 guard，
本轮停在该可审阅差异处。CPU 回放是索引标记与完成时序验证，不能证明真实异步闭环成功或 GPU stream 并发安全。
不同 trace 所消费的标记数不同，也不是原生轨迹动作数的比较。

B 按固定配对顺序共享一个模型，首个 graph episode 在首个 eager episode 之后；记录的实际 cold 组成
不能解释成另一次独立 graph 进程从启动到首动作的测量。观测存档在本次主 wall 区间内，部署中的日志位置需单独固定。

本轮是旧开发状态上的工程复测，成功率 18/20 不替代旧 200 条资格结果。
`baseline_qualified=false`、`realtime_qualified=false`、`predictor_benefit_tested=false`、
`risk_thresholds=null`、`old_confirmation="not_started_untouched"` 均保持不变。

## 原始证据

- A 连续模型合同：`/home/rp/Workspace/SmolVLA_RTC/lerobot/outputs/libero_graph_model_5d45353f`。
- B 原生完整逐步证据：`/home/rp/Workspace/SmolVLA_RTC/lerobot/outputs/libero_graph_native_5d45353f`。
- C CPU 回放与首个失败：`/home/rp/Workspace/SmolVLA_RTC/lerobot/outputs/smolvla_async_timing_5d45353f_290c1a3d`。

A/B 目录保留 registration、实际 worker 命令/PID、supervisor 退出和原始数组；B 每条 final 均在 cleanup 后写入。大数组不进入 Git。
