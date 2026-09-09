# C：严格 deadline 固定 CPU 时序回放（验收勘误版）

日期：2026-09-09。本版依据用户的《SmolVLA_A_B审阅与C合同勘误_20260909.md》纠正旧C验收。
`late_policy=whole_discard_any_late`。正式依据为
[2026-08-30 裁决5471357164](https://github.com/Lebron-233/lerobot/issues/1#issuecomment-5471357164)与
[2026-09-05 裁决5554561794](https://github.com/Lebron-233/lerobot/issues/1#issuecomment-5554561794)，本轮已读取原评论确认。
旧C的exit2、contract_gap及expected=101保留在原目录，属于验收要求冲突；生产整块丢弃实现正确。
旧A/B接受结论沿用。B 已在 `5d45353ff98fc448bc514b284c1a19609405585b` 完整通过并退出 0。
输入固定为 `outputs/libero_graph_native_5d45353f` 的 40 条已审计 episode，按 manifest ordinal 0–39，
逐条采用 `request_timing.engineering_seconds` 的原始顺序，共 6,318 个延迟；包含首次请求与图准备。

直接复用 `PredictiveAsyncInferenceEngine` 的 request、planner、`_run_request`、startup 和 publication，
以及它创建的 `ScheduledActionQueue`。单进程事件驱动替代实际线程等待：fake policy 尚在运行时
继续调用真实控制入口；完成事件先于同时间的控制 tick。线程行为另由原有针对性测试验证。
CPU fake policy 每次返回 50×7 的索引标记，8D identity state；标记仅用于验证队列索引，没有物理动作含义。
没有载入模型权重、创建环境或派发 native 动作。

每条 trace 新建 engine；20 Hz、queue threshold 30、latency quantile 0.9、window 50、margin 1、
min/max delay 0/8、guard 2、max_late_steps 2、identity context、identity fallback，均为现有参数。
max_late_steps只保留guard sizing/诊断用途，不能解释为接受轻微晚到块的阈值。
每 tick 先冻结观察与 next_action_index，再尝试消费动作。启动未 ready 的 tick 单列；ready 后
空队列调用 get_action 并记录 underflow。两种无动作情形都继续推进虚拟 wall ticks，不 hold-last。
只在计划已经生成后把本次完成延迟交给 fake policy；检查当前 latency 尚未进入历史、prefix 没有被改写。
每条回放在最后一个 trace 请求完成 publication 后结束；不推定尚未消费的末尾 chunk 已执行。
异常停止后续 trace，保存首个失败及已完成部分。

独立固定边界：prefix 私有副本、提前 staging/第 3 个绝对索引接管、晚1/2/3步、reset、task epoch。
对late1/2/3分别同时要求：实际late_steps等于1/2/3、deadline_miss、下一旧动作等于4/5/6、
匹配plan已清除、没有staged新块。提前/准时接管、旧返回不得清除较新plan、epoch失效及耗尽后
None/underflow的边界复用现有定向测试。

晚到期间实际执行旧guard动作，并未执行新块被跳过的行。仅裁行号不能证明状态或连续性等价，
identity context也不能消除这一差异。无补偿裁剪需要另立continuity/residual-RTC研究协议。
本轮只修改边界验收、测试和结果说明；生产queue、engine、policy及Graph数值路径保持不变。

执行环境是已有 `/home/rp/miniconda3/envs/smolvla-rtc/bin/python`，torch 2.11.0+cu128、
transformers 5.5.4、datasets 4.8.5；所有本轮 tensor 与 engine device 为 CPU，未安装或升级包。
修订源码、协议和定向测试记录先提交推送，在Issue #1登记新exact HEAD和绝对输出目录并回读，
再执行一次下面的命令。登记时将NEW_EXECUTION_HEAD8展开为实际提交短号。
外层监督限时300秒；超时终止本次CPU子进程并保存退出/日志与已有tuple记录。技术失败停止、无重试或补样本。

```text
scope: 40 traces / 6318 completion delays, unchanged manifest order
late_policy: whole_discard_any_late
pass: 40 traces + 6318 completed publications + all strict boundaries + index/in-flight/history checks + exit 0
old_C: outputs/smolvla_async_timing_5d45353f_290c1a3d/ remains unchanged
```

实际回放CLI：

```bash
env -u PYTHONPATH HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 /home/rp/miniconda3/envs/smolvla-rtc/bin/python -u examples/advanced/predictive_async/replay_smolvla_async_timing.py --output /home/rp/Workspace/SmolVLA_RTC/lerobot/outputs/smolvla_async_timing_strict_deadline_NEW_EXECUTION_HEAD8
```

输出名称在执行时展开为新提交短号；result.json 保存实际C execution HEAD、固定B HEAD、late_policy及正式裁决链接。
监督回执保存完整命令、开始/结束时间、300秒上限和真实exit code。逐tuple核对请求账目后才发布新的C通过结论。
结果与下一轮审阅文档另作提交。真实延迟 trace 不能被当作未来部署分布或 predictor 收益证据。
