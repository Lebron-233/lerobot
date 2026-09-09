# C：固定 CPU 时序回放

日期：2026-09-09。B 已在 `5d45353ff98fc448bc514b284c1a19609405585b` 完整通过并退出 0。
输入固定为 `outputs/libero_graph_native_5d45353f` 的 40 条已审计 episode，按 manifest ordinal 0–39，
逐条采用 `request_timing.engineering_seconds` 的原始顺序，共 6,318 个延迟；包含首次请求与图准备。

直接复用 `PredictiveAsyncInferenceEngine` 的 request、planner、`_run_request`、startup 和 publication，
以及它创建的 `ScheduledActionQueue`。单进程事件驱动替代实际线程等待：fake policy 尚在运行时
继续调用真实控制入口；完成事件先于同时间的控制 tick。线程行为另由原有针对性测试验证。
CPU fake policy 每次返回 50×7 的索引标记，8D identity state；标记仅用于验证队列索引，没有物理动作含义。
没有载入模型权重、创建环境或派发 native 动作。

每条 trace 新建 engine；20 Hz、queue threshold 30、latency quantile 0.9、window 50、margin 1、
min/max delay 0/8、guard 2、max_late_steps 2、identity context、identity fallback，均为现有参数。
每 tick 先冻结观察与 next_action_index，再尝试消费动作。启动未 ready 的 tick 单列；ready 后
空队列调用 get_action 并记录 underflow。两种无动作情形都继续推进虚拟 wall ticks，不 hold-last。
只在计划已经生成后把本次完成延迟交给 fake policy；检查当前 latency 尚未进入历史、prefix 没有被改写。
每条回放在最后一个 trace 请求完成 publication 后结束；不推定尚未消费的末尾 chunk 已执行。
异常停止后续 trace，保存首个失败及已完成部分。

独立固定边界：prefix 私有副本、提前 staging/第 3 个绝对索引接管、晚 1 步、晚 3 步、reset、task epoch。
晚 1 步的任务书预期是裁掉新 chunk 第一行；当前 MVP 的真实实现整块丢弃。
该项按不满足合同记载，保留原接管规则，交下一轮审阅明确接受/丢弃语义，不将其计为通过。

执行环境是已有 `/home/rp/miniconda3/envs/smolvla-rtc/bin/python`，torch 2.11.0+cu128、
transformers 5.5.4、datasets 4.8.5；所有本轮 tensor 与 engine device 为 CPU，未安装或升级包。
先提交本实现，再执行：

```bash
env -u PYTHONPATH HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 /home/rp/miniconda3/envs/smolvla-rtc/bin/python -u examples/advanced/predictive_async/replay_smolvla_async_timing.py --output /home/rp/Workspace/SmolVLA_RTC/lerobot/outputs/smolvla_async_timing_NATIVEHEAD_CHEAD
```

输出名称在执行时展开为两个提交短号；result.json 保存实际 C execution HEAD 和固定 B HEAD。
结果与下一轮审阅文档另作提交。真实延迟 trace 不能被当作未来部署分布或 predictor 收益证据。
