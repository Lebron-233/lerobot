# C 严格 deadline CPU 回放结果

2026-09-09。**新的 C 严格 deadline CPU 回放通过：40/40 条 trace、6,318/6,318 次请求完成发布，7 项边界全部通过，正常 exit 0。**
执行一次，进程墙时 9.429921 秒，限时300秒，未超时、未重试或补样本。

执行源码：`3ae1dc569007e612bca459ba6680dd94204268a6`；固定 B 延迟来源：`5d45353ff98fc448bc514b284c1a19609405585b`。
[预注册评论5595075017](https://github.com/Lebron-233/lerobot/issues/1#issuecomment-5595075017)在执行前发布并逐字回读一致。
[勘误协议](SMOLVLA_ASYNC_TIMING_REPLAY_PLAN.md)、[定向测试记录](SMOLVLA_ASYNC_STRICT_DEADLINE_TESTS.md)、
[机器结果与40条账目](SMOLVLA_ASYNC_STRICT_DEADLINE_RESULT.json)。

## 合同勘误与改动

`late_policy=whole_discard_any_late`，依据先于旧 C 实验的
[正式裁决5471357164](https://github.com/Lebron-233/lerobot/issues/1#issuecomment-5471357164)和
[复核5554561794](https://github.com/Lebron-233/lerobot/issues/1#issuecomment-5554561794)。
现有生产队列的整块丢弃行为正确。本轮修改回放的错误裁剪预期，增加 late=2 边界，记录所依据的合同；
生产 queue、engine、policy 和 Graph 数值路径未改。max_late_steps=2仍用于guard sizing/诊断。

| 边界 | 结果 |
|---|---|
| 私有 committed prefix | worker副本修改未影响queue中的prefix |
| 提前 staging / 原索引接管 | 发送标记0、1、2、100，到接管点才切换新块第0行 |
| late=1 | 实际late_steps=1，deadline_miss，下一旧动作4，匹配plan清除，无staged新块 |
| late=2 | 实际late_steps=2，deadline_miss，下一旧动作5，匹配plan清除，无staged新块 |
| late=3 | 实际late_steps=3，deadline_miss，下一旧动作6，匹配plan清除，无staged新块 |
| reset epoch失效 | 旧返回stale |
| task epoch失效 | 旧返回stale |

14项定向测试通过，另覆盖准时接管、单在途/guard、耗尽后None/underflow且索引不前进、
A→B→A旧返回不清更新plan、task provenance及只取消匹配计划。两个改动Python文件的lint/format通过。

## 固定 trace 实测与完整账目

仍使用CPU fake policy、20Hz虚拟tick及现有engine/planner/queue，按B manifest的0–39原顺序输入全部延迟。
逐请求核对了来源索引、延迟值和顺序；首请求、冷准备及慢样本均保留，未重新执行A/B。

| 指标 | eager来源 | graph来源 | 合计 |
|---|---:|---:|---:|
| 完成trace | 20 | 20 | 40 |
| 输入完成延迟 / 已发布请求 | 3,159 / 3,159 | 3,159 / 3,159 | 6,318 / 6,318 |
| 虚拟wall ticks | 82,498 | 71,607 | 154,105 |
| 已消费动作 | 82,179 | 71,257 | 153,436 |
| 启动无动作ticks | 319 | 350 | 669 |
| ready后underflow | 0 | 0 | 0 |
| deadline_miss整块丢弃 | 4 | 0 | 4 |
| prediction cap exceeded / stale | 0 / 0 | 0 / 0 | 0 / 0 |

全部40条均completed；6,318条记录逐项publication_completed=true，来源延迟列表完全一致。
每条请求只在规划完成后获得本次延迟，历史因果断言通过；请求不重叠、ID未复用、单在途检查通过。
动作标记与绝对索引在逐tick执行时吻合，索引连续无重复；wall ticks=已消费动作+无动作ticks。
首个失败为空。四次实际迟到均由既有整块丢弃路径处理，不影响此次严格合同通过。

## 原始证据

新目录：`/home/rp/Workspace/SmolVLA_RTC/lerobot/outputs/smolvla_async_timing_strict_deadline_3ae1dc56/`。
`result.json`保存回放汇总及7项边界，`tuple_000.json`至`tuple_039.json`保存全部请求和无动作tick；
`execution.json`保存实际命令、起止UTC时间、300秒上限与退出状态，`replay.log`保存进程输出。
`accounting.json`和`collect.py`保存完整账目及其收集方法；`supervise.py`保存本次限时运行方法。
登记/正式裁决回读和定向测试原始日志也在新目录中。

旧目录 `outputs/smolvla_async_timing_5d45353f_290c1a3d/` 保持不变：原exit2、contract_gap和expected=101永久保留。
该首例takeover_index=3、next_action_index=4时返回deadline_miss和旧动作4，符合此前正式裁决；
旧结果中的失败源自交接验收要求冲突。新结果独立保存，未把旧JSON改为passed。

## 已知限制与后续工程

本结果验证固定延迟输入下的CPU时序合同。真实异步Graph路径还需已编码token入口、
同一worker拥有Graph生命周期和processor/GPU reset，以及独立输出、CPU可消费动作与设备完成屏障。
这些工程接入和新的模型/GPU/native验证尚未执行，见[下一轮审阅](SMOLVLA_ASYNC_NEXT_REVIEW.md)。

A/B沿用已接受的工程结论；B约245.30ms→83.22ms、约2.95倍，全部graph请求仍超过50ms。
本次CPU回放不提供实时资格或future-latent收益证据，也不把每次消费1步的B等价性推广到异步连续多步。
`baseline_qualified=false`、`realtime_qualified=false`、`predictor_benefit_tested=false`、
`risk_thresholds=null`、`old_confirmation=not_started_untouched`保持。
