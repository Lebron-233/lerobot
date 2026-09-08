# LIBERO 独立参考资格验证结果

**baseline_qualified=false**。开发队列未通过。确认队列 not_started。

执行源码：[`1cd7d222c9c49d89c6d385006e7967f825f64a90`](https://github.com/Lebron-233/lerobot/commit/1cd7d222c9c49d89c6d385006e7967f825f64a90)。
[执行前预注册](https://github.com/Lebron-233/lerobot/issues/1#issuecomment-5580879265) 已在首个正式 tuple 前发布并回读一致。
代码、候选、依赖和控制合同在执行期间保持固定。

开发裁决依据：task 5 成功 14/20 低于 16/20。

结果提交 [`680a930c`](https://github.com/Lebron-233/lerobot/commit/680a930c313c5bbc77ef2619c35205680739b7ba)
已推送；[闭合评论 5583169242](https://github.com/Lebron-233/lerobot/issues/1#issuecomment-5583169242)
已查重后发布一次，并按真实 ID 回读确认正文一致。发布回执保存在原始目录的
`publication_receipt.json`，也收录于机器可读总结果。

## 队列记账

| 队列 | scheduled | completed | technical_failure | not_run | timeout | 原生成功 | worker exit |
|---|---:|---:|---:|---:|---:|---:|---:|
| development | 200 | 200 | 0 | 0 | 15 | 185/200 | 0 |

确认队列的 200 个槽位已预注册，均未打开；not_started 不表示 0/200 实测失败。

## development：逐任务统计

| task | object | 成功/20 | 观测 n | Wilson 95% | 受限模拟均值 s | tuple wall 合计 s |
|---:|---|---:|---:|---|---:|---:|
| 0 | alphabet_soup | 19/20 | 20 | 76.39–99.11% | 8.8250 | 1079.289 |
| 1 | cream_cheese | 19/20 | 20 | 76.39–99.11% | 7.4050 | 898.560 |
| 2 | salad_dressing | 20/20 | 20 | 83.89–100.00% | 6.7725 | 799.837 |
| 3 | bbq_sauce | 19/20 | 20 | 76.39–99.11% | 7.0400 | 825.688 |
| 4 | ketchup | 19/20 | 20 | 76.39–99.11% | 8.7050 | 1012.874 |
| 5 | tomato_sauce | 14/20 | 20 | 48.10–85.45% | 10.0250 | 1163.989 |
| 6 | butter | 19/20 | 20 | 76.39–99.11% | 8.3500 | 1001.936 |
| 7 | milk | 18/20 | 20 | 69.90–97.21% | 7.8750 | 908.968 |
| 8 | chocolate_pudding | 20/20 | 20 | 83.89–100.00% | 8.4500 | 925.259 |
| 9 | orange_juice | 18/20 | 20 | 69.90–97.21% | 7.3050 | 797.518 |

宏成功率 92.50%；固定十任务内 bootstrap 95%：89.00–95.50%，20,000 次，seed=910001。

受限模拟完成时间均值 8.0753 s；worker wall 9424.932 s。成功按首次成功动作/20，非成功为 14 s；未运行槽位不计入观测时间或置信区间。

原生测量返回 32301；调用方测量记录 32301；settling 返回 2000；物理 step 未知 tuple 0。

环境关闭成功/失败/未知：200/0/0。原生 truncated 观测 15；表中 timeout 只计完成且未成功的超时。

有限越界动作 17928 个，共 17973 个分量，原样交给原生控制器。累计策略选择及反归一化耗时 8127.819 s。

原始记录一致性检查 records_verified=true。逐 tuple 核对原生调用/返回、观测索引、动作、50/1/10、终止及 cleanup；详见 tuple_accounting.json。

## 固定合同与原始产物

完整 LIBERO-Object 原生十任务、task_order_index=0；开发初始状态 1–20，条件性确认 21–40。各组独立门槛：完整 200 个、总成功至少 180/200、每任务至少 16/20、零技术失败。原生 relative Panda OSC_POSE，20 Hz，50/1/10，每 tuple hard reset、十个 settling 后设置策略 seed，最多 280 测量动作。

Policy revision `6721902bc4d61e50a3bfdb11dfb4cb626f05d102`；VLM revision `7b375e1b73b11138ff12fe22c8f2822d8fe03467`；assets revision `0b3ea86be5fe169d0fd036ae63d1070ec09e90f6`。

原始目录：`/home/rp/Workspace/SmolVLA_RTC/artifacts/libero_reference_qualification_20260908T064953Z`。每个已打开阶段包含运行注册、strict policy load、worker 进程和退出记录、完整日志、逐 tuple accounting 和 summary；每个已启动 tuple 包含启动标记、步事件、实际双相机无损观测和 cleanup 后结果。`run_qualification.sh` 保存完整启动命令。

实现验证：20 项定向测试通过，Ruff lint/format 与 diff check 通过；测试日志与首次 fixture 失败日志均保留。

闭合复算直接读取全部 200 个原始步日志，成功数、动作数、逐任务受限时间与 wall 合计均与汇总一致；200 个 reset/settling/policy-seed 顺序通过。共保存 32,501 个双相机无损观测档案。本次没有第 280 步成功的样本；该边界已由定向测试覆盖。

运行后源码仍为预注册 clean HEAD；监督进程 2146633、工作进程 2146798 均已退出。worker exit=0；supervisor exit=2 表示资格门槛未通过。确认目录不存在，没有启动确认样本或执行重试。

[机器可读总结果](LIBERO_REFERENCE_QUALIFICATION_RESULT.json) 与 [闭合检查及全部 200 个 tuple 结果](LIBERO_REFERENCE_QUALIFICATION_AUDIT.json) 随本报告提交。独立复算脚本和日志位于原始目录的 close_records.py、closure_audit.log；执行版 runner 的完整逐步核对保存于 development/tuple_accounting.json。

## 已知限制

全部十任务名称存在于已确认训练来源中；确认状态和 seed 只对本项目准备/开发独立，没有证明与原始训练示范去重。完整历史训练配方及作者评测 manifest 未公开，不能称为作者分数严格复现。

`realtime_qualified=false`、`predictor_benefit_tested=false`、`risk_thresholds=null`。本结果未建立实时能力或预测器异步收益。
