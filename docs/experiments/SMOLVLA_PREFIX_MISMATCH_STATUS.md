# F-PFX1：模型计算完成，独立CPU审计启动受阻

2026-09-10，接续Issue #1评论5619941300。

## 已完成

- 执行提交：`a6a6e9973a3780d8266a3b89de4c0f72321767a9`，已推送；五份F-PFX1计划/源码/测试文件，未混入其他pending文件。
- 本轮仅修正审计器既有UP034冗余括号；15项供体/归约单测通过（1.67s，exit0），Ruff check通过（exit0）。
- 预登记评论：`5620068396`。实际ID单次GET成功；supervisor在建立输出与启动worker前完成登记正文exact门。
- 输出：`outputs/smolvla_prefix_mismatch_a6a6e997`。
- worker PID：3162763；DevSpace session：277；worker/supervisor及外层均exit0，无强制终止、未闭合phase或运行停止原因。
- 运行时间：2026-09-10T14:12:36.880841+00:00至14:13:12.972706+00:00；supervisor记录36.09190571284853秒。
- 运行器完成261次预测器前向与261次完整十步解码，176份真实/无动作的量化token及完整输出exact复现原已选模型评估；graph capture 8次，运行器报告graph已释放、预测器权重未变、VLA冻结。
- VLA加载1、预测器加载2；attempt1/retry0；更新、反传、新图像编码、Env、native、真机和test读取均0。

以上来自本次正常返回的执行回执，不等于独立CPU审计已接纳。原结果、预测数组、事件及capture记录由运行器保存；本会话没有重跑模型。

## 当前断点

模型退出后，以原固定uv离线解释器启动`examples/advanced/predictive_async/audit_libero_prefix_mismatch.py`时，DevSpace.exec_command返回：

> 因 OpenAI 无法确定请求的安全状态，已拦截此工具调用。

没有该审计调用的PID、退出码或结果。具体触发原因未知；未重试、拆写或换工具实施受阻审计。此处是执行断点，不是审计阴性结果，也不是已完成审计。

因此，本轮尚未独立归约真实/错配/无动作的误差、方向与敏感性，`stable_development_action_utility`未判定。不能把176份exact复现或261次完成称为动作效用提升或重大科研成果。

## 接续范围

保留此唯一输出，不重跑F-PFX1，不重新登记同一批模型计算。下一步仍是原计划指定的保存数组独立CPU审计；只有它在正常授权环境成功执行后，才据实际指标形成正式RESULT并决定下一独立实验。

原F-SCL1-r1的`scale_row0_improved=true`、`action_increment_observed=false`、`development_candidate_gate_passed=false`保持；生产默认和闭环资格false，risk_thresholds=null，旧confirmation不动。不读取task8/9测试或confirmation21–40，不干预其他GPU进程。

独立拦截事实已追加到原`/home/rp/.devspace/Reason_for_interception/2026-09-09_devspace_exec_command_safety_interception.md`。本轮自有实验进程已确认退出，没有后台实验。
