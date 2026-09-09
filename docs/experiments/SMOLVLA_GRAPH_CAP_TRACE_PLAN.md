# E-RCV3-trace：独立、可定位首错的原生恢复机制实验

用户在当前会话明确要求依据 Issue #1 / 5603596852 继续实验。
接续 HEAD 为 `4d62316f`。E-RCV2 已终止，3/4 完成和整体 false 永久保留；
本实验不是补第4条、resume、replacement，也不把两轮样本拼成成功队列。

## 预先固定的问题

先补齐上一轮的可观测性缺口，再在同一固定条件下取得一组**新的完整对照**，
或保存可定位到模型阶段的首个新技术失败。两者均如实报告，不按结果追加样本。
本轮不试图用一次运行归因旧超时、不宣称性能改善、不训练预测器。

新入口 `libero_graph_cap_trace_native.py` 复用原 NativeEngine / Graph / worker /
queue / run_episode / source audit。原源码与旧报告不改。
新增 CPU-only、逐条 flush 的模型阶段起止、request terminal snapshot、cleanup 起止；
reset 返回后、第一次模型请求前保存本条真实初态；policy load 回执提前保存。
原 watchdog 触发时先请求 all-thread faulthandler 栈，再按原 TERM / 5秒 KILL 处理。
监督器从自己的 stop_reason 和 call journal 报首错及部分账目，不把缺失解释成无失败或零调用。
不新增 CUDA event / synchronize、不改变时钟，不在暂停中持有 queue/request 锁。
这是带额外观测开销的独立重复，不能与 E-RCV2 wall 直接做性能对比。

## 唯一四行及预算

| ordinal | task/state | Env seed | policy seed | arm |
|---:|---|---:|---:|---|
| 0 | 0/41 | 940041 | 950041 | disabled |
| 1 | 0/41 | 940041 | 950041 | candidate |
| 2 | 2/41 | 940241 | 950241 | candidate |
| 3 | 2/41 | 940241 | 950241 | disabled |

均为 graph_identity_async。candidate 仍为 same_path_discard_probe_v1，生产默认 disabled。
两臂首个 planned 在原独立 CPU chunk/完成屏障之后、采样接纳之前实际 sleep 0.6秒；
startup/bootstrap/probe 不暂停。记录实际时长，不篡改或扣除 tracker 样本。
RTX4070TiSUPER；policy/VLM/assets revisions 与 E-RCV2 exact；50/1/10、20Hz、
threshold30、原 float32/linear P90/window50/margin1/cap8/guard2、any-late whole-discard、
identity/fallback identity、compile=false 均保持。reset/task 不清慢历史。

Env≤4，settling≤40，measured≤1120（每条≤280），主请求≤640（每条≤160），
candidate probe≤50/条、≤100总计（含主请求）；capture≤8，内部 setup/warmup/capture≤8/24/8。
startup≤30秒、单model≤15秒、单native≤30秒、每条ready后1200 slots/60秒，外层870/900秒。
attempt=1，retry/resume/replacement=0。技术首错停止后续条目，缺触发/未恢复/TimeLimit为有效负结果。
无真实机器人、训练、reference 或 predictor 调用，不干预同卡其他进程，不按负载择时重试。

## 准备与发布

只使用既有解释器及 uv offline/no-project/no-download，不安装、sync或升级。
新增 CPU 测试覆盖原 worker 中的 pause/phase/terminal、错误与未知区分、强杀汇总不丢首错、
初态检查点不二次reset，以及空队列不假阳性。保留实际失败日志和修正，再冻结。
模型环境仅 import/--help 准备；环境 metadata、GPU与磁盘背景分别保存。
计划、入口、测试先提交推送；Issue #1 登记完整执行 HEAD、唯一 output 和展开命令，
按实际返回 ID 回读正文 exact 后才执行。结果提交和发布回执另行保存。

## 事先固定的判读

新字段使用 `trace_*`，不改 E-RCV2 的 `stress_*`。
四条完成、两对初态 exact、source audit/清理/预算成立、未知调用0、child正常退出，
才置 `trace_four_episode_contract_passed=true`。同epoch的
paused planned→超cap→有效丢弃probe→原P90回cap→新planned→实际native row0
按原证据函数识别，CPU回读本轮数组独立核对。
两个disabled均需快bootstrap installed却不入history、history仍超cap且无后续planned。
四条合同和上述两个双臂observed同时成立，才置 `trace_mechanism_contrast_passed=true`。
任何缺证据保持 false。自然负载恢复、持续恢复、成功率/时延收益及统计显著性均不由此成立。
baseline_qualified/realtime_qualified/predictor_benefit_tested=false；risk_thresholds=null；
old_confirmation=not_started_untouched，旧E与D历史字段不变。
