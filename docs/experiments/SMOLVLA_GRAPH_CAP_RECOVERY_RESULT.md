# E-RCV1：四条诊断完成，本轮未触发恢复探针

2026-09-09。已完成E-L1旧证据/CPU判别、默认关闭的候选实现与准备门、提交登记及唯一一次四条native诊断。
**原 planned 恢复闭锁已在CPU复现，候选CPU恢复证据通过；真实四条中raw始终为3，没有触发恢复探针。**
本次4/4条、2/2对完成；2次success、2次TimeLimit，未知调用0，清理和进程退出确认。

执行HEAD `2544bf404d0698ac60bcf1fee6d302fec1455a4a`，分支`codex/smolvla-graph-native-equivalence`。
[登记5601625997](https://github.com/Lebron-233/lerobot/issues/1#issuecomment-5601625997)按发布返回实际ID回读一次，正文exact后执行。
交接HEAD `223156f3d41dd37bc982bd691123f8490ed93445`；实现、结果、发布回执分开提交，后两者身份见独立RECEIPT。

## E-L1结论与代码变化

仅从旧E原件重建task0/task2 async的82+31个planner事件，没有重做315条总审计或读取旧大数组。
task0五个接纳样本的原float32/linear P90为0.714749813079834秒，20Hz原取整加margin1得raw16；
两次普通bootstrap分别345.023415ms和646.778322ms，均installed但不进tracker，之后没有新的planned。
task2最终P90为0.5179850459098816秒、raw12。两项超cap机会均无尚在途planned，稳态epoch保持1/0。
[AUDIT](SMOLVLA_GRAPH_CAP_RECOVERY_AUDIT.md)及[机器审计](SMOLVLA_GRAPH_CAP_RECOVERY_AUDIT.json)保留独立重建结果。

原路径CPU实际完成cold/probe/fresh、一次慢planned及两次快bootstrap，历史不变且无法恢复planned；
无慢样本正控制持续接管。当前reset和切换task保留tracker，新engine才创建空窗口，这一边界按真实源码记录。
E-S1 startup的旧9>8合法、新3≤8单次通过结论保持；没有把稳态问题回写成startup计算缺陷。

新增`same_path_discard_probe_v1`，生产构造默认disabled，factory/default配置不变。
原cap_wait机会可派发同identity planned计算路径的recovery_probe，永不install/stage输出；
有效同task/reset epoch完成才接纳一次时延，stop/失效不接纳；上限50且reset/task不重置预算。
bootstrap仍排除，原50窗口、P90、慢样本、margin1和cap8均保留；不做额外CUDA同步/event或GPU取值。
新增四条专用入口复用原worker/controller/queue，补充CPU历史前后审计。
同时修正实验归档边界：Env或Graph/processor/metrics关闭未确认时不保存大数组；正常路径保持。

## CPU准备的真实结果

候选修改前4项characterization通过。当前代码25项独立定向覆盖按4+21完成；模型环境新入口1项通过；ruff/format通过。
具体命令、退出码与首错见[TESTS](SMOLVLA_GRAPH_CAP_RECOVERY_TESTS.md)。
首轮I001导入顺序及测试把journal列表按ndarray读取的TypeError均保留；只修正导入/张量转换，候选算法和数值期望未因失败改变。

恢复正例通过真实worker的planned完成构造满50成员窗口，6个400ms样本使其超cap；
随后45个10ms有效probe自然滚动窗口，仍保留5个400ms慢样本时原linear P90已回cap，接着planned并合法row0接管。
持续慢500ms的50个probe则保持超cap，第51个机会派发前返回cap_wait；预算耗尽后的bootstrap不更新历史。
还通过staged/active/plan/index保留、有效接纳一次、同路径CPU完成、reset/task/stop失效、fatal首错、
两条件等待差异、固定预算以及close失败不保存数组等测试。
夹具显式使用CPU NativeEngine、FakeGraph和可控时钟，CPU恢复不是native恢复观测。

## 唯一四条native结果

四条均启用候选，task/state/seeds/顺序按已提交[MANIFEST](SMOLVLA_GRAPH_CAP_RECOVERY_MANIFEST.json)，
模型/环境/数值身份和完整停止合同见[PLAN](SMOLVLA_GRAPH_CAP_RECOVERY_PLAN.md)。
原RTX4070TiSUPER、固定模型解释器、policy/VLM/assets revisions、strict loader、bf16/fp32、AMP=false和50/1/10保持。
严格加载missing/unexpected/shape mismatches均空；推理trainable=0。
两对第二条件的双图、8Dstate、raw quaternion/EEF/gripper在推理前exact；退出后保存初态的CPU回读也exact。

| ordinal | task/condition | outcome | measured动作 | measured wall秒 | 主调用 | 正常planned接管 | recovery probe |
|---:|---|---|---:|---:|---:|---:|---:|
| 0 | task0 serialized | success | 200 | 10.371901 | 11 | 8 | 0 |
| 1 | task0 async | success | 200 | 9.972257 | 11 | 8 | 0 |
| 2 | task2 async | TimeLimit | 280 | 13.965560 | 15 | 12 | 0 |
| 3 | task2 serialized | TimeLimit | 280 | 14.565356 | 15 | 12 | 0 |

startup probe分别72.175770、71.472386、79.652247、71.955319ms，原gate均raw3/pass；
startup至ready分别3.371322、2.448415、2.538589、2.609518秒，均在30秒内。
四条所有稳态planner raw均3；cap_exceeded0、cap_wait0、recovery_probe0、steady bootstrap0。
100次`plan_failed`的原原因全为`plan_in_flight`：已有计划，未再派发另一计划；没有技术错误。

40次正常planned接管均经原chunk/request/index源审计，960次native发送逐值匹配确定的CPU chunk/row，接管从row0开始。
每个owner与controller分离、峰值inflight1，单次owner播种按原engine工厂执行。
async有31个真实model/native区间交集（14+17），serialized为0；这些属于正常planned路径。
serialized错过20个slot均记为serialized_wait，async无无动作slot；
env_busy、scheduler_miss、underflow类别的无动作slot均0，native调用自身耗时仍在wall分母中。
deadline_miss和stale结果均0。

两对A−S measured wall差分别−0.399644秒与−0.599796秒，本次动作数分别相同200和280。
这是候选开启时两对的描述性结果；本轮探针没有执行，不能把差额解释成恢复策略收益。

## 预算、清理、退出与成本

| 资源 | 实际 | 本轮总上限 |
|---|---:|---:|
| Env | 4 | 4 |
| settling发起/返回 | 40/40 | 40 |
| measured native发起/返回 | 960/960 | 1120 |
| 主策略 | 52 | 640，每条160 |
| recovery probe（主调用子集） | 0 | 200，每条50 |
| capture | 8 | 8，每条2 |
| setup / warmup / capture内 | 8/24/8 | 8/24/8 |
| Graph replay | 52 | 随主调用 |
| reference/predictor/训练/真机 | 0 | 0 |
| 队列尝试 / 重试 | 1/0 | 1/0 |

52个请求构成为8个startup bootstrap、4个startup probe、40个planned。
小型journal共2040个intent/2040个return，其中1000个底层native step（40 settling+960 measured）；
960个environment_step是外层调用账目，不额外加成1920次动力学推进。未知调用0。

4/4 worker join、Graph释放、原sampler恢复、metrics关闭、Env关闭均确认；worker_failed=false、first_failure=null。
清理后保存4份arrays，总775,266,588字节；CPU只读回读新初态和档案结构，未初始化CUDA，未读取旧大数组。
原采集日志、计数及环境前后元数据见`collection.json`，其独立命令exit0。

child PID2649614，exit0且wait确认；监督wall79.406426秒。
独立外层2026-09-09T12:13:06.993459+00:00至12:14:30.658069+00:00，83.664648秒，监督exit0确认。
未触发870/900秒外层退出、15秒模型或30秒native超时，无TERM/KILL。
strict模型加载5.468530秒；新增CPU历史快照记录体成本0.014652秒，小型调用日志写入/flush合计0.121284秒，
清理后数组序列化0.585619秒。快照计时不含进入调用前的参数构造，完整wall保留全部实际开销。
模型Python/version/140项metadata前后exact，无安装、sync、升级或环境切换。
登记前GPU1608MiB/13%，退出后1449MiB/12%，两次均保留其他进程背景，未干预其他任务。

## 独立结果字段

| 字段 | 本次 |
|---|---|
| original_cap_latch_reproduced_cpu | true |
| same_path_recovery_passed_cpu | true |
| production_default_unchanged | true |
| recovery_native_diagnostic_started | true |
| recovery_probe_observed_native | **false** |
| recovery_estimator_returned_within_cap_native | **false** |
| planned_takeover_after_recovery_observed_native | **false** |
| recovery_native_four_episode_contract_passed | true |
| recovery_native_two_pair_comparison_complete | true |

四条合同通过表示本次固定队列正常完成；三个恢复observed为false表示没有触发覆盖。
本轮没有同epoch的“超cap→有效probe→回cap→新planned→native row0接管”链，不能报告native恢复成功。

## 原件与限制

原始目录：`/home/rp/Workspace/SmolVLA_RTC/lerobot/outputs/smolvla_graph_cap_recovery_2544bf40`。
保留原driver/worker/逐episode结果、requests/metrics/history证据、calls.jsonl、model.log、CPU arrays、
执行与独立退出回执、GPU前后/模型环境前后、登记/回读、CPU/lint首错及collection。
CPU完整夹具evidence位于单独preparation目录，原始路径保存在测试receipt；大数组不提交Git。
[机器结果](SMOLVLA_GRAPH_CAP_RECOVERY_RESULT.json)与独立发布RECEIPT引用同一执行身份，HANDOVER/NEXT_REVIEW已续接。

两对是有意选择的困难开发案例，没有原算法native对照臂，无法量化候选对原算法的收益或泛化。
本轮无超cap，CPU正例不能代替native恢复，GPU快照不能解释与旧E的时延差异。
同路径指模型输入和完整完成计算；probe不做planned专属的queue plan构造及prefix证据拷贝，真实成本可能不同。
按需Env推进与本次无underflow也不构成真机等待安全或硬实时资格。
没有重复、补跑、扩样或用旧native拼接覆盖；本轮达到固定停止点。

旧E保持7 completed/1 startup失败/12 not_run、4 success/3 TimeLimit、3完整对；
native_closed_loop_contract_passed=false、paired_scheduling_comparison_complete=false，旧三个observed保持true。
D三个历史CUDA通过字段保持true。baseline_qualified/realtime_qualified/predictor_benefit_tested仍false，
risk_thresholds=null、old_confirmation=not_started_untouched；旧A/B/C/D/E/E-S1原报告及三个未跟踪文档保留。
实际DevSpace拦截记录已读取，确认复发条目存在，没有重复追加；本轮未发生新的安全拒绝。
