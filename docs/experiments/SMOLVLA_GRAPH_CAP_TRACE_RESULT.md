# E-RCV3-trace：独立核验通过，固定两对受控暂停恢复机制对照接纳

2026-09-10。原唯一队列完成 **4/4条、2/2对**；本次退出后的独立CPU核验全部通过。两个disabled均形成planned闭锁，两个candidate均经有效丢弃probe回到cap内，并在实际native控制中合法发送新planned块row0。

本任务仅收集既有原件，新增model、Env、native step、reference、predictor和训练调用均0。它补全[运行与核验断点5611313602](https://github.com/Lebron-233/lerobot/issues/1#issuecomment-5611313602)，没有第二次实验。

| 判定 | 运行器原结果 | 退出后独立核验 |
|---|---|---|
| 四条完整合同 | `trace_four_episode_contract_passed=true` | 通过 |
| 两个disabled闭锁 | `trace_disabled_latch_both_observed=true` | 通过 |
| 两个candidate完整native恢复链 | `trace_candidate_recovery_both_observed=true` | 通过 |
| 整体机制对照 | `trace_mechanism_contrast_passed=true` | 接纳 |
| 自然负载恢复 | `natural_latency_recovery_demonstrated=false` | 保持false |
| 生产默认 | `production_default_unchanged=true` | 仍为disabled |

## 身份与原件

执行HEAD：`563a8077b69785e90763be98da4dbce1bea7a996`，分支`codex/smolvla-graph-native-equivalence`。
[预登记5604094500](https://github.com/Lebron-233/lerobot/issues/1#issuecomment-5604094500)的已保存回读body与本地registration.md程序化exact；启动前的独立preflight也记录了exact门。
此次只读Issue最新一条评论，确认仍为5611313602。最终结果和独立发布回执分开提交；完整结果/回执提交身份在后续发布回执与实际HANDOVER中记录。

唯一原始目录：`/home/rp/Workspace/SmolVLA_RTC/lerobot/outputs/smolvla_graph_cap_trace_563a8077/`。
执行/退出、加载和模型记录分别为`execution.json`、`independent_execution_receipt.json`、`worker_result.json`、`policy_load.json`、`calls.jsonl`及`model.log`。
四个`episode_000`至`episode_003`均有`started.json`、`result.json`、`initial_checkpoint.pt`和`arrays.pt`。
正式[机器结果](SMOLVLA_GRAPH_CAP_TRACE_RESULT.json)保留运行器判定、独立判定、完整窗口、恢复时间线和各门回执。

[冻结PLAN](SMOLVLA_GRAPH_CAP_TRACE_PLAN.md)与[TESTS](SMOLVLA_GRAPH_CAP_TRACE_TESTS.md)保持。原15项不同CPU测试最终15 passed in 0.83s，入口import/--help及lint/format均exit0；本次没有重跑这些测试。准备阶段SIM117和格式差异原件保留。
本次使用本地终端；已读取既有拦截记录，没有重复追加历史事件，也没有新的工具拒绝。

## 四条实测与配对比较

两臂均为原graph_identity_async，candidate为已有same_path_discard_probe_v1；task0 disabled→candidate、task2 candidate→disabled，state41，原Env/policy seeds940041/950041及940241/950241。
原policy/VLM/assets revisions、50/1/10、20Hz、float32/linear P90/window50/margin1/cap8、threshold30、guard2、any-late whole-discard、identity/fallback identity和compile=false保持。

| ordinal | task / arm | outcome | measured动作 | measured wall秒 | 主请求 | probe | steady bootstrap | planned接管 | 无动作slots |
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0 / disabled | success | 148 | 7.568293 | 6 | 0 | 2 | 0 | 4 |
| 1 | 0 / candidate | success | 161 | 8.025540 | 14 | 5 | 0 | 5 | 0 |
| 2 | 2 / candidate | success | 136 | 6.770089 | 13 | 5 | 0 | 4 | 0 |
| 3 | 2 / disabled | success | 128 | 6.567890 | 6 | 0 | 2 | 0 | 4 |

两臂各2/2 success。task0的candidate−disabled为动作+13、wall+0.457247458秒、主请求+8、planned接管+5、无动作slot−4；task2分别+8、+0.202198476秒、+7、+4、−4。
两条candidate实测wall均更长；这组结果不支持成功率提升或耗时收益。

四条总slots分别152/161/136/132，全部8个无动作slot均为disabled的underflow；env_busy/scheduler_miss均0，wall_window_overshoot均0。
最长dispatch gap分别0.149369/0.057106/0.061753/0.149392秒。37次plan_failed均为plan_in_flight，没有附加派发或动力学推进。
startup分别3.175779/2.431516/2.516587/2.398896秒；episode wall分别13.555068/12.371860/11.129237/10.701447秒。

## 暂停、闭锁与native恢复链

四条暂停均仅发生于首个planned request3、reset/task epoch1/0。原CPU chunks和完成屏障先返回，host_pause随后开始；原请求时延采样、接纳与terminal仍在原路径。

| ordinal | 实际暂停ms | policy API ms | total chunk ms | late steps | 暂停后P90秒 | raw |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 600.073476 | 54.855437 | 691.522948 | 11 | 0.6291494965553284 | 14 |
| 1 | 600.076444 | 54.464571 | 686.800688 | 11 | 0.6249379515647888 | 14 |
| 2 | 600.032547 | 54.622934 | 673.066429 | 11 | 0.6126479506492615 | 14 |
| 3 | 600.061333 | 55.043309 | 678.558378 | 11 | 0.6174227595329285 | 14 |

四个慢planned均deadline_miss整块丢弃，但真实时延被原tracker接纳。policy API列为原model起止区间，该planned路径的vision另行计时；它与包含主机暂停的total chunk分开保留。

两个disabled的窗口分别保持`[0.06778867589309812, 0.6915229479782283]`及`[0.06720230914652348, 0.6785583780147135]`。
task0后续bootstrap request4/5为87.816571/81.536231ms；task2为83.280164/75.074468ms，按原整数容差换算均raw3。
四次bootstrap均installed但`latency_tracker_admitted=false`，窗口逐值不变，仍raw14；两个disabled后续planned均0，cap_wait分别74和54。两个闭锁证据均成立。

两个candidate各完成5个recovery_probe，request4–8均输出丢弃且有效时延仅接纳一次。raw序列均为14→13→12→10→9→8。
第5个probe request8将task0 P90从0.3888339698314667降至0.3292403817176819秒，task2从0.38770025968551636降至0.33062678575515747秒；原慢样本仍在各自7成员窗口中。

| candidate | 同epoch链 | probe8返回 | planned9请求 / 完成 | takeover index | native49发起 / 返回 | 实际来源 |
|---|---|---:|---|---:|---|---|
| task0 / ordinal1 | paused3→probe8→planned9 | 1104200.602033508 | 1104200.6314421 / 1104200.710700255 | 48 | 1104200.832341207 / 1104200.847018372 | request9 row0 |
| task2 / ordinal2 | paused3→probe8→planned9 | 1104213.076046946 | 1104213.092891432 / 1104213.167143217 | 48 | 1104213.244521946 / 1104213.267225263 | request9 row0 |

表中时间均为原单调时钟。两条probe完成后才发起planned9；原queue合法takeover，source_request_id=9、source_row_offset=0。
CPU逐值比较保存的post_chunk[0]、control.dispatches命令和实际native动作，结果exact；绝对动作索引、来源身份和native起止与恢复记录也exact。
candidate最终raw分别3和4，完整planned接管分别5和4；probe输出没有进入动作来源。

## 独立核验与账目

独立CPU工作复用`initial_difference`、`audit_episode`、`slot_accounting`、`stress.evidence`、`journal_accounting`和原LatencyTracker/latency_to_steps，没有加载policy或创建Env。

- 四个checkpoint与本条arrays中的初态逐值exact；两对checkpoint以及两对保存初态也exact，范围为双图、8D state、raw quaternion、EEF和gripper。四个checkpoint事件均位于首次模型intent之前。
- 用本轮CPU数组重新执行原source audit，四份结果与运行器原审计exact，覆盖全部573个measured动作和9次planned接管。两个恢复row0另作直接数组回读。
- 按39个请求的原接纳标记逐次调用原LatencyTracker，逐项复算全部前后窗口、P90/raw；39份request_snapshot与terminal/请求记录exact。10个probe均只新增一个有效样本，普通bootstrap不入历史。
- journal中的模型阶段538个事件形成269组完整起止，包含39个whole_request、4个host_pause及4个engine_stop区间，无phase error或缺失terminal。
- 四份cleanup与journal确认worker join、Graph释放、sampler恢复、metrics关闭；随后4次Env close返回。冻结run_episode只在这些门全部通过后序列化数组，四份数组与serialization记录齐全。
- 全部613次底层native以ordinal/segment/number及动作、slot、action_index、observation_index对应journal；实际native区间位于外层call intent/return区间内。无额外容差或时钟调整。

| 资源 | 冻结上限 | 独立核对实际值 |
|---|---|---|
| Env | 4 | 4创建、4关闭 |
| settling | 每条10 / 总40 | 10×4=40，全部返回 |
| measured native | 每条≤280 / 总≤1120 | 148+161+136+128=573，全部返回及源审计通过 |
| 主请求 | 每条≤160 / 总≤640 | 6+14+13+6=39，39返回、39snapshot |
| recovery probe | candidate每条≤50 / 总≤100，含主请求 | 5+5=10，全部丢弃接纳 |
| capture | 每条≤2 / 总≤8 | 2×4=8，全部对应startup |
| setup / warmup / capture内部 | ≤8 / 24 / 8 | 8 / 24 / 8 |
| Graph replay | 单列 | 39 |
| reference / predictor / 训练 / 真机 | 0 | 0 |

主请求39=startup bootstrap8 + startup probe4 + planned13 + steady bootstrap4 + recovery probe10。
pre/policy/prepare_images/noise/vision/post各39，reset各4；计数与39份journal snapshot一致。
capture及其内部计数来自8份实际runtime capture记录，关联startup请求和snapshot，并由原audit再次核对；本轮没有用缺失计数的推知值补账。
底层native613=settling40+measured573；外层environment_step573描述同一次推进，不额外相加。

全journal **1253 intent / 1253 return，error0、unknown0**，没有重复intent/terminal或孤立返回，预算与worker/result汇总一致。
最长model watchdog区间1.894540288秒，最长native watchdog区间0.055210598秒；各startup≤30秒、ready≤1200 slots/60秒，全部处于原预算内。

四份大数组共466,153,260字节，四份初态checkpoint共3,160,116字节，仅保存在本地。本次始终map_location=cpu，CUDA未初始化，没有读取旧轮大数组。
原执行记录的history audit0.015578790秒、journal logging0.144417738秒、数组序列化0.328954236秒均保留，wall未扣除这些开销。

## 退出、加载与补充环境快照

child PID2847418实际exit0、退出确认；supervisor PID2847368实际exit0、退出确认。stop_reason=null、forced_termination=false，worker/result四条first_failure均null。
监督UTC01:29:50.346958→01:30:47.917386，wall57.570463162秒；独立外层UTC01:29:46.255966→01:30:48.596945，wall62.341006344秒。attempt1、retry/resume/replacement0。

提前落盘的policy_load与worker/result一致：strict=true，missing/unexpected/shape mismatch均空，加载5.079062667秒；32层VLM/32层expert、expert hidden480，bf16参数600,902,304、fp32参数4,031,872，AMP=false，推理trainable参数0。

补充收集的模型metadata时间为UTC **01:54:54.650868**，GPU时间为UTC **01:54:54.706090**，均晚于实验退出。
模型Python、完整版本及140项包metadata与准备快照、启动前快照均exact。
此次唯一补充GPU快照为RTX4070TiSUPER总16376MiB、使用1159MiB、利用率17%，compute-app查询返回空列表，两个查询均exit0。
启动前原preflight时间UTC01:29:46.250607，记录1013MiB/41%、compute-app空列表及磁盘1,472,687,800,320字节。本任务没有重复采样选择负载，也没有干预其他进程。

## 收集脚本修正与解释限制

首个CPU收集器exit2：假设所有call_return都有elapsed，实际39个model_request return只有timestamp，因此发生KeyError。首个脚本、日志、JSON与独立退出回执原样保留。修正仅从原watchdog所用的intent/return时间戳求差，复用已采集metadata/GPU快照。

第二次CPU回读的数组、窗口、机制、预算、清理和退出检查均通过；仅新增收集器误将外层journal时间与内层native函数时间要求相等，汇总exit2。冻结NativeSession明确在Calls.call内部单独记录native起止，故这两层边界不同。
最后仅用JSON核对613个调用的精确身份/动作和有序嵌套区间，exit0；已通过的数组检查直接复用，没有再次加载数组或运行模型。原两份失败收集回执均保留，最终整合判定为`independent_cpu_acceptance_20260910.json`，无未解决审计差异。

本轮接纳仅覆盖固定两对案例、600ms受控主机暂停下的恢复机制。它不证明自然负载持续恢复、成功率提升、耗时收益、泛化、真机等待安全或硬实时资格。补充GPU快照不是刚退出快照，也不能解释性能因果；新增trace开销使E-RCV3与E-RCV2的wall不能直接作为性能对比。

生产默认disabled，旧E/E-RCV2原结果保持，E-RCV2整体仍false；D历史字段不变。baseline_qualified/realtime_qualified/predictor_benefit_tested均false、risk_thresholds=null、old_confirmation=not_started_untouched。
本任务在正式结果及发布回执提交后结束，没有授权或启动下一轮实验。
