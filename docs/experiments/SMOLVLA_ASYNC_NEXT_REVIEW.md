# SmolVLA 异步接入：下一轮审阅差异

## 当前：E-RCV3-trace独立核验通过，固定两对600ms受控暂停恢复机制对照接纳

2026-09-10，执行HEAD `563a8077b69785e90763be98da4dbce1bea7a996`。
原唯一队列已完成4/4条、2/2对；本次只完成退出后的CPU核验，新增model/Env/native/reference/predictor/训练均0，没有重跑15项准备测试。
本次补全[运行与核验断点5611313602](https://github.com/Lebron-233/lerobot/issues/1#issuecomment-5611313602)，没有第二次实验。
[完整结果](SMOLVLA_GRAPH_CAP_TRACE_RESULT.md)、[机器结果](SMOLVLA_GRAPH_CAP_TRACE_RESULT.json)、
[冻结PLAN](SMOLVLA_GRAPH_CAP_TRACE_PLAN.md)、[原15项测试](SMOLVLA_GRAPH_CAP_TRACE_TESTS.md)。

两个disabled的paused request3接纳后raw14，快bootstrap request4/5均raw3但不入tracker；历史不变、后续planned0，cap_wait74/54。
两个candidate各5个probe，request8将raw9→8；同epoch1/0的paused3→probe8→planned9→takeover index48→native49实际发送request9 row0。
两个native区间分别1104200.832341207→1104200.847018372、1104213.244521946→1104213.267225263。
CPU重新运行原source audit，573个measured动作及9次planned接管全部exact；两个row0直接数组比较exact。
四份真实checkpoint与保存初态、两对checkpoint/保存初态均exact；39份原窗口/P90/raw和terminal snapshot逐项exact。

`trace_four_episode_contract_passed`、`trace_disabled_latch_both_observed`、`trace_candidate_recovery_both_observed`、`trace_mechanism_contrast_passed`的运行器判定及独立核验均通过。
四条success分别148/161/136/128个measured动作；两臂各2/2 success。
两对candidate−disabled wall分别+0.457247458秒/+0.202198476秒，不能写成成功率提升或耗时收益。
natural_latency_recovery_demonstrated=false、生产默认disabled、旧E/E-RCV2及D历史字段保持，全部科学资格仍未提升。

真实账目：Env4、settling40、measured573、main39、probe10（5+5，含main）、capture8、setup8/warmup24/capture内8、replay39。
1253 intent/1253 return，error0/unknown0；613次底层native与外层调用的精确身份/动作及有序嵌套区间全部核对。
4/4 worker join/Graph释放/sampler恢复/metrics关闭/Env关闭确认，清理后数组466,153,260字节；首次模型前checkpoint另3,160,116字节。
child2847418 exit0、supervisor2847368 exit0均已确认；stop_reason和first_failure均null，无强制终止。
监督wall57.570463162秒、独立外层62.341006344秒；attempt1/retry/resume/replacement0。

补充metadata UTC01:54:54.650868，与准备及启动前Python/version/140项packages均exact；唯一补充GPU UTC01:54:54.706090为1159MiB/17%。
这些是收集时刻快照，不是实验刚退出快照；未重采、未干预其他进程，不归因性能。
首个收集器误读缺失elapsed、第二次误把外层journal和内层native时间要求相等，两个exit2原件保留。
按冻结接口修正字段及区间关联后，仅JSON完成剩余检查exit0；已通过数组检查复用，最终独立接纳全部true、未解决差异为空。
原运行器和旧结果没有被覆盖，冻结源码未改。原始目录 `outputs/smolvla_graph_cap_trace_563a8077/` 已到终点，不得再启动队列。
当前工作只剩本报告的结果提交、正式发布及独立回执收尾；下方为历史结果。

## 历史：E-RCV2已执行，两个候选恢复链成立，第四条模型超时，整体未通过

2026-09-09，执行HEAD `9b7aa8685ae1978c553da6e6d7ac3d4b38e3b9e6`。
本地终端完成准备、冻结、[预登记5603304679](https://github.com/Lebron-233/lerobot/issues/1#issuecomment-5603304679)实际ID单次exact回读与唯一四条native队列。
**3/4条完成、1/2对完整**，第4条在首个planned model request3超过15秒后按原监督器TERM/KILL终止；没有retry/resume/replacement。
[完整结果](SMOLVLA_GRAPH_CAP_STRESS_RESULT.md)、[机器结果](SMOLVLA_GRAPH_CAP_STRESS_RESULT.json)、
[实际TESTS](SMOLVLA_GRAPH_CAP_STRESS_TESTS.md)、[固定PLAN](SMOLVLA_GRAPH_CAP_STRESS_PLAN.md)、
[MANIFEST](SMOLVLA_GRAPH_CAP_STRESS_MANIFEST.json)、[执行任务书](SMOLVLA_E_RCV2_CODEX_EXECUTION_PLAN_20260909.md)。
原始目录 `outputs/smolvla_graph_cap_stress_9b7aa868/` 已有执行与退出证据，首次执行机会已消耗，不得重跑。
结果HEAD `f132f6cec76bf952bbbfa014318e6aefb815e650` 已推送；
[结果评论5603596852](https://github.com/Lebron-233/lerobot/issues/1#issuecomment-5603596852)按实际ID单次回读exact。
[独立发布与退出回执](SMOLVLA_GRAPH_CAP_STRESS_RECEIPT.json)随本次独立回执提交保存，实际完整回执HEAD在提交后写入本地HANDOVER。

| ordinal | task / arm | outcome | measured返回 | model intent/return | recovery probe | planned接管 |
|---:|---|---|---:|---|---:|---:|
| 0 | 0 / disabled | success | 148 | 6/6 | 0 | 0 |
| 1 | 0 / candidate | wall_slot_limit | 270 | 31/31 | 20 | 5 |
| 2 | 2 / candidate | success | 127 | 14/14 | 7 | 2 |
| 3 | 2 / disabled | started_return_unknown | 23 | 4/3 | 0 | 未确认 |

前三条实际暂停600.073085/600.223495/600.062056ms，首个planned均deadline_miss整块丢弃但时延接纳，raw分别14/16/16。
task0 disabled的快bootstrap request4/5均raw3、installed而不入历史；历史仍raw14，后续planned0，闭锁观测成立。
两个candidate均同epoch 1/0：paused3→probe13→planned14→index98/native99 row0，paused3→probe11→planned12→index81/native82 row0。
退出后CPU读取实际dispatch/chunk/native动作及起止，逐值exact。首次回cap各用9/7个probe；task0后来再次超cap，最终raw18，task2最终raw5。
因此 `stress_candidate_recovery_both_observed=true`；四条合同、两个disabled闭锁、整体机制对照均false，natural_latency_recovery_demonstrated仍false。
task0配对初态推理前与CPU回读exact；第二对没有完整归档，不能计为完成对照。

实际Env4、settling40、measured568，底层native608次全部返回；1258 intent/1257 return，未知model1、未知native0。
完整三条归档主调用51、capture6、setup6/warmup18/capture内6、replay51、probe27；第4条startup三请求已返回，
按固定源码推知另2capture及2/6/2内部调用，最终planned内部进度与完整API账目未保存。55是journal主请求intent数。
545个完成条目动作通过原source audit，余23个动作没有归档源审计；三个已保存数组合计446,513,691字节。
前三条5项清理均确认，第4条清理均未确认、无Env close intent。child -9、监督2，退出均确认；监督wall144.368160秒，独立外层174.076897秒。
首错原件为execution.stop_reason.expired_calls/calls.jsonl的ordinal3 request3 call1251；冻结汇总first_failure/budget为null的报告缺口已在正式报告中明确补充，原件未改。

准备接纳历史5项CPU，新增3项，受影响回归2项复查通过；当前8项不同CPU用例通过，入口/lint/format exit0，原开发首错保留。
模型Python/version/140项metadata前后exact。GPU各一次背景快照6010MiB/47%→4088MiB/11%，未干预其他任务或选择低负载时机。
本次本地终端命令正常返回，没有新的工具拒绝；旧插件拒绝属于历史事件，不能写成当前仍待放行。
已按首错结束实验，结果发布与exact回读完成，独立回执已落盘。旧E/E-S1/E-RCV1及全部科学资格保持。下方为历史结果。

## E-L1闭锁复现、E-RCV1四条诊断完成，native未触发恢复（历史结果）

2026-09-09，执行HEAD `2544bf404d0698ac60bcf1fee6d302fec1455a4a`。
已按新任务书完成旧task0/task2两条async的最小证据重建、CPU判别、默认关闭的候选与准备门，
[登记5601625997](https://github.com/Lebron-233/lerobot/issues/1#issuecomment-5601625997)实际ID单次回读exact后，完成唯一四条新native队列。
[完整结果](SMOLVLA_GRAPH_CAP_RECOVERY_RESULT.md)、[机器结果](SMOLVLA_GRAPH_CAP_RECOVERY_RESULT.json)、
[E-L1审计](SMOLVLA_GRAPH_CAP_RECOVERY_AUDIT.md)、[固定协议](SMOLVLA_GRAPH_CAP_RECOVERY_PLAN.md)、
[四行清单](SMOLVLA_GRAPH_CAP_RECOVERY_MANIFEST.json)、[实际测试](SMOLVLA_GRAPH_CAP_RECOVERY_TESTS.md)。
结果HEAD `a27a13b9602ca93825b31074f26a00f48e95f734`已推送；
[结果评论5601832578](https://github.com/Lebron-233/lerobot/issues/1#issuecomment-5601832578)按实际ID单次回读，正文exact。
[独立发布和退出回执](SMOLVLA_GRAPH_CAP_RECOVERY_RECEIPT.json)记录完整执行/结果身份、发布及实际退出。

旧task0五个接纳样本按原float32/linear P90得到0.714749813079834秒、raw16；
两次installed bootstrap不入tracker，后续无planned；task2最终raw12也没有后续planned。
CPU真实worker/queue复现该闭锁，健康正控制保持planned；reset/task保留历史、新engine空窗口的边界已记录。
新增`same_path_discard_probe_v1`仅显式实验启用：在原cap_wait机会同路径计算，输出丢弃、同epoch有效完成才接纳一次，
每条50预算派发前生效且reset/task不重置；默认关闭、原cap8/P90/window50/whole-discard和bootstrap排除保持。
受控满窗口正例经45个快probe自然回cap并planned/合法row0接管；持续慢probe不会强制恢复。
当前25项独立CPU覆盖按4+21通过，模型环境新入口1项通过；lint/format通过，开发首错分别保留。

新task0 serialized/async均success@200；task2 async/serialized均TimeLimit@280。4/4条、2/2对完成，初态exact。
**四条稳态raw全部3、恢复probe0；三个native恢复observed字段均false。**
`recovery_native_four_episode_contract_passed=true`、`recovery_native_two_pair_comparison_complete=true`仅表示本次固定队列完成，
不能代替“超cap→有效probe→回cap→planned→native row0”恢复链。本轮没有该链，也没有原算法native对照臂。
两对async−serialized measured wall分别−0.399644秒和−0.599796秒，是本次正常路径的描述性结果。

实际Env4、settling40、measured960、主策略52（startup bootstrap8/probe4/planned40）、recovery0、capture8；
setup8/warmup24/capture内8，reference/predictor/训练/真机0。2040 intent/2040 return，未知调用0。
40次正常planned接管和960次动作来源通过原审计；async正常model/native重叠31次，serialized为0。
4/4 join/Graph释放/sampler恢复/metrics关闭/Env关闭确认，child与监督exit0；监督79.406426秒、独立外层83.664648秒。
数组清理后保存775,266,588字节，退出后仅CPU读取新初态；140项模型环境metadata前后exact。
原始目录为`outputs/smolvla_graph_cap_recovery_2544bf40/`；执行、结果、发布回执分别提交。

唯一队列已到固定停止点，没有追加探针覆盖或重开旧E。旧E仍7 completed/1 startup失败/12 not_run、4 success/3 TimeLimit、3完整对；
旧整体两个flag仍false，三个observed与D三个历史通过flag保持true；科学资格仍false、risk_thresholds=null、旧confirmation untouched。
GPU前后快照只记背景，不能解释旧E的慢slot。DevSpace复发条目已存在，读取后未重复追加；本轮无新的安全拒绝。

## 以下为E-S1原cap8唯一startup诊断的历史结果

2026-09-09，执行HEAD `6ff4fd48098023de89b348d9f384ae82979935c8`。
E-S0已独立读取旧报告/事件/CPU初态/评论/退出并完成边界复算；E-S1准备和测试提交后，
[登记5600354734](https://github.com/Lebron-233/lerobot/issues/1#issuecomment-5600354734)回读一次exact，随后完成唯一新案例。
[E-S1完整结果](SMOLVLA_GRAPH_STARTUP_BOUNDARY_RESULT.md)、[机器结果](SMOLVLA_GRAPH_STARTUP_BOUNDARY_RESULT.json)、
[旧边界审计](SMOLVLA_GRAPH_STARTUP_BOUNDARY_AUDIT.md)、[固定协议](SMOLVLA_GRAPH_STARTUP_BOUNDARY_PLAN.md)。
结果提交`a9bb53f51ad8109710c4702b6817f048cadb0f52`已推送；
[结果评论5600443170](https://github.com/Lebron-233/lerobot/issues/1#issuecomment-5600443170)按实际ID回读一次，正文exact。
[独立发布和退出回执](SMOLVLA_GRAPH_STARTUP_BOUNDARY_RECEIPT.json)记录完整执行/结果身份与实际进程退出。

旧失败的9来自0.36472486099228263秒×20Hz，原取整为8，再加一次margin1；
startup gate使用当前probe，没有P90，未先clamp，失败样本未入tracker。原cap8合法拒绝，未识别/修复源合同缺陷。
本轮新增独立startup入口及CPU主机快照，生产算法、旧E入口及所有冻结条件保持。

新ordinal0复制旧ordinal7：task3/pair3/graph_serialized/state41，Env seed940341、policy seed950341。
10 settling后独立CPU双图/state/quaternion/EEF/gripper与旧失败项exact。
本次probe68.071492ms，换算2步+margin1=3≤8，cold/probe/fresh共3次主调用、2次capture；
setup2/warmup6/capture内2、replay3，原owner seed一次。
到ready后立即stop，get0、measured0、unknown0；所有20个journal intent均有return。
join/Graph释放/sampler恢复/metrics关闭/Env关闭确认，child和监督exit0；监督15.631292秒、独立外层19.116673秒。
CPU定向19 passed，固定模型环境新入口1 passed，ruff/format通过；140项模型环境metadata前后exact。
原始结果位于 `outputs/smolvla_graph_startup_boundary_6ff4fd48/`，CPU数组1,603,297字节，清理退出后回读确认。

一次新startup通过不改变旧超限事实，不证明稳定修复或原E完整合同通过。旧E仍7 completed、1 startup失败、12 not_run，
其中4 success/3 TimeLimit，3完整对；旧三个observed与D历史三项通过标志保持，两个E整体标志仍false。
科学资格仍false、risk_thresholds=null、旧confirmation untouched。诊断已停止，未续跑旧清单或新建E-r1。
本次同卡登记前1337MiB/4%只是背景记录，不能与旧6248MiB/40%组成因果实验。
DevSpace原拦截记录已按本任务书明确事实追加并回读，缺失待追加附件没有被当作已读取原文。

## 以下为 E 原生闭环的历史执行记录

2026-09-09。E 已完成实现、定向测试、预登记及唯一一次固定native队列；在第8项startup首错停止。
执行HEAD `e00b8731b44b80a9e80dc0af91e4434a9af3d7dd`。
7/20 episodes完成、1失败、12 not_run，3/10完整配对；失败后没有修补重跑或改变实验条件。
[E完整结果](SMOLVLA_GRAPH_IDENTITY_NATIVE_RESULT.md)、[机器结果](SMOLVLA_GRAPH_IDENTITY_NATIVE_RESULT.json)、
[E固定协议](SMOLVLA_GRAPH_IDENTITY_NATIVE_PLAN.md)、[20行清单](SMOLVLA_GRAPH_IDENTITY_NATIVE_MANIFEST.json)。

## E原历史停止点：startup要求9步，固定cap为8

ordinal7 / task3 / state41 / graph_serialized，在10 settling后与同对async初态exact。
probe request1实测364.724861ms，按原20Hz换算8步，加安全余量1得到9，
`startup_gate_outcome=cap_exceeded`，`latency_tracker_admitted=false`。
原门限拒绝该请求；fresh_warmed未运行，measured native intent为0。
该项主模型2次、capture1次；worker join、Graph释放、原sampler恢复和Env关闭均确认。
首错原件在 `outputs/smolvla_graph_identity_native_e00b8731/episode_007/result.json`。

全队列实际83主调用、reference0、15 capture；setup15 / warmup45 / capture内15，replay83。
8个Env各10 settling（总80），1585次measured native均返回，未知调用0。
7个完成项各2 capture，失败项1 capture；没有measured阶段capture。
child/supervisor均exit2，退出已确认；监督wall189.522851秒，独立外层194.259808秒，无超时或强杀。
全部8项清理和退出后大数组归档完成；后续12项未启动。

## 已取得与未完成的工程证据

| 接口或合同 | E实际证据 | 当前判断 |
|---|---|---|
| 两条件实现 | 同一原GraphIdentityEngine、完整50行、原worker/queue/planner；差异仅控制端等待 | 生产默认实现未改 |
| 配对初态 | task0–3的4次第二条件均raw双图/8Dstate/quaternion/EEF/gripper exact | 前3对完整；task3只有async完成 |
| 多行消费和来源 | 全部1585次发送通过activation/index定位chunk与row后核对；async各完成项连续多行 | native_multirow_consumption_observed=true |
| 合法接管及反馈 | async34次planned takeover从新row0开始，后续请求使用新native反馈 | native_replanning_takeover_observed=true |
| 模型/control重叠 | async51个真实模型/native区间交集，完成的serialized为0 | native_model_control_overlap_observed=true |
| 整体原生合同 | 第8项startup时延门失败 | native_closed_loop_contract_passed=false |
| 固定10对比较 | 仅3对完成；task3部分完成；task4–9未运行 | paired_scheduling_comparison_complete=false |
| 清理与环境 | 8/8 join/释放/恢复/close确认，140项模型环境metadata前后exact | 无依赖变更、未知调用或重试 |

CPU定向46 passed；固定模型环境实际入口import-only/--help 1 passed；ruff/format通过。
CPU覆盖使用真实engine/queue和可控model/Env，不冒充native实测。
首个开发F811及后续I001 lint失败记录保留于准备目录，修正仅发生在冻结登记前。

## 已完成配对和时隙成本

| task | serialized | async | async−serialized wall秒 |
|---|---|---|---:|
| 0 | 成功@200步，10.371799秒 | 成功@210步，54.583567秒 | +44.211769 |
| 1 | 成功@138步，7.828472秒 | 280步TimeLimit，13.968797秒 | +6.140325 |
| 2 | 280步TimeLimit，15.024410秒 | 280步TimeLimit，21.041416秒 | +6.017006 |

额外task3 async成功@197步、11.539150秒；serialized未ready，不能计作完整比较。
完成项S=3、A=4，分母不同：S无动作48 slots（等待42/Env忙5/调度遗漏1），
A无动作1057 slots（Env忙566/调度遗漏470/underflow21）。等待和遗漏均计入实际wall分母。
A task0最长派发间隔10.476126秒；A底层native.step最大9.220309秒。
所有有限post7D命令按原Panda relativeOSC发送，没有补发、保持动作或二次归一化。

下一轮审阅应以cap_exceeded首错、分阶段实测和上述成本为依据。
新的实测范围与资源条件需由下一份任务书明确；本轮不自动继续第8项或余下12项。

## 历史已接纳结果

A/B执行 `5d45353ff98fc448bc514b284c1a19609405585b`：A连续模型合同通过；
B完成40/40 native、20对逐步exact，是50/1/10下单行selector的计算实现对照。
[B结果](LIBERO_GRAPH_NATIVE_EQUIVALENCE_RESULT.md)。

C首次执行 `290c1a3dbaa90d4e3d900c4c4eb83aa6840b2ce8` 完成6,318请求但原验收要求冲突。
strict-deadline新回放 `3ae1dc569007e612bca459ba6680dd94204268a6` 完成40条/6,318请求、7项边界和账目，exit0。
[C勘误与结果](SMOLVLA_ASYNC_STRICT_DEADLINE_RESULT.md)。

D3-r1执行 `04902a527d684dd43bb55098c8b5f39ae2a96fa2`：20对token与12真实worker事件通过，
52主调用+12reference、15 capture；39项CPU及模型环境导入门通过，环境不变，child/supervisor exit0。
token_graph_equivalence_passed、graph_worker_lifecycle_passed、graph_identity_engine_integration_passed保持true。
[D3-r1结果](SMOLVLA_GRAPH_TOKEN_WORKER_R1_RESULT.md)。旧D datasets导入失败的原结果保持，未覆盖。

## 接管与动作链的既有裁决

[正式裁决5471357164](https://github.com/Lebron-233/lerobot/issues/1#issuecomment-5471357164)和
[复核5554561794](https://github.com/Lebron-233/lerobot/issues/1#issuecomment-5554561794)：
early仅staging，准时从新row0接管，any late>0整块丢弃并继续旧active；耗尽None、index不推进。
stale不得清除更新plan，max_late_steps=2不授权裁剪。本次真实deadline_miss为0，定向测试含late1/2/3。

固定checkpoint完整输出50×32，有效前7维经原postprocessor反归一化，作为Panda relativeOSC的控制命令。
E从完整块按queue连续消费；B单行selector的原历史含义保持。
post动作不是已实现的EEF位姿变化，不能积分为8D未来state；normalized动作、axis-angle与双指qpos不采用delta_sum。
backend对enabled RelativeActionsProcessorStep的限制仍基于尚未定义的anchor/rebase合同，与native relativeOSC模式不同。
identity context/state在E保持；未来predictor或predicted-context实测需要另行固定合同。

## 已知限制

三个E observed字段不能代替完整20项合同，也不能证明async时延或任务收益。
登记时同卡另一项目约4637MiB，GPU总6248MiB/40%利用率；没有干预其他进程。
该背景不足以归因本轮慢slot或startup probe时延，没有在首错后变更负载重试。

baseline_qualified=false、realtime_qualified=false、predictor_benefit_tested=false、risk_thresholds=null，
old_confirmation=not_started_untouched保持。旧开发基线185/200中task5为14/20，原科学资格仍未通过。
E仅开发工程诊断，不作显著性检验；训练0、真实机器人调用0、旧确认队列未启动。
DevSpace原拦截记录已按任务书事实追加并回读；缺失待追加附件未被冒充成已读取原文。
