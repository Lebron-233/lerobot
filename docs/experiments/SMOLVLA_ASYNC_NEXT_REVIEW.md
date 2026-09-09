# SmolVLA 异步接入：下一轮审阅差异

## 当前：E-S1 原 cap8 的唯一 startup 诊断已完成

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
