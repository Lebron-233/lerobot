# E-S1：旧启动门拒绝正确；新单案例 startup 通过

2026-09-09。E-S0原件审计、E-S1实现/定向测试/登记、唯一一次真实诊断和退出后收集均已完成。
**旧9>8可精确复算，未识别源合同缺陷。新诊断在原cap8下所需3步，startup通过后立即停止，measured=0。**

执行HEAD：`6ff4fd48098023de89b348d9f384ae82979935c8`，分支`codex/smolvla-graph-native-equivalence`。
[预登记5600354734](https://github.com/Lebron-233/lerobot/issues/1#issuecomment-5600354734)按发布返回的实际ID同步回读一次，正文exact后启动。
旧执行/结果/回执分别为`e00b8731b44b80a9e80dc0af91e4434a9af3d7dd`、
`b1bf64077f4db736a9c541096d3543bdccf93055`、`9038ba38b89ab3e26671bc042dbeac9b6cdc124c`。
实现、结果和发布回执分开提交；后两者身份由独立RECEIPT记录。

## 9的来源与合同判断

旧ordinal7/task3/graph_serialized/state41的probe request1完成样本为
`L=0.36472486099228263 s`。原`latency_to_steps(L,20)`得到`ceil(7.294497219845653)=8`，
再加一次安全余量1，得到未截断所需9；原gate在clamp前比较`9>8`并拒绝。
这一门直接使用当前probe，**没有P90计算**，tracker前后均为空，失败样本未被接纳。
9不是去噪次数、Env步数或精确450ms的模型时延。

样本从原request.requested_at开始，到原CPU输出与既有设备完成屏障为止，包含派发、观测/pre、vision、policy、post、
独立CPU chunk和原E证据拷贝。旧派发等待7.216001ms、vision212.043553ms、policy78.841263ms、post23.056821ms。
控制端等待没有第二次相加，margin也只加一次；cold/fresh capture时长未进入该gate或tracker。
旧`cuda_completed_at_s`是按原完成定义重建的host端点，不是独立GPU内核终点。

后续控制规划才使用tracker：窗口50的float32成员，经numpy默认linear P90、原整数容差/向上取整和margin1得到raw，
再按cap8与available−guard2截断planned。旧315个planner判据全部复算exact；71次raw>8是规划机会，
由69次cap_wait和2次active耗尽后的identity bootstrap处理，不等于71次模型调用。
[AUDIT](SMOLVLA_GRAPH_STARTUP_BOUNDARY_AUDIT.md)和[机器审计](SMOLVLA_GRAPH_STARTUP_BOUNDARY_AUDIT.json)
保留20项状态、23个startup、65个tracker接纳样本及原控制边界事实。

本轮新增`libero_graph_startup_boundary.py`、CPU测试和审计/协议/报告。
新入口派生旧NativeEngine，复用真实worker/queue/startup、原loader和native工厂，补充9条CPU主机事件。
**生产算法、旧E入口、阈值和所有旧结果均未修改；source_contract_defect_identified/fixed均false。**

## 唯一新案例与原始初态

新ordinal0对应source_ordinal7；libero_object/task_order_index0，task3/pair3，
`pick_up_the_bbq_sauce_and_place_it_in_the_basket`，graph_serialized，state41，Env seed940341，policy seed950341。
原工厂创建1个Env，ensure中必要隐含reset后，原seed/reset/set_init_state各一次，再执行10个原settling动作。
新独立CPU双图、8Dstate、raw quaternion/EEF/gripper与旧ordinal7全部exact，核对发生在首次策略推理前。
退出后的arrays CPU回读再次确认保存内容一致，47个tensor全CPU/finite，CUDA未初始化。

保持policy `6721902bc4d61e50a3bfdb11dfb4cb626f05d102`、VLM `7b375e1b73b11138ff12fe22c8f2822d8fe03467`、
assets `0b3ea86be5fe169d0fd036ae63d1070ec09e90f6`，原NVIDIA GeForce RTX 4070 Ti SUPER与严格loader。
missing/unexpected/shape mismatch均空；bf16参数600902304、fp32参数4031872，AMP=false，推理trainable=0。
50/1/10、20Hz、threshold30、P90/window50、margin1/min0/max8、guard2/max_late2、整块late丢弃、identity、compile=false均保持。
固定模型解释器与140项distribution metadata前后exact；无安装、sync或依赖改变。

## 新请求、gate与tracker的真实顺序

| request | phase | reset/task/obs | 完成样本ms | capture / replay | tracker接纳 | 结果 |
|---|---|---|---:|---|---|---|
| 0 | cold_temporary | 0/0/0 | 1853.728077 | 1 / 1 | false | 临时块installed，ready=false |
| 1 | probe | 0/0/0 | 68.071492 | 复用1 / 1 | true | raw3≤8，probe_discarded，epoch进入1 |
| 2 | fresh_warmed | 1/0/0 | 1185.937174 | 2 / 1 | false | 新块installed，ready=true，立即stop |

本次probe端点为`1050023.863118908 → 1050023.931190400`（原host monotonic秒），
`L=0.06807149201631546`，乘20为`1.3614298403263092`，向上取整2再加1得到3。
gate之前/刚返回时tracker均`[]`；原完成通知前已加入唯一probe样本并进入epoch1；fresh返回时仍仅这一个成员。
这是原同episode校准序列。cold/fresh样本均未被用作probe门或tracker样本。

probe的host阶段：派发0.522523ms、观测0.840574ms、pre0.419941ms、vision12.163093ms、
policy52.344326ms、post1.369061ms；原最终设备完成等待0.019837ms。
完整完成样本68.071492ms照原定义记录，未扣除日志、拷贝或未被单独分段的开销。
3个完整输出均`[1,50,32]` finite，3次原CPU chunk与设备完成屏障确认，3次主调用均Graph replay。
policy/processor调用均归属owner `125290451813952`，与controller `125299514857280`分离；
原owner seed950341恰好一次，峰值inflight1，queue get0。

## 实际预算、清理与退出

| 项目 | 实际 | 固定上限 |
|---|---:|---:|
| Env | 1 | 1 |
| settling发起/返回 | 10/10 | 10 |
| measured native / queue get | 0/0 | 0/0 |
| 主策略 / reference | 3/0 | 3/0 |
| capture | 2 | 2 |
| eager setup / side-stream warmup / capture内 | 2/6/2 | 2/6/2 |
| Graph replay | 3 | 随3主调用 |
| startup至ready | 3.109054秒 | 30秒 |
| 尝试 / 重试 | 1/0 | 1/0 |
| 训练 / 真机调用 | 0/0 | 0/0 |

逐类journal共有20个intent、20个return，其中3个model request、10个底层settling step；
其余为Env factory/ensure/reset、native seed/reset/set state和close。未知调用0，measured intent0。
worker_joined、graph_released、original_sampler_restored、metrics_closed、environment_closed全部true，
worker_failed=false，cleanup_errors=[]，first_failure=null。
清理0.137047秒；清理完成后保存`case/arrays.pt`，1,603,297字节，序列化0.002258秒。

严格模型加载5.091680秒；Env与startup/清理/保存的episode wall5.770040秒。
9条新增主机快照的记录体成本合计0.000021880秒；小型调用日志写入/flush成本0.000953631秒。
快照计时不包含调用进入前的Python参数构造，完整执行wall仍保留全部开销。
capture1/2已有setup时长0.412228/0.215450秒、preparation合计时长1.405950/1.102801秒。
没有新增CUDA同步、event或GPU取值；原有CUDA阶段指标仍保留。

child PID2630799，exit0并wait确认；监督wall15.631292秒。
独立外层从2026-09-09T10:27:05.980825+00:00到10:27:25.097131+00:00，wall19.116673秒，监督exit0确认。
无超时、TERM、KILL、重试或后续派发。

## 测试、原件与交付

CPU定向19 passed（8.33s）；固定模型环境实际入口import-only/--help 1 passed（8.23s），ruff/format均通过。
实际命令、退出码和覆盖范围见[TESTS](SMOLVLA_GRAPH_STARTUP_BOUNDARY_TESTS.md)；本轮没有开发测试失败。
[PLAN](SMOLVLA_GRAPH_STARTUP_BOUNDARY_PLAN.md)、[MANIFEST](SMOLVLA_GRAPH_STARTUP_BOUNDARY_MANIFEST.json)已在执行前冻结推送。

新原始目录：`/home/rp/Workspace/SmolVLA_RTC/lerobot/outputs/smolvla_graph_startup_boundary_6ff4fd48`。
保留`result.json`、`worker_result.json`、`case/result.json`（requests/metrics/host事件/Graph/owner明细）、
`calls.jsonl`、`model.log`、`case/arrays.pt`、`execution.json`、`independent_exit_receipt.json`、
预登记发布/回读、精确launch命令、准备测试/旧审计及环境前后原件。
退出后独立收集见`collection.json`、`cpu_archive_readback.json`及`collect_new_result.py`；大数组不提交Git。
[机器结果](SMOLVLA_GRAPH_STARTUP_BOUNDARY_RESULT.json)引用同一原件并保留本次真实标志，发布回执在独立RECEIPT中。

## 已知限制与保持的旧结论

新的fresh Env/engine不恢复旧前7项的进程/GPU历史。一次68.071492ms通过只证明这一次通过，
不能证明旧失败消失、稳定修复或失败概率降低。旧364.724861ms触发原门的结论保持。
旧23次startup与新3次分别报告，不拼成成功率或配对结果。
新登记前GPU使用1337MiB/利用率4%；旧登记时6248MiB/40%。这些背景记录不建立负载与probe时长的因果关系，
没有干预其他进程。单独warmup/capture起止及CPU copy与post算子的拆分没有原始记录，继续标not_recorded。

旧E保持7/20 completed、1 startup失败、12 not_run，4 success、3 TimeLimit，3/10完整配对；ordinal6的async单边完成不混入配对。
native_closed_loop_contract_passed=false、paired_scheduling_comparison_complete=false；
旧native_multirow_consumption_observed、native_replanning_takeover_observed、native_model_control_overlap_observed仍true。
D历史token_graph_equivalence_passed、graph_worker_lifecycle_passed、graph_identity_engine_integration_passed仍true。
baseline_qualified/realtime_qualified/predictor_benefit_tested均false，risk_thresholds=null，old_confirmation=not_started_untouched。
到此停止，没有恢复旧E、补齐20条或启动E-r1。

DevSpace实际拦截记录已读取，按本任务书给出的复发事实追加一次并回读。
待追加附件`SmolVLA_DevSpace_拦截记录_待追加_20260909_9038ba38后续.md`未找到，新增段明确来源于任务书。
README读取ENOENT与随后`ls -la`安全检查屏蔽分别记录，未推断未知根因或重试原DevSpace动作。
三个旧未跟踪文档及旧A/B/C/D/E原件保留。
