# E-S0：旧 E startup 边界审计

2026-09-09。本轮已读取实际源码、旧报告/机器记录/回执、20行manifest、PLAN/TESTS、原始事件与退出记录，
并从原arrays按mmap读取8项初始CPU双图/state。未启动新的模型或native调用。

结论：**旧第8项的9可精确复算，原cap8拒绝正确，未发现本题范围内的合同实现缺陷。**
9表示本次probe完成时长换算并加一次安全余量后的未截断预测延迟步数。
旧E仍为7 completed、1 startup gate失败、12 not_run，实际任务outcome是4 success、3 TimeLimit。
完整20项合同和10对比较保持false。

## 身份、公开记录与原件

当前交接HEAD `9038ba38b89ab3e26671bc042dbeac9b6cdc124c`，分支`codex/smolvla-graph-native-equivalence`。
旧execution HEAD `e00b8731b44b80a9e80dc0af91e4434a9af3d7dd`是其祖先；
旧result HEAD `b1bf64077f4db736a9c541096d3543bdccf93055`单独标识。
execution到交接只改4个报告/回执文件，生产与实验实现、旧PLAN/MANIFEST/TESTS均未变。
三个旧未跟踪文档保持原样。

本轮对[旧登记5599454489](https://github.com/Lebron-233/lerobot/issues/1#issuecomment-5599454489)与
[旧结果5599623592](https://github.com/Lebron-233/lerobot/issues/1#issuecomment-5599623592)各按实际ID读取一次，
正文与原存档exact。原始目录由报告的`raw_output_directory`及`archives`字段解析：
`/home/rp/Workspace/SmolVLA_RTC/lerobot/outputs/smolvla_graph_identity_native_e00b8731`。
本轮原件读取/复算记录保存在`outputs/smolvla_graph_startup_boundary_preparation_9038ba38/`，不改旧输出。

原8份CPU初始观测的4组配对重新exact比较通过。原第8项arrays只有1个observation、
request_0 / prefix_1 / request_1，control为null；raw双相机均256×256×3。
它与manifest ordinal7、pair/task3、graph_serialized、state41、env seed940341、policy seed950341一致。

## 20项状态重建

intent/return分别从原calls.jsonl及真实底层step记录读取。表中step按settling/measured分别列；
其他Env创建、ensure隐含reset、显式seed/reset/set_init_state、close的逐类计数见本审计JSON。
8个已启动项每类上述调用均1次，全部返回；model_request终点包含1个gate错误，不等同于全部请求通过。

| ordinal/task | 条件 | 任务outcome | startup/measured进入 | settling发起/返回 | measured发起/返回 | 未知调用 | 清理 |
|---|---|---|---|---|---|---:|---|
| 0/0 | S | success | true/true | 10/10 | 200/200 | 0 | confirmed |
| 1/0 | A | success | true/true | 10/10 | 210/210 | 0 | confirmed |
| 2/1 | A | time_limit | true/true | 10/10 | 280/280 | 0 | confirmed |
| 3/1 | S | success | true/true | 10/10 | 138/138 | 0 | confirmed |
| 4/2 | S | time_limit | true/true | 10/10 | 280/280 | 0 | confirmed |
| 5/2 | A | time_limit | true/true | 10/10 | 280/280 | 0 | confirmed |
| 6/3 | A | success | true/true | 10/10 | 197/197 | 0 | confirmed |
| 7/3 | S | startup_gate_rejected | true/false | 10/10 | 0/0 | 0 | confirmed |
| 8/4 | S | not_run | false/false | 0/0 | 0/0 | 0 | not_applicable |
| 9/4 | A | not_run | false/false | 0/0 | 0/0 | 0 | not_applicable |
| 10/5 | A | not_run | false/false | 0/0 | 0/0 | 0 | not_applicable |
| 11/5 | S | not_run | false/false | 0/0 | 0/0 | 0 | not_applicable |
| 12/6 | S | not_run | false/false | 0/0 | 0/0 | 0 | not_applicable |
| 13/6 | A | not_run | false/false | 0/0 | 0/0 | 0 | not_applicable |
| 14/7 | A | not_run | false/false | 0/0 | 0/0 | 0 | not_applicable |
| 15/7 | S | not_run | false/false | 0/0 | 0/0 | 0 | not_applicable |
| 16/8 | S | not_run | false/false | 0/0 | 0/0 | 0 | not_applicable |
| 17/8 | A | not_run | false/false | 0/0 | 0/0 | 0 | not_applicable |
| 18/9 | A | not_run | false/false | 0/0 | 0/0 | 0 | not_applicable |
| 19/9 | S | not_run | false/false | 0/0 | 0/0 | 0 | not_applicable |

7个completed中只有4个success。8个Env已发生80 settling与1585 measured step，1665底层step均返回；
83个model request有终点，其中最后一个以startup gate错误结束。未知调用0，8/8清理确认。
旧child/supervisor均exit2并已收回，监督189.522851s、独立外层194.259808s，无超时/强杀/重试。

## gate的完整计算链

实际接点：`predictive_async.py::_make_request_locked`、`_run_request`（1097–1122）、
`_validate_startup_probe`（942–984）、`latency_replay.py::latency_to_steps`、`latency_tracker.py::percentile`。

1. 每项新建engine与空tracker；非compile identity / cap8采用cold_temporary→probe→fresh_warmed。
2. cold在epoch0 capture并安装临时块，时长不进tracker。probe用原固定8行计划和同一已捕获图，执行完整十步flow。
3. 时间从`request.requested_at`开始，经worker派发、观测/pre、vision、policy、post和独立CPU输出、
   原E证据CPU拷贝，直到既有`_synchronize_policy_device`完成。全程原`time.perf_counter`，单位秒。
4. 本次probe的`latency_s`直接交给原gate；gate不调用tracker/P90，不混合其他startup样本。
5. `latency_to_steps(L,20)`先算`x=L*20`。若x与最近整数满足`math.isclose(rel_tol=1e-7, abs_tol=1e-9)`，
   取该整数；否则`ceil(x)`。随后加固定margin1一次，先与cap8比较，未先clamp。
6. probe通过后才把该单个样本加入tracker、queue reset_epoch递增至1，进入fresh_warmed。
   fresh在新epoch再次capture，完成后ready，其capture时长也不进tracker。失败probe不被接纳。

第8项精确数据：

```text
requested_at_s       = 1046006.486150870
completed_at_s       = 1046006.850875731
L                    = 0.36472486099228263 s
x = L * 20           = 7.294497219845653
latency_to_steps     = 8  （不在整数容差邻域，向上取整）
margin               = 1
required_unclamped   = 9
frozen_cap           = 8
outcome              = cap_exceeded
tracker before/after = [] / []
gate样本数           = 1；quantile不参与
```

`d_actual_wall=8`与`startup_gate_raw_required_delay_steps=9`分别对应转换值和加余量后的门判据。
probe固定plan的8、guard2和实际所需9的用途不同；guard用于队列可用性/保护行，不额外加进gate公式。
`cuda_completed_at_s`由原请求时间加真实完成样本重建，依据既有完成屏障；它不是独立测量的GPU内核终点。
`model_returned_at`只是host API返回记录，不能替代该完成定义。

probe含7.216001ms的request派发等待；此项本来就在原requested_at→完成屏障的冻结样本内，
控制端等待没有作为第二份时长重复相加。没有ms/s混用或重复安全余量。
postprocessor阶段23.056821ms含原处理器、独立CPU chunks及原E请求证据拷贝；这些成本保留在原完成样本中。
vision阶段212.043553ms，policy阶段78.841263ms；不能把任一单阶段当作完整gate样本。

## 已进入startup的23个请求

各行的观测ID均0、task_epoch均0。engine_generation字段原日志`not_recorded`；fresh实例由实际driver每episode构造及ordinal确定。
完整起止时间、CPU完成/通知、host/CUDA阶段值、tracker成员与capture计数见本审计JSON。

| ordinal/条件 | phase | request | reset_epoch | 完成样本ms | 新capture | tracker接纳 | gate未截断所需 | gate结果 |
|---|---|---:|---:|---:|---|---|---:|---|
| 0/S | cold_temporary | 0 | 0 | 1974.014553 | true | false | — | 不检查 |
| 0/S | probe | 1 | 0 | 65.381354 | false | true | 3 | passed |
| 0/S | fresh_warmed | 2 | 1 | 1229.282412 | true | false | — | 不检查 |
| 1/A | cold_temporary | 0 | 0 | 1774.795981 | true | false | — | 不检查 |
| 1/A | probe | 1 | 0 | 203.995342 | false | true | 6 | passed |
| 1/A | fresh_warmed | 2 | 1 | 1706.310255 | true | false | — | 不检查 |
| 2/A | cold_temporary | 0 | 0 | 1858.471659 | true | false | — | 不检查 |
| 2/A | probe | 1 | 0 | 155.690280 | false | true | 5 | passed |
| 2/A | fresh_warmed | 2 | 1 | 1345.280008 | true | false | — | 不检查 |
| 3/S | cold_temporary | 0 | 0 | 1433.044483 | true | false | — | 不检查 |
| 3/S | probe | 1 | 0 | 106.415515 | false | true | 4 | passed |
| 3/S | fresh_warmed | 2 | 1 | 1363.365139 | true | false | — | 不检查 |
| 4/S | cold_temporary | 0 | 0 | 1342.495080 | true | false | — | 不检查 |
| 4/S | probe | 1 | 0 | 128.565002 | false | true | 4 | passed |
| 4/S | fresh_warmed | 2 | 1 | 1249.768678 | true | false | — | 不检查 |
| 5/A | cold_temporary | 0 | 0 | 1169.393561 | true | false | — | 不检查 |
| 5/A | probe | 1 | 0 | 63.214603 | false | true | 3 | passed |
| 5/A | fresh_warmed | 2 | 1 | 1268.714740 | true | false | — | 不检查 |
| 6/A | cold_temporary | 0 | 0 | 1285.521117 | true | false | — | 不检查 |
| 6/A | probe | 1 | 0 | 77.367497 | false | true | 3 | passed |
| 6/A | fresh_warmed | 2 | 1 | 1222.026832 | true | false | — | 不检查 |
| 7/S | cold_temporary | 0 | 0 | 1982.996175 | true | false | — | 不检查 |
| 7/S | probe | 1 | 0 | 364.724861 | false | false | 9 | cap_exceeded |

8次cold、7次fresh的capture/setup/warmup均未进入决定稳态门的probe样本，也未进入tracker。
7个通过probe的原样本有意从同episode的epoch0带入epoch1，作为warm seed；这是原startup合同。
每episode重新构造空tracker，没有跨episode历史混入。失败第8项cold约1.983s未污染probe的364.725ms。
15个capture分别setup1/warmup3/capture内1，均有10个projection记录；83个主请求各replay1。
原记录有eager_setup_seconds和总preparation_seconds；warmup和capture各自独立起止为`not_recorded`。
CPU拷贝与post算子各自独立区间为`not_recorded`，已有combined post阶段及CPU完成标记保留。

## 控制阶段的P90与同一边界

真实LatencyTracker先把窗口成员转成float32数组，再用`numpy.quantile(q=.9)`默认linear插值。
它与报告描述性统计的nearest-rank是不同用途；保持原tracker算法。
compute_delay_plan使用此前完成历史，先得raw required=latency_to_steps(P90,20)+1，再计算
available_after_guard=max(0,available−2)，planned=min(max(raw,0),min(8,available_after_guard))；
prediction_cap_exceeded仍按未截断raw>8判定，不能由planned<=8推出cap满足。

从原事件重建65个接纳样本（7个probe+58个planned），逐一复算315个planner decisions全部exact。
原raw所需分布：3:43、4:35、5:129、6:23、7:9、8:5、12:4、16:67。
raw>8的71个规划机会中69个cap_wait、2个在active耗尽后的identity bootstrap；
ordinal1有67次raw16，ordinal5有4次raw12。它们是既有控制分支的合法预算处理，未改旧结果。
这71次是规划机会计数，主模型调用账目仍为83。

## 旧完整配对与已观察行为

仅旧前缀的task0–2构成3个完整对：task0 S成功200步、A成功210步；task1 S成功138步、A280步TimeLimit；
task2双方280步TimeLimit。A−S wall差分别44.211769s、6.140325s、6.017006s。
ordinal6的A成功197步属于未配对单边，独立列示；未与以上配对或新诊断拼接。
本轮从原request API/native step区间重新计算overlap，与存档一致：S0、A51。
多行/接管的旧完整chunk行来源审计保持原结论；本轮新增复算聚焦startup边界。

## 准备结论与缺失字段

old_failure_recomputed=true，source_contract_defect_identified=false，source_contract_defect_fixed=false。
保留原算法、十步flow、chunk50、cap8、原依赖和所有旧结果；新增独立startup-only入口，最多一次新案例观察。
主机侧快照补充gate参数和前后tracker成员，复用原CPU通知，不增加CUDA sync/event或GPU取值。
新诊断正常三阶段需要3次主调用，capture至多2，settling至多10，measured严格0；通过也立即停止。

本题的拒绝计算已有充分证据。原engine_generation标签、warmup/capture单独时间、
CPU copy与post算子单独时间未记录，不能据总时间拆出这些值。
同机负载/EGL/线程抖动对本次延迟的因果来源未由这些事件识别；不推断根因或尾部概率。

## 拦截记录与限制

已读本地实际记录，追加并回读任务书第2节的ENOENT后目录枚举安全拒绝复发。
README ENOENT与安全拒绝分别记录；本事件不是旧FORBIDDEN，也未归因CUDA/datasets。
待追加附件未找到，追加正文注明来自任务书明确返回，未宣称读取附件原文。

旧E native_closed_loop_contract_passed和paired_scheduling_comparison_complete保持false，
三个旧observed字段和D3-r1历史CUDA通过标志保持；科学资格三个false、risk_thresholds=null、旧确认untouched。
本轮诊断结果独立报告，不恢复旧E或安排新的20项。
