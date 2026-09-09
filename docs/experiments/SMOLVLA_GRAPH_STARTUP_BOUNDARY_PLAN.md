# E-S1：原 cap8 的单案例 startup-only 协议

2026-09-09。授权范围来自《SmolVLA_E结果分析_Startup边界诊断_Codex接续任务书_20260909.md》。
E-S0已完成原件核验与复算，见 [AUDIT](SMOLVLA_GRAPH_STARTUP_BOUNDARY_AUDIT.md)及对应JSON。
旧gate拒绝计算正确，未证明源合同缺陷；本轮不修改生产/旧实验算法或改写旧E结果。

## 目标与执行身份

在原cap8下，对旧失败案例进行一次新的startup观察。记录新的真实gate结果及主机侧参数/完成通知，
判断这一次是否通过原门。保留旧超限事实，不恢复旧7条或补齐旧配对。

旧执行HEAD：`e00b8731b44b80a9e80dc0af91e4434a9af3d7dd`；
交接HEAD：`9038ba38b89ab3e26671bc042dbeac9b6cdc124c`。执行到交接仅报告/回执改变。
新的execution HEAD在实现、协议、单项manifest和测试提交推送后登记。

新入口：`examples/advanced/predictive_async/libero_graph_startup_boundary.py`。
公开参数只有 `--execution-head <40位HEAD> --output <独占绝对路径>`，内部worker由监督入口启动。
入口使用固定 `SMOLVLA_GRAPH_STARTUP_BOUNDARY_MANIFEST.json` 并逐字段比较，无筛选、resume或探针循环参数。

## 唯一案例与保持的配置

新诊断ordinal0，source_ordinal7；旧案例完整身份保存在manifest的old_case中。

| 项目 | 固定值 |
|---|---|
| suite / task_order_index | libero_object / 0 |
| task / pair | 3 / 3 |
| task name | pick_up_the_bbq_sauce_and_place_it_in_the_basket |
| condition | graph_serialized |
| state row | 41 |
| Env seed | 940341 |
| policy seed | 950341 |
| chunk / n_action_steps / flow steps | 50 / 1 / 10 |
| fps / queue_threshold | 20 / 30 |
| latency quantile / window | 0.9 / 50 |
| safety margin / min / max delay | 1 / 0 / **8** |
| guard / max_late_steps | 2 / 2 |
| context / fallback | identity / identity |
| late / compile | whole_discard_any_late / false |

模型/Env/处理器和转换链沿用原E工厂与严格loader。
GPU为NVIDIA GeForce RTX 4070 Ti SUPER；原bf16/fp32、use_amp=false。
policy revision `6721902bc4d61e50a3bfdb11dfb4cb626f05d102`；
VLM revision `7b375e1b73b11138ff12fe22c8f2822d8fe03467`；
assets revision `0b3ea86be5fe169d0fd036ae63d1070ec09e90f6`。

模型Python固定 `/home/rp/Workspace/SmolVLA_RTC/libero-reference-venv/bin/python`；
CPU测试使用 `/home/rp/miniconda3/envs/smolvla-rtc/bin/python`。
uv固定 `/home/rp/miniconda3/envs/smolvla-rtc/bin/uv`，只用
`run --no-config --no-project --offline --no-python-downloads --python <已有解释器>`。
无安装、同步、依赖调整或PYTHONPATH混入另一环境。模型环境前后metadata直接比较。

## 原生流程与原始初态

先读取报告实际指向的旧ordinal7 `arrays.pt` 中第0条CPU观测，路径固定进manifest。
源证据已在E-S0以CPU mmap读取验证；模型读取和新native只按新登记调用。

严格loader载入一次，然后调用原 `make_native_env_factory` 与 `NativeSession`。
原工厂创建1个Env；原 `_ensure_env` 含必要底层创建和隐含初始reset。
随后原 `Env.reset(seed)` 执行一次native seed、reset、set_init_state(row41)，以及恰好10个settling。
这些调用各有原intent/return；没有额外reset、settling或初态重试。

settling后用原转换生成独立CPU双图、8D state、raw EEF position/quaternion及gripper。
与旧ordinal7初态逐值exact比较；首个差异先保存、关闭Env并停止，模型推理不启动。
原模型配置及Panda relativeOSC路径均不改变。

初态相同时新建 `BoundaryEngine`，派生于旧E `NativeEngine`，继续复用
`SmolVLAGraphIdentityEngine`、真实worker-loop、queue/planner、`e.startup`和原gate。
owner内仍由原 `_make_graph_runtime` 播种一次；没有手动 `_run_request`、额外seed、eager reference或warmup。
控制线程用原条件变量等待真实request完成，依次cold_temporary / probe / fresh_warmed。
正常路径共3次主策略调用：epoch0 cold capture1，epoch0 probe同图replay，通过后epoch1 fresh capture2。
probe第一次拒绝立即停止，最多执行cold与probe两次；不补发第三个请求。
即使三阶段通过并ready，也立即stop并关闭Env，**无get、无measured env.step**。

## 只读证据与成本

新增主机记录位于原owner seed返回、原请求前、原gate前/后、原CPU完成通知前。
记录 `time.perf_counter`、request/observation/task/reset标识、当前startup phase、tracker成员及gate实际参数/原始判据。
这些记录不改变原统计样本或截断判据，不新插CUDA sync/event、不读取额外GPU值。
原E已有CPU独立输出、token/noise/full/post证据拷贝及完成屏障照常保留，成本进入原样本；不扣除它们。
新增CPU主机记录开销单列；小型intent/return写入flush成本及整体wall不消失。

gate仍检查本次probe完成时长，先转换再加margin一次，与原cap8比较；不调用P90，不先clamp。
tracker的P90保留float32输入和numpy默认linear插值，仅后续规划使用；此诊断不进入控制阶段。
时间端点均明确为host记录及原完成屏障定义，不把host API返回自动解释为GPU内核完成。
旧日志没有的独立warmup/capture或CPU copy细分区间继续标not_recorded，不额外测到它们为止。

stop使队列/在途结果失效；确认owner退出、join、Graph释放、sampler恢复、metrics关闭后关闭Env。
只有所有清理确认，才保存本次初态和原CPU请求大数组；join/close未确认时保留小型证据，不并发序列化。
首错保留，清理错误单列并优先标cleanup_failure，不能用正常gate拒绝掩盖清理失败。

## 固定上限与停止规则

| 项目 | 本次上限 |
|---|---:|
| fresh native Env | 1 |
| 必要reset / set state | 上述原工厂序列各一次，无retry |
| settling | 10 |
| measured native step / queue get | 0 / 0 |
| 主策略调用 | **3**；正常三阶段所需，低于任务书16的绝对上限 |
| capture | 2 |
| setup / side-stream warmup / capture内调用 | 2 / 6 / 2 |
| reference / 训练 / 真机动作 | 0 / 0 / 0 |
| startup | 30秒 |
| 单请求 / 单native调用 | 15秒 / 30秒 |
| 外层软退出 / 硬停止 | 270秒 / 300秒 |
| 单次诊断 / 重试 | 1 / 0 |

预算在实际派发前检查；正常路径只到ready，早拒绝或错误会更早结束，不用满上限。
监督器只处理其创建的worker进程组，监视原小型调用日志；单调用超时发SIGTERM，
5秒仍未退则SIGKILL；整体270秒请求退出、300秒硬停止。其他项目进程不干预。
native发起后无返回保留intent/未知，不写成0调用，不重发。外层收回子进程退出并另写独立命令receipt。

status区分startup_passed、gate_rejected、technical_failure及cleanup_failure。
原gate拒绝仍保留RuntimeError，worker/监督exit2；正常startup完成exit0。
未知调用、timeout、非预期退出或measured调用会使本次状态为技术失败；所有实际调用均保留。

## 测试、登记和发布

E-S0的旧事实、23个startup和315个planner复算已完成，未识别合同缺陷。
E-S1a使用原gate/tracker、真实worker/queue与CPU fake policy/Env及受控时钟。
覆盖旧364.724861ms样本、8/9边界及原float32容差、margin一次、分位数/单位/window/未截断判据，
phase及epoch样本来源、一次播种、startup通过也0 measured、失败不补调用、预算、manifest、清理未确认不存数组。
仅加必要原join/reset在途和CPU完成屏障回归；不重跑旧46项或旧模型队列。
实际命令、退出码、日志见TESTS；新增入口在模型环境import-only/--help通过才可登记。

准备文件提交并推送后，Issue #1登记新execution HEAD、旧execution HEAD、完整案例身份、
cap8/measured0、精确展开命令、独占输出目录、主调用3/capture2及各预算、首错停止合同。
用返回的实际comment ID只读回读一次，正文exact后启动唯一诊断。
保持原offline/EGL/GLdispatch/GLX环境变量组合，正式输出不提前创建。

真实结果收回后生成RESULT.md/.json、逐请求/主机/原生小型事件与数组、执行和独立退出回执。
实现与结果分开提交推送，结果评论发布后按实际ID回读一次，再提交发布回执；更新HANDOVER与NEXT_REVIEW。
大数组留在新outputs，旧E及A/B/C/D原件不改。

## 结果解释与限制

old_failure_recomputed=true；source_contract_defect_identified/fixed均false。
startup_gate_passed_this_run与required_steps_unclamped取本次真实值，未启动时null。
native_diagnostic_started、settling/measured/unknown、cleanup_confirmed及进程退出据原件填写。
native_closed_loop_contract_passed和paired_scheduling_comparison_complete始终false。
旧三个observed与D历史CUDA标志保持，不用本次诊断覆盖。

新的fresh实例不恢复旧前7项的进程/GPU历史。一次通过只代表这一次通过；旧超限继续成立，
不称为稳定修复或估计故障概率，不把单次结果与旧前缀拼成配对。
同机负载只读记录，不据背景记录直接归因，不停止其他进程测试。
baseline_qualified/realtime_qualified/predictor_benefit_tested保持false，risk_thresholds=null，旧确认队列untouched。
到此停止；不自动恢复旧E、安排E-r1或实现放宽cap/增加启动校准/跨episode缓存方案。
