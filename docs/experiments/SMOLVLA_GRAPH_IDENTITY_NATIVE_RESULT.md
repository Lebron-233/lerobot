# E Graph serialized / identity async 原生闭环结果

2026-09-09。唯一一次固定队列在第8项 startup 首错停止：**7/20 episodes完成、1项失败、12项not_run；3/10完整配对**。
第8项为task3 / state41 / graph_serialized，完成原10 settling和初态exact比对后，probe要求9个delay steps，超过固定cap8。
该项measured动作尚未发起。未改cap、样本、seed、环境或实验条件，未重试。

执行HEAD：`e00b8731b44b80a9e80dc0af91e4434a9af3d7dd`；分支：`codex/smolvla-graph-native-equivalence`。
[正式E协议](SMOLVLA_GRAPH_IDENTITY_NATIVE_PLAN.md)、[固定20行清单](SMOLVLA_GRAPH_IDENTITY_NATIVE_MANIFEST.json)、
[机器结果](SMOLVLA_GRAPH_IDENTITY_NATIVE_RESULT.json)、[实际测试](SMOLVLA_GRAPH_IDENTITY_NATIVE_TESTS.md)。
[预登记5599454489](https://github.com/Lebron-233/lerobot/issues/1#issuecomment-5599454489)在执行前按实际ID回读一次，正文exact。

## 五个E字段

| 字段 | 实际结果 | 证据 |
|---|---|---|
| native_closed_loop_contract_passed | false | 固定20项未全部完成，startup发生技术错误 |
| native_multirow_consumption_observed | true | 4个完成的async均有同块连续多行真实native发送 |
| native_replanning_takeover_observed | true | async共34次合法计划接管，新块从row0开始，后续请求使用新native反馈 |
| native_model_control_overlap_observed | true | async实测51个model/native交集；完成的serialized均为0 |
| paired_scheduling_comparison_complete | false | 仅3对完成；task3只完成async，后6对未运行 |

4次已执行的第二条件初态比对全部exact，含失败项。只有前3对有完整调度结果。

## 首个失败

```text
RuntimeError: Startup probe requires 9 delay steps, exceeding runtime cap 8
```

原 `PredictiveAsyncInferenceEngine._validate_startup_probe` 在
`src/lerobot/rollout/inference/predictive_async.py:981` 按现有合同拒绝probe。
request1 / startup_probe / reset_epoch0 / task_epoch0 / observation0；
`total_chunk_s=0.364724860992`，原时延取整得到8步，再加固定安全余量1，所需9步。
`startup_gate_outcome=cap_exceeded`、`startup_gate_raw_required_delay_steps=9`、`latency_tracker_admitted=false`。
CPU policy/post chunk和设备完成屏障已形成，门限在发布接管前报错，fresh_warmed未运行。

失败项实际2个主模型调用、1 capture、setup1/warmup3/capture内1、10 settling、0 measured intent。
主模型API与输出有限性没有报错；失败来自startup时延门。
probe主机阶段时间：观测准备16.591ms、preprocessor13.194ms、vision encode212.044ms、
policy78.841ms、postprocessor23.057ms；派发等待7.216ms。完整trace和CUDA阶段时间保留在机器结果。
失败原件：`/home/rp/Workspace/SmolVLA_RTC/lerobot/outputs/smolvla_graph_identity_native_e00b8731/episode_007/result.json`、`arrays.pt`、全局`model.log`和`calls.jsonl`。

## 逐项结果与配对描述

S=graph_serialized，A=graph_identity_async；wall为ready后的实际测量时间。
未到ready的项目没有测量wall。success首步为1基的实际native测量动作数。

| ordinal / task | 条件 | 状态 / success首步 | measured动作 | wall秒 | 无动作slot | 最长派发间隔秒 | planned takeover |
|---|---|---|---:|---:|---:|---:|---:|
| 0 / 0 | S | success@200 | 200 | 10.371799 | 8 | 0.139841 | 8 |
| 1 / 0 | A | success@210 | 210 | 54.583567 | 882 | 10.476126 | 4 |
| 2 / 1 | A | time_limit | 280 | 13.968797 | 0 | 0.062353 | 11 |
| 3 / 1 | S | success@138 | 138 | 7.828472 | 19 | 0.237059 | 5 |
| 4 / 2 | S | time_limit | 280 | 15.024410 | 21 | 0.231991 | 11 |
| 5 / 2 | A | time_limit | 280 | 21.041416 | 141 | 0.556870 | 11 |
| 6 / 3 | A | success@197 | 197 | 11.539150 | 34 | 0.475194 | 8 |
| 7 / 3 | S | startup失败 | 0 | — | — | — | — |
| 8–19 / 4–9 | 固定交替 | not_run | — | — | — | — | — |

完整配对：task0双方成功；task1 S成功而A在280步TimeLimit；task2双方280步TimeLimit。
这些任务结果差异按协议保留。task3 A在197步成功，S未通过startup，不能作为完整配对。

| task | A−S动作数 | A−S wall秒 | A−S无动作slot | A−S最长间隔秒 | A−S计划接管 |
|---|---:|---:|---:|---:|---:|
| 0 | 10 | 44.211769 | 874 | 10.336286 | -4 |
| 1 | 142 | 6.140325 | -19 | -0.174707 | 6 |
| 2 | 0 | 6.017006 | 120 | 0.324879 | 0 |

## 消费、时隙与重叠

以下按已完成episodes汇总：S为3项，A为4项，分母不同；配对差仅使用上面的3对。

| 指标 | S | A |
|---|---:|---:|
| 完成episodes | 3 | 4 |
| 成功episodes | 2 | 2 |
| 实际measured动作 | 618 | 967 |
| 计划接管次数 | 24 | 34 |
| 模型/native交集区间数 | 0 | 51 |
| 峰值模型在途 | 1 | 1 |
| 丢弃块，含startup临时块与probe | 6 | 8 |
| 未消费行，含startup块 | 1032 | 1433 |
| serialized_wait无动作slots | 42 | 0 |
| env_busy无动作slots | 5 | 566 |
| scheduler_miss无动作slots | 1 | 470 |
| underflow无动作slots | 0 | 21 |
| 总wall slots，包含所有无动作 | 666 | 2024 |
| 最长连续无动作slot数 | 3 | 209 |
| model/native交集总秒数 | 0.000000 | 0.811901 |

全部1585个measured发送动作按原install/stage/takeover和index区间定位确定chunk/request/row，
随后与独立CPU post chunk及底层native命令逐值核对。全部完成项来源审计通过，未出现复制动作补发或未知动作。
58次计划接管均从新row0；any-late整块丢弃语义未变，本次实际deadline_miss为0。
None的21个slots全部位于task0 A，期间Env未调用、queue index未推进，旧观测年龄如实增长。

块消费行数分布（消费行数:块数，包含startup零消费块）：

- S：0:6，7:1，16:2，23:8，24:5，25:11。
- A：0:8，1:2，2:1，16:1，23:5，24:6，25:14，26:6，27:2，28:1，50:2。

实际消耗覆盖1、2、7、16、23–28、50行等情况；消费量由合法接管、耗尽或episode终止决定。
初始临时块/probe的零消费记录保留。没有固定25/50行消费窗。
A task0有54.583567秒measured wall、882个无动作slots和10.476126秒最长派发间隔；
底层native.step单次最大9.220309秒。全部等待与遗漏slot保留在分母，无追赶派发。

## 实测时间

统一nearest-rank经验分位数；下表单位ms，四元组为P50 / P95 / P99 / max。
model API和请求观测年龄含startup；measured请求/派发/Env只含ready后。

| 指标 | S样本数 | S四元组ms | A样本数 | A四元组ms |
|---|---:|---|---:|---|
| startup_request_seconds | 9 | 1249.769 / 1974.015 / 1974.015 / 1974.015 | 12 | 1222.027 / 1858.472 / 1858.472 / 1858.472 |
| measured_request_seconds | 24 | 89.698 / 174.463 / 184.348 / 184.348 | 36 | 149.188 / 646.778 / 957.529 / 957.529 |
| model_api_seconds | 33 | 59.017 / 1407.537 / 1931.217 / 1931.217 | 48 | 61.186 / 1608.062 / 1820.984 / 1820.984 |
| native_step_seconds | 618 | 12.453 / 21.836 / 42.847 / 53.036 | 967 | 14.866 / 72.751 / 163.716 / 9220.309 |
| serialized_wait_seconds | 24 | 88.609 / 170.958 / 181.136 / 181.136 | 0 | — |
| dispatch_observation_age_seconds | 618 | 37.285 / 52.708 / 195.905 / 3290.734 | 967 | 38.786 / 118.548 / 944.557 / 4293.955 |
| request_observation_age_seconds | 33 | 37.929 / 1995.072 / 2060.623 / 2060.623 | 48 | 38.832 / 1921.245 / 2077.525 / 2077.525 |
| dispatch_jitter_seconds | 618 | 0.348 / 16.784 / 46.019 / 49.187 | 967 | 0.451 / 39.571 / 47.312 / 49.754 |
| measured_request_observation_age_seconds | 24 | 37.603 / 39.945 / 40.003 / 40.003 | 36 | 38.172 / 175.258 / 237.020 / 237.020 |

startup/日志/序列化/整episode单列；失败项startup未完成，不伪造ready时长：

| ordinal | ready startup秒 | measured wall秒 | 小日志flush秒 | 大数组序列化秒 | 整episode wall秒 |
|---|---:|---:|---:|---:|---:|
| 0 | 3.270296 | 10.371799 | 0.031866 | 0.139853 | 16.893627 |
| 1 | 3.697855 | 54.583567 | 0.258070 | 0.213195 | 61.030182 |
| 2 | 3.368619 | 13.968797 | 0.038343 | 0.214742 | 20.090592 |
| 3 | 2.910908 | 7.828472 | 0.029639 | 0.103450 | 12.958294 |
| 4 | 2.732144 | 15.024410 | 0.042965 | 0.161020 | 20.328872 |
| 5 | 2.502600 | 21.041416 | 0.159845 | 0.192162 | 25.599509 |
| 6 | 2.593933 | 11.539150 | 0.052431 | 0.203797 | 16.482852 |
| 7 | — | — | 0.003560 | 0.013117 | 4.718583 |

小日志flush合计0.616719秒，大数组序列化合计1.241336秒；
这些时间进入实际wall。严格模型载入5.601771秒。
监督器自worker启动至退出189.522851秒；独立外层命令194.259808秒。
worker与supervisor均exit2并已收回，未超时、未发出强制停止信号、未重试。

环境相关调用按真实外层/内层分别留证，嵌套时长不相加当作独立wall：

| 调用 | intent / 已返回 | n | P50 / P95 / P99 / max ms |
|---|---|---:|---|
| environment_factory | 8 / 8 | 8 | 5.928 / 12.635 / 12.635 / 12.635 |
| environment_ensure_and_initial_reset | 8 / 8 | 8 | 1188.820 / 1976.661 / 1976.661 / 1976.661 |
| environment_reset | 8 / 8 | 8 | 779.738 / 914.098 / 914.098 / 914.098 |
| native_seed | 8 / 8 | 8 | 0.014 / 0.026 / 0.026 / 0.026 |
| native_reset | 8 / 8 | 8 | 664.654 / 769.072 / 769.072 / 769.072 |
| native_set_init_state | 8 / 8 | 8 | 1.998 / 4.141 / 4.141 / 4.141 |
| native_step | 1665 / 1665 | 1665 | 13.260 / 58.644 / 131.481 / 9220.401 |
| environment_step | 1585 / 1585 | 1585 | 13.576 / 60.257 / 135.850 / 9222.833 |
| environment_close | 8 / 8 | 8 | 165.395 / 261.487 / 261.487 / 261.487 |

`native_step`包含80次settling及1585次measured；原reset嵌套调用的层级保留。
83个model_request均有终点记录，其中1个终点是startup gate错误；不把有终点等同于请求成功。

## 调用预算、清理与环境

| 项目 | 实际 | 固定总上限 |
|---|---:|---:|
| 已启动episodes | 8 | 20 |
| settling | 80 | 200 |
| measured native | 1585 | 5600 |
| 主模型API | 83 | 3200 |
| reference | 0 | 0 |
| capture | 15 | 40 |
| eager setup / warmup / capture内调用 | 15 / 45 / 15 | 40 / 120 / 40 |
| Graph replay | 83 | 每个主请求1次 |
| 额外setup视觉编码 | 0 | 0 |

83次主请求各有一次pre、policy、prepare_images、noise、vision、post审计调用；
8次owner reset均已记录。7个完成项各2 capture，失败项1 capture；无measured阶段capture。
终点账目：installed17、正常probe_discarded7、staged_early50、staged_on_time8、startup gate错误1。
所有单episode和总调用预算均未越界。

8个环境均关闭；worker_joined / graph_released / original_sampler_restored / metrics_closed均8/8确认。
全部8个arrays.pt在相应清理后保存，共1,280,318,664字节；未重读大数组做重复验收。
原始数据保留在outputs，未提交Git。未知调用0，后续12项没有启动标记或native/model调用。
模型Python路径/版本与全部140项包metadata前后exact，无依赖安装/同步/变更。

CPU定向测试46 passed、固定模型环境真实入口import-only/--help 1 passed；ruff/format通过。
首个开发F811及后续I001 lint失败原日志保留。运行源码冻结后未修改代码或再启动实验。
DevSpace拦截记录已按任务书明确事实追加并回读；待追加附件未找到，追加段已注明真实来源。

## 原件与继续审阅

原始目录：`/home/rp/Workspace/SmolVLA_RTC/lerobot/outputs/smolvla_graph_identity_native_e00b8731`。
`result.json`为冻结driver的原始结果，`collection.json`为退出后的描述性汇总；前者未覆盖。
`episode_000`–`episode_007`各有result和arrays；总目录保留model.log、calls.jsonl、worker_result.json、
execution.json、independent_exit_receipt.json、independent_supervisor_final.log、登记回读与环境前后记录。
对应8个归档的路径、字节数和清理状态在本报告JSON中。

当前停止点是startup probe超过已冻结cap8。后续审阅应以该首错和已完成3对结果为依据，
新的实测范围和资源条件须由下一份任务书固定。本轮不自动继续第8项或剩余12项。

## 已知限制

完整20项原生合同和10对比较均未通过。三个observed字段只说明已完成部分出现对应工程行为。
同卡另一项目在登记时使用约4637MiB显存，GPU总占用6248MiB、利用率40%；本轮没有干预该进程。
这些背景记录不足以确定大间隔或364.725ms probe的因果来源；不能据本轮认定async有时延或任务收益。
仅作开发描述，不作显著性检验或科学基线资格判定。

旧baseline_qualified / realtime_qualified / predictor_benefit_tested仍均false，risk_thresholds=null，
old_confirmation=not_started_untouched。D3-r1三个历史CUDA合同通过标志保持true，旧A/B/C/D结果保持。
真实机器人调用0、训练0；没有启动旧确认队列。
