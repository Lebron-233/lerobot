# E-RCV2：两个候选恢复链已观测，第四条模型请求超时，整体合同未通过

2026-09-09。唯一队列已实际执行，**3/4 条完成、1/2 对完整**。task0 disabled 的闭锁和两个 candidate 的完整 native 恢复链有实际证据；task2 disabled 在首个 planned 请求返回前超时。按原首错合同停止，attempt=1，retry/resume/replacement=0。

| 判定字段 | 实际值 |
|---|---|
| `stress_four_episode_contract_passed` | false |
| `stress_disabled_latch_both_observed` | false：task0 已观测，task2 未取得证据 |
| `stress_candidate_recovery_both_observed` | true：两条均有同 epoch 的实际 native row0 |
| `stress_mechanism_contrast_passed` | false |
| `natural_latency_recovery_demonstrated` | false |
| `production_default_unchanged` | true |

## 执行身份与准备

执行 HEAD：`9b7aa8685ae1978c553da6e6d7ac3d4b38e3b9e6`，分支 `codex/smolvla-graph-native-equivalence`。
[预登记5603304679](https://github.com/Lebron-233/lerobot/issues/1#issuecomment-5603304679)在执行前按实际返回 ID 单次回读，正文 exact。
原始目录：`/home/rp/Workspace/SmolVLA_RTC/lerobot/outputs/smolvla_graph_cap_stress_9b7aa868/`。
完整命令保存在该目录 `independent_exit_receipt.json` 和[机器结果](SMOLVLA_GRAPH_CAP_STRESS_RESULT.json)；执行后没有修改冻结源码或清单。
结果提交身份与结果评论回读另记于[独立回执](SMOLVLA_GRAPH_CAP_STRESS_RECEIPT.json)，执行提交和结果提交分开。

本轮使用本地终端。准备阶段保留历史5项 CPU 测试及首错，新增3项缺失覆盖，并复查受影响的2项旧负例；此次5 passed in 0.64s，共8项不同 CPU 用例通过。入口 import/--help、Ruff/format exit0，入口检查 CUDA 未初始化。
首个 AttributeError、F811 和格式差异原件保持，详见[实际测试](SMOLVLA_GRAPH_CAP_STRESS_TESTS.md)。

两臂均为原 `graph_identity_async`；task0 disabled→candidate、task2 candidate→disabled，state41 与登记 seeds 保持。
候选为已有 `same_path_discard_probe_v1`，生产默认仍 disabled。每条首个实际 planned 在原独立 CPU chunks 与设备完成屏障之后、原时延采样/接纳之前请求暂停600ms；startup/bootstrap/probe不暂停。
固定 policy/VLM/assets revisions、50/1/10、20Hz、原 float32/linear P90/window50、margin1、cap8、threshold30、guard2、whole-discard、identity/fallback identity 和 compile=false 均按[PLAN](SMOLVLA_GRAPH_CAP_STRESS_PLAN.md)与[清单](SMOLVLA_GRAPH_CAP_STRESS_MANIFEST.json)执行。

## 四条实际结果

| ordinal | task / arm | outcome | measured 已返回 | measured wall 秒 | 主请求 intent / return | probe | steady bootstrap | planned 接管 |
|---:|---|---|---:|---:|---|---:|---:|---:|
| 0 | 0 / disabled | success | 148 | 7.570034 | 6 / 6 | 0 | 2 | 0 |
| 1 | 0 / candidate | wall_slot_limit | 270 | 59.968964 | 31 / 31 | 20 | 2 | 5 |
| 2 | 2 / candidate | success | 127 | 6.709774 | 14 / 14 | 7 | 1 | 2 |
| 3 | 2 / disabled | started_return_unknown，model timeout | 23 | 未保存 | 4 / 3 | 0（disabled） | 未确认 | 未确认 |

前三条 startup 分别4.393559、3.196376、3.355682秒，episode wall 分别15.536456、70.829710、12.996496秒。
各条均已完成10次 settling。第4条 journal记录 startup request0/1/2返回，以及23次 measured返回；它不构成完成 episode。

| ordinal | 总 slots | underflow | env_busy | scheduler_miss | 无动作 slots | 最长连续无动作 slots | 最长 dispatch gap 秒 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 152 | 4 | 0 | 0 | 4 | 2 | 0.149791 |
| 1 | 1200 | 14 | 450 | 466 | 930 | 252 | 12.625643 |
| 2 | 135 | 2 | 2 | 4 | 8 | 2 | 0.151546 |
| 3 | 未保存 | 未保存 | 未保存 | 未保存 | 未保存 | 未保存 | 未保存 |

前三条 wall_window_overshoot 均0。29次 plan_failed 均为原 `plan_in_flight`，没有额外派发。underflow和跳过的slot不推进Env；实际动力学推进只按底层 native_step 计一次。

task0 的 candidate−disabled：measured动作 +122、wall +52.398930秒、主请求 +25、planned接管 +5、无动作slot +926。该对两臂各完成1条，success分别 candidate 0/1、disabled 1/1。
task2 candidate完成1条且success；disabled没有完成项，success未知，只有23个已返回动作，故该对不计算完整轨迹差值。
task0推理前配对门及退出后新数组中的双图、8D state、quaternion、EEF、gripper均 exact；task2第二条已进入推理，按原执行顺序经过初态门，但没有保存初态回读原件，不能计为第二个完整对。

## 受控暂停与恢复证据

三个已归档首个planned均为 request3、reset/task epoch=1/0。它们按原规则 whole-discard，时延仍被接纳。

| ordinal | 实际暂停 ms | policy API ms | 总请求时延 ms | late steps | 暂停后 P90 秒 | 暂停后 raw |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 600.073085 | 59.759251 | 683.682760 | 11 | 0.622899055480957 | 14 |
| 1 | 600.223495 | 63.741752 | 786.473933 | 11 | 0.7229576110839844 | 16 |
| 2 | 600.062056 | 61.413652 | 779.6106790192425 | 12 | 0.7149956226348877 | 16 |

policy API 列是原 `model_returned_at − model_started_at`，该入口的 vision 另行计时；总请求时延包含真实主机暂停和原路径开销。

task0 disabled：request3接纳后历史为 `[0.07584587996825576, 0.6836827599909157]`。随后普通bootstrap request4/5分别74.817445、78.415915ms，按原整数换算均raw3，installed而 `latency_tracker_admitted=false`；历史逐值不变，仍raw14，后续planned为0，cap_wait为74。因此这一条闭锁成立。

| candidate | 首个回cap probe / 第几个probe | probe前 raw → 后 raw | probe后 P90 秒 | 后续 planned | takeover action index | source request / row | native step |
|---|---|---|---:|---:|---:|---|---:|
| task0 / ordinal1 | request13 / 第9个 | 9 → 8 | 0.3045479357242584 | 14 | 98 | 14 / 0 | 99 |
| task2 / ordinal2 | request11 / 第7个 | 9 → 7 | 0.2923796474933624 | 12 | 81 | 12 / 0 | 82 |

两条链均同epoch 1/0，均从本条paused request3进入超cap。对应probe有效完成，输出丢弃，时延仅接纳一次；原历史保留慢样本，P90自然回cap。probe返回之后才发起表中planned。

- task0：probe13返回1063655.70726284；planned14请求1063655.749602083、完成1063655.93258141；native99发起1063656.319559308、返回1063656.336458535。
- task2：probe11返回1063721.404272467；planned12请求1063721.439025842、完成1063721.529347971；native82发起1063721.789536194、返回1063721.807829089。

原source audit通过；退出后CPU直接读取本轮 `control.dispatches`、planned `post_chunk[0]` 和实际native动作，源request、row0、动作逐值及native起止全部 exact。机器结果保存每个probe的完整前后窗口、P90/raw，以及paused/probe/planned时间线。

task0第一次恢复用了9个probe，episode总计20个。其后planned request18总时延424.143091ms，历史再次超cap；probe19–29未再次回cap，结束时P90=0.8426574468612671、raw18。task2共7个probe，最终raw5。
27个probe均 `discarded_recovery_probe` 且有效接纳，输出没有成为动作源。

## 首个技术失败与退出

第4条 ordinal3 / task2 disabled 的 request3：

```json
{"event":"call_intent","timestamp":1063736.830878878,"call_id":1251,"kind":"model_request","ordinal":3,"limit":15,"request_id":3}
```

该intent在 `calls.jsonl` 没有对应return；原监督器 `execution.stop_reason.expired_calls` 将它记为超时。原监督器发送TERM、等待5秒仍未退出后KILL；child PID2725045的实际退出码为 **-9**，退出已确认。独立记录的监督进程退出码为 **2**，退出已确认。没有触及外层870/900秒预算，也没有追加运行。

监督记录：UTC 14:13:52.114732→14:16:15.829852，原记录wall **144.368160秒**；独立外层 UTC 14:13:47.240552→14:16:41.317386，wall **174.076897秒**。保留各自原始时钟记录，不以UTC差覆盖已有单调时钟wall。

最后一次native调用为第4条measurement23 / action index22 / slot55，于1063737.119437149发起、1063737.224735283返回；外层environment_step随后也返回。
全队列1258 intent / 1257 return，唯一未知调用是上述model_request，未知native为0。

前三条的worker join、Graph释放、sampler恢复、metrics关闭、Env关闭均确认，并在清理后保存数组。第4条上述五项均未取得确认，也没有Env close intent；只保存了started和journal。进程退出确认不替代这些清理确认。

## 实际资源账目与退出后收集

| 资源 | 原预算 | 实际可证实账目 |
|---|---:|---|
| Env | 4 | 4次创建返回；3次关闭返回 |
| settling | 40 | 40次，全部返回 |
| measured native | ≤1120，每条≤280 | 568次，全部返回；545次源审计通过，余23次无归档源审计 |
| 主策略请求 | ≤640，每条≤160 | journal 55 intent / 54 return，逐条6/31/14/4 intent |
| recovery probe | ≤100，candidate每条≤50 | 27=20+7，全部返回并丢弃接纳，包含于主策略 |
| capture | ≤8，每条≤2 | 前三条归档6；第4条已返回cold/probe/fresh按固定源码推知另2，缺最终counter回执 |
| setup / warmup / capture内部 | ≤8 / 24 / 8 | 前三条归档6 / 18 / 6；第4条startup按固定源码推知另2 / 6 / 2 |
| Graph replay | 单列 | 前三条归档51；第4条startup返回对应另3，最后planned内部进度未知 |
| 主机暂停 | 最多4次 | 归档3次，合计1.800358636秒；第4次是否发生未知 |
| reference / predictor / 训练 / 真机 | 0 | 0 |

前三条归档的51次主调用为 startup bootstrap6 + startup probe3 + planned10 + steady bootstrap5 + recovery probe27；pre/policy/images/noise/vision/post各51。第4条只有请求边界journal，没有逐API归档，因此55是请求intent数，不能写成已逐项验证的55次policy API。

只读取前三条已经确认清理的本轮CPU数组，总446,513,691字节；未加载模型/Env、未读取旧大数组，CUDA未初始化。收集exit0、wall5.714826秒。
前三条已归档开销：history audit0.192164秒、journal logging0.384947秒、序列化0.576479秒；第4条对应完整开销未保存，不能把前三条小计当作全队列总数。

模型Python、版本及140项包metadata退出前后直接比较exact；本轮未安装、升级或切换环境。
GPU背景各读一次：登记前 UTC14:08:23.625285，6010MiB/47%；退出后 UTC14:17:33.694833，4088MiB/11%。其他项目进程分别记录4637MiB和2285MiB；未干预其他任务或等候低负载。登记前磁盘可用1,473,396,756,480字节。

## 已知问题与解释限制

1. 第4条超时的内部阶段、是否到达600ms暂停、模型内部完成状态与清理未确认，现有证据不能区分GPU等待、主机执行或其他原因。保留首个超时证据，不归因于受控暂停或其他GPU进程。
2. 强杀前未写出 `worker_result.json`。冻结入口汇总的 `first_failure`、`budget`、`policy_load`、`model_load_seconds` 因而为null；这是实际报告缺口。原件不改，正式机器报告从监督器stop_reason和持久journal补充首错与分层账目，不能把null解释成无失败或零调用。
3. 第4条数组、第二对初态回读、完整API/capture计数和源审计缺失，四条合同与两个disabled对照不成立。两个候选链提供本次受控干预下的局部机制证据；task0后来再次超cap，不能据此声称持续恢复、成功率改善或自然负载恢复。
4. wall与success按各自实际分母描述；本次没有追加样本、因果隔离GPU负载或统计显著性证据。主机发布暂停和policy API计时也不能充当GPU算子延迟。

旧E整体两字段仍false，三个native observed保持true；D三个历史CUDA通过字段保持true。`baseline_qualified=false`、`realtime_qualified=false`、`predictor_benefit_tested=false`、`risk_thresholds=null`、`old_confirmation=not_started_untouched`。旧E、E-S1、E-RCV1报告和确认队列保持原状。

本任务已到预定首错停止点。本轮收集与发布完成后结束，不补第4条、不重跑这次队列。
