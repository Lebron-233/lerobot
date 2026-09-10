# E-RCV3-trace：正式结果发布与退出独立回执

2026-09-10。固定两对600ms受控主机暂停的原生恢复机制对照，经退出后独立CPU核验接纳；正式结果已经提交推送并发布。本任务没有再次执行队列。

## 提交与评论

| 项目 | 实际身份 |
|---|---|
| 分支 | `codex/smolvla-graph-native-equivalence` |
| execution HEAD | `563a8077b69785e90763be98da4dbce1bea7a996` |
| result HEAD（已推送） | `83017092832a53b870aa3491ac4bd1835a10481d` |
| 预登记 | [5604094500](https://github.com/Lebron-233/lerobot/issues/1#issuecomment-5604094500) |
| 本次补全的运行核验断点 | [5611313602](https://github.com/Lebron-233/lerobot/issues/1#issuecomment-5611313602) |
| 正式结果评论 | [5611585394](https://github.com/Lebron-233/lerobot/issues/1#issuecomment-5611585394) |
| POST / GET实际退出 | 0 / 0 |
| 正式结果回读次数 / body exact | 1 / true |
| 发布记录时间UTC | 2026-09-10T02:07:33.103059+00:00 |

实际GET：`gh api repos/Lebron-233/lerobot/issues/comments/5611585394`。
POST使用本地`result_comment_payload.json`的结构化body；GET使用POST实际返回ID，与本地`result_comment.md`逐字比较，无重复POST或再次GET。
发布正文、payload、POST响应、GET响应及stderr、程序化receipt均保存在原始输出目录的`result_comment*`文件。
本回执另作独立提交；提交后的完整回执HEAD记录在实际HANDOVER与本轮`final_commit_identities.json`，不将execution/result身份混为回执身份。

## 接纳门与真实退出

[正式报告](SMOLVLA_GRAPH_CAP_TRACE_RESULT.md)、[机器报告](SMOLVLA_GRAPH_CAP_TRACE_RESULT.json)分别保留运行器原判定与独立审计。

| 独立核验范围 | 结果 |
|---|---|
| 4条/2对与初态 | 4/4完成，四个checkpoint与保存初态、两对checkpoint/保存初态exact |
| 原算法闭锁 | 两个disabled均raw14，快bootstrap不入历史，后续planned0 |
| 候选完整native链 | 两个均paused3→probe8→planned9→index48→native49 source request9 row0，逐值exact |
| 动作来源 | 原source audit在CPU重新执行，全部573个measured动作及9次planned接管exact |
| 请求与原估计器 | 39个前后窗口/P90/raw及snapshot exact；10个probe各接纳一次、输出丢弃 |
| 持久journal | 1253 intent / 1253 return，error0、unknown0、无重复或孤立terminal |
| native与journal关联 | 613个精确身份/动作与内外嵌套区间全部通过 |
| 阶段记录 | 538个phase事件、269组完整起止，无phase error |
| 清理与归档 | 4/4 join、Graph释放、sampler恢复、metrics关闭及Env关闭；归档门成立 |
| 预算 | Env4、settling40、measured573、main39、probe10、capture8、setup8/warmup24/capture内8、replay39，均在上限内 |
| 补充模型环境 | Python/version/140项metadata与准备及启动前快照exact，CUDA未初始化 |
| 未解决审计差异 | 无 |

child PID2847418实际exit0且退出确认；supervisor PID2847368实际exit0且退出确认。
stop_reason=null、forced_termination=false、first_failure=null。监督wall57.570463162秒，独立外层62.341006344秒。
attempt1，retry/resume/replacement0；本次接续新增model/Env/native/reference/predictor/训练0，原15项CPU测试未重跑。

独立收集先后保留两个exit2回执：缺失model return elapsed，以及误要求外层journal与内层native时间相等。两者均为收集器对已存在接口的错误解读。
按冻结源码修正字段和时间区间关联后，最后仅JSON核验剩余613个区间exit0，既有数组检查复用；无容差调整、数据替换或新实验。
最终接纳原件为`independent_cpu_acceptance_20260910.json`；过程日志/脚本/回执全部保留。

## 原件与限制

原始目录：`/home/rp/Workspace/SmolVLA_RTC/lerobot/outputs/smolvla_graph_cap_trace_563a8077/`。
四份arrays共466,153,260字节，四份initial_checkpoint共3,160,116字节，只留本地。
模型metadata和GPU补充采集时间分别UTC01:54:54.650868、01:54:54.706090，晚于native退出；GPU1159MiB/17%、compute-app空列表，只采集一次，不作因果归因。

两臂均2/2 success，candidate−disabled measured wall为+0.457247458/+0.202198476秒。接纳只覆盖固定两对受控600ms干预的恢复机制，不提升自然负载、持续恢复、成功率、耗时、泛化、真机安全或硬实时资格。
生产默认disabled；旧E/E-RCV2整体字段及D历史字段保持；baseline/realtime/predictor资格false、risk_thresholds=null、old_confirmation=not_started_untouched。

工作树原三个历史未跟踪文档未暂存或覆盖，旧拦截记录未重复追加。本任务完成发布回执后停止，未启动下一轮。
