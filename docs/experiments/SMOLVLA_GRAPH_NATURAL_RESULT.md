# E-NAT1：20条无注入延迟原生调度对照结果

## 结论

2026-09-10，本轮新队列20/20条完成、10/10对初态exact，退出后的独立CPU核验通过。
`nat_native_closed_loop_contract_passed`和`nat_paired_scheduling_comparison_complete`接纳为true。
两种调度均8/10任务成功；每对实际动作数相同。异步的10对测量阶段wall均较短，
合计95.342538100→91.663881869秒，减少3.678656231秒，即3.858357775%。
这是固定开发任务上的一轮描述性调度结果，不是成功率提升、预测器收益或泛化结论。

执行HEAD `2d672b5e3c9d56e54b21cc0a9211bab89f825f15`。
预登记评论[5611778706](https://github.com/Lebron-233/lerobot/issues/1#issuecomment-5611778706)在运行前按POST实际ID一次GET，正文exact并保存。
[冻结计划](SMOLVLA_GRAPH_NATURAL_PLAN.md)、[实际准备](SMOLVLA_GRAPH_NATURAL_TESTS.md)、[机器摘要](SMOLVLA_GRAPH_NATURAL_RESULT.json)。
本轮不恢复旧E、不重跑E-RCV3，也不拼接历史样本。

## 对照定义

libero_object/task0–9/state41，环境及策略seed按冻结20行清单配对，调度顺序按task奇偶交替。
serialized等待刚接受的请求完成；identity async继续消费旧active。
两臂使用同一Graph/worker/queue/identity context，并启用相同的same_path_discard_probe_v1；生产默认仍disabled。
没有主机暂停、负载或时钟注入，cap8/P90/window50/margin1及全部模型、动作接口保持。
测量阶段wall为原ready后控制窗口，不含startup、加载和归档；实际等待与遗漏不从分母扣除。

## 完整配对

| task | 两臂任务结果 | 两臂各自动作数 | serialized秒 | async秒 | async−serialized秒 |
|---:|---|---:|---:|---:|---:|
|0|success|200|10.371972|9.971150|-0.400822|
|1|success|136|7.094053|6.784448|-0.309605|
|2|TimeLimit|280|14.566946|13.965857|-0.601089|
|3|success|190|9.873605|9.465700|-0.407905|
|4|success|149|7.727138|7.422069|-0.305069|
|5|TimeLimit|280|14.574577|13.970431|-0.604147|
|6|success|161|8.375483|8.220004|-0.155479|
|7|success|143|7.418814|7.122949|-0.295865|
|8|success|154|7.969701|7.667260|-0.302441|
|9|success|142|7.370249|7.074015|-0.296234|

task2和task5的正常TimeLimit保留在全10对分母内，不改为技术失败、不补样。

| 全10条/每臂指标 | serialized | identity async |
|---|---:|---:|
|任务成功|8/10|8/10|
|measured native动作|1835|1835|
|测量阶段wall合计/秒|95.342538100|91.663881869|
|主请求（含startup）|107|107|
|合法planned接管|75|75|
|无动作wall slots|77|4|
|其中serialized_wait|77|0|
|其中env_busy|0|4|
|underflow / scheduler_miss|0 / 0|0 / 0|
|真实model/native区间交集对数|0|145|
|recovery probe|0|0|

无动作时隙净减少73个，按50ms计约3.65秒，与总wall差3.678656秒相近；
这与等待隐藏的机制一致。145是原模型调用/native step的区间交集对数，不是145次独立任务。

测量阶段完整chunk请求各77次：serialized均值77.691ms、P50/P95/P99为78.063/84.829/86.596ms；
async均值82.041ms、P50/P95/P99为82.064/89.643/93.666ms，分位数为原nearest-rank。
异步完整请求没有更快，控制wall仍较短；不能把该收益写成模型算子加速。
原`dispatch_jitter_seconds`的P99分别31.341657ms/2.833944ms；保留startup指标在独立timing字段，不混入上述稳态chunk统计。

## 独立核验与账目

全部3670个实际命令重新执行原source audit，与运行器审计逐项一致；
150次planned takeover从对应chunk row0接管，连续多行消费和后续native反馈重规划成立。
20份首次模型前checkpoint与归档初态、10个配对初态逐值exact。
214个请求的terminal snapshot及原窗口成员/P90/raw逐项复算一致；无host_pause phase、无host_intervention字段。
3870次底层native的身份/动作及外层journal对真实native区间的包围关系均核对。

实际预算：Env20、settling200、measured3670、主请求214、capture40；
每capture原setup1/warmup3/内部capture1，合计40/120/40；每主请求一次Graph replay。
7894 intent/7894 return，error0、unknown0；所有单条及总预算通过。
外层environment_step 3670与底层native measurement 3670是同次推进的两层记录，不重复加总。
20/20 worker join、Graph释放、sampler恢复、metrics关闭及Env关闭均确认。
清理后数组2,971,610,852字节，首次模型前checkpoint另15,800,580字节；大数组未入Git。

child PID2857688、supervisor PID2857634均exit0并已收回，first_failure/stop_reason均null，无强制终止。
运行UTC 2026-09-10T02:32:49.998062至02:37:32.167281；监督278.579488485秒，独立外层282.169210750秒。
attempt1/retry0，无resume/replacement或追加样本。训练/predictor/reference/真实机器人调用均0。

独立核验仅CPU，CUDA未初始化，模型环境Python/version/140项包metadata前后exact，无依赖变更。
核验器首轮在第0条遇到归档`cpu(control)`把NumPy命令存成Tensor的接口差异，exit1日志保留；
仅将CPU Tensor视图恢复成原审计器期待的NumPy表示，不改值/原件/来源判据。随后新数据核验全部通过，
最终工具exit0，核验内层3.563261秒；未重跑模型、Env或任何native step。
准备阶段5项新CPU测试通过，首个格式差异及该读取错误均保留于preparation目录。

## 资格与限制

本轮recovery probe为0，`nat_recovery_probe_observed=false`、`nat_recovery_row0_observed=false`。
因此自然负载下的恢复链仍未在本轮观察到；E-RCV3已经接纳的是受控600ms暂停的机制对照，二者不互相替代。
每任务只有同一开发初态的一对，没有显著性或跨seed泛化结论；相同结果/动作数不意味着已核验逐动作轨迹一致。
按需推进Env不能证明真机等待安全或硬实时；模型/资产/负载记录不作跨轮性能因果归因。

新5个`nat_*`闭环/接管/重叠/配对字段均true，但旧E整体false、E-RCV2整体false和E-RCV3受控机制true保持。
baseline_qualified/realtime_qualified/predictor_benefit_tested仍false，risk_thresholds=null，old_confirmation=not_started_untouched。
这轮完成正常运行的工程调度证据，不把identity结果冒充future latent预测补偿收益。

原件：`outputs/smolvla_graph_natural_2d672b5e/`的result/execution/manifest/calls、20份episode结果与arrays/checkpoint、
independent_execution_receipt、independent_audit、publication_summary及环境/资源快照。
准备、登记、核验器和原失败日志：`outputs/smolvla_graph_natural_preparation_4ab3a03e/`。
固定队列已结束；下一阶段应独立冻结predicted-context输入和对照，不因本轮未触发probe追加恢复样本。
