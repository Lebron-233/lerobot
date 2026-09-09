# D3-r1 Graph/token/worker 固定 CUDA 合同通过

2026-09-09。**20对token等价性和12个真实worker事件全部完成，三个真实CUDA标志均为true。**
执行提交 `04902a527d684dd43bb55098c8b5f39ae2a96fa2`，分支 `codex/smolvla-graph-native-equivalence`，源码和协议已先推送。
[登记评论5597278767](https://github.com/Lebron-233/lerobot/issues/1#issuecomment-5597278767)在启动前按真实ID回读，正文exact。
[协议](SMOLVLA_GRAPH_TOKEN_WORKER_R1_PLAN.md)、[定向测试](SMOLVLA_GRAPH_TOKEN_WORKER_R1_TESTS.md)、
[机器结果](SMOLVLA_GRAPH_TOKEN_WORKER_R1_RESULT.json)、[发布回执](SMOLVLA_GRAPH_TOKEN_WORKER_R1_RECEIPT.json)。

## 实测与退出

| 判定 | 结果 | 实际覆盖 |
|---|---|---|
| token_graph_equivalence_passed | true | 20/20对noise、完整[1,50,32]、有效[1,50,7]、post-policy chunk均shape/dtype一致、exact、finite |
| graph_worker_lifecycle_passed | true | 单owner、CPU发布、ready后reset、A→B→A、在途stop及退出全部完成 |
| graph_identity_engine_integration_passed | true | 原12事件、语言长度12/13/12、调用和capture固定账目全部满足 |

只运行一次，墙时 **37.979632秒**；child exit **0**、supervisor exit **0**，均已收回。
UTC `2026-09-09T06:42:19.197867+00:00` 至 `2026-09-09T06:42:57.177489+00:00`，子进程PID `2516895`。
没有首个模型失败、重试或超时，295/300秒终止均未触发。

实际 **52主调用+12reference=64**；token/worker分别40/12主调用，Graph replay共32次。
**15 capture=token10+worker5**；单列eager setup15、side-stream warmup45、capture内调用15，
setup额外视觉编码0。每次capture均记录十次[1,50,32]投影；未触及120调用/30 capture上限。
每个主请求prepare_images、视觉编码和noise各一次；显式noise reference没有新增采样。
20个token Graph主请求内部额外视觉编码为0，持有的输出经过后续请求与任务切换仍未被覆盖。

## 原12事件实测

每个事件均完成一次主请求和一次发布后的显式noise eager reference，完整/去pad/post输出均exact/finite。
表中capture编号为worker内部编号；epoch列为请求发出时的reset/task世代。

| 请求 | reset/task epoch | 语言长度 | capture | 真实终点 |
|---|---|---|---|---|
| 0 cold | 0/0 | 12 | 1 | installed |
| 1 probe | 0/0 | 12 | 1 | probe_discarded |
| 2 fresh | 1/0 | 12 | 2 | installed |
| 3 planned | 1/0 | 12 | 2 | staged_early |
| 4 reset_inflight | 1/0 | 12 | 2 | stale |
| 5 new_epoch | 2/0 | 12 | 3 | installed |
| 6 new_epoch_planned | 2/0 | 12 | 3 | staged_early |
| 7 a_to_b | 2/0 | 12 | 3 | stale |
| 8 b_bootstrap | 2/1 | 13 | 4 | installed |
| 9 b_to_a | 2/1 | 13 | 4 | stale |
| 10 a_return | 2/2 | 12 | 5 | installed |
| 11 stop_inflight | 2/2 | 12 | 5 | stale |

原startup probe耗时70.629227ms，使用真实8行prefix，raw_required_delay_steps=3，原gate通过。
冷bootstrap真实耗时1.210510秒，按原规则不进入tracker；没有改写实际时间、历史、索引或delay cap。
两次正常planned均staged_early，并通过原get_action在takeover处消费新第0行。
四次注入均在CPU chunks及设备屏障完成、尚未queue publication时发生，终点全部stale。
reset使CPU epoch从1立即增至2；A→B→A的task epoch为0→1→2；stop使reset epoch从2增至3。

## CPU边界与owner证据

12/12请求均记录cpu_chunks、device_completion_barrier、queue_prefix_cpu和worker inference_mode为true。
policy/post为独立CPU副本；控制夹具消费186个CPU动作，旧active任务标签在切换时保持正确。
worker统计：12次请求、4次stale、deadline_miss0、underflow0、prediction_cap_exceeded0。

worker owner为 `139752221177408`，控制线程为 `139760828454720`；
102条worker模型/processor审计调用全部属于该owner，Graph capture和生命周期记录也在同一线程。
policy/pre/post reset各2次，reset没有重新播种。结束记录
worker_joined=true、graph_released=true、original_sampler_restored=true、metrics_closed=true，failed=false。
先关闭owner并join，随后保存数组；85项档案已在CPU读取，记录名与driver清单一致。

## 准备修复与环境

本轮实现提交只有五文件：正常rollout入口的可选datasets边界、三个隔离导入测试、具名监督CLI和两份R1准备文档。
既有39项CPU通过、模型环境2通过/1跳过以及正常D3 --help证据沿用；包入口和导入测试没有再次修改。
补齐的三文件ruff/format、监督--help和git diff检查全部exit0。监督脚本的UP017首例已保留，
只改等价UTC别名；没有修改原worker、queue、planner、sampler、模型数值或D3请求表。

专用Python和RTX4070TiSUPER保持；policy revision `6721902bc4d61e50a3bfdb11dfb4cb626f05d102`，
VLM revision `7b375e1b73b11138ff12fe22c8f2822d8fe03467`，原本地snapshots、chunk50/denoising10及processor不变。
strict loader缺失键、额外键和shape mismatch均为空，use_amp=false，保留原bf16/float32混合参数。
运行前后Python路径/版本和全部140项包版本记录都与原清单逐项一致；环境未安装、同步或变更。
唯一输入仍为task0–9/state41/observation0，十帧的原双相机及8D state重建全部exact。

## 原始证据

目录：`/home/rp/Workspace/SmolVLA_RTC/lerobot/outputs/smolvla_graph_async_contract_04902a52/`。
`result.json`为原driver结果；`request_arrays.pt`含85项输入/noise/token/完整及post chunk、prefix和消费记录；
`execution.json`与`synchronous_supervisor_receipt.json`记录真实子进程及监督退出；
`model.log`、登记/回读、环境前后清单和比较、测试及首次lint日志均已保存。
`collection_summary.json`是退出后的汇总，未重跑模型或重复数值实验。大数组未提交Git。
准备目录 `/home/rp/Workspace/SmolVLA_RTC/lerobot/outputs/smolvla_graph_async_contract_r1_preparation_da6ecd7b/` 中原日志也保留。

## 适用限制与停止点

这是固定记录帧和确定性交错下的模型/线程合同；186个CPU夹具动作没有发送给Env或机器人。
发布前Event注入不代表取消正在执行的CUDA kernel；耗时和零underflow不能作为20Hz连续控制资格或native多行消费收益。
旧D导入失败、A/B/C和三个旧未跟踪文档原样保留。

`baseline_qualified=false`、`realtime_qualified=false`、`predictor_benefit_tested=false`、
`risk_thresholds=null`、`old_confirmation=not_started_untouched`保持；新增native episode/机器人动作均0。
本轮完成后停止，未开启新的native、确认或predictor工作。
