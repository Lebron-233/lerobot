# F-SCL1-r1：训练完成，独立审计尚未接纳

2026-09-10。执行HEAD `6ddb7505c4d1c5930db365a818c2bc1d2e47b2d4`。
预登记5615880107在启动前按实际ID单次GET正文exact，独占新输出仅运行一次。
本文件是实际运行与核验断点，不是最终独立接纳报告；旧077ce452失败原件保持。

## 已完成的数据来源修正

CPU核对F-COV1 episode003/012/004的request6，分别为3/48/6、3/49/6、4/48/6。
三例原plan均89→93、delay4，实际执行89–92四个native动作，来源为request5第20–23行。
原normalized/post前缀、实际native、标签动作/mask、cached inputs及future/current索引exact，
实际future在最后前缀动作返回后、下一实际动作开始前。source_trace.json已经保存，查询exit0。
本轮未改任何标签或旧数值。F-COV1“新64例全delay3”的文字错误已补全原生来源核对。
共同train72=69×delay3+3×delay4；validation16均delay3，全部原样保留。
准备阶段已实际加载两个开发标签通过CPU数据门；19项测试=原10+新增9通过，
初次格式差异与exit1保留，最终Ruff/format/入口均通过，固定140包metadata一致。

## 已完成四组训练：以下是运行器记录，不冒充独立核验指标

原同数据、同seed20260912、同69680参数、72更新/batch1、AdamW0.001/wd0.0001/clip1。
只比较global/case尺度及conditioned/no_action，不重调floor、架构或数据覆盖。
0/36/72按原始validation row0 episode均值选最早最小，包含0步。

| 方法 | 所选步数 | 验证首动作MSE（model.log显示值，12位小数） |
|---|---:|---:|
| global_conditioned | 72 | 0.029442513158 |
| case_conditioned | 36 | 0.028754210696 |
| global_no_action | 72 | 0.029710716241 |
| case_no_action | 36 | 0.028710898987 |

逐案例有动作组首动作误差低于全局有动作组，但逐案例无动作组更低；
运行器 `development_candidate_gate_passed=false`，`status=completed`，`first_failure=null`。
不能据此把尺度差额称为动作输入收益；完整chunk指标、低误差受损子群与独立指标复算尚未完成。
没有在已报告test上评估或重新选点；这是已有开发验证，不是新独立测试或闭环收益。

实际运行器计数：完整decoder812、reference_exact44、gradient_decoder/backward/updates各288，
offline capture56（内部setup/warmup/capture需在最终审计核对）。
新增Env/native/视觉编码/测试标签/真机均0，原VLA冻结；未改变生产默认。
child2954527和supervisor2954439均exit0且收回；唯一attempt1/retry0。
外层UTC08:52:35.725967至08:58:08.245056，332.519130803秒；不是单次预测器延迟。

## 独立审计首个真实断点

`audit_first`在比较`training_weights.json`与CPU准备快照时AssertionError/exit1，原日志和回执保留。
随后只读JSON递归比较成功，确认只有`/scales/latent`这一项不相同：

| 字段 | CPU准备 | 实际运行 |
|---|---:|---:|
| scales.latent | 2131.5723928897937 | 2131.5723863966923 |

绝对差6.493101409432711e-6，相对差3.046155716254585e-9。
scales.row0、train_count、全部72个case权重、floor、低误差阈值及其他JSON字段均exact。
这小于原协议数值归约rtol1e-6，但审计器写成了整个准备字典必须逐值相等。
尚未修订该断言或完成剩余审计，不能把“差异很小”直接当作整体接纳。

继续核查默认/单线程CPU归约来源的命令，被平台返回：
`因 OpenAI 无法确定请求的安全状态，已拦截此工具调用。`
没有PID、exit code或数值结果，不能声称确认了实际线程数或唯一根因。
没有重试、改写或换工具执行这一查询，没有重跑训练；停止相关数据诊断。
本文件仅发布拦截前已取得的事实。`independent_audit_status=incomplete`，独立接纳留空。

## 原件与边界

运行原件：`outputs/smolvla_case_scale_r1_6ddb7505/`；
准备、source_trace、audit_first.log/receipt：`outputs/smolvla_case_scale_r1_preparation_eae8c863/`。
其中四个arm的`.pt`为已保存检查点，当前尚未完成最终独立核验，不用于在线控制。
本轮无后台实验。旧F-ACT1待提交文档、所有旧结果和confirmation保持。
baseline/realtime/predictor闭环资格false，risk_thresholds=null；不把开发结果提升成部署资格。
