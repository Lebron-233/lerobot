# F-ACT1：首动作监督四组对照结果

2026-09-10。执行HEAD `1666c067d9ea76dfb410dda0ac0c64334135ad38`。
预登记Issue #1评论5613998294按实际ID单次GET正文exact后执行唯一队列。
原件：`outputs/smolvla_action_objective_1666c067/`；准备/核验器：
`outputs/smolvla_action_objective_preparation_b7f0831f/`。

## 结论

**实验完整完成并独立接纳，但没有得到优于identity的已学习联合监督检查点。**
纯token模型略微降低新测试token误差，却增加首动作和完整动作块误差。
两组联合监督在验证集的30/60步均不及0步，按预登记规则选回0步，
两个检查点up_projection权重/偏置均为0；全部30个test的视觉token和完整50×32动作与identity逐值一致。
这是验证选点保留identity基线，不是在线风险门，也不是训练改善。

冻结运行器的 `action_objective_row0_improved=true` 只是joint_conditioned数值优于退化的token_conditioned；
joint_conditioned本身就是identity，因此**不能把4.662408%的该比较差额写成学到的首动作收益**。
`joint_action_input_increment_observed=false`、`joint_candidate_gate_passed=false`。
独立语义核验新增 `nonidentity_joint_checkpoint_selected=false`、
`learned_action_objective_benefit_demonstrated=false`；原运行器输出不改写。

## 数据与训练

12条新identity async轨迹：task0–7/state46，task8/9/state46–47；
训练task0–5、验证task6–7、测试task8–9，按轨迹隔离。所有旧state41–45数组、旧权重和confirmation21–40未读取。
每条只选最先8个完整承诺前缀，得到train47/validation12/test30，共89个delay3样本。
全部12条正常结束，其中4 success、8 TimeLimit；采集由identity控制，与预测器效果无关，不跨轮比较成功率。
四组各69680参数、同初始化seed20260912、同60步/batch1样本序列、AdamW lr0.001/wd0.0001/clip1。
token目标=Lz/Sz；joint目标=Lz/Sz+La/Sa。训练集定标Sz=2131.5723863966923、Sa=0.058981226729922634。
Lz是native bf16视觉token MSE；La是原十步decoder有效7D首动作到oracle视觉参考的MSE。
四组共同按验证首动作episode-macro在0/30/60选择最早最小值；四检查点固定后才提取test未来标签与评估。

| 组 | 验证0步 | 验证30步 | 验证60步 | 选定步数 |
|---|---:|---:|---:|---:|
| token_conditioned | 0.022993765277 | 0.023509044627 | 0.021120431811 | 60 |
| token_no_action | 0.022993765277 | 0.023249903485 | 0.021317135676 | 60 |
| joint_conditioned | 0.022993765277 | 0.024385937242 | 0.025161860355 | 0 |
| joint_no_action | 0.022993765277 | 0.023880073301 | 0.026899326030 | 0 |

## 新测试结果：四episode等权

| 指标，越低越好 | identity | token_conditioned | token_no_action | joint_conditioned | joint_no_action |
|---|---:|---:|---:|---:|---:|
| token MSE | 2499.013622284 | 2481.264559428 | 2478.083722432 | 2499.013622284 | 2499.013622284 |
| row0 oracle MSE | 0.013628059533 | 0.014294528849 | 0.014987963387 | 0.013628059533 | 0.013628059533 |
| chunk oracle MSE | 0.066969463210 | 0.070400962461 | 0.069632216279 | 0.066969463210 | 0.066969463210 |

token_conditioned/no_action相对identity的token误差下降0.710243%/0.837526%，
首动作误差却增加4.890420%/9.978705%，整块误差增加5.123976%/3.976070%。
token_conditioned首动作在全部4个test episodes均劣于identity；不是仅由一个episode反转。
两组joint在30个完整动作输出上与identity exact，没有动作输入增量。
oracle仍为真实未来视觉+当前state+同语言/噪声的冻结策略输出，不是专家/最优动作或成功率上界。
表中为四episode等权；sample均值及逐episode数值完整保留于原件independent_audit.json。

## 真实反传与独立核验

两个joint组各60次原十步decoder可导前向/反传，仅更新predictor。两组首步零残差decoder与原生完整动作exact；
action-only token梯度范数各7.865038267525824e-5，预测器首步完整梯度范数0.026587272063/0.026656383649，VLA梯度数0。
不是用CUDA Graph反传；Graph仅用于no-grad标签、验证和测试。原VLA参数始终冻结，未改state/动作转换、denoise步数或生产默认。

退出后CPU一次独立核验通过：2878个measured动作重建原source audit，121次planned接管、12份初态、
全部89个prefix/cache、训练集定标、四组相同60个样本索引、验证选点与30例identity动作、834项数值比较。
指标CPU float64对保存float32结果最大相对差1.988978498e-7，归约rtol1e-6/atol1e-7；不放宽任何动作/token exact门。
另一次仅CPU核对两个joint零权重、30/30 token和30/30完整动作的identity关系，不重复模型或native。

实际Env12、settling120、measured2878、采集main157、native capture24及内部setup/warmup/capture24/72/24。
额外encode101=89future+12current；正式离线decoder562=开发标签118+验证144+联合训练120+测试180；
其中Graph442/可导120；predictor更新240、decoder backward120、额外action-only VJP2。
offline capture34及内部34/102/34，全部预算内。6117 intent/6117 return、错误和unknown0；2998次底层native与外层journal身份及嵌套区间通过。
12/12 worker join/Graph释放/sampler恢复/metrics关闭/Env关闭确认；911组外层/离线phase均正常返回。
child2884010、supervisor2883962均exit0且收回，无超时/强杀；监督349.144896233s，独立外层352.837374063s。
UTC06:09:12.046147–06:15:04.883494。attempt1/retry/resume/replacement0。
12项新CPU测试通过，首个E731/格式差异已在冻结前修复并保留；模型环境140包前后exact，无依赖变更或GPU干预。
独立审计第一次exit0，审计CUDA未初始化、forward/training/native全0。

## 限制与下一步

这是单训练seed、每训练任务一条新初态、60步和一组固定联合权重的小规模实验；不能据此否定所有动作监督方案。
全部delay3，test只有两任务四条轨迹且已用于本报告，不继续拿它们选点/调参。
训练/验证曲线证明本配置没有选择出非零联合检查点，尚不能唯一归因于学习率、目标权重、数据覆盖或过拟合。
下一步先限于训练/验证的可学习性诊断与多初态数据覆盖，记录同样本优化前后动作误差，
在验证能选出优于identity的非零检查点之前，不消耗新的test、不直接接入在线控制。
baseline_qualified/realtime_qualified/predictor_benefit_tested仍false，risk_thresholds=null，old_confirmation untouched；旧结果全部保持。
