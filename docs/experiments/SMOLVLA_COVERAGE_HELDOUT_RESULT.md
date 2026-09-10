# F-COV2正式结果：独立评估未复现首动作优势

2026-09-10。执行HEAD `4ee28ccf7b129f2cd7e736184bbb90227bd9ed7c`。
本报告补全Issue #1评论5615181227的报告整理断点，不是第二次实验。
预登记5615127445正文在启动前exact回读；唯一队列、退出和原独立CPU审计已完成。
本次接续只整理既有证据并发布，新增模型前向、训练、编码、Env和native step均为0。

## 一、固定问题与数据范围

冻结F-COV1四个第72步检查点：single_conditioned、multi_conditioned、
single_no_action、multi_no_action，各69680参数；权重前后逐张量exact。
没有训练、重新选点、调整残差强度或删除原验证首动作最强的single_no_action。

按(8,48)、(9,48)、(9,49)、(8,49)采集四条identity async轨迹。
Env seed=1020000+100*task+state，policy seed=1030000+100*task+state。
准备阶段只查本项目`outputs/smolvla*/episode_*/started.json`，76份身份中没有这四个task/state组合；
该结论不覆盖本机其他目录或远端历史。任务名称此前参与过项目开发，不能称全新任务benchmark。
每条取最先4个完整planned承诺前缀对应的未来观测，共16例，全部delay3。
四条各4例，故sample均值与episode等权均值在本轮相同；推断单位仍不能当成16个独立episode。

所有采集仍由identity async控制，模型/转换/20Hz/50-1-10/cap8/P90/window50等按原协议固定。
两条task8成功，两条task9为TimeLimit；采集2/4 success不归因于预测器，也不与旧轮次合并。
未来观测只用于标签及oracle视觉参考，不进入预测器输入。oracle是实际未来视觉token、
当前model-ready32Dstate、相同语言与噪声输入原十步策略的输出，不是专家、最优动作或闭环成功率上界。
动作指标使用normalized有效7D，row0为第0行动作，chunk为完整50行；不将relativeOSC积分成未来state。

## 二、五方法完整结果

下表为四episode等权平均，误差越低越好。百分比均为本轮内部比较，不与F-COV1验证拼接。

| 方法 | 未来token MSE | 首动作参考MSE | 50行动作块参考MSE | 首动作相对identity | 整块相对identity |
|---|---:|---:|---:|---|---|
| identity | 2396.411827087 | 0.023649880924 | 0.059474715781 | 基线 | 基线 |
| single_conditioned | 2400.479537964 | 0.029711701802 | 0.061122294916 | 恶化25.631507% | 恶化2.770218% |
| multi_conditioned | 2399.634735107 | 0.026361419492 | 0.056412612787 | 恶化11.465337% | 改善5.148579% |
| single_no_action | 2400.575088501 | 0.029115281057 | 0.060649216233 | 恶化23.109631% | 恶化1.974790% |
| multi_no_action | 2399.728141785 | 0.025756284798 | 0.056183623412 | 恶化8.906615% | 改善5.533599% |

**首动作全表最优是identity；完整动作块最优是multi_no_action。**
四个预测器的首动作宏平均均劣于identity；四者token误差也均略高于identity。
multi_conditioned的token误差增加0.134489%，不是更准确的未来视觉预测。

| multi_conditioned相对参照 | 首动作误差降低 | 整块误差降低 | 首动作/整块胜出episode数 |
|---|---:|---:|---|
| identity | -11.465337% | 5.148579% | 2/4、2/4 |
| single_conditioned | 11.275969% | 7.705342% | 3/4、3/4 |
| single_no_action | 9.458475% | 6.985422% | 3/4、3/4 |
| multi_no_action | -2.349464% | -0.407573% | 3/4、3/4 |

负数表示恶化。候选击败原验证首动作最强的single_no_action，但没有击败identity，
也没有击败同数据multi_no_action的两项动作宏平均；不能只挑胜出的基线。
对multi_no_action虽3/4条胜出，剩余条目的幅度足以改变平均方向，胜出数量不能替代预设均值。

## 三、主门未通过的具体原因

| 冻结主门条件 | 结果 |
|---|---|
| 四episode均有完整样本 | 通过，16例 |
| multi_conditioned首动作严格优于identity | 不通过，恶化11.465337% |
| multi_conditioned首动作严格优于single_conditioned | 通过，降低11.275969% |
| multi_conditioned首动作严格优于multi_no_action | 不通过，恶化2.349464% |
| multi_conditioned整块不劣于identity | 通过，降低5.148579% |

`independent_contract_accepted=true`、`four_episodes_have_samples=true`；
`heldout_primary_gate_passed=false`、`heldout_all_comparators_better=false`。
两个`strongest_no_action_*_better`字段均true，其特定参照是single_no_action，
不是“击败本轮所有无动作模型”或“本轮主门通过”。没有事后修改判据。

## 四、逐episode：不剔除大幅退化案例

| task/state | identity首动作 | multi_conditioned首动作 | 首动作相对identity | identity整块 | multi_conditioned整块 | 整块相对identity |
|---|---:|---:|---|---:|---:|---|
| 8/48 | 0.020691970130 | 0.026040733355 | 恶化25.849463% | 0.010949536372 | 0.011956911883 | 恶化9.200166% |
| 8/49 | 0.014936703141 | 0.013314933400 | 改善10.857615% | 0.023224739882 | 0.023041295004 | 改善0.789868% |
| 9/48 | 0.033544010308 | 0.058188697847 | 恶化73.469711% | 0.099769440480 | 0.102656762116 | 恶化2.893994% |
| 9/49 | 0.025426840119 | 0.007901313365 | 改善68.925304% | 0.103955146391 | 0.087995482143 | 改善15.352452% |

候选相对identity两条改善、两条退化。恰好state48均退化、state49均改善只是四条固定记录中的模式，
不是初态ID的因果效应、可上线门控规则或可据以挑选后续样本的结论。
本轮不按失败、变化幅度、任务或状态筛掉记录；全部方法逐episode原始数值保存于机器结果。

## 五、运行账目与既有独立审计

| ordinal | task/state | 采集终态 | measured动作 | 主请求 | 所选样本 | planned接管 |
|---:|---|---|---:|---:|---:|---:|
| 0 | 8/48 | success | 153 | 9 | 4 | 6 |
| 1 | 9/48 | TimeLimit | 280 | 15 | 4 | 12 |
| 2 | 9/49 | TimeLimit | 280 | 15 | 4 | 12 |
| 3 | 8/49 | success | 156 | 9 | 4 | 6 |

原独立审计已核验869个measured动作来源、36次planned接管、4份真实初态checkpoint、16个prefix/cache，
以及四个checkpoint运行前后与来源权重的exact关系。运行时每条首个current双相机token/state重编码exact；
16例identity完整50×32与本轮native归档逐值exact，退出后再次CPU回读验证。
240项数值归约检查通过，CPU float64对GPU float32最大相对差2.608950826e-7，rtol1e-6/atol1e-7；
该归约容差不替代输入/动作的逐值exact门。既有审计首次exit0，本次没有重跑审计器或大数组扫描。

实际Env4、settling40、measured869、native主48、native capture8及内部setup/warmup/capture 8/24/8；
底层native共909次，外层environment_step不重复计作动力学推进。
新增双相机encode20、predictor64、正式decoder96（16例×identity/四模型/oracle六上下文）；
offline capture2及内部2/6/2单列。训练/反传/真机均0。
1854个intent、1854个return，错误/未知0；332组native模型phase、119组外层/离线phase完整。
四条worker join/Graph释放/sampler恢复/metrics关闭/Env关闭确认，环境原Python/version/140包一致。

child2908691、supervisor2908645均exit0且已收回；first_failure/stop_reason为空，无强制终止。
UTC 2026-09-10 07:55:38.111461至07:57:05.766072，监督83.670802686秒，独立外层87.654648160秒。
原审计UTC07:58:04.980632启动，进程外层6.039533秒，审计内部0.528511秒，exit0。
attempt1/retry0。上述是整段执行耗时，不是预测器单次延迟或实时资格。

## 六、报告接续、结论与下一步

原native运行与审计结束后，上一会话写汇总脚本被工具拦截，其原始记录和评论5615181227保留。
本次按用户继续指令直接读取既有审计与退出回执，补全正式报告；不修改原结果、源码、权重或已登记实验。
结果提交、结果评论及exact发布回读、独立发布回执分别记录，不把未执行写成已完成。
旧F-ACT1三份待提交文档与原有未跟踪文档保持，不混入本次提交。

**F-COV1验证选点的首动作优势没有在本次独立评估复现；当前不推进在线7D接入或部署。**
结论限定为四个固定模型、两项开发中出现过的任务、四episode、每条前4个样本、delay3和单训练seed。
不能由此否定所有未来视觉预测方案，也不能唯一归因于覆盖不足、学习率或某个损失项。
这16例现在是已报告的评估数据，不用于继续选检查点、调残差强度、制定样本筛选或训练门控。
下一步回到train0–5/validation6–7开发范围，独立预登记单因素的逐案例动作损失尺度对照；
不同时换架构、加数据、调整多项权重，也不再追加新test来追逐正结果。

原件：`outputs/smolvla_coverage_heldout_4ee28ccf/`；准备与原审计：
`outputs/smolvla_coverage_heldout_preparation_23d3932a/`。
机器结果保留五方法全部逐episode指标、比较方向和原账目。
生产默认保持，baseline/realtime/predictor闭环资格仍false，risk_thresholds=null，confirmation untouched。
