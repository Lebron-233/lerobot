# F-OPT1：训练可拟合，但首动作收益尚未迁移到验证集

2026-09-10。执行HEAD `0b584f0c1f1fca3a9dc98e5b1942f66a1ecde4b2`；
预登记 Issue #1 评论5614295152，实际ID一次GET正文exact后执行。
[协议](SMOLVLA_OPTIMIZATION_PROBE_PLAN.md)、[准备](SMOLVLA_OPTIMIZATION_PROBE_TESTS.md)、
[机器摘要](SMOLVLA_OPTIMIZATION_PROBE_RESULT.json)、[接续](SMOLVLA_OPTIMIZATION_PROBE_NEXT.md)。

## 问题与数据

F-ACT1联合监督的验证选点退回identity。本轮不再用新test检验候选，
只诊断优化能否改变实际动作、降低训练样本误差，以及这些变化是否迁移。
只读取F-ACT1的development_labels.pt：原47训练和12验证标签。
按task0–5最小request_id取6个anchor，实际均request3；每组顺序循环四遍，共24updates。
其余41训练例只参与原Sz/Sa尺度计算，未用于本轮梯度或末步效果评估。
验证为已使用过的task6–7各6例，共12例/2episode；不能称为盲测或新任务泛化。
均state46/delay3，不读取test30、旧test12/63或确认集。

四组同seed20260912、69680参数、零残差初始化，AdamW wd1e-4、clip1、batch1。
原joint lr1e-3；小步长joint lr1e-4；小步长action-only；小步长joint屏蔽动作。
joint=Lz/Sz+La/Sa，action-only=La/Sa；Sz=2131.5723863966923，Sa=0.058981226729922634。
原十步sampler、bf16 cast、normalized7D prefix、当前32D state和每样本保存noise保持。
VLA冻结，标签已存在；无新teacher、视觉编码、Graph capture、Env或native调用。

## 结果：训练与验证必须分开

表中“下降”是相对本组第0步identity的误差降低；“上升”表示恶化。
所有均值先episode内部平均，再episode等权；训练六个anchor各来自不同episode。

| 设置 | 训练row0前→后 | 训练row0下降 | 验证row0前→后 | 验证row0上升 | 单步同样本下降次数 |
|---|---|---:|---|---:|---:|
| joint_original，lr1e-3 | 0.026789098→0.007037037 | 73.731714% | 0.022993765→0.030600349 | 33.081071% | 13/24 |
| joint_small，lr1e-4 | 0.026789098→0.025965117 | 3.075808% | 0.022993765→0.024203091 | 5.259363% | 11/24 |
| action_small，lr1e-4 | 0.026789098→0.025595562 | 4.455305% | 0.022993765→0.023153851 | 0.696215% | 12/24 |
| joint_small_no_action，lr1e-4 | 0.026789098→0.025892423 | 3.347162% | 0.022993765→0.024464969 | 6.398274% | 14/24 |

四个第24步检查点的残差权重均非零，但其验证row0宏平均均未优于identity。
action-only只是在这组开发比较中退化最小，不能写成正收益或胜出的部署模型。
没有在评估后追加训练、修改学习率、选择新测试样本或评估中间检查点。

## 三个机制层面的观察

1. **原目标有训练拟合能力，但均值掩盖部分样本退化。**
   joint_original训练row0下降73.73%，仅3/6锚点变好：task1/2/3下降，task0/4/5上升。
   两个验证task6/7也都恶化，分别0.016405611→0.029849474、0.029581919→0.031351225。
   当前证据支持“训练均值可优化而验证未受益”，不能把所有训练样本都称为学好了。
2. **低学习率缩小了变化幅度，但没有建立验证优势。**
   joint_small的训练拟合较弱、验证恶化较小；去掉token项后验证恶化0.70%。
   在固定24步条件下，步长及目标改变都会影响结果，但不能据此唯一归因或断言长期最优设置。
   屏蔽动作组训练下降3.35%、验证恶化6.40%，当前没有建立动作条件的稳定增量收益。
3. **不存在“所有更新都被bf16量化抹掉”的现象。**
   96/96次更新前后保存的量化token均实际变化，四组均产生非零残差权重。
   单步row0下降/上升为13/11、11/13、12/12、14/10，无相同项。
   这不证明每步都是有效下降，也不排除量化对梯度近似或目标曲面的影响。

首动作与整块依旧不同：joint_original验证整块误差由0.057171758降至0.034711892（下降39.284896%），
但row0恶化33.081071%。其训练token误差略升0.142422%，验证token略升0.138519%。
小步长joint/action-only/no_action验证整块误差相对identity分别恶化1.402257%、改善0.498833%、恶化1.068702%。
不能以整块均值或token误差替代接管首动作指标。

oracle始终是实际future视觉+当前state+原language/noise输入原策略所得参考，
不是专家、最优动作或成功率上界。本轮未让预测器控制环境。

## 独立核验与真实执行账目

四组各24updates，合计96真实反传/更新；完整原十步decoder336：96更新前、96更新后、144个0/24评估。
全部更新前后保存双相机token和完整50×32输出；CPU从源开发标签复算1008项指标，
最大相对归约差2.343120979e-7，rtol1e-6/atol1e-7。该容差不用于放宽输入或动作exact门。
72个0步评估完整chunk与原生归档逐值exact；四组初始参数逐张量一致；循环顺序和设置均核对。
246组phase完整（开发加载1/模型加载1/四arm/96update/144assessment），无错误、未知或超时。
VLA梯度一直None；环境原Python/version/140包exact，无安装升级或其他进程干预。
独立CPU核验首次exit0，CUDA未初始化，没有追加forward/训练/native。

child2888422和supervisor2888376均exit0且收回；first_failure/stop_reason为空、无强制终止。
监督129.090555717秒，独立外层132.589440357秒；UTC06:40:51.190075至06:43:03.779492。
这些是完整离线诊断时间，不是实时推理性能。attempt1/retry0。
准备9项CPU首次通过，首个C420/格式差异留存，最终Ruff/format/import --help通过。

原件：`outputs/smolvla_optimization_probe_0b584f0c/`，四个末步检查点为`*_checkpoint.pt`。
独立审计：该目录`independent_audit.json`；准备/登记/审计器在
`outputs/smolvla_optimization_probe_preparation_d4488fe1/`。
旧F-ACT1待提交回执及NEXT_REVIEW工作树修改原样保留，本轮不将其纳入提交；新接续另存。
旧科学结论、生产默认、baseline/realtime/predictor闭环资格false、risk_thresholds=null、confirmation untouched保持。
