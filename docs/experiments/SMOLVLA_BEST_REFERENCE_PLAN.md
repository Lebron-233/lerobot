# F-BRP1：只改变软退化惩罚的训练参照

2026-09-17。接续已实际完成、并于本次首次独立CPU审计接纳的F-ITC1-R1。
其固定guarded候选没有通过开发推进条件；不修改该结论、不重跑旧训练。
用户再次要求继续实验；本轮是新的一次固定开发实验，不是旧输出的恢复。

## 依据和唯一变量

ITC-R1中guarded的训练目标均值下降，但未加权首动作均值劣于冻结I+delta。
训练5/48/6从冻结0.005773退化到0.643131，仍优于Identity0.818460，因此相对Identity的首动作hinge为0。
这证明原超额项不保护已经优于Identity的冻结收益，不证明整个loss没有惩罚或它是唯一根因。
验证7/49/5有类似现象，仅作已知结果披露，不将验证样本用于构造训练约束。
最坏训练4/46/5的hinge实际激活，不能解释为漏计算。

新候选bestref与原guarded保持同一个F-ACR1 centered第72步起点、同72个训练样本/顺序、
72次batch1更新、seed20260912、AdamW lr1e-4/wd1e-4/betas0.9,0.999/eps1e-8/foreach=false、clip1。
仍I+(h(a)-h(0))、幅度1、FP32先减后加再BF16往返，无alpha搜索、不选点，只取最终72步。
唯一变化：对每个训练样本和每项动作指标，参考值R=min(L_identity,L_frozen_IAR)。
L=原plain + L_chunk/S_chunk + relu(L_row0-R_row0)/S_row0 + relu(L_chunk-R_chunk)/S_chunk。
三项系数仍1，scales/样本权重完全沿用ITC，仅训练样本形成R；不同指标可取不同参考，不声称存在共同可达到的硬边界。
冻结参考较坏时仍用Identity，不保护原来有害的行为。这仍是软惩罚，不是安全保证。

## 来源与对照

原72训练/18episode、16反复使用开发验证/4episode；task0–7/state46、48、49。
F-ACQ1-R1的32资格样本、task8/9标签、confirmation保持封存，不加载、不训练或重新资格评估。
新增预测器只训练bestref一组；plain/guarded使用刚完成CPU审计的ITC-R1保存指标，属历史匹配对照，不称同期重新随机实验。
固定同split/task/state/delay循环供体；训练69、验证16可错配；单例保留true/zero。
旧ITC首次技术失败和恢复的总成本保留：VLA2、预测器4、decoder933、predictor1510、更新/反传144。
新实验另记，不能将累计训练重置为零。旧4份ITC-R1阴性报告仅本地保留，与其他13份pending共17份按状态/哈希冻结。

## 运行、审计与预算

代码/计划/CPU测试提交推送后按新HEAD生成唯一PREP/OUT。CPU准备只计算训练参照和来源哈希，CUDA/Env/模型前向0。
先对88样本重构旧h(a)/h(0)、I+delta及解码I/旧B+delta/冻结I+delta，共264次完整输出exact重放。
然后72次更新和261次最终true/zero/mismatched解码；88次zero raw/完整输出exact回Identity。
VLA1、预测器1；正式decoder597=264+72+261、predictor842=176+144+522；更新/反传72；Graph capture<=16，内部setup/warmup/capture每次1/3/1另计。
新Env/图像编码/真机/资格读取0。环境沿用已验原offline/EGL/LIBERO_CONFIG_PATH/LD_PRELOAD、删除PYTHONPATH，uv offline/no-project/no-python-downloads固定解释器。
开发加载30秒、VLA90秒、初始重放180秒、训练180秒、评估180秒、单decoder/更新30秒；总soft600/hard630秒、工具690秒；仅自有worker组，TERM后5秒才KILL。
只允许本输出attempt1/retry0；首技术错误停止并保留，不注册第二次恢复，不增加更新或换样本。
预登记F-BRP1-REGISTER:<HEAD>先GET查重复，再gh literal argv单独POST一次，实际ID独立GET并正文exact。
保存真实POST/GET规范化投影及来源，不伪造HTTP字节。明确安全/权限拒绝停止，不换工具绕过；未知POST状态不重发。

独立CPU/NumPy审计一次：全部来源、动作/供体/参照/组合/zero、指标、目标、clip后梯度与AdamW更新算术、固定最终步及预算/phase/退出。
科学指标rtol1e-6/atol1e-7、更新算术rtol1e-5/atol1e-7，exact不放宽；不独立重新求导。
保存每步完整证据与权重，无中间权重验证选择；预期674个phase闭合。

## 预先固定开发推进条件

bestref验证首动作优于Identity及自身错配，且不劣冻结IAR；验证chunk严格优于冻结IAR、原guarded、plain。
相对Identity至少3/4episode和>8/16样本改善；训练最大正超额首动作误差严格小于冻结/guarded/plain；验证最大正超额不劣冻结。
12项全部满足才development_followup_supported；方向容差1e-7+1e-6*abs(control)。不是统计显著性、独立资格、风险安全阈值。
完整报告全部12组可用指标/各分母、逐样本/episode、不利样本、留一与旧成本。失败不改选别的组、不改系数、不再次训练直到通过。
所有结果向用户如实报告；未通过仅本地保存，不发结果评论；通过才显式提交结果并发布/回读一次。
无新资格采集、时延/闭环或部署。baseline_qualified/realtime_qualified/predictor_benefit_tested保持false、risk_thresholds=null。
