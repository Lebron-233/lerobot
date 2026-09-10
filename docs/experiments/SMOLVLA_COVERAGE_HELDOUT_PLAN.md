# F-COV2：四个冻结检查点的独立新初态评估

接续 F-COV1 / 评论5614752189，用户本轮要求继续实验。
冻结来源为 outputs/smolvla_coverage_6249b03d 下四个第72步检查点，
不重新训练、选择检查点、改变残差强度或删除single_no_action。
本实验不改F-COV1验证为测试，也不读取旧test12/63/30或confirmation21–40。

## 问题与固定身份

原验证选点的收益能否迁移到此前未运行的task8/9×state48/49？
按(8,48),(9,48),(9,49),(8,49)唯一四条identity async采集。
Env seed=1020000+100*task+state，policy seed=1030000+100*task+state。
准备时仅查本项目已保存started.json的身份，若任一(task,state)已有原生运行则停止，
不换seed规避重复；同时记录索引覆盖范围，不宣称查过本机之外的历史。
任务名称在此前项目开发出现过，新的只是此task/state组合和seed，不称全新任务benchmark。

保持原strict loader/policy/VLM/assets、50/1/10、20Hz、cap8/margin1/P90/window50、
threshold30/guard2、any-late整块丢弃、current-state identity、原7D relativeOSC转换、
same_path_discard_probe_v1、compile=false。每条new Env/engine，原10 settling与3段startup。
predictor始终不控制Env；采集success/TimeLimit不能归因于任何预测器。

每条按request_id取最先4个完整planned前缀→future，复用F-COV1对齐及选择函数。
不足4个用全部，0个停止；不按成功率、误差、变化幅度或delay挑样。
离线比较identity、single_conditioned、multi_conditioned、single_no_action、multi_no_action、oracle。
两相机原生scaled token，normalized7D已执行承诺前缀，当前model-ready32Dstate；
四个模型参数相同为69680、seed20260912、best_step72且非零，权重运行前后逐tensor exact。
未来图像仅作标签与oracle参考；oracle=真实future视觉+当前state+同language/noise的原十步输出，
不是专家/最优动作、成功率上界或反事实物理执行。
每条首个current token/state重编码exact；所有样本identity完整50×32重放与本轮native归档exact。

## 固定判读

主指标为row0有效7D MSE；同时报告50行有效7D MSE与token MSE。
逐episode平均后四episode等权为主，完整披露sample平均、每episode值与胜出数。
延续开发推进门：multi_conditioned首动作严格优于identity、single_conditioned、multi_no_action，
且chunk不劣于identity；新字段heldout_primary_gate_passed，不能冒充统计显著性。
另独立报告multi_conditioned对single_no_action的首动作和chunk方向，
以及对所有四个参照、两项动作指标都严格较好的heldout_all_comparators_better。
没有隐藏“最强基线”或用综合分数抹去冲突；不根据新测试结果更改门。
实验合同通过与模型假设通过分开。负结果也是有效完成；不反复训练/扩样到通过。

## 预算与终止

仅一次attempt，retry/resume/replacement0。Env≤4、settling≤40、measured≤1120且每条≤280；
native main≤640且每条≤160、probe每条≤50含main；native capture≤8，内部setup/warmup/capture≤8/24/8。
每episode ready后1200slots/60秒，startup30秒、model15秒、native调用30秒。
样本≤16；新增双相机编码≤20=16future+4current；正式离线decoder≤96=16×6。
离线capture≤2，内部setup/warmup/capture≤2/6/2；predictor forward≤64（每模型每例一次）。
训练、反传、额外reference、真机调用均0；oracle解码已包含96内。
load60秒、采集阶段480秒、编码/解码合计180秒、单离线调用15秒；外层870秒TERM/900秒KILL。
原watchdog记录首错，仅退出自有进程组；TERM后5秒未退出才KILL。未知调用不重发。
首个技术错误/输入不一致/缺回执停止后续，保留部分结果，不改条件重跑。

## 准备与交付

使用原uv离线/原Python；不安装升级。新CPU测试针对身份、四checkpoint、指标门、选择和预算。
记录HEAD/工作树、环境metadata、一次GPU背景；既有F-ACT1待提交三文件不动。
PLAN/入口/测试先提交推送，实际GitHub登记ID单次GET正文exact再执行。
退出后仅CPU核验本轮native来源与清理、prefix/cache、冻结权重、所存预测与动作、计数/日志/退出。
数值归约rtol1e-6/atol1e-7；不放宽任何输入/identity动作逐值exact。
结果和独立回执分别提交推送，旧结果不覆盖。生产默认与所有闭环资格false、risk_thresholds=null保持。
