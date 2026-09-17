# E-GFB1：将已验证Graph接入实际环境反馈的观测原点闭环

2026-09-17。接续E-GCR1结果70085ebf、评论5711788239。用户已要求实际集成与持续实验，包含必要代码冻结/预登记/结果发布。

## 问题与唯一比较

E-GCR1证明206份完整输出exact且原生固定负载下Graph完整P99=122.374ms；它不是新策略rollout。E-OBS1原串行7/8、异步6/8及375.470ms尾延迟不改写。
本次真实新环境反馈决定每次模型输入，新模型输出经队列实际控制Env，不读旧观测/噪声/命令作为当前控制。两臂都使用Graph，唯一调度差别仍是serialized等待、async继续消费旧动作。不是对eager的本轮速度比较，也不是guided RTC三臂合同完成。

## 最小实现

复用E-OBS1原control_loop、estimate_delay、should_submit、RTCExecutionQueue和NativeSession；不复制控制器、不修改模块全局切换合同。
InferenceOwner新增显式on_owner_close和request_kind；episode新增predictor_factory注入；默认eager行为保持。Graph创建、重放、恢复原sampler与释放均由同一owner执行，最终join后才关Env，释放失败也必须上报。
GraphFeedbackPredictor复用已核验FullPath和SmolVLAGraphRuntime，每episode专用CUDA Generator按固定policy seed产生全新noise。每请求重新编码当前双相机，留完整50x32、归一化50x7、后处理50x7、8项动态输入和当前输入指纹。
每episode首次bootstrap只捕获一次，其完整成本保留并在控制t0前完成。后续runtime丢失、签名变化、额外capture立即失败，不能静默重捕获。Graph replay证据为元数据与bootstrap capture，不能伪造每次触发10次Python投影hook。
新块row i对应请求观测索引+i；实际消费几行只trim一次；提交与安装在Env返回边界，空等不推进动作索引。原threshold30/初始delay7/P90/window50/cap8/margin1/guard2、50行/十步/20Hz保持。

## 固定样本与配置

八对顺序(task,state)：(0,10),(2,10),(6,10),(7,10),(7,11),(6,11),(2,11),(0,11)。偶数对serialized先，奇数对async先。
Env seed=1160000+100*task+state；policy seed=1170000+100*task+state；配对同初态/种子。
截至准备前仓库357条历史started.json身份无冲突，正式prepare再次查重并绑定元数据；独立性仅覆盖本仓库记录，有外部重复披露须停止。登记后不替换或补正常失败。
原LIBERO_OBJECT工厂、256双相机、relative OSC、20Hz、10settling、280measured/episode；同冻结SmolVLA，无RTC/预测视觉补偿/训练/真机/依赖或驱动更新。IQ1/ACQ资格及旧confirmation不读取。

## 预算与停止

16Env、160settling、最多4480measurement；每episode最多160完整请求、总最多2560。VLA加载1，每正式请求一次真实双相机编码、一次Graph replay。恰好16捕获，内部setup16/warmup48/capture16，共80次额外sampler调用另列，不能隐藏。
每episode1200个20Hz wall时隙、100秒phase；模型90秒、完整请求15秒、native30秒。soft1500/hard1530/工具1620秒。唯一attempt1/retry0，无重启/续跑旧输出。
技术首错、非有限、来源变化、预算、形状或捕获不符即封存；自有进程组TERM后等5秒才KILL。全部退出/待返回调用与资源释放核验。明确安全权限拒绝不换工具绕过，普通接口问题在冻结前最小修复。

## 固定报告与解释

独立CPU审计复用原动作/时间来源核验并显式注入Graph证据检查；8对初态、首次noise/完整chunk exact；每条请求对当前记录的输入指纹、动态shape、单owner/单在途、全部动作来源/一次trim/命令、非native调用中途安装、Graph在同owner释放且早于Env.close、所有预算/调用/phase闭合。
完整列出每臂success、动作数、control wall、无动作slots、underflow/expired/stop-cancelled、实际delay、模型/native交集、完整请求及owner排队P50/P95/P99、首次bootstrap/捕获成本。全部非首次请求含停止时已执行但未安装者，不删慢请求。
线上时延条件：两臂nearest-rank ceil(P99_seconds*20)+1<=8（350ms）；样本小P99通常取最大值，不代表总体硬实时保证。不能继承重放时延，不降频/扩cap。
任务报告8对全部结果，both_success/serialized_only/async_only/neither，另报no_serialized_success_lost_in_pilot描述字段；它不是统计非劣检验。任务失败不得因wall短被写成改善。
动作切换跳变只报告后处理7D向量L2及最后坐标绝对差描述，坐标尺度混合，不能作为机器人安全或成功替代终点。
技术链、时延、任务保持分别判断。不把相同输入exact推成不同反馈轨迹相同；新模型实际控制是本轮区别。

## 冻结、审计、发布

保留21份旧pending。新源码/回归/本计划显式提交并正常推送原分支；CPU prepare绑定真实HEAD、源码/权重/环境、已接纳GCR结果与历史身份。未知输出禁止覆盖。
Issue#1唯一E-GFB1-REGISTER:<HEAD>先单次只读查重，缺失才独立gh literal argv POST一次，实际ID独立GET/body exact保存后启动。未知发布不重发。
退出后一次CPU独立审计，不运行模型、编码、后处理或物理仿真；输入指纹只绑定保存输入，不能替代从RGB重算token。结果/完整不利样本/限制保存，已审计有意义结果正常提交推送并在Issue回报。
无论结果方向，固定16episode结束即停止本合同；强任务保持、泛化或部署资格另需未用确认集、样本量和容忍差异。旧阴性、risk_thresholds=null和生产资格false不自动更改。
