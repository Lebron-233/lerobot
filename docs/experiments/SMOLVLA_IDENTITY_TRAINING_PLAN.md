# F-ITC1：Identity中心化微调与软退化惩罚的配对开发实验

2026-09-17。接续F-IAR1结果5705906022。用户要求继续；新合同只使用原开发集，不重跑旧实验。
原F-ACQ1-R1阴性及32个资格样本继续封存、不载入；所有旧输出/pending保持字节不变。

## 问题与固定设计

F-IAR1的I+delta首动作开发均值更好，但完整chunk比原B+delta恶化且4/46/5等严重退化未消失。
现在检验：从相同旧centered第72步检查点出发，在I+(h(a)-h(0))下微调，加入固定软惩罚是否
改善chunk和最坏训练退化，同时保留首动作收益。plain和guarded只有目标函数不同，
不是为全部可能原因做因果识别，不把软惩罚叫硬安全约束。

数据固定原72训练/18episode、16反复使用开发验证/4episode；原task0–7/state46、48、49，
同split/task/state/delay循环供体；3个训练delay4单例仍无错配，训练M69、验证M16。
不读取R1样本、task8/9标签或confirmation；不增加样本、VLA训练、Env或图像编码。
原centered.pt（第72步、69680参数、seed20260912）分别严格加载两组。
两组均72个batch1更新，原multi_conditioned轮转顺序，lr=1e-4、AdamW wd=1e-4、
betas=(0.9,0.999)、eps=1e-8、foreach=false、clip_norm=1。固定最终第72步，不选点或早停。
两组均I+(h(a)-h(0))先减后加、FP32组合后BF16往返，幅度1，无训练后alpha搜索。

令Lz为有效token MSE，La为首动作7D MSE，Lc为50x7D chunk MSE。
Sz、Sa和逐样本w沿用旧training_weights.json；Sc为仅72训练样本identity chunk MSE平均。
plain = Lz/Sz + w*La/Sa。
guarded = plain + Lc/Sc + relu(La-La_identity)/Sa + relu(Lc-Lc_identity)/Sc。
新增三项系数均固定1，不搜索。两个hinge对所有训练样本等权，不删除/特调4/46/5。
这里同时加入chunk与超额惩罚，是一个目标函数组合的配对检验，不独立归因某一项。
Oracle沿用真实future视觉+当前state+相同language/noise的冻结策略输出，不是专家或安全上界。

## 运行和证据

仅新增runner、独立CPU审计器、CPU测试与本计划，不修改旧代码/结果。
代码测试提交推送后绑定真实HEAD；新PREP/OUT按HEAD唯一命名，CPU准备一次，保存全部源哈希、
manifest、训练权重和scales。工作树必须等于原13份pending的状态/哈希快照。
恢复固定uv offline/no-project/no-python-downloads解释器和offline/EGL/LIBERO/LD_PRELOAD环境，
删除继承PYTHONPATH；不改系统依赖、配置或驱动。必要预登记已授权。
只读GET查F-ITC1-REGISTER:<HEAD>；不存在才gh literal argv独立POST一次，再按真实ID独立GET。
正文exact、POST/GET规范化回执与来源保存后，才启动一个正式监督入口；未知POST不重发。

先用第一组真实预测器重构88份旧h(a)/h(0)、I+delta，与F-ACR1/F-IAR1逐值exact；
每例解码I、原B+delta、初始I+delta，与旧50x32归档exact。第二组初始权重必须同源exact。
任何差异首错停止。随后两组各固定训练，再评估全部88例true/zero及85例mismatch。
每组zero raw和完整输出必须exact回I；保存所有三种上下文，不仅报告好看均值。
每步保存实际动作/mask、h(a)/h(0)、raw/BF16视觉、完整输出、损失、clip前norm、clip后梯度
和更新后完整权重。CPU审计使用NumPy重算指标/目标函数，并从原始权重和保存clip后梯度
独立重构AdamW算术；这不独立验证梯度确由autograd产生，后者仍依赖审计过的执行路径。
优化器算术对比预先固定rtol1e-5/atol1e-7；科学指标仍rtol1e-6/atol1e-7，exact不放宽。

预算：VLA加载1、预测器加载2；正式decoder930=264初始控制+144训练+522最终评估；
预测器前向1508=176初始+288训练+1044最终；反传/更新各144；Graph capture<=24，
每次内部setup/warmup/capture=1/3/1另列，训练使用原可微十步decoder，评估使用原Graph。
开发加载30秒、模型90秒、初始重放180秒、每组训练180秒/评估180秒、单更新/decoder30秒，
外层soft900/hard930秒、工具990秒。attempt1/retry0，TERM后5秒才KILL，仅管理自有worker组。
原系列消耗另外披露；不重启原输出、不覆写审计。权限/安全拒绝停止相应动作，不绕过。

## 预先固定的开发推进条件

仅guarded是预定候选，不依据结果改选plain：验证首动作优于I和同组mismatch；
首动作不劣于冻结F-IAR1；验证chunk严格优于冻结F-IAR1和plain；相对I至少3/4episode、
超过8/16样本改善；训练全体最大正超额首动作误差严格小于冻结F-IAR1和plain；
验证最大正超额误差不劣于冻结F-IAR1。全部同时满足才development_followup_supported。
方向容差1e-7+1e-6*abs(control)。这些是本轮开发推进条件，不是显著性/独立资格/安全门。
逐split报告全部对照三指标、配对分母、样本和episode方向、最差样本、留一episode；
同时完整展示plain及guarded不利结果，记录4/46/5但不据此再次训练。

退出后一次独立CPU审计及完整REPORT。结果全部向用户如实反馈；开发条件失败时仅本地
保留新结果，不新增GitHub结果评论；成功时显式提交结果并POST一次/实际ID回读。
无论结果如何，本合同不自动追加训练、系数搜索、新资格集、时延或闭环。
baseline_qualified/realtime_qualified/predictor_benefit_tested仍false，risk_thresholds=null。
