# E-GFC1：冻结Graph真实反馈候选，独立身份确认

2026-09-17。用户明确要求冻结已成立候选，不再改参数，开展独立确认。候选固定为a5e9fddefc0cfa03ea1509493e8b304ec7c06d92（执行实现8112f8d7）；E-GFB1的8对只作先导，不并入本轮。

## 冻结与范围

不改GraphFeedbackPredictor、FullPath、SmolVLAGraphRuntime、InferenceOwner、control_loop、estimate_delay、should_submit、RTCExecutionQueue、NativeSession、模型/processor源码及权重。本轮仅新增编排、统计审计、测试和文档。逐次核对冻结提交下原src/lerobot与predictive_async已追踪文件未变，并保存来源哈希；新增编排文件不算算法改动。旧21份pending原状态与字节保持。
两臂同Graph/冻结权重/当前真实反馈/50行动作块/十步解码/20Hz/cap8/余量1/guard2/threshold30/P90-window50/initial-delay7。真实当前观测与新noise产生动作并控制Env，非旧命令重放。每episode单owner首次捕获一次，control内禁止捕获，结果仅在Env返回边界安装，按实际消费只trim一次；关闭先停止发布、收回请求、同owner释放/恢复、join后关Env。

## 固定样本、顺序与预算

32对=四个已见任务(0,2,6,7)各八个未用初态12..19；64episode。按state递增，偶数state任务顺序0,2,6,7，奇数反序7,6,2,0；pair_index偶数串行先、奇数异步先，使每task交替先后。Env seed=1160000+100*task+state，policy seed=1170000+100*task+state，沿用先导的固定映射而非搜索种子。配对相同初态/seed，独立生成器及episode reset隔离。
首次只读查373条历史身份无冲突；prepare再次查全仓库started.json身份，只读元数据，不读IQ1/ACQ或confirmation21..40标签。旧先导10/11、E-OBS1的8/9及所有旧资格不进入确认分母。身份独立性仅覆盖当前仓库；外部重复披露即停止，不在登记后替换。
每episode原280测量动作、10settling、1200个20Hz wall时隙；每episode请求上限160。总上限64Env/640settling/17920measurement/10240请求，VLA加载1。64capture；内部setup64/warmup192/capture64=320次sampler额外调用独立计账，启动成本全部报告。各请求真实双相机编码和Graph replay一次；训练/RTC/预测器/真机0。
模型90秒/native30秒/request15秒/episode100秒保持；总体soft2400/hard2430/工具2520秒，因确认样本量扩大只增加总预算，不改变单episode条件。attempt1/retry0/replacement0。技术首错保存并停止；正常任务失败全部保留，不提前因阳性/阴性停科学采集，不追加到通过。

## 主要终点：保守的配对任务保持确认

L_i=1表示同一身份串行成功而异步失败；G_i=1表示反向。报告全部32对与双方成功率，净差Dhat=(sum G-sum L)/32。主门不让额外获益抵消已有成功任务的丢失：检验平均回退概率qbar的95%单侧上界U是否<0.10。
10个百分点是本轮预先设定的工程确认容忍上限，不是用户批准的生产容忍度，不声称零损失。这个标准比仅检验净成功率差更保守，因为E[S_async-S_serialized]=gbar-qbar>=-qbar。即使通过也不意味着每task都非劣、泛化到新任务或真机。

为避免把四任务异质性当成同一个i.i.d.比例，主上界使用独立Bernoulli（允许各q_i不同）的Chernoff/KL反演：x=k/n，解 n*[x log(x/U)+(1-x) log((1-x)/(1-U))]=log(1/0.05)，U>=x；k=0时U=1-0.05^(1/n)，k=n时U=1。独立性是统计解释的条件，不是通过一次审计就证明的事实；固定初态/种子下结果首先是已测队列的事实，不外推LIBERO总体。硬件时变/跨episode依赖会限制概率解释，必须披露。
推导：E exp(t sum L_i)=prod(1-q_i+q_i exp(t)) <= (1-qbar+qbar exp(t))^n（对log用凹性）；t<0优化得P(K<=nx)<=exp(-n KL(x||qbar))，反演构成保守单侧界。零丢失时，乘积上界亦直接给出(1-qbar)^n。
取n=32是四任务均衡且超过零丢失95%上界低于10%所需最少29对，不是完整高功效非劣试验。n=32零丢失时U约8.94%，有一次丢失即无法通过本固定保守门。不改变置信度/界算法来取得通过；即使净收益正且主门未过也据实报告。零丢失接受概率在q=0.02/0.05的i.i.d.规划例中分别约52.4%/19.4%，所以未通过不能当成已证明劣化。
辅助报告单臂及四类配对计数、每task成败，不进行事后最优子集选择。样本单位是身份对，不是模型请求或动作。使用旧开发数据不做power拟合；本轮新结果只判读一次。
区间基本参考：NIST https://www.itl.nist.gov/div898/handbook/prc/section2/prc241.htm；其小样本二项反演仅作背景，以上异质独立KL界的推导明确给出，不冒称为精确Clopper-Pearson或标准配对净差区间。

## 线上延迟及工程联合判读

两臂分别按全部非bootstrap请求（包括停止时完成但未安装者）nearest-rank P50/P95/P99，要求ceil(P99_seconds*20)+1<=8；原350ms门不改。同时为本轮保守确认要求两臂所有非bootstrap请求均<=350ms，防止请求数多时P99隐藏少数超预算点。后者是更严格的报告/确认要求，不修改控制器或旧门，两个字段独立呈现。
报告每episode最大值、全部>350ms请求、bootstrap/捕获/owner等待/取消、actual delay、underflow/expired、接管及命令跳变。请求时间相关，不把数百请求当作独立Bernoulli样本建立硬实时概率保证。
技术来源与退出接纳、两臂线上尾延迟门、全部请求预算、真实异步重叠与等待比率下降、上述配对回退上界门全部通过，才标记bounded_confirmation_passed=true。绝不将不通过的一项被其他均值替代。realtime_qualified=false、risk_thresholds=null、broad_generalization_claimed=false保留；确认的是冻结四任务新初态范围，不是生产认证。
总wall包含任务步数/结果差异，不当成同轨迹加速；全部失败都保留。旧阴性和旧实验成本原样报告。

## 冻结、运行、审计和发布

新增代码/测试/本计划显式提交，正常非强制push至Lebron-233/lerobot当前分支。一次CPU prepare绑定新执行HEAD、原候选HEAD、来源/权重/专用环境和32对身份。Issue#1唯一E-GFC1-REGISTER:<HEAD>只读GET查重，缺失才gh literal argv POST一次，实际ID独立GET并body exact落盘。未知POST不重发，明确安全权限拒绝停止相应动作，不换通道规避。
用原uv+专用Python及offline/EGL/LIBERO_CONFIG_PATH/LD_PRELOAD、删除PYTHONPATH；不安装/升级。一个监督worker仅管理自己的进程组，TERM后等5秒才KILL。执行后一次独立CPU审计，逐episode复用已接受的动作/Graph/输入指纹/生命周期核查，重新核全部调用账、64条episode输出覆盖/32对初态与bootstrap、66阶段、独立重算置信界和指标。这里是64条episode而非64个配对身份。
CPU审计不重算模型/图像编码/物理仿真，也不把输入指纹当成从RGB重建token。结果无论方向均保留并可发布有意义的确认结论，不重跑本合同。
