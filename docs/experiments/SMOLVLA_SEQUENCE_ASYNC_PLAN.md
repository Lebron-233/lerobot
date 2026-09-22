# E-SEQ1：保序接管候选的全程真实反馈三臂开发实验

2026-09-22。接续 E-PSI1 的 ff21ccb7。原候选 a5e9fdde/8112f8d7、E-GFC1 32对任务保持未通过及所有既有源码/权重保持不变。用户已授权继续、必要代码冻结、原分支普通推送及Issue#1预登记和报告。

## 候选定义，先于执行冻结

新队列 SequenceExecutionQueue 是独立版本，不是原时间对齐算法的等价修复。请求仍看到当前环境图像/状态；推理中执行旧计划。新结果在Env返回边界且实际消费delay<=8时替换旧未执行尾段，新计划从真实row0顺序执行。下一次替换可明确丢弃旧未执行尾段，不拼接积累落后计划；不允许重复发布相同请求/跨episode结果。
模型计划row、输入观测索引、安装时物理索引和实际执行索引分别记录，明确 nominal_minus_actual。不同计划中数值相同的动作不盲目去重，因为连续保持/重复命令可能有意义；只保证(epoch, request_id, row)不重复、已开始计划内部无洞。并不声称不存在语义重复、状态过时或动作计划退化。
现有cap8保留为观测到安装之间的实际动作进度限制，不将row0谎称对应当前观测时刻。全部旧尾段的superseded与终止未执行尾段均计数；原时间对齐臂跳过的新块前缀另计。无标签/未来状态/task-ID门控，不按7/19特判。

## 固定三臂与身份

同冻结Graph权重、50行动作块、10步解码、20Hz、relative OSC、256双相机、原factory/资产、10settling、280测量上限。三臂为 serialized / aligned_async / sequence_async。两异步均全程调用原冻结control_loop，不人为固定延迟，不在后续请求阻塞等待。串行按原control_loop等待。
所有臂共享 GraphFeedbackPredictor、InferenceOwner/关闭钩子、estimate_delay/should_submit；只有sequence替换新实例中的队列。原threshold30、P90/window50、initial-delay7、guard2、margin1、cap8不变。
保序队列替换后剩余50行，而原对齐队列为50-delay，因此qsize触发下后续请求节奏可能不同。这是本候选的一部分；完整记录观测索引/请求间隔，不将结果宣传为仅改变单一row因素的因果实验，不暗中换固定网格。
固定10个已见开发身份：原E-GFB1全部8对 [(0,10),(2,10),(6,10),(7,10),(7,11),(6,11),(2,11),(0,11)]，加既有诊断[(7,18),(7,19)]。前8对与后2对分别报告；没有读取E-GFC1其他30对或新独立资格。共30episode，每格一次。按identity index mod3轮转三臂顺序，0:S/A/Q，1:A/Q/S，2:Q/S/A。
沿用Env seed=1160000+100*task+state、policy seed=1170000+100*task+state；每身份三臂初态/首次完整输出/noise exact。所有请求为真实新反馈与专用Generator新noise，不由存档命令驱动；不强制完整轨迹等同历史（并发接管时刻可变化）。不搜索种子、不换失败、不加样本。

## 固定验收与报告

开发候选推进条件：30episode技术审计全接纳；sequence相对本轮serialized和aligned均没有丢失成功配对；三臂全部非bootstrap请求均<=350ms且原ceil(P99*20)+1<=8；sequence有真实模型/native重叠且每动作空时隙比率低于serialized；sequence无underflow/过期安装、每个实际执行计划从row0开始且行号连续且全程无重复身份。仅描述性开发门，不是统计非劣或部署资格。失败保留，不放宽门或只选择修复7/19解释成功。
完整报告所有30任务结果、动作数、控制wall/启动、空slot、请求/取消、实际delay、输入观测到动作发出的年龄、名义偏移、各计划头部跳过/执行/被替换尾段/结束尾段分区、相邻跨计划数值重复和7D命令L2、请求节奏。语义重复或安全不由L2或身份无重复证明。动作数/轨迹不同，总wall不是同轨迹纯加速。

## 工程、预算与停止

仅新增队列、runner、audit、tests和计划，不修改已冻结文件，不升级依赖或驱动。先CPU合成回归（row0/连续性、拷贝隔离、重复结果、reset/close、超过cap、前缀不足、全三臂episode/审计篡改）及Ruff；普通提交推送；CPU prepare绑定源码/权重/环境/历史/原pending；唯一登记 E-SEQ1-REGISTER:<HEAD>查重，单次gh literal argv POST，再真实ID独立GET/body exact；未知发布不重发。
VLA加载1；30Env/300settling/最多8400measurement；每episode<=160请求，总<=4800。每正式请求真实双相机编码/Graph replay一次；30bootstrap capture，内部setup30/warmup90/capture30共150次sampler额外计账。在control前完成首次捕获，禁止控制内重捕获，同owner释放恢复再join再关Env。
每episode1200个20Hz时隙、100秒phase；模型90秒/request15秒/native30秒；总soft1500/hard1530/工具1620秒，attempt1/retry0。不因正常失败提前停或继续测到阳性。技术首错封存，不修后重启本合同；监督仅管理自有进程组。
退出后一次独立CPU/NumPy审计：当前输入指纹、全部动作原始行/实际命令/年龄、时间关系/单在途、三臂bootstrap、计划全50行分类计账、30capture/32phase/调用/退出。不重算模型/视觉编码/后处理/物理仿真。结果及所有不利样本推送报告，旧候选锁和独立确认判据不更新。新方案获支持后才能另行冻结和设计未用身份确认，不能将本10开发身份重新称独立样本。
明确安全权限拒绝停止对应动作，不换通道绕过；普通网络语法问题核实后修正。保留21份旧pending，不reset/clean/stash/强推，不运行真机。本轮GitHub连接器最新评论GET返回FORBIDDEN，不重做该拒绝读取；必要的新登记/发布属于独立操作。
