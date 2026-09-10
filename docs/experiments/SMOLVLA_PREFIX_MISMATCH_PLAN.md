# F-PFX1：冻结预测器是否利用正确的承诺动作前缀

2026-09-10，接续Issue #1评论5619360430及F-SCL1-r1 NEXT。
这是独立开发诊断，不重跑原训练，不重试先前受阻的目录清单或回执比较。

## 问题与冻结对象

主对象为原F-SCL1-r1已选36步case_conditioned；对照为原36步case_no_action。
权重、69680参数结构、视觉/state/语言/noise、原十步decoder及归一化7D动作不变。
不改floor、损失、残差强度、检查点或训练数据，不加入新训练。
同一有动作模型分别输入真实前缀和错配前缀，另运行原无动作模型。
正确前缀比错配好回答固定输入干预下的开发效用；并不证明反事实物理收益。

## 样本与供体

仅加载既有F-ACT1 development_labels.pt及F-COV1 new_labels.pt，复用r1选择：
训练72例/18episode，验证16例/4episode；不读取task8/9或confirmation21–40。
按(split,task,initial_state_id,delay)分组；组内按request_id升序排列，
每个recipient取循环下一例的完整已存actions，最后一例取首例；不按误差或success挑选。
不替换mask/delay/current/future/state/视觉/语言/noise，不使用供体未来观测。
此为离线输入干预，不是可部署的供体检索策略，不声称供体在recipient时刻可用。
单例组的错配标为不可评估，真实与无动作主比较仍保留。
预期训练69例可错配、3个delay4单例不可错配；验证16例全部可错配。
若实际身份或数量不符，模型加载前停止，不改分组凑数。

## 计算与判读

真实/无动作各88例，错配85例，共261次预测器前向及完整decoder；无额外参考解码。
真实/无动作的176份量化token与完整输出必须逐值重现原已选检查点评估数组，
这也是本轮运行时加载和解码路径的复现门，不再另跑旧实验参考集合。
保存FP32量化前token、BF16往返token、完整[1,50,32]输出及实际前缀，
分别报告量化前后敏感性、首动作/整块输出变化和到原oracle的误差。
oracle仍为未来视觉+当前state+原语言/noise的策略输出，不是专家动作。

主指标为原归一化7D首动作MSE的episode等权均值；同时报告token MSE及50行整块MSE。
全体真实/无动作比较与可错配子集三方比较分别列出，不混分母。
训练和验证分开，报告每episode与每sample的方向；误差差值定义为对照减真实，正值才是收益。
严格方向使用delta > 1e-7 + 1e-6*abs(对照值)，界内记tie。
“稳定开发动作效用”仅在验证中真实首动作均值优于错配和无动作、
对每个对照至少3/4episode严格更好，且整块不劣于无动作及identity时为true。
不以训练敏感性代替验证效用，不以token发生变化代替动作收益。
这是反复使用过的开发验证，任何正结果也不升级为独立测试或闭环收益。

## 执行和交付

准备仅CPU身份/供体/冻结检查点元数据；新增针对性单测通过后提交并推送，
按实际Git HEAD和唯一输出路径登记Issue #1并单次GET正文exact，之后才加载模型。
固定原uv offline/no-project/no-python-downloads解释器，不安装依赖，不干预同卡进程。
VLA加载1、冻结预测器加载2；优化/反传/编码/Env/native/真机/test读取均0。
Graph capture上限64，内部setup/warmup/capture单列，不能隐藏在261次正式解码内。
单次解码30秒、模型加载90秒、全轮soft600/hard630秒；仅管理自有worker，TERM后5秒才KILL。
attempt1/retry0，首个真实差异保存后停止，不放宽exact、不改输入或重选模型。
退出后只对本轮保存数组做独立CPU归约和供体/分母/实际输入/预算核验，
归约容差rtol1e-6/atol1e-7；结果、原始记录、发表回执均保留。
生产默认与闭环资格false，risk_thresholds=null，旧confirmation不变，其他pending文件不动。
