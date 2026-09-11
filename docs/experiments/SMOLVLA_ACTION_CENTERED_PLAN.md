# F-ACR1：固定无动作基底，分离动作增量与通用视觉校正

2026-09-11，接续已独立接纳的F-PFX1及评论5627262353。此为独立结构假设，
不是重跑F-PFX1，不把前缀敏感性当作稳定动作收益；原结果三份pending文档原样保留。

## 假设与冻结内容

基底B为F-PFX1已选36步case_no_action的量化前FP32视觉token缓存。
比较中心化`B + (h(z,a,s,d)-h(z,0,s,d))`与普通`B + h(z,a,s,d)`。
减法必须先于加基底，以保证零动作时逐值等于B。h直接输出delta，不再加一次当前z。
两个h均复用69680参数结构、seed20260912、零输出初始化；不修改原基底权重。
冻结VLA及原十步decoder；将组合FP32 token经BF16往返后送入decoder，包括训练路径。
固定原无动作基底还依赖当前视觉/state/delay；不声称它是真实世界的零动作反事实。

使用同一72训练/18episode与16开发验证/4episode；每条固定前四例，原三个delay4训练单例保留。
仅加载原F-ACT1 development_labels及F-COV1 new_labels，不读取task8/9或confirmation21–40。
来源身份、延迟、当前视觉/state/语言/noise、oracle及原case权重、floor和损失尺度不变。
oracle仍为未来视觉+当前state+同语言/noise的冻结策略输出，不是专家动作。

## 训练、选点及对照

两组各72更新、batch1，固定原多初态顺序、AdamW lr0.001/wd0.0001/clip1。
目标仍为Lz/Sz + wi*La/Sa，直接读取原已审计training_weights，不重新估计或调权。
验证仅在0/36/72步；按原FP32首动作episode-macro选最早最小值，零步合法，不按错配结果重选。
相同的是参数/初始化/优化更新/完整decoder预算；中心化需要两次h前向，普通仅一次，
不宣称FLOPs或训练耗时相等，也不向普通组添加无用前向凑数。

选中后两组均评估真实、零动作和同split/task/初态/delay组内循环下一request错配。
供体规则完全沿用F-PFX1，不按误差挑选。训练错配69例、验证16例；单例仅没有错配输出。
真实/零动作保留全88例。原无动作基底和identity均保留；训练/验证、全体/可错配分母分开。
零动作不仅是对照，还要求中心化组88份FP32 token及完整输出exact回到原无动作基底。

主指标：验证首动作归一化7D MSE，episode等权，方向容差1e-7+1e-6*abs(control)。
候选门要求中心化选择非零步，真实均值同时严格优于原无动作、同模型错配和普通真实；
相对原无动作和错配各至少3/4episode严格更好；整块误差不劣于原无动作及identity。
token、整块和逐样本指标均报告，不用于替换主门或删例。开发验证已反复使用，
通过也仅是开发候选，不是独立泛化或闭环成功；阴性结果同样保存。

## 执行预算与证据

准备只做CPU身份/来源/缓存/权重核对，不加载新模型或Env。
新PLAN、runner、独立CPU审计器及针对性测试提交并推送后，按实际HEAD/唯一输出预登记并单次GET正文exact。
保留此前待提交的F-PFX1 RESULT/NEXT/STATUS，不将它们混入新实验提交；仅允许既有pending文档。
使用原uv offline/no-project/no-python-downloads及固定libero-reference-venv解释器、RTX4070Ti SUPER；不安装依赖。

新VLA加载1、支路初始化2；基底模型加载0，依赖已审计缓存。
正式完整decoder850 = 基底逐例exact复现88 + 梯度训练144 + 验证曲线96 + 所选三方评估522。
支路前向1143 = 中心化762 + 普通381；优化与反传各144。Graph capture上限64，内部setup/warmup/capture另记。
额外图像编码、Env/native、真机、test读取均0。模型加载90秒、单次预测/解码/更新30秒，
总soft1200/hard1230秒；attempt1/retry0。只管理自有worker，TERM后5秒才KILL，保存首差异，不重试模型运行。

保存初始/0/36/72/所选权重、每次训练与评估的实际动作/mask、h(a)/h(0)、量化前后token、完整输出、损失、
固定序列、事件、计数和退出。独立审计仅用CPU保存数组，重构供体、组合代数、零动作不变量、指标、
权重/损失/训练顺序、选点、所选验证exact复现、预算/phase/退出；归约rtol1e-6/atol1e-7，输入/输出exact。
缓存省略基底模型计算，不能用本轮速度宣称部署延迟；真实部署开销需独立实验。
生产默认与闭环资格false、risk_thresholds=null、旧confirmation untouched。
