# E-OBS1：无RTC、观测时刻原点的同eager原生异步闭环

2026-09-17，接续b505cb44的路线A，不是原未通过前置的24episode三臂试验。
上轮完整请求base P99 292.692ms、含一行动作余量需7步，满足cap8；RTC P99 731.916ms需16步，不满足。
本轮只推进无RTC工程线，不训练、读取封存IQ1/ACQ标签、重跑旧试验或声称学习补偿有效。

## 固定对照、身份与预算

顺序固定八对task/state：(0,8),(2,8),(6,8),(7,8),(7,9),(6,9),(2,9),(0,9)。
每对偶数序号先serialized、奇数先async，共16episode；不是根据任务表现调整顺序。
环境seed=1140000+100*task+state，策略seed=1150000+100*task+state；配对两臂相同。
初步325个仓库started.json身份查重无冲突；prepare须再次冻结元数据，注册后不得换例，外部运行身份未覆盖。
原LIBERO_OBJECT完整工厂、relative OSC、20Hz、256双相机、十步settling、280个measured动作上限保持。
每episode 1200个20Hz控制wall时隙上限；正常time_limit/失败不补例。每episode请求<=160，总<=2560，Env16/settling160/measured4480。
同固定VLA加载1次、chunk50/解码10、无torch.compile/Graph、无RTC/预测器/优化器/参数梯度；权重和原专用环境不变。

## 唯一实验变量和时间语义

两个条件共享同一InferenceOwner、同一EagerPredictor、同一control_loop、同一RTCExecutionQueue（仅借用替换/trim能力，RTC引导关闭）。
每episode一个模型owner线程。主控制器只在Env返回边界发起请求/安装结果；worker只计算并返回CPU数组，不在native.step中途安装。
新chunk第i行对应请求观测索引+i。安装时从实际已成功pop并已执行的动作数计算trim，一次丢弃；空等不推进动作索引，禁止继续沿用未来视觉起点row0约定。
原始归一化7D、完整模型50x32、后处理7D和实际native命令分别保存。完整32维输出只做来源/解padding证据，不作为有效32D机器人动作。
每条原始观测和配对初态checkpoint保存。配对初态及首个bootstrap noise/完整输出要求逐值exact。
每条新请求用按episode相同seed的专用CUDA Generator生成noise；不允许native线程的随机状态干扰模型噪声。不同轨迹的后续同序号噪声不代表同一物理情境。
先单次真实bootstrap得到50行动作，不额外预热或丢弃失败；startup单独报告，控制wall从bootstrap安装后计时。
请求阈值30，延迟初始7；以后完整请求时长P90/window50，ceil(20*P90)+1，guard2；超过cap不截断为已合格，等待可用队列耗尽后以bootstrap请求恢复。
remaining=0可请求；非空时只有delay<=8且delay+2<=remaining<=30且无pending才发起。任何时刻只允许一个请求在途。
serialized提交后阻塞等待，async继续消费旧动作。唯一调度差别为这一阻塞；推理完成后都按实际消费行安装。超过8行或整块已过期则拒绝，不覆盖旧块。
每个wall时隙最多一次native命令；迟到不追赶突发发送。停止时先关闭队列、等待已派发请求返回、拒绝其安装、join线程，再关闭Env。
这是按需推进仿真，不是环境在等待时连续演化的真机验证；50ms nominal tick不等于已证明硬实时。

## 可核验证据和判读

独立CPU审计全部16条：原始初态/配对bootstrap、逐request和观测身份、前缀源、一次trim、完整输出/解padding、每个归一化/后处理/native命令、时间嵌套和单在途、owner退出/Env关闭、预算与全局intent/return。
不重新运行模型或仿真。动作来源逐值exact，不放宽数值；控制wall/分位数独立NumPy复算。审计独立验证数组和时间关系，不冒称从RGB独立计算策略正确性。
每臂报告8episode的success、正常timeout、action数、wall、无动作slots、underflow、过期/停止丢弃、接管数、真实model/native区间交集、完整请求和owner排队P50/P95/P99。
首个bootstrap成本单独保留，非bootstrap时延统计包含停止时已执行但未安装的请求；不删慢请求。
wall不包含加载/startup/归档，实验总wall另报；无动作slots=全部实际覆盖时隙-实际dispatch，不从分母扣掉等待。
工程机制观察：全部技术接纳，serialized无真实交集、async有交集；等待减少按每实际动作的空时隙率描述，同时完整列出每对差异。成功率差异和不同动作数可能混淆总wall，不将总wall下降直接当相同轨迹加速。
不据8对声称统计非劣/普遍泛化/部署安全。RTC没有臂，不能声称RTC效果；旧所有阴性与风险阈值null保持。
没有按结果增加seed/重复试验/扩频率cap。任何科学方向都完整报告，有意义的已审计工程发现可提交推送并反馈Issue #1。

## 实施与冻结

保留21份旧pending状态/字节；只提交新增runner、audit、测试与本计划。开发先合成线程/控制循环/来源审计测试，不创建科学Env。
CPU prepare绑定HEAD、全部相关源/权重/配置哈希、原专用offline/EGL/GL库环境、历史身份与旧pending。
先查E-OBS1-REGISTER:<HEAD>唯一标识，单次gh原生POST后按真实ID独立GET/body exact；数据来自工具真实返回，不虚构登记。
一个正式监督入口，attempt1/retry0；模型加载90秒、每完整模型请求15秒、每native调用30秒、每episode100秒；soft1500/hard1530，工具1620秒。TERM后5秒才KILL，只管理本worker组。
首个技术错误封存输出与已消耗身份，不修改后自动重启该合同。故障时保存能取得的全部partial控制、native和模型账本。
正式退出后只运行一次独立CPU审计；若审计读取器自身有错误，保留失败文件，最小修复和新命名复核不得重跑模型或修改原件。
明确安全/权限拒绝不更换通道绕过。普通接口/参数错误透明修复；未知POST先查状态而非重发。没有reset/clean/stash/force push、升级依赖、其他任务干预或真机操作。
本轮GitHub连接器最新读取FORBIDDEN，未重试。依据本地已落盘路线A而非声称已读远端最新指令；必要新登记/结果发布是独立动作。
