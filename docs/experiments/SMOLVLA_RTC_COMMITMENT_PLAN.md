# E-RCC1：将 RTC 预计前缀变成明确的执行承诺

2026-09-22，接续993442a3的E-RDC1/E-RDF1。旧候选、原32对确认阴性和全部既有结果保持不变。只测试一个预先固定的安装合同，不搜索长度、延迟阈值、引导强度或成功身份例外。

## 假设与唯一候选

E-RDF1中79次RTC接管有78次预计延迟大于实际消费。该现象不证明失败原因。本轮把原预计覆盖值作为提交时声明的承诺C：沿用initial7、P90最近50次完整请求、ceil(20*P90)+1、cap8。该C仍作为完整RTC的inference_delay，模型、EXP/horizon10/guidance10、50行十步与所有设备权重/前缀刷新代码不改。

新CommittedPrefixOwner只改变接收时机。bootstrap仍阻塞完成后开始控制。后续提交冻结当前观测、真实归一化旧前缀和C；实际消费少于C时不安装早到结果，继续原旧命令；到达C时调用原receive，若结果晚到则阻塞等待，不执行第C+1条旧命令。等待不推进物理索引。原RTCExecutionQueue仍按实际消费一次trim，所有正常安装必须actual_delay=source_row=C。关闭时解除等待承诺只用于收回请求，原队列close拒绝发布，不能为完成承诺继续已经结束的episode。

这不是从未来读取实际延迟，不是改成row0、假装等待为零、硬前缀训练或硬夹紧动作。RTC仍为软引导，并且horizon10的软尾约束保留。C绑定的是执行调度范围，不声称精确模型前缀已等于旧动作。超时/承诺越界是技术错误，不静默取消约束。声明范围实现成功也不代表任务泛化。

## 同期四臂，40个已见开发episode

10身份顺序固定为0/10、2/10、6/10、7/10、7/11、6/11、2/11、0/11、7/18、7/19。前8先导开发与后2已知诊断分开报告。每身份serialized、aligned_async、rtc_async、committed_rtc各一次；按pair%4轮转顺序。Env seed1160000+100*task+state，policy seed1170000+100*task+state，无替换、补样或种子搜索。

四臂同DynamicRTCGraphRuntime、同无前缀/有前缀双图bootstrap、同RTCPrefixOwner载荷、同原control_loop和RTCExecutionQueue；两个RTC变体均传真实前缀，两个无RTC参照不引导。spec.arm表示原计算臂，spec.variant显式区分四臂，不能将两个RTC结果合并。只在committed_rtc启用承诺安装，其他三臂直接继承原receive。所有既有实现文件保持字节不变。qsize触发和一次trim不改，无计划行平移。模型线程实际重叠而非存档命令重放。

## 固定判据与计账

主要开发门：独立技术审计；新候选不得丢失三个同期参照中任何已成功身份；四臂全部非bootstrap完整请求<=350ms且ceil(20*P99)+1<=8；候选存在真实模型/native重叠、空slot/动作率低于serialized、无underflow/expired；每次正常承诺接管的声明C、RTC输入C、实际消费和source_row一致。任一任务保持门失败不追加样本、不调整本合同。

逐请求记录计算覆盖估计、声明承诺、RTC条件步数、实际消费、原归一化前缀来源、安装边界时间、结果是否已就绪、早到保持秒数、晚到剩余计算等待秒数、完整receive耗时及episode结束取消。后者不混称纯模型时间。所有完成请求含关闭时取消者纳入时延分母。报告每臂全部任务/动作/控制wall/无动作时隙、预计实际分布、在线请求和提交至完成时间、观测年龄、引导次数及资源关闭。不同轨迹成败的wall差异不是等质量纯加速。

## 预算、冻结、监督与审计

1次VLA、40Env/400settling/最多11200测量动作；每episode280动作、160请求、1200个20Hz时隙，总6400请求。每请求真实双相机编码，控制中不重捕获。每episode2图，共80capture；内部setup80/warmup240/capture80，共400额外sampler独立计账。所有臂首次双图时间另报。模型90秒/request15秒/native30秒/episode100秒，soft1500/hard1530/工具1620，attempt1/retry0。技术首错封存，正常成败完整跑完固定40例。

先合成测试早到、边界、晚到、承诺耗尽、越界、原cap拒绝、提交快照、episode提前结束与原代码路径、审计篡改。测试通过后普通提交并非强制推送原分支；CPU prepare绑定真实HEAD、源码权重、728来源基础、原21份pending、历史和offline/EGL/LIBERO配置。唯一E-RCC1-REGISTER:<HEAD>只读查重，独立literal-argv gh POST一次，真实ID GET/body exact落地才启动。不得覆写原输出。

退出后一次独立CPU审计，复用原动态前缀/动作来源审计，增加承诺、早到/晚到、历史延迟估计的独立检查；40episode、10组四臂首次全部输入/噪声/完整输出exact、42阶段、调用/80图及同owner退出。CPU不重算模型/VJP/视觉编码、后处理、物理或GPU计时。所有完整请求的模型计数保留真实分母，不报告新独立资格或生产部署。

授权沿用用户要求的研究、必要原分支提交推送和Issue#1完整登记/结果发布。保留旧pending、不reset/clean/stash、不升级依赖、不操作真机或改默认。明确权限安全拒绝停止对应操作；普通网络错误核实后处理，未知POST不盲目重发。

参考语义核查：Hugging Face官方RTC文档 https://huggingface.co/docs/lerobot/rtc （2026-09-22读取）：inference_delay决定前缀权重，execution_horizon决定后续一致性范围。固定提交承诺是本轮实验假设，不声称来自官方默认实现。
