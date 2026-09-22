# 完整 RTC Graph 接入动态前缀：E-RDC1 与 E-RDF1

2026-09-22，接续 cd63157e / E-RGC1。两个独立登记的有界阶段，不改既有源码、模型或动作起始行规则，不覆写旧资格结论。

## 固定语义

使用原RTCProcessor.denoise_step、完整Jacobian/VJP、7D有效维度、EXP/horizon10/guidance10、chunk50/十步/20Hz。所有实际接管继续由原RTCExecutionQueue按真实已消费数trim一次，原control_loop不变。无保序row0策略、训练或身份特例。

DynamicPrefixRTC用原CPU公式预计算delay0..8和有效horizon1..10的精确权重；每请求将选定值copy到固定设备缓冲。真实前缀长度L零填充到50x7，只有原有7坐标有效。捕获中使用canonical key(3,10,50)，返回值严格为实际W(delay,min(L,10),50)，不是固定delay3；捕获包含完整求导。值变化不重捕获，只在bootstrap准备无前缀/有前缀两图。未覆盖的图像/语言形状显式失败，不静默用慢路径。

RTCPrefixOwner仅新增submit请求载荷，继承原serve、receive、关闭与原队列。frozen RTCRequest封装request stamp、深拷贝的当前观测、独立归一化CPU float32前缀、来源request ID。避免控制器继续消费后读到移动的尾段，不使用后处理命令作为条件。预计delay来自原P90窗口；实际delay仅接管时测量，两者差异报告而不隐去。

## A / E-RDC1：先证明动态输入的真实模型等价

复用原16份RTC开发输入，不读取新资格样本。原submit在非空队列仅在delay+2<=remaining<=30时发起；穷举delay0..8及L=delay+2..30共225种可提交组合，循环分配16样本。取存档归一化30行尾段的前L行作为接口探针，明确不是225条新物理轨迹。另16个无前缀校准，每组合native_eager与graph各一次，共482请求。按组合序号奇偶交替臂顺序，不选速度或结果。

原生eager传真实delay及Lx7前缀；Graph传50x7零填充和动态精确权重，全部482输入、32无前缀参考、241对完整50x32/归一化/后处理输出必须逐值一致，不放宽容差。CPU解析测试另外覆盖短前缀截断、长度1/50、预计和实际delay不同、prefix不足及关闭路径；合成GPU验证动态刷新与无recapture。

一次VLA，2capture及额外setup2/warmup6/capture2共10sampler另计；482真实图像编码，每有前缀Graph十步VJP，225次共2250被捕获引导步。模型90秒，10组各90秒，每请求30秒，原监督soft1500/hard1530、工具1620。attempt1/retry0，技术首错停止封存。效率门：Graph全部225次<=350ms且ceil(20*P99)+1<=8。只报告本接口分布，不作环境并发或任务保持结论。

## B / E-RDF1：动态输入门通过后另行登记30个开发episode

保持同一执行HEAD，但仅在A独立CPU审计和Graph预算门通过后prepare与发布B登记。已有10身份固定：0/10、2/10、6/10、7/10、7/11、6/11、2/11、0/11、7/18、7/19。前8与后2分别标明开发范围，不加入旧32对确认。每身份serialized、aligned_async、rtc_async各一次，按pair%3轮转顺序。原环境种子1160000+100*task+state与策略种子1170000+100*task+state不变。

三臂都走同一DynamicRTCGraphRuntime与同样两图bootstrap流程；基线传None而RTC臂传该请求真实旧归一化尾段及预计delay。每请求重新处理当前双相机、状态、语言，新Generator噪声；真实新动作控制Env，非重放命令。无后续人为等待，不改变threshold30、P90/window50、初始delay7、cap8、余量1、guard2。

每episode280动作上限、10settling、1200个20Hz时隙；30Env/300settling/最多8400动作，每episode160请求/总4800，60capture及setup60/warmup180/capture60共300额外sampler计账。模型90秒/request15秒/native30秒/episode100秒，soft1500/hard1530/工具1620。正常任务失败不提前停止或补样。

固定推进条件：独立技术审计；RTC无丢失两个同期对照成功身份；三臂所有非首次完整请求<=350ms与原P99门；RTC有模型/native重叠、空时隙/动作比低于串行、无underflow/expired且真实前缀确已传入。启动成本、所有成败/失败、预计与实际delay、前缀来源/值/长度、权重、裁剪一次、图生命周期与结束取消全部审计。前缀贴合不能替代任务结果，开发成功也不构成独立确认。

## 完整性与退出

两阶段各一次prepare、唯一标识查重、独立literal-argv gh POST、实际ID GET与body exact落地后运行；不重用旧目录。绑定真实HEAD、源码、权重、原21份pending、历史与offline/EGL/LIBERO配置。明确安全拒绝停止对应动作；普通网络故障先核实实际状态，未知POST不盲目重发。

分别一次退出后CPU审计，只复算保存的数值、输入来源、命令时序、日志、统计和资源闭合，不重算模型/自动微分/图像/物理仿真。A共11阶段，B共32阶段。所有图在同owner准备/重放/恢复/释放，先收回请求再join再closeEnv。禁止控制中capture、依赖升级、真机、生产默认更改、扩大确认样本或修改旧判定。任一技术失败封存；任务阴性完整报告，不调参挽救本合同。
