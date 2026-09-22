# 前缀目标范围实验：E-RPI1 / E-RPF1

2026-09-22，接续65ec2f8d。用户明确选择：保持Graph、真实反馈与原时间对齐，把变量从安装时机转为旧动作目标范围。两个阶段分别登记、各attempt1/retry0，不扩充旧实验、不覆写旧阴性或生产默认。

## 固定算法与不变量

新候选仅使用ZEROS权重w[j]=1(j<C)，否则0；不是关闭全部RTC，不是硬夹紧动作，不宣称完整VJP不会通过耦合改变尾段。原RTCProcessor.denoise_step、十步Euler、7D有效坐标、权重、Graph完整VJP、50行/20Hz/cap8/一行余量不改。独立PrefixOnlyGraphRuntime显式校验ZEROS配置，不放宽旧EXP构造器。图使用原CPU公式精确缓存权重，实际C及长度每请求刷新，不重新捕获。

C始终在请求前由原P90/window50、初值7、余量1估计，三个承诺臂都执行完C步旧动作再安装：早到保留结果并继续旧动作，迟到在C边界等待，停止时取消不继续移动。原RTCExecutionQueue仍按实际消费量trim一次，原control_loop不改。禁止取未来实际delay作为条件，禁止身份特例或搜索C/horizon/strength。

## A / E-RPI1 动态数值与成本

同原E-RDC1的16个开发输入与225种可提交d=0..8,L=d+2..30组合，每组合原生ZEROS eager/Graph各一次，加16对无前缀校准，共482请求、241对。按原固定input_schedule循环分配样本并交替臂顺序。短前缀仅为接口探针，不是新物理轨迹。

每次从原始双相机重新处理编码并刷新8项输入及真实归一化前缀；482输入exact、32无前缀原参考exact、241对完整50x32/归一化/后处理输出共723数组exact。新公式的参考是同ZEROS eager，不要求与EXP相等。225个Graph有前缀请求全数<=350ms且ceil(P99*20)+1<=8；不删慢值。225相关探针不构成长尾统计保证。

VLA1，2capture（无前缀/有前缀）且额外setup2/warmup6/capture2共10sampler；241正式Graph重放/2250被捕获VJP步。模型90秒、10组各90秒、request30秒、11阶段；软1500硬1530工具1620秒。技术首错停止封存，退出一次独立CPU审计通过及预算通过后才允许准备B。

## B / E-RPF1 同期五臂真实开发闭环

身份固定0/10、2/10、6/10、7/10、7/11、6/11、2/11、0/11、7/18、7/19。每身份五臂各一次：serialized、aligned_async、committed_base（承诺无RTC）、committed_exp、committed_prefix（新候选），顺序按pair%5轮转，共50episode。Env seed1160000+100task+state、policy seed1170000+100task+state固定；前8和后2分开报告，全部均为已见开发，不并入旧32对或未用确认。

三承诺臂使用同一接收规则；计算耗时和反馈不同仍可能使每次C不同，全部记录，不把终局差异称为每步C相同的纯mask因果结论。所有臂同原动态请求快照、归一化前缀来源、模型、50行/十步/20Hz和原控制循环。两基线及committed_base不施加RTC，EXP和ZEROS保持同引导强度10/horizon10的原完整VJP；ZEROS在C后无显式旧目标。所有臂控制前准备两图，启动成本不省略。

固定推进门：独立技术审计；committed_prefix不丢失其余四同期对照任何成功身份；五臂全部非首次完整请求<=350ms且ceil(P99*20)+1<=8；候选真实模型/native重叠、空slot/动作比低于串行、无underflow/expired、承诺C=实际消费=trim、真实前缀和权重已应用。正常任务失败不提前停止、不补样、不改判据。全部终局、动作数、wall、取消、承诺/预计与实际、早到保持/迟到等待、来源观测年龄、提交至完成时间、图释放及不利配对报告。前缀误差不能替代任务保持。

预算VLA1、50Env、500settling、最多14000测量动作（每episode280）、最多160请求每episode/8000总、1200个20Hz时隙。100bootstrap capture，额外setup100/warmup300/capture100共500sampler另计；控制中禁止capture。模型90秒/request15秒/native30秒/episode100秒，软1500硬1530工具1620秒，52阶段；只管理自有进程组，TERM后5秒才KILL。

## 完整性、登记与审计

两阶段同一冻结执行HEAD，每阶段一次prepare绑定源码/权重/环境/历史/原21pending；B额外绑定A的result和audit SHA。必要发布已有授权，唯一标识查重后一次独立gh literal argv POST，真实ID独立GET及正文exact留存才启动。普通网络错误先核实状态，未知POST不得盲重发；明确安全权限拒绝停止该动作，不换工具规避。

不改已有src或执行器/队列/owner源码。分别一次退出后CPU审计，不重跑模型/视觉/VJP/物理仿真或GPU计时。输入审计检查新ZEROS公式和完整eager/Graph结果；闭环审计同时检查实际配置、权重、提交前缀、所有命令来源与一次trim、承诺无引导对照的conditioned_steps=0、50实例同owner恢复释放后join和Env关闭。

开发推进失败则冻结阴性，不追加到阳性、不继续改接管行号或静默切trained模式；开发门通过才另立新独立确认合同，不能并入本10例。无真机、训练、依赖升级、无限后台任务或生产默认变化。
