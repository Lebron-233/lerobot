# F-LAT1：原生承诺前缀上的未来视觉 token 预测试验

接续 E-NAT1 结果评论5611860645、回执HEAD195ff5fa。用户2026-09-10要求继续实验。
不重跑旧队列。本轮独立离线试验，只读E-NAT1的10条identity async轨迹；新增Env/native/真机为0。

## 问题、数据与边界

检验现有轻量残差预测器能否预测执行承诺前缀之后的原生视觉token，以及动作输入是否优于屏蔽动作。
来源固定 `outputs/smolvla_graph_natural_2d672b5e`，执行HEAD `2d672b5e3c9d56e54b21cc0a9211bab89f825f15`。
每条所有planned请求按request_id顺序处理；每条输入使用归档的两相机scaled token、原32D model-ready state、
原policy-normalized 7D committed prefix及delay。前缀pad到8，mask为真实连续长度；post7D仅用于核对实际native命令。
要求request observation index等于prefix起点，完整prefix每行与已发送来源的normalized/post动作逐值一致。
目标为该前缀执行完后、接管行执行前的真实观测；native返回时间必须早于目标观测及接管dispatch。
只排除结束时未执行完prefix或缺未来观测的请求，记录所有排除；其他首个契约差异停止，不跳过坏样本。
准备阶段仅CPU完成这项新target-alignment核对并保存一次prepared_pairs；正式GPU运行直接复用，不重复读取全轨迹。
不读取其他旧数组、confirmation、SO101 predictor权重或新增轨迹。

固定任务划分：train=0,1,2,3,4,5；validation=6,7；test=8,9。禁止同任务/同轨迹跨集合。
这是已知开发任务和state41上的小规模任务留出，不是全新benchmark测试集或统计泛化证明。
本轮不把Panda relativeOSC命令积分为位姿；不预测state。预测/identity/oracle视觉比较均保持同一当前state。
future RGB及其token只作监督或privileged离线reference，不进入预测器输入，不用于在线动作。

## 固定训练及消融

复用 LightweightFutureLatentPredictor；token_dim960/action_dim7/state_dim32，rank16、action_hidden64、
state_hidden32、delay_embedding16、fusion_hidden64、max_cameras2、depthwise1d，risk_head=false。
两臂同架构/初始化seed20260910/采样seed20260911。conditioned使用真实normalized动作；no_action将有效动作置零，
保留相同mask/delay/state/token。identity直接复制当前token。不得把未校准的risk零输出当置信度。
每臂AdamW lr0.001、weight_decay0.0001、batch8，固定200次更新，gradient norm clip1；无超参扫描/重训。
FP32训练，以mask下当前token+残差与未来scaled token的MSE监督；VLA/视觉编码器全部冻结。
验证在step0/25/50/75/100/125/150/175/200进行，按sample-mean native-bf16 token MSE选最早最低点，
含step0的identity初始解。两个checkpoint固定后各评估test一次，不据test回选训练或更改超参。

## 留出诊断与预算

报告每样本/任务和sample-mean/task-macro latent MSE、相对identity变化、两臂差异、最佳step。
测试集以原noise、language、当前state，在原十步Graph sampler中比较identity/conditioned/no_action/oracle视觉。
输出保留完整50x32，主要动作距离只算有效7D；分别报告row0与50行MSE到oracle视觉输出。
oracle输出不是专家动作、最优动作或success上界；距离降低也不证明闭环成功率改善。
identity必须与原归档full_chunk逐值exact；不同则本轮首错停止，不继续比较。此门检查新离线回放是否保持原计算。

上限：来源planned≤77；future双相机编码≤77，首样本current-token回放每task1次≤10，共≤87编码批次；
两臂各200更新/总400；test样本≤24，每样本4次正式decoder≤96；每test任务1次capture≤2，
每capture原setup/warmup/capture内部1/3/1，总2/6/2另记。只用归档noise，不新增随机policy noise。
固定原模型/revisions、offline uv及原解释器，不安装升级；GPU背景只记一次，不择低负载或干预其他进程。
单编码/decoder调用15秒，load/数据提取/训练各阶段120秒，外层600秒；超时仅终止本轮自建子进程并保留unknown。
先CPU定向测试和model入口import-only，冻结提交推送及GitHub登记exact回读，再唯一attempt运行；失败不改后重跑。
输出固定为 `outputs/smolvla_libero_future_latent_<execution_HEAD前8位>`；已存在则禁止启动。

## 判读

数据对齐通过、执行完整、训练/验证/test隔离及资源账目成立才接纳离线合同。
token或oracle动作距离的改善均单独如实报告，不自动转为predictor_benefit_tested或baseline/realtime资格。
首轮小数据负结果同样结束本轮，不追加seed/epoch/episode。生产默认、旧历史结论、risk_thresholds=null和旧confirmation保持。
