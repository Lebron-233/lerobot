# F-SCL1：仅开发数据的动作损失尺度对照

接续 bb2931dd / Issue #1 评论5615328437。依据用户继续实验指令。
旧F-COV2及所有已报告test不重跑、不参与训练、选点或尺度计算；不新增Env/native或测试标签。

## 唯一变量与固定数据

复用F-ACT1 development_labels.pt的task0–5/state46每任务前4例（24例），
和F-COV1 new_labels.pt的task0–5/state48/49（48例），共同训练72例/18episode。
共同验证仅F-COV1 task6–7/state48/49的16例/4episode；它已用于开发，不称盲测。
旧59例开发文件中的12例旧验证只验证split身份，不用于训练、定标或评估。
保护所有task8/9测试及confirmation21–40。仅加载两个明确标签文件，不扫描其他旧轨迹。
源文件SHA256在准备阶段保存，启动时比较；不读取或修改旧预测器权重。

四组global_conditioned、case_conditioned、global_no_action、case_no_action。
均从同seed20260912初始化原69680参数rank16预测器，零up projection，risk关闭。
同72次更新、batch1、原F-COV1 multi schedule，每个task/初态4次，每个样本一次。
同AdamW lr0.001/wd0.0001/clip1；同72例训练和16例验证；不增数据、换架构或加入identity保持项。
normalized7D承诺动作、两相机原scaled tokens、当前model-ready32Dstate、语言/noise、delay3不变。
无动作组只屏蔽动作值，mask、delay和其他输入不变。VLA冻结，原10步采样器反传，Graph仅no-grad。

## 尺度（全部只由训练数据确定）

Sz、Sa保持F-ACT1原47例训练identity平均误差：由开发标签按原train_scales重算，不用验证/test。
每个训练例Ei为其identity首动作到已有oracle输出的有效7D MSE，以CPU float64计算。
固定floor=max(0.1*Sa,1e-12)，qi=Sa/max(Ei,floor)，wi=qi/mean_train(q)。
global目标=Lz/Sz+La/Sa；case目标=Lz/Sz+wi*La/Sa。
Lz和La的量化/归约定义保持F-COV1。权重均值为1，避免把平均权重放大当作样本重权效果；
仍不能排除非线性优化中的有效步长变化。floor系数0.1和归一化预先固定，不做扫描。

低identity误差定义为Ei不超过72例train Ei的25%线性分位数，阈值仅来自训练；
同阈值用于描述训练/验证子群（若空则报告null）。不按该子群挑样或选checkpoint。
报告低误差子群的案例数、首动作误差、绝对增加、损害案例数；不以相对百分比遮蔽近零分母。

## 固定评估与判读

共同validation原始row0 episode-macro在0/36/72步选最早最小值，包含identity起点。
每组选定checkpoint后评估共同72例train；不运行任何test。
报告identity+四模型的train/validation row0、chunk、token、逐episode值、低误差子群、非零权重。
尺度效应case_conditioned对global_conditioned、同目标动作增量case_conditioned对case_no_action分开。
开发候选门：case_conditioned非零且步数>0，validation row0严格优于identity/global_conditioned/case_no_action，
chunk不劣于identity。global_no_action完整披露，另报所有参照是否都胜出，不把指定门当全面领先。
门或负结果均不允许在本轮追加步数、换设置、重跑或上线。

## 运行预算与复现

启动后按task/state排序，每个开发episode首例重放identity及oracle各一次：22*2=44完整解码，逐值exact。
validation=4组*3时点*16例=192；梯度训练=288；选定train评估=4*72=288；合计812正式解码，上限900。
Graph捕获<=64，内部setup/warmup/capture单列；梯度解码/反传/更新各288。
模型load一次；新增图像编码、标签生成、Env、native、训练VLA、真机、test读取均0。
单解码/更新30s，单组600s，参考120s，外层1400s TERM/1430s KILL，首错停止。
仅监督自有子进程组，TERM后5s未退才KILL；attempt1/retry/resume/replacement0。
原uv offline/no-project/no-download及模型Python/version/140包不变；一次GPU背景记录，不干预其他进程。

先CPU测试/模型入口无CUDA导入门，提交推送PLAN/代码/TESTS；GitHub预登记按实际ID单次GET正文exact后执行。
保留原dirty F-ACT1文档，不混入本轮；保存源hash、尺度、顺序、初始/选定权重、训练日志、每次评估完整输出。
退出后仅CPU复算保存数组/目标/权重/选择/预算与清理，指标归约rtol1e-6/atol1e-7，输入/参考exact不放宽。
本轮结果与发布回执分别提交，所有旧结果/生产默认保持，baseline/realtime/predictor闭环资格false，risk_thresholds=null。
