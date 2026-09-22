# F-PTP1：固定同预算离线前缀条件优化试验

接续F-PTI1接口和F-DTC1数据时间诊断。仅数值学习试验，不是完整策略闭环/部署资格；原Graph/控制器/源检查点不变。原数据350命令回放支持20Hz原始行，四任务原始RLDS来源/划分已独立核对，但不宣称四任务所有物理状态逐值重现。

## 数据、监督与范围

固定openvla/modified_libero_rlds@6ce6aaaaabdbe590b1eef5cd29c0d33f14a08551的object首4分片。全部57episode检查CRC，按语言保留4任务20条唯一动作轨迹，已见episode814对应原始轨迹排除。cohort manifest SHA=d1a60540fce2056d10c29ea361329da2730e2e5f89415f86731c9e57c68d7bbf；train12/dev4/sealed4。每任务train5/1/4/2，不夸大样本量；本次不解码sealed图像、不做sealed模型评价。原始RLDS同记录的状态、动作、两相机直接绑定，native row order，无FPS插值或动作重复。使用基线检查点已有归一化，前后处理roundtrip检查，不重估全数据归一化。

## 固定两个训练臂

ordinary（无前缀条件，所有输入动作正常加噪）和prefix（已声明C步为干净条件、逐动作time0，其余加噪）。两臂都只在同一C之后的有效7D后缀计算损失，分母和目标完全相同。故ordinary是**同后缀监督的无条件续训对照**，并非默认全50行损失的标准BC；清楚区分，不把改变损失mask本身归为条件收益。

两臂各128次更新/batch1，固定种子20260922，任务每4步轮换；轨迹/20步间隔观测anchor按固定随机序列选，C均匀0..8，time=0.001+0.999*Beta(1.5,1)，每步noise seed固定。两臂同一schedule和原始参数起点，任务2仅一条train，不能声明泛化。仅训练action_in_proj、action_time_mlp_in、action_time_mlp_out、action_out_proj共8张量；视觉/语言/Transformer专家权重冻结，但梯度经冻结Transformer回传。保持eval dropout模式，两臂一致。

AdamW lr2.5e-6，betas(.9,.95)，eps1e-8，weight_decay0，clip_grad_norm1，foreachFalse。无scheduler/搜索/中间验证选模，只用最终128步检查点。每臂训练后保存独立投影检查点和optimizer state，不覆盖base；最后恢复base参数/求导标志。

## 固定开发评价

每dev轨迹在可用anchor的1/3、2/3位置各取一份观测，C3和8，共16个case。固定time.5及noise。对frozen、ordinary末态、prefix末态各用三种输入：无条件、正确演示前缀、循环取下一dev任务的错配前缀。保留同一接收者的观测/后缀目标，不改语言为供体语言。总144次完整评价forward；训练256forward/backward，共400forward。没有十步采样或新Env，所以报告的是流速度后缀误差，不是最终动作MSE或任务成功率。

正确前缀是离线专家条件，线上旧策略前缀会不同；错配是诊断对照，不是新训练数据。按trajectory等权汇总，报告所有4条dev及初始/最终/正确/错配。固定探索推进门：prefix正确条件均值优于ordinary无条件和frozen正确条件；相对ordinary至少3/4轨迹改善；prefix正确条件优于其错配条件；全部工程/有限值/预算/恢复通过。未通过不追加更新或选其他检查点；通过也只支持下一步采样/Graph验证，不是任务保持资格。

## 完整性与预算

运行前新HEAD冻结、sources包括检查点/数据/代码，真实唯一预登记及ID独立GET核验。一次模型加载；256更新，400forward，144评价，Env0，Graph0，sealed评价0，软SIGALRM900秒、工具960秒。技术首错保存，不重跑到阳性。独立CPU审计保存velocity/target/mask/损失、两臂完全相同数据目标与预算、optimizer最终step128和参数已变、base恢复、源哈希；不声称CPU重算Transformer或所有AdamW逐步算术。

任何推进不改原E-GFC1确认阴性、不混入旧10例/32对、不启用当前候选部署/真机。旧21pending不动。明确权限拒绝停止相应动作，不规避；普通网络失败先核实真实状态。结果包括负面和限制。
