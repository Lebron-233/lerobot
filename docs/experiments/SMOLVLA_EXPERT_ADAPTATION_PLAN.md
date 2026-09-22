# F-EAF1 / F-EAT1：动作交互参数是否限制前缀学习

2026-09-22，接续3526e68b。用户已选择受控专家轻量适配。不是追加旧128步、不是新生产默认，不读取sealed图像/模型指标。旧模型文件、Graph、控制器、一次trim与21份pending原样保留。

## 固定新变量

仅四个既有动作接口模块继续可训练，加上实际执行动作自注意力的专家Q/V的LoRA。按运行分支及expert layer映射选择，当前32层/每2层自注意力意味着全局0,2,...,30；不适配交叉注意力、不适配K/O/MLP/state_proj/VLM。rank8、alpha8、dropout0、A固定本地CPU种子2026092201+i初始化、B0，adapter参数float32，输出回到原Linear dtype。原base不变，不作merge，不宣称部署Graph资格。

## A：F-EAF1 固定训练可行性

一次模型加载；仅取旧train schedule第一条的观测/动作，把C固定3、flow time0.5、noise seed2026092202。projection与projection+LoRA各3次同数据更新，分别从同原始投影起点开始，后者零B起点；AdamW沿用lr2.5e-6/betas(.9,.95)/eps1e-8/wd0/clip1/foreachFalse。合计6个完整训练前向/6反向/6更新，仅为技术探针，产生的权重不进入B或任务评价。

检查：零B首步完整velocity/loss及8投影梯度逐值相同；32个目标Linear均实际被调用；全部梯度有限，首步A=0为正常初始化性质、B非零，后续A梯度非零；优化器步数3；冻结原始权重未变，临时adapter拆除，原参数和求导标志恢复；每臂peak allocated<=8GiB，记录reserved和前/反向/更新耗时。没有以3次训练损失改善做能力判据。

技术首错停止、保存，不换配置继续。一次CPU审计通过后才能准备B。总运行限300秒、每前向/反向/更新组合30秒；工具360秒。无Env、采样、封存评价或Graph捕获。

## B：F-EAT1 同期2x2对照

A=projection ordinary，B=projection prefix，C=projection+LoRA ordinary，D=projection+LoRA prefix。四臂均从原始base重新开始，LoRA两臂A初始化相同且B0，不继承前置探针或前轮末态。各128更新，batch1，原12train/4dev/4sealed划分和原schedule完全复用。相同数据/anchor/noise/time/C/合法后缀目标和分母，同lr/optimizer/clip；合计512更新。普通臂所有动作加噪，前缀臂前C动作干净、逐动作time0，两者都只监督同一合法7D后缀。

按A,B,C,D固定顺序运行，不中途看dev挑模型；同更新数不是同FLOPs，报告参数量、forward/backward成本和显存。此次没有参数量匹配对照，不能声称收益仅来自位置而非多参数；一个rank/种子/预算不能证明整个专家训练方法族。

冻结末态后评价frozen+A+B+C+D五检查点。每个用原16dev case，native无条件/正确前缀/固定错配，各16，并额外16C0校准；共320个十步解码、3200速度步，240评分结果、80C0配对。每次重编码双相机；采样复用已验证smolvla_prefix_sampling，保留干净前缀和原十步，不给采样器目标后缀/未来观测。所有旧dev及不利结果保留，不读取sealed评价。流速度训练loss只作辅助，不代替最终动作。正确专家前缀是离线条件，不是线上恢复保证。

主候选D，不事后改选C。主要报告归一化suffix和first(row C)，辅助命令空间suffix/first/short，block含硬复制前缀不作门。固定探索推进门：D正确前缀的suffix/first均优于B正确前缀、C正确前缀、frozen正确前缀及A无条件；相对B的两个主指标各至少3/4轨迹改善；D正确均优于自身错配；前缀训练增量(C-D)在两主指标均大于(A-B)，四者均同正确前缀输入；命令空间D的suffix/first/short不劣于A无条件和B正确。任一未满足保留阴性，不改门/补样/追加到阳性。

这些是小开发集的路线探索判据，不是总体因果、任务非劣或真机安全结论。改善后也必须另行验证新权重Graph和真实队列前缀下的闭环。技术预算：VLA1、512训练前向/反向、320解码、0Env/0Graph/0sealed；单调用30秒，总软1500秒，工具1620秒；首技术错误停止，正常不利结果完整保留。原模型和adapter对象退出前还原。

## 完整性与发布

两阶段分别一次prepare，绑定实际源码/数据/checkpoint SHA及合同。正式前唯一标识GET查重、一次literal argv gh POST、实际comment ID独立GET/原始正文核对；必要发布与原分支push沿用用户授权。明确安全拒绝停止对应动作，未知POST先核实状态，不循环重发。

各阶段退出后一次CPU审计；B核验512目标/分母四臂匹配、checkpoint与optimizer计数、全部320来源/C0/积分记录及240评分。CPU审计不运行Transformer、不重新求参数梯度、不重测GPU或物理仿真。旧科学阴性不改，所有新增开销和跳过项据实报告。

方法来源：LoRA原论文 arXiv:2106.09685；本地实际dispatch为smolvlm_with_expert.py。未安装/升级PEFT，使用独立最小低秩Linear分支并进行零初始化与梯度测试。
