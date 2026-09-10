# F-ACT1：首动作监督的四组离线对照

接续 b7f0831f / Issue #1 评论5612797920，依据用户本轮继续实验指令。
旧F-LAT1/F-LAT2结束；不读取其训练/测试数组或权重，不复用state41–45结果选点。

## 问题、数据和唯一运行

比较纯token目标与首动作联合目标，并分别保留conditioned/no_action，构成2×2对照。
新建12条identity async轨迹：task0–7各state46，task8/9各state46/47。
前6条train，中2条validation，最后4条test；Env seed=980000+100*task+state，
policy seed=990000+100*task+state。全部原libero_object/task_order_index0。
每条按request_id取最先8个完整planned前缀→未来观测；不足8个用全部，0个则停。
只按完整性排除，不按成功、误差或延迟筛样。sample按原绝对动作索引对齐。
test标签和指标仅在四组检查点全部固定以后提取；不输入训练或选点。
保护confirmation21–40，所有旧实验报告与输出保持。

12条使用原NaturalEngine/worker/queue、严格loader及原native转换，不注入暂停。
50/1/10、20Hz、threshold30、P90/window50/margin1/cap8/guard2、any-late整块丢弃、
identity context/current state、same_path_discard_probe_v1、compile=false保持。
predictor不控制Env。每条fresh Env/engine，10 settling，原3阶段startup，owner播种一次。

## 四组与损失

token_conditioned、token_no_action、joint_conditioned、joint_no_action。
均为原69680参数rank16预测器，risk关闭、零初始化up projection、独立同seed20260912初始化。
不使用旧检查点。两台相机scaled tokens、normalized7D committed actions、32D model-ready state、
真实delay；no_action仅将动作置零，保留mask/delay/state/视觉。

每组60 updates，batch1，AdamW lr0.001/wd0.0001、clip1；相同样本索引序列。
Lz为native bf16量化后future token MSE。La为原十步decoder的有效7D row0与oracle输出的MSE。
oracle=真实未来视觉+当前state+同language/noise，不是专家或最优动作。
Sz、Sa仅为train样本identity的平均Lz/La（至少1e-12）；
token目标=Lz/Sz，joint目标=Lz/Sz+La/Sa。固定权重1，无扫描。
通过原model.sample_actions对输入token反传，VLA参数requires_grad=false且无梯度；
不用CUDA Graph反传、不改denoise步数、不构造未来state。bf16 cast使用PyTorch原生梯度路径。
每个joint组首个实际训练步记录action-only token梯度、完整预测器梯度及VLA梯度数；
首步零残差的可导decoder输出必须与原生归档exact，否则首错停止，不放宽阈值。

所有组使用相同validation row0 episode-macro标准在0/30/60步选择最早最小值，包含identity起点。
不按各组不同训练目标选checkpoint，不读取test再回训练。

## 判读

主要比较joint_conditioned vs token_conditioned的test row0 episode-macro，
动作输入增量比较joint_conditioned vs joint_no_action；分别报告，不混为一项。
候选门：4个test episodes均有样本、joint_conditioned row0同时优于identity、token_conditioned、
joint_no_action，且chunk不劣于identity。该门不等于显著性或闭环成功率。
报告五组（identity+四模型）的token/row0/chunk、每episode值、胜出数与sample/episode两个分母。
模型目标改变的比较只发生在本新实验内部，不与旧表拼接选优。

## 固定预算与停止

Env12；settling120；measured每条280/总3360；native main每条160/总1920；
recovery probe每条50/总600且计入main；native capture每条2/总24，内部setup/warmup/capture24/72/24。
每条ready后1200 slots/60s，startup30s，单native30s、主request15s。
每条≤8样本，总≤96，其中train≤48/validation≤16/test≤32。
额外双相机encode≤108（96future+12current）；离线完整十步decoder总≤640，
其中train-label≤128、validation≤192、joint训练≤120、test≤192（六上下文），不额外参考重试。
offline Graph仅no-grad reference/validation/test，capture≤34，内部setup/warmup/capture≤34/102/34。
predictor更新240，完整decoder backward120，首步额外action-only VJP2；无VLA训练。
native collection≤960s，每offline forward/backward≤30s、每组训练≤600s、外层1800s TERM/1830s KILL；
单调用/phase超时先记录全线程栈、TERM，自有子进程5s未退才KILL，不操作其他进程。
attempt1/retry/resume/replacement0；首个技术错误停止，负结果/未触发不扩样。

## 准备、核验与交付

固定既有uv offline/no-project/no-download与模型Python/140包，不安装升级。
CPU测试覆盖split、四组、目标和macro、动作隔离、预算、零残差和可导路径的fake用例。
当前代码/PLAN/TESTS先提交推送；按GitHub实际登记ID单次回读exact才执行。
保存实际native源证据、选样、预测token和完整action arrays、训练步损失及梯度、validation各次输出、检查点。
退出后仅CPU核验新native来源/清理/预算、cache、固定选点、五组指标和执行回执，不重跑模型。
输入token/identity动作exact；CPU/GPU指标归约rtol1e-6/atol1e-7，不用此容差放宽动作exact。
baseline/realtime/predictor闭环资格仍false；risk_thresholds=null；旧confirmation untouched。
