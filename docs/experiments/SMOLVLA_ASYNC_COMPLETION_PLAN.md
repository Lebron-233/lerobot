# SmolVLA异步探索的收敛计划：先验证最佳冻结候选，而非继续盲调

2026-09-17。接续F-TUA1（结果90c54a5f，评论5707186673）。用户要求分析、制定可执行计划并实际推进至有意义结果。

## 当前判断

1. 异步调度与学习式补偿必须分开。E-NAT1已在单轮10对开发初态上完成原生identity异步闭环：两臂8/10成功，控制测量wall下降3.858%，没有模型本体加速或泛化声明；不重跑该队列冒充新成果。
2. IAR是目前旧开发集上的冻结比较基准，后续plain/guarded/BRP/TUA均没有在原固定门上超过它。TUA实际保留3/72更新，训练门保证的下降不是验证证据。继续在同16个开发验证样本上试loss，边际信息价值低，也增加自适应过拟合风险。
3. 一个尚未补齐的直接证据：IAR的I+(h(a)-h(0))尚未在未用过的初态上独立评估。ACQ1-R1评估的是B+(h(a)-h(0))，不能代替IAR，也不能把它的32个已看样本再次用作新资格。
4. 本轮暂停训练研究。冻结IAR原第72步权重、幅度1、真实两次预测器前向，开展一次F-IQ1新初态资格；不以缓存残差获得虚假端到端资格。

官方概念对照（只用于边界，不升级本地库）：https://huggingface.co/docs/lerobot/async 区分预测/执行解耦；https://huggingface.co/docs/lerobot/rtc 在异步之外处理动作块重叠一致性。本文检索于2026-09-17。RTC不是未来视觉预测器，其效果需单独对照，不能借用官方叙述声称本项目已成功。

## F-IQ1唯一执行合同

模型固定 outputs/smolvla_action_centered_2926678f/centered.pt（arm=centered,best_step=72,seed=20260912,69680参数），不加载plain/guarded/BRP/TUA作为可选候选。
输入与原IAR相同：I当前视觉，实际承诺的7D动作前缀、原mask/state/delay；FP32先h(a)-h(0)再加I，BF16往返，幅度1。
对照仅identity、iar_true、iar_zero、iar_mismatched及future-visual oracle。无权重训练、幅度搜索、检查点选择、学习门、真机。
oracle是未来视觉+当前state+同language/noise的冻结策略输出，不是专家/成功率上界；不据离线MSE宣称闭环改善。

新固定顺序(task,state)：(6,4),(7,4),(7,5),(6,5),(6,6),(7,6),(7,7),(6,7)。
Env seed=1120000+100*task+state；policy seed=1130000+100*task+state。这仍是已见task6/7的新初态，不是全新任务泛化。
本仓库outputs下started.json只读身份清单已查317条，八个拟用身份无冲突；正式prepare再次冻结并复核。有冲突即停止，绝不替换初态。外部未挂载运行记录无法排除，身份独立性声明只覆盖当前已登记仓库；有外部重复披露即停止。
原ACQ1-R1 task6/7 state0..3、task8/9标签、confirmation21..40不读取；历史只查身份元数据，不查旧观测/标签。旧开发16例只作exact锚点，不新选模型。

原native identity async工厂、队列、50/1/10、20Hz、cap8/margin1/P90/window50、threshold30/guard2、any-late整块丢弃、relativeOSC、same_path_discard_probe_v1和10settling保持。
仅identity控制新Env，预测器不控制环境。每episode按原selected_pairs规则选最先4个完整承诺前缀，N<=32；不足保留全部，0个技术停止；不补例。供体为同split/task/state/delay下一request，单例无错配；所有错配对比使用同接收者子集。
先CPU环境/工厂预检（不创建Env），再加载VLA1/预测器1，重构原16验证锚点h(a)/h(0)、IAR raw/BF16与完整输出exact；identity完整输出亦exact。随后8新Env采集，再离线评估。

预算：锚点decoder32/predictor32；新数据decoder=4N+M、predictor=4N+2M；总decoder<=192、predictor<=224、双相机编码=N+8<=40。无复用零支路隐藏前向。
新Env<=8、settling<=80、measured<=2240、native main<=1280、native capture<=16、offline capture<=8；Graph内部setup/warmup/capture每次1/3/1另列。所有旧训练成本保留，不算作本轮新增。
预检30秒、加载90秒、锚点180秒、native960秒、offline240秒，原单native30秒/request15秒、单decoder30秒；总soft1500/hard1530秒，工具1620秒。attempt1/retry0/replacement0；仅自有worker组TERM后5秒才KILL。
固定原uv和libero-reference-venv解释器，offline/EGL/LIBERO_CONFIG_PATH/LD_PRELOAD、删除PYTHONPATH；不安装/升级、不改驱动/全局配置。

## 一次性判读

主指标归一化首动作7D MSE，同时完整50x7D chunk、有效token MSE；episode等权、CPU float64。科学rtol1e-6/atol1e-7；方向容差1e-7+1e-6*abs(control)，exact不放宽。
primary：8episode均有样本和错配；iar_true首动作macro优于identity及自身错配；相对二者各至少6/8episode改善；chunk不劣identity。
robustness在primary基础上：相对identity及自身错配，各严格改善样本数超过配对样本的一半；相对identity的8种留一episode平均收益均为正且越过固定方向容差。
全部三指标、逐样本/episode、最大退化、最佳例、delay分层、贡献集中度和留一结果保留。门未过不选择好看的子集，不追加seed/state直到通过。
本轮诊断的意义是决定冻结IAR能否进入后续工程评估，不要求结果阳性。旧R1阴性、TUA阴性不改写；所有现有部署资格false、risk_thresholds=null。

## 收敛分支与最终完成标准

A. 独立合同接纳且IQ1主门/稳健门均过：冻结候选进入另立合同的真实完整路径时延对照，再进入identity与prediction受控仿真闭环。时延必须包含真实h(a)/h(0)、数据搬运、解码、同步和排队，报告P50/P95/P99、deadline/underflow；闭环报告配对success、控制wall、动作接管/前缀一致性及所有失败。两者未过不得说完成预测式异步。
B. 科学门未过：封存该新资格结果，不再围绕这一69,680参数候选及同16验证例继续小步调loss。将学习未来token路径记为当前不受支持；工程主线转为已成立的identity async与标准RTC/无预测补偿对照，先核对本地实现与动作时间语义，再新登记小规模闭环。不是放宽IQ1门或改名为成功。
C. 技术失败：保存首错、退出及已消耗身份，无自动重启本合同。
本轮只执行F-IQ1；A/B是明确后继路线而非承诺后台执行，不自动跨入未定义的闭环、真机或新增数据预算。

## 来源、发布和安全

保留21份既有pending的路径/状态/哈希及所有旧输出。新源/测试/本计划显式提交推送到Lebron-233/lerobot的codex/smolvla-graph-native-equivalence。prepare绑定真实HEAD、源码/依赖/检查点/锚点哈希、环境、历史和pending。
预登记F-IQ1-REGISTER:<HEAD>，Issue #1单次只读GET查重；不存在才gh executable literal argv POST一次，真实ID独立GET，正文exact及来源回执保存后启动监督入口。
已授权必要发布不再重复询问；未知POST先核状态不重发。明确权限/安全拒绝停止对应动作，不换工具绕过；普通语法/兼容错误可在不越界时透明修正。
正式退出后一次独立CPU审计，重新核native选样/动作接管与调用账本、输入/供体、残差代数、指标和门。审计不执行模型或新Env；不把重算聚合和预算当成独立验证模型前向真实性。
技术失败仅本地保存；完整、已审计的关键资格结论可按用户最新“有意义成果”要求发布，即使阴性也必须清楚标注，不称达成模型收益。结果评论POST/真实ID GET一次、保留原始不利结果。
