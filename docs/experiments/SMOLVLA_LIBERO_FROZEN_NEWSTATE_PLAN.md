# F-LAT2：冻结预测器的新初态检验

接续 Issue #1 / 5612169518 与 HEAD 8db97488。F-LAT1-r1 的两个检查点、选点和结果保持；
本轮不训练、不按新结果调整模型、alpha、风险门、样本或延迟。目的为复核动作输入增量，
不是重复旧12例、部署7D predicted路径或证明闭环收益。

## 固定新数据

libero_object/task_order_index=0，task8、task9，state42、43、44、45；每state各两个task，
偶数state顺序8→9，奇数9→8，共8条。Env seed=970000+100*task+state，policy seed=960000+100*task+state。
与原训练/验证/测试state41分离；旧qualification confirmation保留state21–40，不读取或运行。
使用原strict policy/VLM/assets、NaturalEngine、原worker/queue、identity context/state与
same_path_discard_probe_v1（仅实验启用）；50/1/10、20Hz、P90/window50/margin1/cap8、
threshold30、guard2、any-late整块丢弃全部不变。每条fresh Env/engine与原10 settling、cold/probe/fresh。
不人工暂停、不干预其他进程、不按负载重试。

每条按request_id取前12个完整对齐的planned前缀，最长96例。先按原aligned_pairs验证：
当前observation index=prefix起点；每个承诺normalized动作、post命令、实际native动作一致；
future在最后承诺动作返回之后、接管首动作之前。episode结束前没有完整future仅记排除；
其余超出12的完整对齐例记预算排除。不根据success、误差或动作幅度筛样。
动作前缀长度保持真实1..8，padding到8×7；模型state仍为当前model-ready32D，不积分relativeOSC。

## 冻结模型与对照

读取 outputs/smolvla_libero_future_latent_r1_63f1f306/conditioned.pt、no_action.pt；
config及元数据必须符合原F-LAT1-r1，step125/175、seed20260910、每臂69680参数，strict state_dict。
只eval/冻结，训练更新0；检查点在新轨迹前读取，并把原CPU权重快照留存供退出后直接比较。

四个主要context：identity、conditioned、no_action、oracle_future_visual。
每例原noise、当前state、language、mask及完整十步解码一致。identity完整50×32必须与本轮
采集归档逐值exact；每条首例的当前双图token/state须经原worker_batch重新编码后exact。
future只用于标签/reference，不进入predictor输入。

额外mismatched_action：对同episode、同delay的已选样本，按request_id循环移位一个动作前缀；
保持当前图像/state/mask/delay不变，用conditioned检查点。组内不足2例则本项缺失，主要比较保留。
记录donor身份、前缀是否实际不同及输出变化。它是输入敏感性诊断，不是反事实物理轨迹或因果成功率。

## 指标与预先判读

逐例记录native-bf16未来token MSE，以及有效7D row0/50行输出到同state/noise的oracle视觉输出MSE。
主汇总为先episode内平均、再八episode等权；同时报告样本平均、每task与每state，不做显著性声明。
仅八条均有数据且独立核验通过，并且conditioned在两个动作MSE上均严格优于identity和no_action，
才置 frozen_action_increment_observed=true；任意一项不满足为false，不能把token改善替代动作增量。
错配动作单列，不参与挑选检查点。两task曾出现在旧开发报告中，新的是固定初态/seed，不宣称全新任务泛化。
采集成功率属于identity行为策略；没有任何预测器控制Env。predictor_benefit_tested（闭环）保持false。

## 预算、停止与发布

唯一attempt，retry/resume/replacement=0。Env≤8、settling≤80、measured≤2240且每条≤280；
采集主请求≤1280且每条≤160，含最多50 recovery probe/条；采集capture≤16，内部setup/warmup/capture≤16/48/16。
ready后每条1200slots/60秒，startup30秒、model15秒、native30秒。None不推进Env、不补发。
离线：编码≤104双相机批次；predictor≤3批前向；decoder≤480正式调用，capture≤2、内部≤2/6/2。
模型加载≤60秒、单编码/decoder≤15秒、整体离线≤240秒；外层900秒TERM/930秒KILL，仅自建进程组。
单调用超时后原TERM等待5秒再KILL；未知返回保留，不重发。首个技术错误停止，不按缺样/负结果追加。
源审计及五项清理完成才归档数组。部署/训练/真机0，生产默认及旧所有报告、risk和confirmation不变。

先提交入口/测试/本PLAN并在Issue登记完整HEAD、八行清单、固定命令和独占output，按实际ID单次GET正文exact后执行。
退出后CPU重算保存的预测/动作误差、核对实际动作来源/初态/预算/退出及冻结权重，新增forward/native=0；
正式结果和发布回执分别提交。一次检查只解决会改变结论或下一步的具体问题，不重复旧已接纳审计。
