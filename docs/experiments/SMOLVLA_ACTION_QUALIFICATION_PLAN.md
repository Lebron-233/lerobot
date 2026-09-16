# F-ACQ1：冻结中心化候选的新初态资格实验

2026-09-16。依据 Issue #1 评论 5683274000 与 F-ACR1 完整报告制定。
这是新合同，不是恢复/重跑 F-ACR1、F-PFX1、prepare 或旧 confirmation。

## 1. 为什么先做这一轮

F-ACR1 在反复使用的开发验证集上，中心化真实首动作 MSE 为 0.013862918700，
较无动作基底下降 51.7155%，4/4 episode 改善。但仅 8/16 样本改善，单样本
(7,49,5) 贡献净收益 78.6290%；训练样本 (4,46,5) 恶化 0.624101963214。
token MSE 也未改善。因此当前证据支持“值得独立评估的开发候选”，不支持
普遍逐样本改善、视觉预测整体更准、部署加速或闭环收益。

本轮问题：**不改变已选权重、残差强度、尺度或门，在此前未使用的初态上，
中心化的动作指标优势是否存在，且不是少数大收益 episode 掩盖大量退化？**
先回答这一问题，不增加训练步数、不学风险门、不直接部署。

## 2. 冻结身份与准备门

模型来源固定：

- 基底：`outputs/smolvla_case_scale_r1_6ddb7505/case_no_action.pt`，第36步。
- 中心化、普通支路：`outputs/smolvla_action_centered_2926678f/{centered,ordinary}.pt`，均第72步。
- 三者均原69680参数结构、seed20260912元数据，strict加载，运行前后权重exact；不重新选点。

新采集顺序固定为 `(6,0),(7,0),(7,1),(6,1),(6,2),(7,2),(7,3),(6,3)`。
Env seed=`1100000+100*task+state`；policy seed=`1110000+100*task+state`。
**0–3 的未使用资格尚未在本次开发中得到确认，必须先通过正式 CPU 准备检查。**
未注册草案曾考虑42–45，但历史检查发现旧 `tuple` 格式记录已使用这些组合，因此已弃用；
当时没有预登记、训练、图像编码或仿真。后续一次组合只读盘点被安全检查拦截，未绕过。

`--prepare` 同时识别 `started.json` 的 `spec` 与 `tuple` 两种历史格式，扫描范围仅为
本仓库 `outputs`；未知格式、任一拟选(task,state)已有记录、源审计未接纳、源文件变化均停止。
不宣称检查过本机其他项目或其他机器。对未纳入本仓库的已知重复必须人工披露并停止。
不因为冲突自动换state/seed，不读取已有任务8/9标签或confirmation21–40。
仅可读取身份metadata，以及已经反复使用的F-ACT1/F-COV1开发来源，后者只用于16个旧验证锚点exact重放。

代码/计划/测试先提交推送。CPU准备保存完整manifest、来源SHA256和历史身份索引；
准备前后不初始化CUDA、不加载新VLA、不启动Env。按实际execution HEAD、唯一输出和
`preparation.json` SHA256预登记到Issue #1；POST返回实际ID后单次GET正文exact才可执行。
运行前再核对源哈希、历史索引、HEAD与工作树。历史/条件变化不是允许自动重选的理由。

## 3. 执行顺序

**A. 旧锚点兼容性检查。** 只加载一次VLA和三个冻结预测器。对原16个验证样本重新计算
FP32无动作基底B，必须逐值等于F-PFX1缓存；基底、中心化真实、普通真实的token和完整
50×32输出重放必须exact。此阶段48次decoder、64次预测器前向；不得用锚点重新选模型或调容差。
首差异即停止，新Env为0。该阶段不是重跑旧训练实验，不覆盖旧输出或消费新的科学样本。

**B. 仅identity async采集。** 8个新的Env/engine，每条沿用原strict policy/VLM/assets、
50/1/10、20Hz、cap8/margin1/P90/window50、threshold30/guard2、any-late整块丢弃、
current-state identity、原7D relativeOSC转换、same_path_discard_probe_v1、compile=false、
10 settling与三段startup。预测器始终不控制Env，采集success不能归因于中心化模型。

**C. 固定对齐。** 每episode按request_id取最先4个完整planned prefix→future，
不足4个取全部，0个为技术失败；不按成功率、误差、delay或状态变化筛选，不替换episode。
新样本总数N≤32，复用原对齐逻辑。新数据允许原cap范围内的实际delay，
不套用旧开发样本特有的delay3/4身份表。核对真实动作、mask、零填充和future=current+delay。
每episode首个current token/state重新编码exact；每样本identity完整输出与native归档exact。

**D. 冻结离线对照。** B必须由原无动作模型重新前向计算，不能将旧缓存B套给新样本。
保留identity、B、中心化true/zero/mismatched、普通true/zero/mismatched及oracle。
中心化组合为 `B+(h(a)-h(0))`，先减后加，再BF16往返；普通为 `B+h(a)`。
中心化zero的FP32 token及完整输出必须exact回到同样本B。
错配仍是同split/task/state/delay内按request_id循环下一供体，保留接收者mask；
单例不生成错配，但保留其true/zero。M≤N为可错配数，所有配对比较都只使用同一M例子集。

oracle仍为真实future视觉+当前state+同language/noise的冻结十步decoder输出，
不是专家、成功率上界或零动作物理反事实。只评估保存的离线预测，不做环境干预。

## 4. 预先固定的判读

主指标：归一化首动作7D MSE，每episode内部平均、8个episode等权。
同时完整报告50×7D chunk MSE与有效视觉token MSE。归约采用保存数组的CPU float64；
数值一致性rtol1e-6/atol1e-7，方向容差沿用`1e-7+1e-6*abs(control)`，exact检查不放宽。

`heldout_primary_gate_passed` 需要同时满足：

1. 8个episode均有样本；8个episode均有至少一个可错配样本（不要求每样本可错配）。
2. 中心化真实首动作episode-macro严格优于基底、普通真实及同模型错配；错配使用同M例真值子集。
3. 相对基底与错配分别至少6/8 episode严格改善。
4. 中心化真实chunk不劣于基底和identity。

`heldout_robustness_gate_passed` 除主门外，还要求相对基底严格改善的样本超过N/2，
并且8种留一episode后的平均收益均大于基底完整macro对应的方向容差。
这两个附加条件是本次新数据前制定的工程推进门，不是统计显著性检验或安全风险阈值。
样本相关、只有两个已见任务，不能外推为全新任务benchmark、跨机器人泛化或普遍安全。

固定报告每episode/每sample方向、最不利样本、最大episode与sample净贡献、
留一episode结果及delay分层描述；不删异常、不依据集中度重新选权重或门。
没有足够错配episode时保留完整有效结果，但主门未过/机制证据不足，不重采直到满足。
实验合同通过与模型假设通过分开：技术完整、模型阴性也属于有效完成。

## 5. 预算与退出

| 项目 | 固定上限 / 公式 |
|---|---:|
| 新Env / settling / measured steps | 8 / 80 / 2240；每条1 / 10 / 280 |
| native main / native captures | 1280 / 16；每条160 / 2；recovery probes每条≤50、计入main |
| native内部setup / warmup / capture | 每次capture按原1 / 3 / 1执行，全部另列实际日志 |
| 新双相机编码调用 | N+8≤40（N个future、8个current检查） |
| 新数据离线decoder | 7N+2M≤288，含oracle与所有可用对照 |
| 总正式离线decoder | 48+7N+2M≤336，含旧锚点48 |
| 总预测器前向 | 64+7N+3M≤384，含真实计算B及旧锚点64 |
| VLA / 预测器加载 | 1 / 3（基底也实际加载） |
| offline Graph captures | ≤8；其内部setup/warmup/capture另计，不伪装成正式decoder数 |
| 新训练/反传/重选/真实机器人/旧test标签读取 | 全部0 |
| attempt / retry / replacement | 1 / 0 / 0 |

load≤90秒；旧锚点阶段≤180秒且单锚点≤90秒；native采集阶段≤960秒；
离线编码/评估≤240秒；单次编码/decoder≤30秒、单样本离线评估≤120秒。
原native阶段每条ready≤1200slots/60秒、startup≤30秒、request≤15秒、native调用≤30秒。
外层soft1500/hard1530秒。只终止自有worker进程组，TERM后等5秒才KILL；
保存首错、日志、计数、active/pending、退出码，不重发未知状态调用。
工具或权限安全拦截不是可绕过的实验异常：停止对应动作，不换工具/降级防护规避。

## 6. 审计、交付和后继路线

保存manifest、源哈希、锚点数组、native原始归档、完整aligned cache、固定供体、
每样本真实动作/mask/h(a)/h(0)/raw与BF16 token/完整输出、计数/phase和退出。
退出后独立CPU审计重构native前缀来源、固定选样、供体、组合/精度/零动作不变量、
全部动作与token指标、各分母、逐episode统计、权重前后、预算和phase/调用闭合。
主门另有独立实现交叉核对，审计不初始化CUDA、不执行模型前向。
未来视觉编码与oracle正确性依赖冻结、被审计的执行路径与保存来源；CPU审计不会重新运行视觉模型。

报告脚本固定生成REPORT.md/REPORT.json，包含不利样本。发布到原Issue #1，并按实际ID回读。
旧pending文档原样保留；不git add全部，不覆盖旧实验、不提交大tensor/权重或凭据。

后续分支固定：技术失败先修真实首错；主门失败则暂停部署推进并回开发集研究；
主门过、稳健门不过则在开发集研究残差收缩/训练约束，不能拿本轮新样本调参后再称其独立资格集；
双门通过才另立完整B+h计算路径的配对时延实验，随后再审阅受控闭环资格。
本合同不自动授权后两轮。生产默认、baseline_qualified/realtime_qualified/predictor_benefit_tested
保持false，risk_thresholds=null，旧confirmation untouched。
