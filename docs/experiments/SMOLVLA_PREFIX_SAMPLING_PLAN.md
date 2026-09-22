# F-PSD1：冻结小试检查点的十步条件动作诊断

2026-09-22，接续bf9e5474与SMOLVLA_PREFIX_TRAIN_PILOT_NEXT.md。F-DTC1和F-PTP1已完成，不重复其回放/训练。新合同只回答：流速度损失的比较，是否与十步积分后的动作误差一致。原F-PTP1主门失败不改写，未晋级为闭环候选。

## 不变量与采样语义

原始检查点、两份128更新的absolute projection检查点、normalizer和四条dev冻结，不增加训练、不读取sealed图像/模型结果、不修改src/lerobot、原Graph/control_loop/queue及21份旧pending。ordinary不是prefix训练检查点，载入时核验arm、base policy、manifest、version、128更新与8参数张量；禁止将绝对权重相加当delta。

新增默认不接入部署的条件采样器：只接受当前观测、独立noise、提交前C和BxCx7归一化旧前缀。无target suffix或episode终局/valid_steps参数。每次Euler步，前C行7维真实值及25维补齐零保持干净，流时间0；剩余行使用原十步1→0时间和原速度积分。不启用guided VJP，不切换生产trained模式。不修改原观测原点或实际消费一次trim。

C0必须与同检查点native sample_actions完整50x32逐值一致。新采样器独立执行prefix prefill和原Euler，不直接返回native完整调用结果；C0步级使用原denoise_step，非零C使用训练已验证的逐动作时间embedding与原缓存裁剪规则。合成测试覆盖0/3/8、混合batch、前缀/尾段填充值、错误配置、checkpoint reload、固定前缀与十步算术。

## 固定总体

沿用F-PTP1已封存的16个dev case（4轨迹×2anchor×C3/8），相同接收者、noise种子、正确动作和错配供体。每个检查点frozen/ordinary/prefix：16case×native_unconditioned、correct_prefix、mismatched_prefix=48正式解码，加16个新采样器C0校准。总192次完整解码、1920个denoise步、192次真实双相机编码；48对C0完整输出/后处理输出exact。每次重新编码，非缓存视觉替代。记录新采样器144次调用的全部十步x/time/velocity以CPU审核算术。

正确前缀取离线专家演示，仅作动作诊断；线上旧策略前缀可能偏离分布。本轮不据此称为恢复能力。错配用原固定donor的前C行，只改变前缀，目标/有效后缀/噪声不变。生成时不传终端标签；合法末尾mask只用于事后评分。至少保留一个未执行合法动作。

## 指标与预定描述性推进门

在相同case上同时报告：(a)50行中实际有效的7D整块MSE（前缀被硬固定，非主要证据）；(b)C之后有效7D后缀MSE；(c)row C的7D首个将执行动作MSE；(d)十行内首个可执行短段MSE；(e)后处理命令单位下对应误差。均先按case、再按四轨迹等权。整块均值不能抵消首动作退化。

下一步仅可考虑进一步开发的条件：prefix/correct在后缀和首动作两个主要均值均优于ordinary/native_unconditioned与frozen/native_unconditioned；相对ordinary两个指标均至少3/4轨迹改善；相同正确前缀下优于ordinary、frozen；自身正确前缀两个主要均值优于错配。任何一个不满足均不晋级。所有16case/4轨迹和不利pair保存，禁止改选checkpoint/供体/步数、补测或拼入sealed。即使门通过，也不等于新权重Graph/实时/任务资格。

## 执行预算与审计

正式模型加载1，模型全参数冻结，无backward/optimizer、Env、Graph capture、下载或生产变更。soft alarm600s，工具总预算720s，attempt1/retry0。每call记录意图/返回，首技术错停止封存；非优势不提前停止。所有检查点只修改进程内投影副本，结束恢复原参数和求导标志。原normalizer/权重/来源hash前后相同。

代码测试后冻结提交，一次prepare绑定源码/四dev包/原训练preparation与两个checkpoint及其审计。必要登记已有授权，唯一标识F-PSD1-REGISTER:<HEAD>先GET查重，一次literal gh POST再独立GET真实ID与正文exact后启动。明确安全拒绝停止对应动作；网络未知先核实，不盲目POST。

退出后一次独立CPU审计：192/192调用、144评分总体、48C0配对、checkpoint完整性；以原F-PTP1冻结target逐值核验noise减去归一化truth，核验前缀/评分mask（不在CPU冒称重演CUDA随机数生成），重算每个Euler步算术和全部MSE、供体及目标同源。CPU不重跑Transformer或物理仿真，不把算术核验当独立重算模型。使用旧dev，不宣称独立确认；后续先看真实动作证据再决定训练路线，不追加训练到阳性。
