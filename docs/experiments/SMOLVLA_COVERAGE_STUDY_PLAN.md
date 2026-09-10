# F-COV1：训练初态覆盖 × 动作输入的开发对照

接续7b9d527d及Issue #1评论5614368170，依据当前用户继续实验指令。
只比较训练覆盖，不增加identity保持项、逐案例重加权或新损失。旧结果及F-ACT1待提交文档不改。

## 固定数据与四组

旧数据仅允许读取F-ACT1 `development_labels.pt` 的59个开发样本，检查task0–7/state46身份；
只取task0–5每任务request_id最先4个训练样本。原47个train仍用于共同Sz/Sa，旧validation不选点。
新采集16条identity async轨迹，顺序为state48的task0→7，随后state49的task7→0。
task0–5为train，6–7为validation；不访问task8/9、旧test或confirmation21–40。
Env seed=1000000+100*task+state，policy seed=1010000+100*task+state。
每条取request_id最先4个完整planned前缀→future；不足4用全部，0个则停止后续训练，不补样。
新标签提取使用已接纳的原native worker输入路径，归档current token/state逐值exact。
旧开发标签中的future/reference可复用；每训练task首个旧样本重放identity和oracle均须exact。

四组：single_conditioned、multi_conditioned、single_no_action、multi_no_action。
single使用state46；multi使用state46/48/49。两组共同新验证为task6/7×state48/49，最多16样本。
相同rank16/69680参数、seed20260912、零残差初始化；各72updates、batch1、AdamW lr0.001/wd0.0001、clip1。
每步task=step%6；每轮每task一次。single固定state46，multi按轮循环46/48/49。
每episode按request_id循环其最多4个样本；每task总12次、多初态下每state每task4次。
两种动作设置共享完全相同的样本序列；no_action只把7D normalized前缀置零，不改变mask/state/delay。
相较覆盖控制，按episode轮转是两边共同规则，不把新覆盖与旧实验直接做因果对照。

## 目标、验证和解释

四组共同L=Lz/Sz+La/Sa，Sz/Sa复用原47个训练样本identity均值，分别
2131.5723863966923 / 0.058981226729922634（源数组重新计算核对）。不根据新数据或验证重定标。
Lz为native bf16输出token MSE；La为有效7D首动作对oracle视觉参考MSE。
oracle仍是实际future视觉+当前model-ready32Dstate+同language/noise的原十步策略输出，不是专家或最优动作。
只训练predictor，VLA冻结；训练走原可导eager sampler，Graph仅no-grad标签/评估，state/risk不训练。
共同验证row0 episode-macro在0/36/72步选最早最小，保留0步；每次保存token和完整动作。
最终用所选检查点评估该组的全部训练样本；验证末步与所选值都报告，不能把不同训练分母作同一总体。
主要覆盖比较为multi_conditioned对single_conditioned；动作增量为multi_conditioned对multi_no_action。
推进门：所选multi_conditioned非零，验证row0同时优于identity、single_conditioned、multi_no_action，
且完整chunk不劣于identity。单纯选回0步不算学习收益；报告每episode和胜出数、sample/episode两种分母。
本轮仅开发验证，不评估任何test。无论结果如何，不追加样本、seed、权重或训练步。

## 执行与预算

原strict loader/RTX4070TiSUPER/模型revision及Python环境保持，50/1/10、20Hz、P90/window50/margin1/cap8、
guard2、threshold30、identity/fallback、same_path_discard_probe_v1、any-late整块丢弃保持，不注入暂停。
每条新Env/engine，原10settling和三段startup，无predictor控制Env。
新Env16、settling160、measured每条≤280/总4480、native main≤160/总2560（含probe≤50/总800）、
native capture≤32，内部setup/warmup/capture≤32/96/32。startup30s、主请求15s、native30s、ready后60s。
新样本≤64、原取样≤24；新双相机编码≤80。离线decoder≤900，梯度decoder/backward=288，updates=288；
no-grad上限为12个旧identity/oracle、128个新identity/oracle、192个验证、192个最终训练=524，总≤812。
offline capture≤64，内部setup/warmup/capture≤64/192/64；每个capture的额外5次内部调用单列，不算正式decoder。
native阶段≤1440s，训练每组≤600s，forward/backward阶段≤30s，外层2400s TERM/2430s KILL；
单调用超时仅管理自有子进程组，5s未退出才KILL，保留未知调用，不重发。
attempt1/retry/resume/replacement0；首技术错停止，TimeLimit为有效数据，不据负结果扩样。

## 交付

先CPU定向测试、固定环境入口--help和metadata核对，再冻结提交、GitHub登记并按实际ID一次exact回读。
退出后仅CPU核验新native来源/预算/清理、前缀和cache、采样均衡、288步证据、验证选点和数值。
指标归约rtol1e-6/atol1e-7，输入token与identity/reference动作保持exact。
旧训练/测试科学结论不变，baseline/realtime/predictor闭环资格false、risk_thresholds=null、confirmation untouched。
