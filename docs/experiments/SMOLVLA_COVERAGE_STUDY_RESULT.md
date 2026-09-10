# F-COV1：获得非零验证候选，但不是所有指标全面领先

2026-09-10。执行HEAD `6249b03d716e9578ac45f36b677c158ed93806ac`；
预登记5614574855实际ID单次GET正文exact后，唯一运行16条新轨迹和四组训练。
运行器与退出后独立CPU核验均通过。本轮只开发验证，没有读取或生成test任务8/9标签。

## 数据与比较

新采集task0–7×state48/49，共16条identity async轨迹，全部正常结束。
采集10 success/6 TimeLimit均保留；预测器未控制Env，采集成功率不能归因于预测器。
每条按request_id取最先4个完整planned前缀→future，共64例；真实delay均为3。
旧F-ACT1开发标签仅取task0–5/state46的24个样本。single训练24例/6episode，
multi训练72例/18episode；共同新验证为task6/7×state48/49的16例/4episode。
旧validation不选点、旧test不读取，确认集21–40不动。

四组同seed20260912、69680参数、零残差初始化、72updates/batch1、AdamW lr0.001/wd0.0001/clip1。
每task均12次更新曝光，multi下每初态每task4次；同覆盖的有/无动作组样本顺序exact。
共同目标Lz/Sz+La/Sa，尺度仍由原47个train固定为2131.5723863966923和0.058981226729922634。
没有同时新增逐样本归一化、identity保持项或改变目标权重。
相同验证row0 episode-macro在0/36/72步选最早最小；四组均选择72步，残差权重均非零。

## 验证结果：四episode等权，越低越好

|方法|所选步|token MSE|首动作MSE|50行动作块MSE|
|---|---:|---:|---:|---:|
|identity|—|2670.828865051|0.031709703660|0.053248021140|
|single_conditioned|72|2674.843589783|0.029763374143|0.058854652416|
|multi_conditioned|72|2674.252922058|0.029442513158|0.049140438008|
|single_no_action|72|2674.772369385|0.029402667045|0.052969111275|
|multi_no_action|72|2674.301811218|0.029710716241|0.049999032271|

multi_conditioned相对identity首动作误差下降7.149832%，整块下降7.714058%；token误差反而增加0.128202%。
相对同覆盖multi_no_action，首动作/整块分别低0.902715%/1.717222%；
相对single_conditioned分别低1.078040%/16.505432%。
所选候选非零、首动作优于预设的identity/single_conditioned/multi_no_action且整块不劣于identity，
故`validation_candidate_gate_passed=true`，不是选回0步制造的正结果。

**首动作全表最好的是single_no_action，不是multi_conditioned。**
multi_conditioned的首动作比它差0.135519%，但整块好7.228124%。
预设推进门没有要求击败single_no_action；门通过不能改写为“四组全面胜出”或动作输入普遍有效。

|验证episode|identity首动作MSE|multi_conditioned首动作MSE|相对identity误差降低|
|---|---:|---:|---:|
|task6/state48|0.030254652025|0.039607416664|−30.913476%（恶化）|
|task6/state49|0.016374576022|0.012830489228|21.643839%|
|task7/state48|0.013116091490|0.012457482633|5.021380%|
|task7/state49|0.067093495105|0.052874664107|21.192563%|

四组各3/4 episode首动作优于identity，不隐去task6/state48的退化。
multi_conditioned的验证row0为0.031709703660→0.032244153379→0.029442513158（0/36/72步），
36步尚未受益；不声称训练单调改善或全程稳定。
所选模型在自身训练群体上的row0相对identity分别下降17.303455%/19.053448%/16.915387%/18.759412%，
顺序同上四组；single/multi训练分母不同，这些百分比不构成覆盖因果效应的共同总体比较。

## 原件与独立核验

16条新native的3517个measured动作、146次planned接管、16份初态checkpoint和64个所选prefix/cache通过原source审计。
运行时16条首例current两camera token/state对归档exact；新64个identity decoder、旧6个identity及6个oracle参考重放exact。
退出后CPU核对同初始化、288步任务/初态曝光及目标计算、0/36/72验证选点、保存预测token和完整输出；
1440项数值比较通过，最大相对归约差2.286721696e-7（rtol1e-6/atol1e-7）。
64个零步验证完整动作对原生归档逐值exact，输入/动作exact门未放宽。
独立CPU审计首次exit0；解释计算也只用已保存数组，无新增forward、更新或Env。

实际Env16、settling160、measured3517、native main195、native capture32及内部32/96/32。
离线encode80，正式decoder812=旧参考12+新标签128+验证192+梯度训练288+所选模型训练集评估192；
梯度decoder/backward/updates均288，no-grad decoder524，offline capture62及内部62/186/62另记。
7501 intent/7501 return，error/unknown0；1187组外层/离线phase和1349组native模型phase完整。
所有新episode worker/Graph/处理器/metrics/Env清理确认，VLA始终冻结，Python/version/140包metadata前后exact。
child2892454及supervisor2892405均exit0并收回，first_failure/stop_reason为空，无强制终止。
监督552.437030627s、独立外层556.406346276s；UTC07:07:57.844519–07:17:14.250866，attempt1/retry0。
准备10项新CPU一次通过，首个format差异原件保留，最终Ruff/format及固定入口通过，无依赖变更或同卡干预。

原始目录：`outputs/smolvla_coverage_6249b03d/`；四个检查点为对应arm的`.pt`。
准备/独立核验/发布脚本：`outputs/smolvla_coverage_preparation_7b9d527d/`。
旧F-ACT1待提交文档和所有旧结果保持，只有本轮新文件提交。

## 结论边界与下一步

本轮得到达到预设开发门的非零候选；验证用于选点，不能当作盲测或统计显著性证据。
仅两个验证任务、四episode、每条前四个样本、delay3、单训练seed；当前收益异质且相对动作消融幅度小。
oracle为真实future视觉+当前32Dstate+同language/noise的原十步策略输出，不是专家/最优动作或成功率上界。
本轮是三初态训练数据策略与单初态重复的对照，不把数据量、重复次数和初态多样性拆成独立因果结论。
不与旧验证表拼接计算收益。生产默认不变，baseline/realtime/predictor闭环资格仍false、risk_thresholds=null。
下一步冻结四个检查点，在未用于训练或选择的新评估数据上比较，必须保留首动作最强的single_no_action。
本轮未启动独立test或在线部署；原生7D接口和state/risk仍需独立处理。
