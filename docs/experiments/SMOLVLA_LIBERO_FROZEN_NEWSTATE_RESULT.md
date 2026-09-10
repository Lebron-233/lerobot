# F-LAT2：八条新初态完成，动作条件优势仍是混合结果

2026-09-10。执行HEAD `48e3c62a8fca3eb56d0f51e9027e31635957d6f7`；
预登记Issue #1 / `5612668685`在启动前按实际ID单次GET正文exact。
8/8条新native采集完成，63个对齐样本，退出后独立CPU核验通过。
`independent_contract_accepted=true`，`frozen_action_increment_observed=false`。
结果提交、正式评论和回读身份见独立RECEIPT。原运行器结果未改，attempt1/retry0，无补样。

## 问题与冻结条件

F-LAT1-r1在旧state41仅12个留出样本上，no_action的两项动作距离优于conditioned。
本轮冻结原conditioned step125与no_action step175检查点，每臂69680参数，不训练、不调参。
task8/9在state42–45上采集；这些任务曾出现在旧开发报告，新的是初态和seed，不能称全新任务泛化。
两模型在新轨迹之前读取；原权重、本轮起始快照和结束权重CPU逐张量exact。
旧confirmation的state21–40和旧实验目录不动。

原identity异步行为策略采集，predicted模型从未控制Env。当前双相机scaled token、
normalized 7D承诺动作、当前model-ready32Dstate为输入；不积分relativeOSC。
future图像只用于标签与oracle视觉参考；同一当前state/language/noise和原十步Graph生成50×32输出，
比较有效前7维。参考不是专家/最优动作，也不构成成功率上界。

## 新数据与采集结果

|ordinal|task|state|采集结局|measured动作|主请求|对齐样本|
|---:|---:|---:|---|---:|---:|---:|
|0|8|42|success|165|10|7|
|1|9|42|success|131|8|5|
|2|9|43|TimeLimit|280|15|12|
|3|8|43|success|175|10|7|
|4|8|44|success|209|12|9|
|5|9|44|success|132|8|5|
|6|9|45|TimeLimit|280|15|12|
|7|8|45|success|156|9|6|

63例全部delay3，每条5–12例；两个TimeLimit保留，不补样、不按成功筛选。
6/8 success只是identity行为策略的采集结果，不能归给任何预测器。
每例承诺前缀与真正发送的normalized源行、post命令和native动作对齐；future在
最后一个承诺动作返回之后、接管首动作之前，当前索引与prefix起点一致。

## 冻结主要指标

预先指定先episode内平均、再八条等权，避免长失败轨迹凭样本多主导指标。越低越好。

|八episode宏平均|identity|conditioned|no_action|mismatched_action|
|---|---:|---:|---:|---:|
|future token MSE|2008.911386532|1992.141095579|1997.271429135|1994.713130815|
|row0到oracle视觉输出MSE|0.053979883281|0.053084572353|0.056335776540|0.056986795761|
|50行有效7D到oracle视觉输出MSE|0.051731220731|0.042044258153|0.040704378435|0.042975111638|

相对identity，conditioned三项误差分别降低0.834795%、1.658601%、18.725563%。
no_action分别降低0.579416%、**增加4.364391%**、降低21.315643%。
conditioned的首动作宏平均优于identity和no_action，但整块误差仍高于no_action。
按预先要求“两个动作指标均优于两个基线”的共同标准，动作条件增量不通过。

一致性也有限：conditioned相对identity只在3/8条的row0误差更低；相对no_action为6/8。
conditioned的整块误差在8/8条优于identity，但仅4/8条优于no_action。
task8的首动作宏平均比identity差3.071427%，task9好3.669105%，不可只报整体正均值。
完整样本平均、逐episode/task/state结果保存在原始independent_audit.json，不做显著性推断。

## 错配动作前缀诊断

同episode、同delay按request_id循环移位一个前缀，其他predictor输入不变。
63/63前缀实际改变，63/63解码首动作输出改变。
真实动作前缀相对错配前缀，row0误差低6.847592%，整块误差低2.166029%（宏平均）。
因此不能说模型完全忽略动作；但敏感性及均值改善不等于整体优于无动作模型。
错配没有进入Env，这不是反事实物理轨迹或动作条件的因果成功率证明。

## 独立核验与账目

1528个measured动作重新执行原source audit、63次planned接管、8份初态checkpoint及
本轮所有已选前缀与缓存输入逐值核对。8条首例的当前token/state在运行时重编码exact；
63例identity完整50×32输出与新native归档exact，退出后CPU直接回读确认。
两个冻结模型原件/开始/结束权重exact。

从保存预测token和五臂完整输出，以CPU float64独立复算819项数值比较；
对运行器float32归约最大相对差2.587146144e-7，在rtol1e-6/atol1e-7内。
此归约容差不用于输入或identity动作exact门。所有donor身份、同episode/delay、前缀确实变化均核对。
390组外层/离线phase完整；3279 intent/3279 return，error/unknown0；
1608个底层native（80 settling+1528 measured）与journal身份/动作/嵌套区间全部一致。

实际Env8、settling80、measured1528、采集主请求87、native capture16及内部setup16/warmup48/capture16。
离线71个双相机编码批次（63future+8current）、3批冻结predictor前向、315正式decoder、
2个offline capture及内部2/6/2。训练/真机0；核验新增forward/native0。
八条worker join/Graph释放/sampler恢复/metrics关闭/Env关闭均确认。
child2868990与supervisor2868947均exit0并收回，first_failure/stop_reason为空，无强杀。
监督138.296484986秒，独立外层141.776977823秒，UTC03:55:27.432658–03:57:49.209587。
这些是整条实验管线时间，不是predictor单次延迟。固定Python/version/140包前后exact。

准备11项新CPU一次通过；首次格式差异保留，修正后ruff/format/model-entry通过。
独立审计首轮因重建CPU engine遗漏metrics属性而AttributeError，原exit1日志保留；
补上保存的metrics字段后审计exit0，仅修读取器，没有改样本、算法或判据、没有重跑native/模型。

## 当前判断与下一步

轻量视觉残差对整块参考动作距离的改善在新初态上再次出现；动作条件的优势仍依赖指标和episode，
不应删除no_action基线，也不能据此部署或宣称闭环收益。
下一阶段优先在独立训练/验证划分比较纯token目标与首动作相关监督，保持no_action对照；
当前63例已用于报告，不继续用它们选检查点或调损失。新目标需要新的未用于选型的评估数据。
原baseline/realtime/predictor闭环资格仍false，risk_thresholds=null，生产默认及旧confirmation保持。

原件：`outputs/smolvla_frozen_newstate_48e3c62a/`。
准备/核验器：`outputs/smolvla_frozen_newstate_preparation_8db97488/`。
