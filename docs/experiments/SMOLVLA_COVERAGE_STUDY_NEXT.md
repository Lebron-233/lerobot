# F-COV1之后：冻结候选与最强消融，进入独立评估而非继续验证调参

本文件为F-COV1最新接续点，旧F-ACT1待提交NEXT_REVIEW/回执和各轮原件保持。
执行6249b03d，16条新开发轨迹和四组72步训练完成，独立CPU审计通过。
验证选定四组均72步且非零。multi_conditioned对identity row0/chunk降低7.149832%/7.714058%，
对single_conditioned和multi_no_action的预设比较通过，validation_candidate_gate_passed=true。
但single_no_action的row0仍比multi_conditioned低0.135519%；后者不是全表最好。
task6/state48仍比identity差30.913476%，其余3条变好，不能宣称稳定部署收益。

## 下一项问题

保持本轮四个检查点权重与72步选点冻结，不根据后续评估继续训练或调权重。
独立登记新的评估身份、数量与预算后，比较identity和四个固定预测器；single_no_action不得删除。
优先只做独立离线泛化验证；任务8/9的state48/49本轮未读取，可作为待登记的候选评估范围，
但应先确认具体(task,state,seed)未用，不将旧state41–47的已报告样本重新称为盲测。
任务名称此前用于过项目开发，因此即使新初态未见，也不称为完全新任务benchmark。

主要报告multi_conditioned对identity及同数据multi_no_action，并完整报告对最强single_no_action的比较；
row0与chunk分别报告、逐episode列示，不以综合分数遮蔽冲突。
验证16例已经参与检查点选择，不改标test。独立新评估数据只读一次，不按结果追加样本或改判据。
若未复现收益，保存负结果并回到开发；若复现，再单独处理原生7D在线入口与延迟/风险边界。

旧test30/12/63、confirmation21–40不作为新方案选点数据。本轮没有启动独立test或在线控制。
生产默认及baseline/realtime/predictor闭环资格false，risk_thresholds=null保持。
