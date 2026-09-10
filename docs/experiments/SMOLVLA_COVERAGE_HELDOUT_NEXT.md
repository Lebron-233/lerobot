# F-COV2后：独立评估负结果，回到开发，不直接部署

本文件接续F-COV2完整报告，不修改旧F-ACT1待提交NEXT_REVIEW和回执。
正式报告HEAD `4d98ad31e759172023dc6b83b9ead42f52ada432`已推送；
正式评论5615328437已按实际ID单次GET，正文程序化exact。发布证据见本轮RECEIPT。
执行HEAD 4ee28ccf；四episode/16例完成，独立审计接纳。
multi_conditioned首动作比identity差11.465337%，比multi_no_action差2.349464%；
整块比identity好5.148579%，但比multi_no_action差0.407573%。
原验证最强single_no_action被击败不代表主门通过；identity是本次首动作最优。
主门false、所有参照全面胜出false，保留F-COV1验证true的历史含义，不合并分母。

## 数据与执行边界

本轮16例、F-ACT1 test30、F-LAT1 test12、F-LAT2 test63及confirmation21–40均不参与后续选点。
不能按state48/49的测试表现筛选样本、调整残差强度或学习在线fallback。
当前四个72步检查点保持冻结，不重跑F-COV2，不新增测试，不直接在线部署。
本次仅结束CPU报告与发布，新增forward/训练/Env/native为0，无后台任务。

## 下一项开发问题：减少对原本已准确案例的损害

待检验而非已证实的解释：全局尺度的绝对动作损失，可能让误差较大的训练案例主导更新，
在降低部分案例误差时损害identity原本较准确的案例。不能仅凭本次测试就确认这一机制。

建议下一份独立预登记只改变动作损失尺度：全局训练尺度 vs 逐案例identity误差尺度，
每种均保留conditioned/no_action，构成同数据、同预算的2×2开发对照。
仅用既有task0–5的state46/48/49训练与task6–7开发验证；明确验证集已用于开发，不称盲测。
分母下限、采样顺序、更新次数、检查点选择和停止条件在读取新评估结果之前固定；
所有尺度与下限只由训练数据确定，不能由验证或测试误差决定。
固定VLA、7D动作与当前state语义、模型结构、学习率及token项，不同时添加identity保持项或扩数据。

开发报告至少给出：共同训练群体与验证群体的首动作/整块误差、逐episode变化、
identity低误差案例是否被损害、非零检查点选择与同数据no_action差额。
这一方案尚未登记或执行，不能写成损失已修复或已得到新模型。
通过开发检查也不直接获得baseline/realtime/predictor闭环资格；risk_thresholds仍为null。
