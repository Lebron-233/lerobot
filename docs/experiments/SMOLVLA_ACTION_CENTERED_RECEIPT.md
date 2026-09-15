# F-ACR1发布与退出回执

2026-09-15，execution HEAD `2926678f0f1c0e62f53f1d4c46fcb71c9dd47cf5`；result HEAD `4f082062c359b08a8e76d6909d3ad518490feb66`，结果提交已推送。
[正式结果评论5683274000](https://github.com/Lebron-233/lerobot/issues/1#issuecomment-5683274000)已发布，POST exit0；按实际返回ID单次GET exit0，body与本地publication.md逐字相等。
[预登记5683122737](https://github.com/Lebron-233/lerobot/issues/1#issuecomment-5683122737)亦为POST/单次GET exit0、body exact，正式启动前门已通过。

唯一运行目录：`outputs/smolvla_action_centered_2926678f`。
worker292375及监督入口292328均exit0；独立CPU审计exit0，contract accepted与development candidate gate均true。
监督wall196.384365488秒，外层200.713109050秒；audit外层2.699466258秒。
first_failure/stop_reason=null，active_at_exit空，无强制终止，Graph已释放，VLA保持冻结。
独立核验762条保存预测及2448项数值比较；真实decoder850、支路前向1143、更新/反传144，attempt1/retry0。

原始POST/GET、正文及机器回执留在上述目录；预登记回执与监督外层exit留在`outputs/smolvla_action_centered_preparation_11c81ae1`。
此发布步骤新增模型前向、Env/native与test读取均0。完整[结果和局部退化](SMOLVLA_ACTION_CENTERED_RESULT.md)及[522条所选指标](SMOLVLA_ACTION_CENTERED_PER_SAMPLE.json)已归档。
原九个dirty文档保留，未暂存到本轮提交。实验源码未修改，生产默认及闭环资格false、risk_thresholds=null、旧confirmation untouched。
