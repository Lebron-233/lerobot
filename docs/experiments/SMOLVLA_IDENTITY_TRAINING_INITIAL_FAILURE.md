# F-ITC1 原尝试：初始离线指标设备不一致，训练未开始

2026-09-17。execution HEAD 266b14666864dbabb590b1c2ce4468810b3140c4；预登记5706253063。
唯一输出outputs/smolvla_identity_training_266b1466永久保留，attempt1/retry0，不重启/清空。
真实首错：libero_identity_training.py:276调用p.metrics时visual在cuda:0而future/mask在cpu，
q.metrics的token误差相减触发RuntimeError: Expected all tensors to be on the same device。
这是本次新增代码遗漏的离线设备转换，不是安全拦截、模型阴性或训练优化失败。

worker618248 exit2，监督入口exit2，exit_confirmed=true；UTC23:51:53.218229至23:52:04.644740，
wall11.426565829984611秒，first_failure完整trace保留；无pending/active/强制终止，Graph释放。
实际VLA加载1、预测器加载2、predictor前向2、decoder3、capture1。
已累计identity_exact1、old_centered_exact1。第三个冻结输出比较位于异常前，但计数和
control文件尚未写出，不能报告完整锚点阶段通过。训练/反传/新Env/资格集读取0。
独立CPU审计执行一次exit2（Run not completed），REPORT.md/REPORT.json及审计原样保留。
三个结论属于技术未完成，开发候选未评估。result.json SHA256：
df44d1c4dfd796892caa83229fca2f4484fb4e37ee649787057073585f50d4be。

用户要求持续推进的范围内，只做一次公开记录的技术恢复：新HEAD、新输出、新预登记，
仍原同起点/数据/目标/预算；原失败保留并计入ITC系列总尝试，不伪装为原worker恢复。
最小修复把初始重放和最终评估的离线指标输入统一到CPU，训练可微指标仍原CUDA路径。
正式重启前需CPU回归和不加载模型的合成GPU设备检查；不以修改loss、gate或数据解决技术错误。
此文件不是GitHub科学结果评论，旧未提交文档不纳入修复。
