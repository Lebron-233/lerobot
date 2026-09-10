# F-COV2实际准备

2026-09-10；原件 outputs/smolvla_coverage_heldout_preparation_23d3932a/。
11项新CPU测试首次通过（11 passed in 0.41s，exit0）：
固定新身份与四臂、主门与最强消融区分、部分/identity平局不得通过、整块退化、episode宏平均、
四种错误checkpoint身份、派发前预算、复用原first-four对齐选择。
这些是CPU合同测试，不冒充真实GPU模型或native实验。

首次Ruff C420（测试字典构建）与格式差异已记录；使用等价dict.fromkeys及格式修改，
最终Ruff与format --check均exit0。没有修改数值期望，没有重跑已通过11项。
固定模型Python实际入口--help通过，CUDA未初始化；未加载模型或创建Env。
Python路径/版本/140项metadata与F-COV1准备逐项一致，无安装升级。

只读outputs/smolvla*/episode_*/started.json中的spec身份，成功索引76项，
task8/9×state48/49命中0；未读取旧测试数组或指标。该范围不代表本机之外的历史。
一次GPU背景为1729/16376MiB、5%；其他任务只记录、不干预、不等待低负载择时。
精确命令、UTC与exit保存在各独立receipt。准备阶段新模型/native/训练调用均0。
