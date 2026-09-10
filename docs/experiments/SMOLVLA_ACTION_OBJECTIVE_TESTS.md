# F-ACT1 准备结果

2026-09-10。准备目录：`outputs/smolvla_action_objective_preparation_b7f0831f/`。

新增12项CPU测试一次通过，pytest `12 passed in 0.38s`，命令exit0。
覆盖新初态与task split、四组与派发前预算、仅train定标、四组损失数值及梯度、
joint缺失动作loss拒绝、episode宏平均、训练目标收益与动作输入增量分开、
真实轻量预测器零初始化/动作隔离、最先8个完整样本的固定选择。
测试没有真实模型加载、CUDA初始化或Env；可导解码器真实非零梯度由首次joint更新确认。

首次ruff发现E731局部lambda，已等价改为def；首轮格式差异保留。
最终ruff/format均exit0；无改变测试期望、实验样本和损失设计。
模型解释器执行实际新入口--help并确认CUDA未初始化，exit0。
模型Python/version/140项包metadata与F-LAT2准备快照exact，无安装或升级。
一次登记前GPU背景845MiB/16376MiB、0%，compute-app列表为空；只记录背景，不据此选择重试时机。
精确命令、UTC起点、退出码和wall均在各receipt.json；磁盘及GPU记录在resource_before.json。

仅冻结准备门；没有提前运行native、视觉编码、训练或decoder。旧已通过用例未重复运行。
