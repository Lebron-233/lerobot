# F-OPT1 实际准备证据

2026-09-10。原件 `outputs/smolvla_optimization_probe_preparation_d4488fe1/`。
9项不同CPU用例首次 `9 passed in 0.36s`，进程exit0；未重新执行旧实验测试。
覆盖固定6-anchor/12-validation隔离、test污染/缺样/重复身份/错误state/delay拒绝、
单因素对照与action-only梯度、预算及相同循环顺序、独立CPU快照不别名。
fake/CPU张量测试，不称为真实VLA训练或native验证。

首个Ruff C420和格式差异保留。改为dict.fromkeys并仅按格式修正后，
最终Ruff/format --check均exit0。模型解释器真实入口--help通过，CUDA未初始化。
原Python/version/140项包metadata与F-ACT1准备逐项一致，没有依赖操作。
背景GPU1592/16376MiB、6%，另awesun438MiB；只记录，不等待特定负载或干预其他进程。
本准备没有模型加载、视觉编码、训练、解码或Env调用。

旧F-ACT1未提交回执和NEXT_REVIEW修改保留原样；本轮仅冻结新增实验文件。
