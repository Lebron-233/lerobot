# F-COV1实际准备记录

2026-09-10，接续7b9d527d。原件：`outputs/smolvla_coverage_preparation_7b9d527d/`。

新CPU测试一次通过：10 passed in 0.37s。覆盖16条固定新初态、无test任务、训练/验证隔离、
四臂各72步和每task12次曝光、multi每state4次、两动作条件共享顺序、缺覆盖拒绝、
episode-macro、零检查点不算候选收益、预算派发前检查、共同联合损失梯度。
这是CPU合同测试，不冒称native验证。格式修订后未重复已经通过的测试。

首轮Ruff通过；format --diff发现两个文件排版差异，原日志和exit1回执保存。
只按格式差异调整，最终Ruff和format --check均exit0。
固定模型解释器运行真实入口--help通过，CUDA前后未初始化，未加载模型或创建Env。

Python/version/140项metadata与F-OPT1准备快照逐项一致，无安装/sync/升级。
一次GPU准备背景：RTX4070TiSUPER，1397/16376MiB，6%；awesun432MiB。
未干预其他进程、未按负载选时重复采样。精确argv、UTC、退出码及wall见各receipt。

旧F-ACT1三份回执/交接文件及三个旧未跟踪文档保持。只有本轮新文件纳入提交。
本记录仅是运行前准备，不预填模型、native或训练结果。
