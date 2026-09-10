# F-LAT2 准备实测

2026-09-10，原件 outputs/smolvla_frozen_newstate_preparation_8db97488/。

新CPU定向11项一次通过：11 passed in 0.36s。覆盖新八条身份/旧确认状态排除、
同episode同delay的确定性错配、episode等权汇总、两项动作指标与两个比较对象同时成立的门、
五类派发前预算、未知调用保留，以及按request取前12个完整对齐例。无CUDA或真实Env；
pytest默认打印DEVICE=cuda不是GPU执行证据。

首次ruff通过，首次format --diff报告两文件格式差异，原日志与退出码保留。
按其差异修正后，ruff和format --check均exit0；未改变算法/判据，未重复11项已通过测试。
固定模型解释器运行新入口--help、前后CUDA未初始化，exit0；模型未加载、未编码、未训练。
原Python/version/140包metadata与F-LAT1-r1结束直接比较exact，没有安装/升级/sync。
一次GPU背景快照：RTX4070TiSUPER，1561/16376MiB，4%；compute列表仅记录awesun409MiB。
该快照不用于挑选执行时机，不干预同机进程。完整命令/UTC/退出/实际耗时在独立receipt中。

运行前此处没有native或预测效果结果。原F-LAT1-r1/旧E系列源码与报告保持。
