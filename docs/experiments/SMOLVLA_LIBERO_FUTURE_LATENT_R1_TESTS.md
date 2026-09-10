# F-LAT1-r1：准备实测

2026-09-10，准备原件 `outputs/smolvla_libero_future_latent_r1_preparation_c5950d51/`。
新2项worker_batch测试通过：`2 passed, 11 deselected in 0.37s`，exit0，外层4.218s。
测试确认使用保存的worker uint8和原features、不改变原数组、不重复raw旋转、保留task与robot_type。
旧11项测试及76对CPU对齐结果复用，未重跑；不是本轮又运行13项。
Ruff check/format、固定模型解释器真实入口--help均exit0，入口前后CUDA未初始化。
Python/version/140项packages与原F-LAT1准备exact；没有安装/sync/升级。
本轮唯一GPU/进程/磁盘背景及实际argv/UTC/wall/exit回执分别保存；未选择负载时机或干预其他进程。
到准备结束本轮新模型加载、编码、训练、decoder、Env和native均0。

原F-LAT1已实际编码1批后token exact门失败，child/supervisor均exit2、训练0；不改其报告。
本准备仅验证修改路径可用，数值一致仍待独立登记的r1真实执行，不提前标通过。
