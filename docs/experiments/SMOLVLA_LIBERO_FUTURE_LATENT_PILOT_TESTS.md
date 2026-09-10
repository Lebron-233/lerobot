# F-LAT1：实际准备证据

2026-09-10；准备目录 `outputs/smolvla_libero_future_latent_preparation_195ff5fa/`。

新增CPU定向测试11项：固定任务划分、正常prefix到future映射、index/policy/post/mask/time五种真实边界错误、
末尾未完整前缀显式排除、no_action只屏蔽动作不读取目标、原小模型零残差初始化/risk关闭、mask下逐样本MSE。
实际pytest：**11 passed in 0.41s，exit0**，DevSpace外层4.584s。CPU设备、无Env、无VLA加载；pytest横幅DEVICE=cuda不代表GPU运行。

一次CPU读取固定E-NAT1的10条async原件，新的target-alignment核对通过。
76对全部delay3；train51/validation13/test12，对应固定task0–5/6–7/8–9。
唯一排除为task1/request8，episode结束前没有完整未来目标；不按success或误差选择。
每行normalized prefix等于原动作来源、post prefix等于实际发送/native命令、未来观测位于前缀返回之后且首个接管动作之前。
准备缓存140,599,471字节，CPU处理0.795846s，CUDA未初始化。正式运行复用缓存，不再次扫描旧全轨迹。

首次Ruff check通过；format --diff指出两个文件格式差异，原diff及exit1保留。仅应用格式修改，最终Ruff/format均exit0。
固定模型解释器真实入口runpy --help通过，前后CUDA未初始化，无模型加载。11项已通过CPU测试未重跑。
固定Python/version/140项包metadata与E-NAT1准备快照直接比较exact，没有安装/sync/升级。
一次GPU背景：RTX4070TiSUPER，1468/16376MiB、6%；awesun406MiB。磁盘可用1,469,143,728,128字节。
所有准备命令、退出码和UTC/wall回执保留；不干预其他进程、不根据负载择时。

准备阶段训练/编码/decoder/native调用均0。正式两臂试验须在冻结HEAD及GitHub预登记exact回读之后唯一启动。
