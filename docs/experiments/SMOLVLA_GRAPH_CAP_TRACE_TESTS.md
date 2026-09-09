# E-RCV3-trace：实际准备证据

2026-09-09 UTC。原件：`outputs/smolvla_graph_cap_trace_preparation_4d62316f/`。

新增7项CPU测试：固定四行不变；disabled/candidate真实CPU worker各1项；
失败phase先落盘且原异常传播；error/unknown/重复terminal区分；
缺worker回执的超时不丢首错、不写零budget；真实reset初态在首次模型前落盘且不重复reset。
另复查原E-RCV2的8项CPU测试，包括实际异步暂停晚完成、同epoch/native row0来源与缺证据负例。

首次命令实际 `15 passed in 0.75s`，exit0。最终准备命令实际
`15 passed in 0.83s`，exit0，独立外层5.521383秒；不是30项不同测试。
FakeGraph/FakeEnv/CPU设备和受控时钟被明确使用，pytest打印DEVICE=cuda不构成GPU实验。
候选继续由真实worker/queue保留慢样本、丢弃probe输出、回cap后从planned row0接管；
disabled快bootstrap仍不入tracker。新增journal不增加时延样本或策略调用。

首轮Ruff有1项SIM117（测试嵌套with），格式检查有2个文件差异，原日志保留。
只合并with并按格式差异修订；最终Ruff exit0，format --check为2 files already formatted、exit0。
没有掩盖或覆盖这些准备阶段失败。

固定模型解释器通过真实入口runpy `--help`，前后CUDA均未初始化，exit0，外层4.177150秒。
公开参数仅 `--execution-head` 和 `--output`；没有模型加载或Env。
使用原uv及原Python，offline/no-project/no-download；没有安装、sync、升级。

模型Python路径、完整版本和140项包metadata与E-RCV2前快照直接比较全部exact。
登记前背景快照UTC15:03:30：RTX4070TiSUPER，5958MiB/16376MiB，利用率20%；
同卡其他项目4484MiB，未干预。磁盘可用1,472,840,339,456字节。
另有本轮最初只读可达性检查（4585MiB/13%），不据此筛选时机或因果归因。
全部精确argv、UTC起点、退出码及外层耗时保存在各独立receipt。

此处仅记录冻结前准备；native结果不预填。旧E-RCV2源文件、结果、manifest和旧队列未修改。
