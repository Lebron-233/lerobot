# E-NAT1：实际准备结果

2026-09-10。原件 `outputs/smolvla_graph_natural_preparation_4ab3a03e/`。

新定向CPU测试：`5 passed in 0.49s`，命令exit0，DevSpace实测外层5.043秒。
覆盖20行固定清单/配对条件、两个调度各自不调用暂停注入、慢策略下原恢复仍有效、
未知请求与缺worker回执不误判整体通过。使用真实CPU worker/queue及FakeGraph/FakeEnv，
不是CUDA或native实测；pytest的DEVICE=cuda标题不改变这个事实。
既有E-RCV3已接纳测试与数组审计不重做，新CPU测试也没有重复运行。

首轮Ruff check通过；format check发现仅一个print表达式需要合并行，exit1原差异保留。
只修订格式，最终Ruff/format均exit0。实际模型环境runpy入口--help exit0，
前后均断言CUDA未初始化，没有加载模型/Env。

固定解释器路径、完整Python版本和140项包metadata与E-RCV3准备快照逐项exact；
使用既有uv offline/no-project/no-python-downloads，无安装/sync/升级。
登记前一次GPU背景：RTX4070TiSUPER，1652/16376MiB，6%；compute-app为awesun445MiB。
磁盘可用1,472,281,968,640字节。未干预其他进程或按负载择时。

CPU原命令、stdout和真实退出来自本轮DevSpace记录；后续检查的argv、UTC起点、
wall与exit独立保存于`*_receipt.json`。`preparation_gates.json`四项均true。
本文件仅记准备，不预填native结果；执行仍需冻结提交和GitHub登记正文exact门。
