# F-SCL1-r1准备证据

2026-09-10。新原件目录 `outputs/smolvla_case_scale_r1_preparation_eae8c863/`。

三个delay4实际来源只读CPU核对通过：3/48/6、3/49/6、4/48/6均计划89→93，
真实prefix四动作与标签/原native逐值一致。source_trace.json保存逐例来源与时序；
工具exit0，CUDA未初始化，没有模型/Env/native。

真实标签数据门在准备中实际执行exit0，保持原两文件SHA256不变；
train69×delay3+3×delay4、validation16×delay3，72/16分母未变。
四步指定身份、future/current索引、原[1,8,7]actions、连续mask及zero padding全部通过。

测试实际 `19 passed in 0.43s`：原10项尺度测试＋9项新增混合延迟正负例，
不是19项新增用例；原pytest打印DEVICE=cuda不表示初始化GPU。
新增包括完整保留、错delay/mask/future/padding/NaN/split、额外四步身份、删例负例。
首次Ruff通过；首次format仅新测试文件排版差异，exit1日志保留。
只修排版后Ruff/format最终均exit0，未重复跑已通过测试。

固定模型Python运行原入口--help，前后CUDA未初始化，exit0。
Python/version/140项packages与原F-SCL1准备exact，未安装/升级或变更模型。
一次GPU背景快照6385MiB/16376MiB、42%，存在其他项目4637MiB进程；未干预或按负载择时。
所有CPU/格式/入口/metadata命令与UTC起点、退出码、外层耗时保存在独立receipt。

当前只是CPU准备，不预填训练结果。旧失败077ce452和旧测试/确认集保持。
