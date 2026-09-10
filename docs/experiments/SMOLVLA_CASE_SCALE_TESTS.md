# F-SCL1实际准备

2026-09-10，原件 `outputs/smolvla_case_scale_preparation_bb2931dd/`。
新增10项不同CPU用例一次运行：10 passed in 0.38s，exit0。覆盖零误差分母下限、
权重均值1、非法误差、只改变动作项梯度、拒绝验证数据定标、72例各一次的task/初态曝光、
候选门不等于击败最强消融、空低误差子群为null、解码预算派发前拒绝。
pytest打印DEVICE=cuda，但这些测试仅构造CPU张量，没有加载VLA/Env或初始化CUDA。

Ruff首次exit0；format首次有两个文件差异，原diff保留。仅按差异排版；
最终Ruff/format均exit0（2 files already formatted），没有重复计作新的CPU用例。
固定模型解释器入口runpy --help正常退出，前后CUDA未初始化。Python/version/140包metadata与原快照exact。
两个明确开发标签文件SHA256已保存，未读取旧test或加载预测器权重。
登记前GPU背景一次：RTX4070TiSUPER 6274/16376MiB、37%；其他计算进程4637MiB，未干预或挑选空闲时机。
此为准备证据，不预填实验结果、复现exact次数或模型收益。
