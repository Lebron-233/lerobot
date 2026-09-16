# F-ACQ1 开发交付状态

2026-09-16。本轮用户要求分析F-ACR1、开发下一轮脚本，并将执行提示词发到Issue #1给Codex。
来源评论：5683274000；工作起点HEAD：e70f7297a93aa6887556206bf1b7f43aad31d06d。

## 已完成

新增冻结实验计划、监督运行器、独立CPU审计器、完整/逐样本报告生成器、针对性测试及Codex执行文档。
不修改旧实验代码、旧结果、生产默认或旧pending文档。
新实验为“冻结基底36步+支路72步，先验证旧锚点exact，再采集新初态identity轨迹并离线对照”。

固定参考解释器下，`tests/test_libero_action_qualification.py`与
`tests/test_libero_action_centered.py`合计**43项通过，pytest exit0，1.37秒**。
其中31项为新测试，12项为旧F-ACR1回归。覆盖历史两种schema、冲突即停、源哈希/登记回读、
固定权重、所有主门必要条件、错配分母/空集合、episode-macro、样本退化、留一episode、
预算、真实delay/mask、非有限指标、BF16组合回归、独立归约、合成端到端评估及不利结果报告。
Ruff check四份新Python文件通过，exit0；git diff --check通过。
测试里的合成数据不代表真实GPU重放或新初态模型结果；CUDA未初始化断言通过。

## 数据身份修正与未完成事项

未预登记草案的task6/7×state42–45，被历史`tuple`格式记录判定已经使用；
检查在任何新仿真之前停止，没有把旧数据冒充新数据。
一次后续组合只读盘点被平台安全检查拦截；没有重发或换工具绕过同一被拦截操作。
因此新计划提出固定state0–3，但**未使用资格必须由Codex按新合同的正式CPU准备检查确认**。
若冲突或未能读取合法身份记录，应停止，不替换样本、seed或阈值。

本轮没有执行新`--prepare`、预登记或正式worker；没有新增GPU实验、Env、新资格样本、
训练或闭环结果。16个旧锚点的GPU exact重放以及实际原生采集/独立审计尚待正式执行。
CPU测试通过不等于这些运行时门已经通过，不保证新模型科学假设成立。

## 交付入口

- 计划：`SMOLVLA_ACTION_QUALIFICATION_PLAN.md`
- Codex提示词：`CODEX_F_ACQ1_EXECUTION.md`
- 执行：`examples/advanced/predictive_async/libero_action_qualification.py`
- 审计：`examples/advanced/predictive_async/audit_libero_action_qualification.py`
- 报告：`examples/advanced/predictive_async/report_libero_action_qualification.py`

执行CODE_HEAD与评论URL以本次实际Git提交/Issue发布回执为准，不提前伪造。
负结果和技术失败同样要求反馈到原Issue。后续时延/闭环需要另一份合同，不自动启动。
