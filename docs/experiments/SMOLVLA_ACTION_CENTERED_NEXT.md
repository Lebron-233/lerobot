# F-ACR1完成后的接续入口

2026-09-15，固定F-ACR1单次实验和独立CPU审计全部完成，开发候选门通过。
execution HEAD：`2926678f0f1c0e62f53f1d4c46fcb71c9dd47cf5`。
唯一输出：`outputs/smolvla_action_centered_2926678f`；两组所选检查点为`centered.pt`及`ordinary.pt`，均第72步。
[报告](SMOLVLA_ACTION_CENTERED_RESULT.md)、[完整机器结果](SMOLVLA_ACTION_CENTERED_RESULT.json)、[所选逐样本指标](SMOLVLA_ACTION_CENTERED_PER_SAMPLE.json)。

下一步提交本轮证据供独立审阅，结合验证收益集中、逐样本恶化、训练4/46退化及token误差变差，决定后续独立资格实验的具体范围。
当前没有新样本、训练、闭环、test或confirmation的冻结合同；本轮原始数据和一次执行机会保持封存。
无待恢复worker，不重跑F-ACR1、F-PFX1或prepare，不利用结果重新选择步数/供体/容差。
原F-PFX1三个pending文档及其他既有未提交文件继续保留，未混入本轮结果提交。
生产默认及闭环资格false，risk_thresholds=null，旧confirmation untouched。
