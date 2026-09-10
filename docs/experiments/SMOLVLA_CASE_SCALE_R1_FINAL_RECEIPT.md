# F-SCL1-r1最终结果发布回执

本回执对应完成独立CPU核验后的正式RESULT，不替代或覆盖旧STATUS发布回执。

- 冻结执行：`6ddb7505c4d1c5930db365a818c2bc1d2e47b2d4`，原训练没有重跑。
- 正式结果提交：`e0897a1f6a4dd81c71dcf06735f5c4bda8e17623`，已推送。
- 正式评论：5616215138；创建UTC2026-09-10T09:20:01Z，按实际ID一次GET，正文exact。
- 回读UTC2026-09-10T09:20:08.335736+00:00；正文SHA256见配套JSON。
- 审计标签`audit_final_reduction_contract` exit0，2558项数值检查，独立接纳true。
- 公共机器结果与既有审计的35个共有字段、四组均值/逐episode/低误差子群逐项一致。

仅对计算token尺度采用原先规定的rtol1e-6/atol1e-7，其他准备字段仍exact；
原审计失败和线程来源未确认的事实保留，没有重发受阻查询或修改原运行值。
尺度改善true；动作输入增量false；开发候选门false，不能把审计接纳当作科学假设通过。
本次核验/报告/发布新增forward、训练、Env/native、test均0；原实验子进程及监督已exit0。
旧F-ACT1待提交文件保持不动。没有新实验或后台任务，生产默认及闭环资格不变。

原始发布回读和回执：`outputs/smolvla_case_scale_r1_preparation_eae8c863/final_result_*.json`。

接续说明：上一会话追加的两份发布回执JSON一致性比较被工具拒绝，未执行；本次不重试该比较。
本提交仅归档既有发布回执和NEXT索引，不新增核验接纳声明，不重跑训练或独立审计。
