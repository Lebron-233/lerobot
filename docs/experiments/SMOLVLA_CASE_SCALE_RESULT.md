# F-SCL1：启动数据门失败，模型与训练尚未开始

2026-09-10。执行HEAD `077ce452d1cc0c2b17fd1266f0b7aa2b16f8c3d9`。
预登记评论5615533186按实际ID单次GET正文exact后，仅执行一次。
本文件是技术停止结果，不是动作尺度方法的正/负效果评估。

## 实际停止点

`select_data(old, new)` 抛出 `ValueError: Development split or native delay changed`。
发生在CPU开发标签加载后、VLA加载前。运行器status=technical_failure，counts为空，
development_candidate_gate_passed=false；该false不能解读为尺度假设被实验否定，因为尚未训练或评估。
模型加载、视觉编码、predictor forward、解码、反传、优化更新、Env/native/真机均0。
child2933692与supervisor2933649均exit2且已收回；没有timeout/强杀。
外层UTC2026-09-10T08:26:24.484003至08:26:32.096740，7.612760213秒，attempt1/retry0。
原结果和日志保存在 `outputs/smolvla_case_scale_077ce452/`，不改写/续跑该独占目录。

## 已确认的源标签事实及报告勘误

随后一次只读CPU查询原 `outputs/smolvla_coverage_6249b03d/new_labels.pt`，实际exit0，CUDA未初始化：

| split | delay | 数量 |
|---|---:|---:|
| train | 3 | 45 |
| train | 4 | 3 |
| validation | 3 | 16 |

64条标签的split均符合task0–5训练/task6–7验证。三个非3的标签身份为：

| task | initial_state_id | request_id | 保存的delay |
|---:|---:|---:|---:|
| 3 | 48 | 6 | 4 |
| 3 | 49 | 6 | 4 |
| 4 | 48 | 6 | 4 |

因此，本轮把所有新开发样本限定为delay3的准备假设不正确；F-COV1已发布文案中“新64例真实delay均3”也应勘误。
这里证明的是保存标签中的真实字段分布，不是确认底层native计划/已执行前缀的全部来源关系。
没有把4改成3、删掉三例、改mask、重新训练旧模型或重新计算旧测试指标。
不能仅凭此文案错误推翻或宣称重新接纳F-COV1的全部数值，旧数值与原件保持。

## 当前核验断点

继续只读查询F-COV1的episode003/012/004原result.json中request6的plan和观测索引时，
DevSpace.exec_command返回：`因 OpenAI 无法确定请求的安全状态，已拦截此工具调用。`
没有查询结果、PID或exit code；无法确认该请求执行，具体拦截原因未知。
没有重试/换工具读这三个原生请求，未放宽数据门启动修正版。此前完成的标签查询不受影响。

## 准备与下一步

10项新CPU用例、Ruff、format、固定模型入口通过，140包环境metadata相同；
但CPU准备没有加载真实标签验证delay分布，因而没能提前发现这一假设错误。
下一次应先在CPU准备中核对真实身份/前缀/mask/delay，再独立登记保留全部原数据的F-SCL1-r1。
修正版尚未实现、登记或执行。它不是本轮失败队列的resume/retry。
本轮没有消耗test或产生方法效果指标，不能提供首动作改善百分比。
旧F-ACT1待提交文档保持。生产默认/所有闭环资格不变，risk_thresholds=null，confirmation untouched。
