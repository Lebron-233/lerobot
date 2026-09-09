# E-RCV2 接续状态：代码与测试已核对，资源准备等待平台解除记录

2026-09-09。最新依据为用户附件
`/home/rp/Downloads/SMOLVLA_E_RCV2_CODEX_EXECUTION_PLAN_20260909.md`，已完整读取并纳入仓库。
本次接管准备HEAD `1242c09f4e496708d7d19b6063c18d0245936e45`，
分支`codex/smolvla-graph-native-equivalence`。

## 本次完成

适用AGENTS.md已读取；本次核对当前HEAD、相关源码/清单及已知E-RCV2输出路径。
仅发现原preparation目录，其中没有登记或执行证据；没有E-RCV2运行目录。当前准备提交不等于已登记execution HEAD。
此前5项CPU测试及首错原件继续接纳，未重跑整套测试。
按新计划第6.1/6.2节修正实验取证：快bootstrap使用原整数换算；恢复链显式关联同epoch暂停请求、
probe/planned、source_request_id/source_row_offset和native起止，缺native row0不能成立。
生产恢复算法、controller/queue、600ms干预及四行条件不变。

新增3项CPU用例，并运行受取证改动影响的2项原回归，实际5 passed in 0.64s、exit0；
当前共有8项不同CPU用例通过。新入口import/--help、Ruff/format再次通过，CUDA未初始化。
CPU实测暂停planned总时延625ms、late1整块丢弃，但原tracker仍接纳并达到raw13；
4个probe后，原源审计确认planned request8在index31、第32次FakeEnv step发送row0。
缺native记录或暂停epoch不同的负例不能报告恢复。实际命令、退出和证据见[TESTS](SMOLVLA_GRAPH_CAP_STRESS_TESTS.md)。
这些是CPU受控结果；本次没有新CUDA推理、真实Env或native动作。

代码满足预定干预位置：两臂均为原graph_identity_async，首个planned在原独立CPU输出及设备完成屏障后暂停600ms，
然后按原计时和接纳路径继续。startup/bootstrap/probe不受暂停；生产默认、cap8、P90/window50和bootstrap排除保持。
没有改动候选、冻结参数或干预时长。
原CPU AttributeError、Ruff F811和格式差异原件完整保留，未改名冒充通过。

## 未满足的接续条件

旧交接附件第6节记载，DevSpace工作区`ws_85671cfbfc`的模型版本/包元数据、nvidia-smi、磁盘剩余空间只读准备查询被拦截：

> 因 OpenAI 无法确定请求的安全状态，已拦截此工具调用。

没有返回PID、exit code、快照或具体触发因素。它不是已执行实验的技术失败，也不表示资源检查通过。
此前已请求维护者解除记录或确认，目前尚未取得；没有改写、分拆或通过其他通道重试该查询。
最新执行计划第2节仍要求保留权限拒绝并停止相应操作，不换通道绕过。本轮使用正常本机终端完成独立的代码/CPU工作，未调用插件。
此前CodexPro的Unknown workspace_id与这次平台拒绝分别保留，不推断共同根因。
上述事件原文保留在用户附件，本次未改写既有Reason_for_interception历史记录。

## 待条件满足后的固定接续点

完成原计划的资源/环境准备，保存实际快照；随后确认最终源码、PLAN、MANIFEST、实际TESTS和新任务书的冻结提交，指定完整execution HEAD。
以该HEAD展开精确命令和独占输出`outputs/smolvla_graph_cap_stress_<HEAD前8位>`，发布Issue #1预登记并按实际返回ID单次回读exact。
随后仅运行固定四条队列一次：task0 disabled/candidate、task2 candidate/disabled，原state41和seeds保持。
每条首个planned的600ms主机暂停保持，Env4、settling40、measured≤1120、主调用≤640、capture≤8、候选probe总≤100。
startup30秒、单model15秒、单native30秒、外层870/900秒；无retry/resume/replacement或为覆盖追加样本。
按首错停止、确认退出与清理，收集真实RESULT与原件，再分别提交结果和发布回执、更新交接。

四条完成不是机制通过。必须两个disabled均有快bootstrap不更新历史的闭锁，两个candidate均有同epoch的
“超cap→有效probe丢弃接纳→原P90回cap→新planned→native合法row0接管”，加上两对初态exact、清理/退出确认，
才允许`stress_mechanism_contrast_passed=true`。
`natural_latency_recovery_demonstrated`在本轮固定false；旧E整体资格、科学资格和旧confirmation保持。

目前没有本轮execution HEAD、登记评论ID、native输出目录、RESULT或实测退出回执。
这份文件记录准备进展与未满足条件，不是新的native结果。
