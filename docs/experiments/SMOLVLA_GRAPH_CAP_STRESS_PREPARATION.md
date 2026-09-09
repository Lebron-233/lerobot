# E-RCV2 接续状态：代码与测试已核对，资源准备等待平台解除记录

2026-09-09。依据用户附件
`/home/rp/Downloads/SMOLVLA_E_RCV2_PROGRESS_AND_HANDOFF.md`完整读取并接续。
接管时HEAD `5e2679a95adefa0de57e10b05506d20f35c7cac2`，
分支`codex/smolvla-graph-native-equivalence`。

## 本次完成

读取仓库AGENTS.md、当前工作区状态、新入口和测试完整源码、固定PLAN/MANIFEST、原始测试日志和退出回执，
以及与暂停位置有关的原worker/Graph完成和采样路径。
已有四个新增文件保留，生产源文件和旧实验报告没有修改。
本次准备代码与记录作为独立准备提交保存，其提交身份见本地HANDOVER；尚未被指定为登记后的execution HEAD。
5项不同CPU测试、模型环境入口、最终Ruff/格式的实际结果与附件一致，正式[TESTS](SMOLVLA_GRAPH_CAP_STRESS_TESTS.md)已补齐。
本次没有重跑已有测试，也没有新CUDA推理、Env或native动作。

代码满足预定干预位置：两臂均为原graph_identity_async，首个planned在原独立CPU输出及设备完成屏障后暂停600ms，
然后按原计时和接纳路径继续。startup/bootstrap/probe不受暂停；生产默认、cap8、P90/window50和bootstrap排除保持。
源审阅未发现需改动候选、冻结参数或干预时长的问题。
原CPU AttributeError、Ruff F811和格式差异原件完整保留，未改名冒充通过。

## 未满足的接续条件

用户附件第7节原文：

> 仅在平台/工具维护者已处理拦截、执行环境明确获准后接续；不以换工具或改写请求绕过本次拒绝。

附件第6节记载，DevSpace工作区`ws_85671cfbfc`的模型版本/包元数据、nvidia-smi、磁盘剩余空间只读准备查询被拦截：

> 因 OpenAI 无法确定请求的安全状态，已拦截此工具调用。

没有返回PID、exit code、快照或具体触发因素。它不是已执行实验的技术失败，也不表示资源检查通过。
本次已向用户请求维护者解除记录或确认，目前尚未取得；没有改写、分拆或通过其他通道重试该查询。
此前CodexPro的Unknown workspace_id与这次平台拒绝分别保留，不推断共同根因。
上述事件原文保留在用户附件，本次未改写既有Reason_for_interception历史记录。

## 待条件满足后的固定接续点

完成原计划的资源/环境准备，保存实际快照；随后提交推送实现、PLAN、MANIFEST及实际TESTS，取得完整execution HEAD。
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
