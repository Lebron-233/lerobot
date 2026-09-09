# E-RCV2 准备记录（已执行队列的历史准备快照）

当前队列已于执行HEAD `9b7aa8685ae1978c553da6e6d7ac3d4b38e3b9e6` 完成唯一尝试，3条完成、第4条模型超时；
见[实际结果](SMOLVLA_GRAPH_CAP_STRESS_RESULT.md)。下面的“尚未执行”等状态仅指冻结登记之前，不是当前接续入口。

2026-09-09。最新依据为用户附件
`/home/rp/Downloads/SMOLVLA_E_RCV2_CODEX_EXECUTION_PLAN_20260909.md`，已完整读取并纳入仓库。
本次接管准备HEAD `24b687bfa100e06f05fd562f5972a62b98a7df36`，
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

## 本机环境与资源准备

用户明确要求使用当前本地终端后，准备命令实际正常返回，均exit0；本轮未调用DevSpace或CodexPro。
模型解释器、Python版本及140项包metadata与E-RCV1退出快照直接比较exact。
固定policy/VLM/assets/config/EGL路径均存在，磁盘可用1,473,396,756,480字节。
登记前GPU只读快照为2026-09-09T14:08:23.625285+00:00，RTX4070TiSUPER总16376MiB、使用6010MiB、利用率47%。
其他进程包括PID2004/awesun 413MiB和PID2719319另一项目Python 4637MiB；未干预、未等待负载变化或重复采样。
`model_environment_before.json`、`model_environment_vs_e_rcv1.json`和`gpu_before.json`保留实际结果。
既有8项不同CPU证据、新入口与最终lint/format继续复用，源码与测试在本轮资源准备时没有再改。

## 历史工具中断记录

旧交接附件第6节记载，DevSpace工作区`ws_85671cfbfc`的模型版本/包元数据、nvidia-smi、磁盘剩余空间只读准备查询被拦截：

> 因 OpenAI 无法确定请求的安全状态，已拦截此工具调用。

没有返回PID、exit code、快照或具体触发因素。它不是已执行实验的技术失败，也不表示资源检查通过。
该记录对应旧插件会话。此前把它当作当前本机终端仍需额外放行的依据不准确；没有据此虚构维护者解除回执。
当前本机授权操作的实际返回与该历史记录分别保留；本轮没有收到新的工具拒绝。
此前CodexPro的Unknown workspace_id与这次平台拒绝分别保留，不推断共同根因。
上述事件原文保留在用户附件，本次未改写既有Reason_for_interception历史记录。

## 冻结前的固定接续点（已执行）

资源/环境准备已完成；确认最终源码、PLAN、MANIFEST、实际TESTS和新任务书的冻结提交后，指定完整execution HEAD。
以该HEAD展开精确命令和独占输出`outputs/smolvla_graph_cap_stress_<HEAD前8位>`，发布Issue #1预登记并按实际返回ID单次回读exact。
随后仅运行固定四条队列一次：task0 disabled/candidate、task2 candidate/disabled，原state41和seeds保持。
每条首个planned的600ms主机暂停保持，Env4、settling40、measured≤1120、主调用≤640、capture≤8、候选probe总≤100。
startup30秒、单model15秒、单native30秒、外层870/900秒；无retry/resume/replacement或为覆盖追加样本。
按首错停止、确认退出与清理，收集真实RESULT与原件，再分别提交结果和发布回执、更新交接。

四条完成不是机制通过。必须两个disabled均有快bootstrap不更新历史的闭锁，两个candidate均有同epoch的
“超cap→有效probe丢弃接纳→原P90回cap→新planned→native合法row0接管”，加上两对初态exact、清理/退出确认，
才允许`stress_mechanism_contrast_passed=true`。
`natural_latency_recovery_demonstrated`在本轮固定false；旧E整体资格、科学资格和旧confirmation保持。

上述准备快照形成时，没有本轮execution HEAD、登记评论ID、native输出目录、RESULT或实测退出回执。
这份文件保留冻结前准备事实；当前状态以RESULT及NEXT_REVIEW为准，不能再次启动队列。
