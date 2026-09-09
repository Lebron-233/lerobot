# SmolVLA E-RCV2：受控暂停恢复机制对照——Codex 执行计划书

日期：2026-09-09

执行对象：在正常授权的本地开发环境中工作的 Codex

项目：`/home/rp/Workspace/SmolVLA_RTC/lerobot`

目标分支：`codex/smolvla-graph-native-equivalence`

## 0. 任务与完成标准

接续已有 E-RCV2，不另起恢复算法。在确认该队列尚未执行后，完成必要准备、冻结提交、GitHub 预登记及唯一一次四条 native 机制对照，收集并发布实际结果。

**本轮要回答：同样经历一次受控短暂停顿后，原算法是否维持 planned 闭锁，而已有候选是否通过真实恢复探针回到 planned，并在 native 控制中合法消费新块 row0。**

达到四条完整合同、两个原算法闭锁观测、两个候选完整恢复链，才算本轮工程里程碑。若覆盖不足、未恢复或出现技术首错，也必须按预定停止点交付真实结果；不得为获得正结果追加运行。

这是一次“受控主机完成发布暂停”实验，不是自然 GPU 慢请求实验、成功率确认实验或生产默认切换。本文给出后续执行安排，不宣称新的 native 结果已经产生。

## 1. 依据、已知进度与必须先核对的现态

### 1.1 已有结论

依据用户提供的 `SMOLVLA_GRAPH_CAP_RECOVERY_RESULT.md`：

- E-L1 已在 CPU 复现原路径闭锁：慢 planned 时延进入原窗口后 raw 超 cap；planned 被禁止，后续普通 bootstrap 不更新 tracker。
- `same_path_discard_probe_v1` 已有 CPU 恢复正例及持续慢探针负例，生产默认仍为 `disabled`。
- E-RCV1 完成 4/4 条、2/2 对；960 次 measured native、40 次正常 planned 接管，未知调用为 0。四条 raw 始终为 3，恢复探针为 0。
- 所以 E-RCV1 既没有 native 恢复链，也没有候选对原算法的 native 对照。其 async/serialized wall 差不能归因于恢复策略。

对应历史结果评论：
<https://github.com/Lebron-233/lerobot/issues/1#issuecomment-5601832578>

依据 `SMOLVLA_E_RCV2_PROGRESS_AND_HANDOFF.md`，上一开发阶段已写入以下四个文件：

```text
examples/advanced/predictive_async/libero_graph_cap_stress_native.py
tests/test_smolvla_cap_stress.py
docs/experiments/SMOLVLA_GRAPH_CAP_STRESS_MANIFEST.json
docs/experiments/SMOLVLA_GRAPH_CAP_STRESS_PLAN.md
```

历史准备目录：

```text
outputs/smolvla_graph_cap_stress_preparation_5e2679a9/
```

历史记录为 5 项不同 CPU 测试按“先通过 2 项、修正夹具后通过剩余 3 项”完成；模型环境入口 exit 0 且 CUDA 未初始化；最终 lint/format/diff 检查通过。当时尚未提交、登记或启动 native。

**上述“尚未提交、尚未执行”只代表旧交接时刻，不是本计划核实过的当前状态。** 旧 HEAD `5e2679a95adefa0de57e10b05506d20f35c7cac2` 仅作定位，不作为新执行 HEAD。

### 1.2 当前核验边界

本计划编写时，CodexPro Local 的打开操作返回目标项目，但后续 selected-workspace 读取仍落到插件默认目录；使用本轮返回 ID 也得到 Unknown workspace_id。GitHub 读取返回会话级 FORBIDDEN。因此未读到最新源码、NEXT_REVIEW 或评论。

该次打开返回的 Git 状态仅列出三个历史未跟踪文档，没有列出上述四个 E-RCV2 文件。**这不能单独证明它们已经提交、被移除或已经执行。** 执行端必须先核对，避免重复运行。

### 1.3 接续入口

读适用的 `AGENTS.md`，再按已知路径读取当前 E-RCV2 源码、manifest、PLAN、准备回执，以及 `SMOLVLA_ASYNC_NEXT_REVIEW.md` 最新段落。仅核对相关 HEAD、差异、登记和执行状态，不重读全部历史实验。

| 实际状态 | 下一步 |
|---|---|
| 尚未登记、未执行 | 接续现有实现，完成本计划准备与唯一队列 |
| 已登记，且已有证据确认从未启动，独占输出目录也不存在 | 使用原登记身份核对一致性后执行首次队列，不重复登记 |
| 已有 native/model 执行 intent、started/execution/result 回执，或队列已完成 | 不重跑；只读收集该次实际结果、补齐报告和发布 |
| 输出目录已存在，但是否启动不能确定 | 保留原件，报告状态不确定，不删除目录或换新目录重试 |
| 当前代码或登记已改变实验设计 | 明列差异；不把本计划覆盖到既有冻结实验上，不启动第二套条件 |

必要实现确未落地且确认没有已登记/已执行队列时，才按下文补齐。不要因为旧交接写着“未提交”而重做已经完成的工作。

## 2. 工具、授权与工作区约定

每一轮通过 CodexPro Local 开始连续开发时，先调用一次 `open_workspace(root="/home/rp/Workspace/SmolVLA_RTC/lerobot")`；后续优先省略 ID 使用 selected workspace。不跨轮次保存或复用 `ws_xxx`。若需要显式 ID，只能使用本轮实际返回值。

不使用 DevSpace。Codex 在其本来就获授权的原生本地环境中执行时，使用该环境的正常文件、命令和 GitHub 能力；本任务不包含修复 ChatGPT 插件或改变平台权限。

如再次遇到安全检查拒绝或 `FORBIDDEN`，保留原始错误并停止相应受阻操作，不换通道、改写同一请求或转交其他代理绕过拒绝。工作区 ENOENT/Unknown-ID 与权限拒绝分别记录，不归因于模型、CUDA 或数据。

GitHub 发布与回读只使用当前执行环境已获授权的入口。若使用 `gh`，保持单次同步请求，使用实际返回 comment ID 精确回读；不得默认扫描整个 Issue 评论历史。认证信息不写入日志、文档或提交。

## 3. 冻结实验设计

### 3.1 唯一对照变量及四行清单

所有条目使用原 `graph_identity_async`。两臂仅在 `recovery_policy` 上不同：原算法 `disabled` 与已有 `same_path_discard_probe_v1`。不再混入 serialized/async 调度比较。

| ordinal | pair | task / state | Env seed | policy seed | arm | recovery_policy |
|---:|---:|---|---:|---:|---|---|
| 0 | 0 | 0 / 41 | 940041 | 950041 | disabled | disabled |
| 1 | 0 | 0 / 41 | 940041 | 950041 | candidate | same_path_discard_probe_v1 |
| 2 | 1 | 2 / 41 | 940241 | 950241 | candidate | same_path_discard_probe_v1 |
| 3 | 1 | 2 / 41 | 940241 | 950241 | disabled | disabled |

沿用 `libero_object`、task_order_index=0。task0 为 alphabet_soup，task2 为 salad_dressing；完整任务名称从原固定工厂读取，不另造映射。每对两条件使用同一初态和种子，顺序不得根据负载或先前结果调整。

### 3.2 唯一干预：每条首个 planned 暂停 600 ms

在每条 episode **首个实际执行的 planned 请求**中，完成原模型、postprocessor、独立 CPU chunks 与原设备完成屏障后，在原请求总时延采样及结果接纳之前，执行一次 `time.sleep(0.6)`。

要求如下：

1. 只发生于已有 model worker，不暂停 controller，不持有 queue/request 锁休眠；不新增 worker、队列、GPU 服务或捕获。
2. 不作用于 cold/probe/fresh startup、普通 bootstrap 或 recovery probe。不为了触发而增加请求，不随 reset/task 再次注入。
3. 记录请求 ID、kind、epoch、CPU 完成时间、暂停起止和实际 elapsed；保留原 `requested_at`，真实总时延和 wall 包含暂停。
4. 不修改时钟、tracker 成员或原时延公式。实际暂停可能长于 600 ms，按实测保留，不扣除或事后校准。
5. async 下慢 planned 可以按原 any-late 规则整块丢弃。**不要求这个慢块成功接管；要求其真实时延按原规则被接纳并导致超 cap。** 若实际没有形成该链，记录覆盖不足，不强塞慢样本。

干预的含义是控制“CPU 计算完成后到原结果接纳之间”的主机停顿。模型 API 时间不因此成为 GPU 算子延迟；报告必须把两者分开。

### 3.3 候选保持原样

继续使用已经实现的 `same_path_discard_probe_v1`：原 cap_wait 机会、最新反馈、原单 worker 和单在途约束；探针输出永不 install/stage；同 task/reset epoch 且有效完成才接纳一次时延。保留 active/staged/plan/index 语义和 stop/失效处理。

不清历史、不删慢样本、不改变 P90 算法、不提高 cap、不接纳普通 bootstrap、不新增 recovery 策略。探针预算在派发前计数，reset/task 不清预算。

“同路径”指原模型输入和完成计算路径；探针不做 planned 专属 queue plan / committed-prefix 证据拷贝。不得声称两种请求 CPU 成本必然相等。

### 3.4 固定模型与控制条件

| 项目 | 固定值 |
|---|---|
| GPU | NVIDIA GeForce RTX 4070 Ti SUPER |
| policy revision | `6721902bc4d61e50a3bfdb11dfb4cb626f05d102` |
| VLM revision | `7b375e1b73b11138ff12fe22c8f2822d8fe03467` |
| assets revision | `0b3ea86be5fe169d0fd036ae63d1070ec09e90f6` |
| 模型路径与数值 | 原 strict loader、processor、相机/state/action 转换、bf16/fp32、AMP=false |
| chunk / n_action_steps / flow steps | 50 / 1 / 10 |
| 控制 | 20 Hz、queue threshold=30、P90、window=50 |
| 延迟 | margin=1、min=0、cap=8、原 float32 成员/linear 分位数及原整数容差 |
| 队列 | guard=2、max_late_steps=2；任意 late>0 仍整块丢弃，不授权裁剪 |
| 上下文 | identity / identity fallback；compile=false |

相同初态比较仅针对每对推理前的双图、8D state、raw quaternion/EEF/gripper，要求 exact。不同 arm 的请求数、RNG 消耗和轨迹可能分化，不要求逐动作 exact；owner 仍按原逻辑播种一次，不回滚 RNG。

## 4. 预算与停止合同

| 资源 | 每条上限 | 四条总上限 |
|---|---:|---:|
| Env | 1 | 4 |
| settling | 10 | 40 |
| measured native step | 280 | 1120 |
| 主策略请求 | 160 | 640 |
| recovery probe | candidate ≤50；disabled=0 | ≤100，包含于主策略 |
| capture | 2 | 8 |
| setup / warmup / capture 内调用 | 每 capture 为 1 / 3 / 1 | ≤8 / 24 / 8，单列记账 |
| 受控暂停 | 首个 planned 一次 | 最多4次，每次请求600 ms |
| reference / predictor / 训练 / 真机 | 0 | 0 |

每条 startup ≤30 秒；ready 后 1200 slots，即 60 秒；单 model request ≤15 秒；单 native call ≤30 秒。沿用原监督器外层 870 秒请求退出、900 秒硬停止及原单调用超时处理，只处理自建进程组，不干预其他任务。

**attempt=1，retry/resume/replacement=0。** 不追加第三对、第三臂，不改干预时长重试，不重开旧 E 或旧确认队列。

success、TimeLimit、已约定动作/wall 上限、正确 underflow/late 丢弃、探针预算耗尽、未触发或未恢复均可成为有效实验结果。技术异常、初态不一致、非法动作来源、未知调用或清理未确认则停止后续条目，剩余记为 not_run。

underflow 不推进 Env、不补零/保持动作、不追赶旧时隙；未知 native 调用不重发。stop、worker join、Graph 释放、sampler 恢复、metrics 关闭、Env 关闭按原合同确认后才归档大数组。超时、TERM/KILL 或退出未确认不得报告成功。

## 5. 执行步骤

### A. 一次现态核对与最小实现审查

按第1节确定是否仍有首次执行机会。读取上述四个文件及必要的原 `NativeEngine` / `RecoveryEngine` 接口，只判断以下会改变本轮执行的事项：

- 暂停位于真实完成屏障之后、原时延接纳之前，且只作用于首个 planned；不阻塞 controller 的 queue/request 锁。
- 两臂确实共用相同 async 控制和干预，仅恢复开关不同；生产默认、原 controller/queue/estimator 不变。
- 四行固定清单、预算及首错停止在派发前生效；汇总不会把“完成四条”误判成“恢复机制通过”。
- native 恢复证据能关联具体 probe、planned、queue takeover 和 native source row，而不是只看 installed 或 request_count。

已对的直接保留。冻结前发现实质缺口时，只补实验入口、证据提取或对应定向测试；若需要改生产恢复算法或科学条件，停止并指出具体问题，不自行扩展本轮。

不得新建通用 harness、发布框架、校验和体系或第二层监督包装。复用已有入口和回执形式，不把本轮转成插件维修或全仓审计。

### B. 接纳已有测试，补齐真正缺失的准备门

读取历史 preparation 中的实际日志及独立退出回执。重点保留：

```text
cpu_first.log / cpu_first_receipt.json
cpu_remaining.log / cpu_remaining_receipt.json
model_entry.log / model_entry_receipt.json
ruff_final.log / ruff_final_receipt.json       # 历史首个 lint 失败，保留原名
ruff_pass.log / ruff_pass_receipt.json
format_pass.log / format_pass_receipt.json
diff_check.log / diff_check_receipt.json
preparation_gates.json
```

历史 CPU 首错是使用不存在的 `ScheduledActionQueue.next_index`；历史 lint 首错是 fixture 同名 F811。不得删除原件，也不把它们写成候选算法失败。

若代码与已通过回执对应，直接接纳已完成的“2+3 项”，不为获得一条整齐的 `5 passed` 重跑。不要重新运行会重复全部测试的 `finish_preparation.py`。核对 gates 与真实最终回执；不能仅因 JSON 写着 true 就跳过缺失证据，也不能将首错日志当作最终状态。

只有相关代码确实改变、回执缺失，或现有测试没有覆盖本轮可达缺口时，才运行受影响的定向节点。新增检查必须能回答具体问题，例如：async 暂停造成晚完成时是否仍按原规则接纳时延，或“有 probe 但无 native row0”是否被正确判为未恢复。失败阻止冻结；修正仅在冻结前进行，首错保留。

固定解释器：

```text
CPU Python:   /home/rp/miniconda3/envs/smolvla-rtc/bin/python
Model Python: /home/rp/Workspace/SmolVLA_RTC/libero-reference-venv/bin/python
uv:           /home/rp/miniconda3/envs/smolvla-rtc/bin/uv
ruff:         /home/rp/Workspace/SmolVLA_RTC/libero-reference-venv/bin/ruff
```

Python 使用既有 uv 的 `--no-config --no-project --offline --no-python-downloads --python <既有解释器>`，清除 PYTHONPATH。不得安装、sync、升级或替换环境。

补齐真实环境/资源准备：沿用原 Python/version/140项包 metadata 快照进行直接比较，登记前保存一次 GPU/其他进程背景与磁盘可用空间。目的分别是防止环境身份变化、记录运行背景和确认数组归档空间，不用于选择低负载时机；不结束他人任务，不反复探测等待低负载。查询失败原样记录，不伪造快照。

模型环境入口检查只做实际入口 import / `--help`，CUDA 未初始化；不额外加载模型、预热或创建 Env。已有有效且对应当前入口的回执可复用。

### C. 冻结、提交与预登记

将 E-RCV2 入口、测试、固定 PLAN/MANIFEST、实际 TESTS 说明及本任务书纳入执行提交；只提交本轮有关路径，不使用全仓 `git add .`。原三个未跟踪文档保持不动：

```text
docs/experiments/CODEX_D_GRAPH_TOKEN_WORKER_CONTRACT_20260909.md
docs/experiments/CODEX_NEXT_TASK_GRAPH_NATIVE_ASYNC_20260909.md
docs/experiments/SMOLVLA_GRAPH_AB_REVIEW_C_CONTRACT_CORRECTION_20260909.md
```

不提交大数组、模型缓存、密钥或整个 outputs。按原 source 门确认当前完整 HEAD 与已推送的实验分支身份。没有实测前不能生成声称 native 成功的 RESULT。

输出目录固定为：

```text
/home/rp/Workspace/SmolVLA_RTC/lerobot/outputs/smolvla_graph_cap_stress_<execution_HEAD前8位>
```

要求启动前不存在；不删除或复用已有结果目录。

Issue #1 预登记包含：完整 execution HEAD、四行展开清单、唯一600 ms干预及其插入位置、两臂、生产默认不变、精确命令、独占输出路径、准备门实际结果、全部预算/停止规则、正负结果定义、旧资格保持。

发布后使用实际返回 comment ID 单次同步回读，并直接比较正文 exact。回读失败、不一致、发布结果不明或权限拒绝均阻止 native；不得盲目重复 POST。已登记的队列按第1节处理。

精确读取某条评论的命令形式为：

```bash
gh api repos/Lebron-233/lerobot/issues/comments/<实际返回的comment_id>
```

登记后的源码/manifest 不再修改。若冻结后发现实质错误，记录并停止，不边跑边修。

### D. 唯一 native 执行

以下是需在登记时展开为具体 HEAD/路径的命令模板，不是当前已登记的命令。变量在冻结时取值，登记后不重新取另一个 HEAD。

```bash
cd /home/rp/Workspace/SmolVLA_RTC/lerobot
EXECUTION_HEAD="$(git rev-parse HEAD)"
OUTPUT="$PWD/outputs/smolvla_graph_cap_stress_${EXECUTION_HEAD:0:8}"
```

准备门和预登记 exact 回读均通过后，才使用已有独立退出记录方式执行一次：

```bash
env -u PYTHONPATH \
  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  MUJOCO_GL=egl PYOPENGL_PLATFORM=egl \
  LIBERO_CONFIG_PATH=/home/rp/Workspace/SmolVLA_RTC/libero-reference-cache/config \
  LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libGLdispatch.so.0:/usr/lib/x86_64-linux-gnu/libGLX.so.0 \
  /home/rp/miniconda3/envs/smolvla-rtc/bin/uv run \
  --no-config --no-project --offline --no-python-downloads \
  --python /home/rp/Workspace/SmolVLA_RTC/libero-reference-venv/bin/python \
  python -u -X faulthandler \
  examples/advanced/predictive_async/libero_graph_cap_stress_native.py \
  --execution-head "$EXECUTION_HEAD" --output "$OUTPUT"
```

使用入口原监督器，不绕过它直接调用内部 worker。保留真实 child 与 supervisor 退出码和 elapsed；不得把命令已启动、工具返回了 PID 或发生超时解释为完成。

按固定顺序完成四条，或在首个技术失败处停止。每条仍为原 Env 创建/reset/set_init_state、恰好10 settling、cold→probe→fresh；每对第二条在任何推理前核对初态 exact。

### E. 收集、判定、发布

正常清理退出后，仅 CPU 读取本轮小型结果和需要的本轮 arrays，核对请求/动作来源及初态；不重新加载模型或 Env，不读取旧大数组拼接覆盖。

保存退出后模型环境和 GPU 背景快照，与登记前按原项目方式直接比较。环境不一致要明确报告，不用“运行结束”掩盖；GPU快照只作背景，不能证明旧慢 slot 的因果。

按第6节判定字段。RESULT.md / RESULT.json 单独提交推送；Issue #1 发布结果后按实际 ID 单次 exact 回读；再提交独立 RECEIPT，并更新当前 HANDOVER 实际文件与 `SMOLVLA_ASYNC_NEXT_REVIEW.md`。HEAD 分清执行、结果和发布回执，不伪造自身提交号。

如 GitHub 发布受阻，保留本地结果和真实错误，报告“实验已执行、发布未完成”，不重跑实验。

## 6. 证据与验收口径

### 6.1 disabled：原算法闭锁证据

每个 disabled 条目必须同时具备：首个 planned 的暂停实际发生；其完成时延被原 tracker 接纳并使 raw>8；之后至少一个普通 bootstrap installed，按原 `latency_to_steps(total,20)+1` 判定已在 cap 内，但 `latency_tracker_admitted=false`；历史仍不变、仍超 cap，且没有后续 planned。

使用原换算函数判定“快”，不另造近似350 ms阈值替换整数容差。若 episode 提前结束而没有足够 bootstrap 证据，判为未观测，不把逻辑推断当作 native 闭锁实测。

### 6.2 candidate：完整 native 恢复链

同一个 task/reset epoch 内，必须逐段关联：

```text
受控暂停的真实 planned 样本接纳 → raw>8
  → recovery_probe 实际执行、完成且输出丢弃
  → 有效时延仅接纳一次，原 P90 自然回到 raw≤8
  → 在该 probe 完成之后产生新 planned
  → 原 queue 合法 takeover
  → 实际 native.step 发送该 planned 输出块 row0，源审计逐值通过
```

具体保留 paused/probe/planned request ID、epoch、tracker 前后成员/P90/raw、takeover action index、source_request_id/source_row_offset、native 发起/返回与动作逐值匹配。不能把普通 bootstrap 接管、probe dispatched、估计下降或仅 queue_get 当成完整 native 恢复。

### 6.3 新结果字段

| 字段 | 置 true 的条件 |
|---|---|
| `stress_four_episode_contract_passed` | 四条有效结束、两对初态 exact、原源审计/预算成立、清理与退出确认、未知调用0 |
| `stress_disabled_latch_both_observed` | 两个 disabled 均满足6.1 |
| `stress_candidate_recovery_both_observed` | 两个 candidate 均满足6.2，并有有效受控暂停/超cap证据 |
| `stress_mechanism_contrast_passed` | 上述三个字段全部 true |
| `natural_latency_recovery_demonstrated` | 本轮固定 false |
| `production_default_unchanged` | 仅在实际源码差异确认默认及 factory 未改时为 true |

汇总字段必须由实际记录计算，不按“测试通过”硬编码 native 结果。队列完成但机制未覆盖时，四条合同可以 true，机制字段必须 false；技术失败时四条合同 false。

保留逐条结果，至少含 outcome、measured actions/wall、干预实际时长、超cap/恢复时点、probe 数、steady bootstrap、planned 接管、underflow/no-action slots、未知调用和清理状态。按 task 对比 candidate−disabled，明确两臂各自分母；wall/成功率只作描述，不作显著性或泛化收益声明。

主策略调用与 capture 内 setup/warmup 分开列；recovery 是主策略子集。外层 environment_step 与底层 native_step 不能重复累计成两次动力学推进。

### 6.4 历史字段与解释边界

旧 E 的 `native_closed_loop_contract_passed=false`、`paired_scheduling_comparison_complete=false` 保持；D 三个历史 CUDA 通过字段、旧 E 三个 observed 保持原值，不用本轮结果覆盖旧报告。

`baseline_qualified=false`、`realtime_qualified=false`、`predictor_benefit_tested=false`、`risk_thresholds=null`、`old_confirmation=not_started_untouched` 保持。

即使机制对照通过，也只支持两个开发任务在本次固定600 ms主机干预下的恢复机制结论。不宣称自然负载稳定恢复、真机等待安全、硬实时资格、任务成功率提升或未来 predictor 有效。

## 7. 最终交付

仓库内建议沿用这些既有命名：

```text
docs/experiments/SMOLVLA_GRAPH_CAP_STRESS_PLAN.md
docs/experiments/SMOLVLA_GRAPH_CAP_STRESS_MANIFEST.json
docs/experiments/SMOLVLA_GRAPH_CAP_STRESS_TESTS.md
docs/experiments/SMOLVLA_GRAPH_CAP_STRESS_RESULT.md
docs/experiments/SMOLVLA_GRAPH_CAP_STRESS_RESULT.json
docs/experiments/SMOLVLA_GRAPH_CAP_STRESS_RECEIPT.json
docs/experiments/SMOLVLA_ASYNC_NEXT_REVIEW.md
```

存在已发布同名原件时不覆盖其历史身份，按第1节处理，而不是另起目录规避“唯一一次”。原始输出保留 calls、逐episode结果、metrics/history、暂停记录、准备首错、model.log、环境/GPU快照及实际退出回执；大数组只留本地。

面向用户的最后汇报应直接给出：本轮是否实际执行，四条/两对完成情况，两个 disabled 和两个 candidate 的真实证据，机制判定，实际预算和退出码，三个提交身份与结果评论，以及限制或首错。

**在本计划范围内连续完成准备→登记→执行→收集→发布，不逐步等待重新审批；达到里程碑、有效负结果或真实阻断后停止。不得将“持续到重大成果”解释为反复试到成功。**

## 依据材料

1. 用户上传 `SMOLVLA_GRAPH_CAP_RECOVERY_RESULT.md`：E-L1/E-RCV1 已有结果、固定模型/控制、预算和解释边界。
2. `SMOLVLA_E_RCV2_PROGRESS_AND_HANDOFF.md`：E-RCV2 既有四行设计、600 ms 干预、历史本地文件、测试回执及中断位置。
3. Issue #1 评论5601832578：上一轮已读取的对应结果；本计划编写时未能重新获取，须由执行端在授权环境核对。
4. 当前用户约定：新轮次先打开工作区、优先 selected workspace、不跨轮复用 ID、不使用 DevSpace。
