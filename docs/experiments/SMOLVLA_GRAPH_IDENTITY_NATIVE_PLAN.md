# E：Graph serialized / identity async 原生闭环协议

2026-09-09。执行授权和固定范围来自
`SmolVLA_D3r1接纳_E阶段原生闭环_Codex接续任务书_20260909.md`。
本协议将该任务书的 E0→E1→E2→E3 纳入仓库；D3-r1 已接纳的 CUDA 合同和旧 A/B/C/D 结果保持历史原件。
执行 HEAD 在实现、定向测试和本协议提交推送后，于 Issue #1 预登记。登记前模型和 native 调用为零。

## 问题与实现边界

比较同一十步 Graph sampler、同一 50 行完整预测块、同一生产队列与 planner 下的两种调度：

| 条件 | 控制线程接受到新请求后的行为 |
|---|---|
| `graph_serialized` | 等待该 request ID 的实际完成通知，再 get 和 native dispatch |
| `graph_identity_async` | 不等待，按正常队列继续取旧 active 的下一行 |

两条件均使用 `SmolVLAGraphIdentityEngine` 的原 `_worker_loop`、单 owner、原 startup 和原队列。
实验 subclass 只增加 owner 的每 episode 一次 seed、只读记录、完成通知和预算。
控制线程不调用 `_run_request`，不持 request/queue lock 等待。
没有替换生产默认入口、SyncEngine、worker-loop、queue 或 planner；不使用 B 的单行 selector，也不执行 D 的事件夹具。

入口为 `examples/advanced/predictive_async/libero_graph_identity_native.py`。
唯一公开运行入口启动一个自有进程组中的 worker；内部 `--worker` 只由监督入口调用。
CLI 没有任务子集、恢复、重试、加样本、延迟扫描或重跑参数。

## 冻结资源

| 项目 | 固定值 |
|---|---|
| 模型 Python | `/home/rp/Workspace/SmolVLA_RTC/libero-reference-venv/bin/python` |
| CPU 测试 Python | `/home/rp/miniconda3/envs/smolvla-rtc/bin/python` |
| uv | `/home/rp/miniconda3/envs/smolvla-rtc/bin/uv`；仅 `--no-config --no-project --offline --no-python-downloads --python <已有环境>` |
| GPU | NVIDIA GeForce RTX 4070 Ti SUPER |
| policy revision | `6721902bc4d61e50a3bfdb11dfb4cb626f05d102` |
| VLM revision | `7b375e1b73b11138ff12fe22c8f2822d8fe03467` |
| assets revision | `0b3ea86be5fe169d0fd036ae63d1070ec09e90f6` |
| dtype / AMP | 原 bf16 / fp32 混合；`use_amp=false` |
| 模型配置 | `chunk_size=50`, `n_action_steps=1`, `num_steps=10` |

直接调用原 `predict_action_chunk` 获得完整 50 行，保留 checkpoint 的 `n_action_steps=1`。
沿用已有 `libero-reference-cache/hub` 的固定快照及严格 `load_runtime`。
不安装、升级、降级或链接依赖，不以 `PYTHONPATH` 混用另一环境。
不改变 attention、精度、compile、量化、RTC、未来 state/context 或 late 裁剪规则。
模型环境 metadata 在前后记录并直接比对；测试环境导入成功不能替代模型环境入口门。

## 固定清单与配对

唯一机器清单为 `SMOLVLA_GRAPH_IDENTITY_NATIVE_MANIFEST.json`，入口与内建固定清单逐字段直接比较。
共 20 个新 native episodes / 10 对：`libero_object`、`task_order_index=0`、task 0–9，全部 state row 41。

| task | environment seed | policy seed | 顺序 |
|---|---:|---:|---|
| 0 | 940041 | 950041 | serialized → async |
| 1 | 940141 | 950141 | async → serialized |
| 2 | 940241 | 950241 | serialized → async |
| 3 | 940341 | 950341 | async → serialized |
| 4 | 940441 | 950441 | serialized → async |
| 5 | 940541 | 950541 | async → serialized |
| 6 | 940641 | 950641 | serialized → async |
| 7 | 940741 | 950741 | async → serialized |
| 8 | 940841 | 950841 | serialized → async |
| 9 | 940941 | 950941 | async → serialized |

每次创建 fresh Env 和 engine，调用原 reset/指定 state 41，原 reset 内含恰好 10 次 settling，无额外 settling。
`_ensure_env` 的环境创建和隐含初始 reset 单列记时；原 reset 内的 seed/reset/set_init_state 和 settling 实际调用均留日志。
两条件 settling 后的原始双相机、EEF quaternion/position、gripper 和转换后的 8D state 必须 exact。
第二条件完成 reset/settling 后、启动模型前比对；首个不一致即停止后续队列，保留证据，不换 seed/state。
开始模型推理后允许两条件产生不同轨迹、请求数和任务结果。

## 共同消费与接管

```text
consume_policy = queue_until_valid_takeover_or_exhaustion
chunk_size = 50
queue_pops_per_native_dispatch = 1
fps = 20
queue_threshold = 30
latency_quantile = 0.9
latency_window = 50
delay_safety_margin_steps = 1
min_prediction_delay = 0
max_prediction_delay = 8
committed_guard_steps = 2
max_late_steps = 2
late_policy = whole_discard_any_late
context_mode = identity
fallback_mode = identity
use_torch_compile = false
```

不规定每块固定消费 25 或 50 行。旧 active 继续逐行消费至合法接管、耗尽或 episode 终止。
early 只 staging，on-time 在 takeover index 从新块第 0 行开始；任何 late > 0 整块丢弃，继续旧块。
耗尽返回 None，不推进 queue index。stale 返回不能清除更新的 plan。
保留原 tracker、请求时间、延迟估计、guard、horizon、CPU epoch 和 index 语义。

## Startup 与 wall slot

每 episode 在 settling 后仅由 owner 调用一次 `torch.manual_seed(policy_seed)`。
fresh engine 正常走 `cold_temporary → probe → fresh`，30 秒内 ready，每个请求至多 15 秒。
startup 不消费临时块、不调用 native step，使用 settling 后同一份真实观测。
正常 startup 的 epoch 0/1 各 capture 一次；每次单列 eager setup 1、side-stream warmup 3、capture 内调用 1。
测量阶段不允许新 capture。不同 episode 不复用 Graph cache。

ready 后记 `t0`，20 Hz 第 k 个 slot 为 `[t0+k*0.05, t0+(k+1)*0.05)`。
每 slot 最多发送一次原生动作；按最新实际返回观测 notify，再按条件等待，再 get 至多一次，再 env.step。
serialized 等待跨 slot 时保留全部跳过 slot，恢复后按实际当前 slot 派发；等待后至首次 get 之间不重复 notify 同帧。
慢 Env、等待或调度跨越 slot 时不补发。日志和处理开销进入实测 wall 时间。
None 不调用 Env、不发送零值或保持动作，index 不变，下一 slot 可继续用未更新观测，其年龄继续增长。
只有一次真实 env.step 返回才能形成下一个 observation index。
CPU 观测拥有独立双图 buffer，Env 后续复用或修改 buffer 不会改写在途输入。

每次 dispatch 保存当前 slot、时间、jitter、观测 index/age 和 queue action index。
slot 结算优先归类 dispatch，其次实际 get=None 的 underflow；其余按实际阻塞区间覆盖区分
serialized_wait / env_busy，无阻塞覆盖的空 slot 归 scheduler_miss。
episode 结束所在部分 slot 仍计入分母；等待或 Env 返回超出 60 秒的实际耗时保留为 overshoot，不再派发动作。
若正常 get 后本地准备恰好跨出最终 wall window，保留该次 queue get 为未派发，不调用 Env；不补取/补发。

CPU post-policy 7D 命令原样进入现有 Panda relative OSC 路径。派发前检查有限值，
不二次归一化、不新增 clipping、gripper 映射或动作复用。原控制器对有限超界值的处理不变。

## 预算、终止与退出

| 项目 | 每 episode 上限 | 本轮总上限 |
|---|---:|---:|
| episodes | 1 | 20 |
| 原 reset settling step | 10 | 200 |
| measured native step | 280 | 5600 |
| ready wall slots | 1200 / 60 秒 | 每 episode 独立 |
| 主模型 API 调用，含 startup / 丢弃 / 已开始的错误调用 | 160 | 3200 |
| eager reference 调用 | 0 | 0 |
| capture | 2 | 40 |
| eager setup / warmup / capture 内调用 | 2 / 6 / 2 | 40 / 120 / 40 |

每次受限真实调用前先守住计数，不把未派发计作已调用。
startup 最多 30 秒，每请求最多 15 秒，每次 native/环境调用最多 30 秒。
监督器从小型 intent/return 日志监视实际调用：超时只停止自有 worker 进程组，先 SIGTERM，
调用超时后 5 秒仍未退出则 SIGKILL；整体运行至 3570 秒 SIGTERM，3600 秒硬停止。
不接触其他项目进程，不改变同机负载；执行前的 GPU 占用留证。

success 优先于同时到达的 TimeLimit；native success、TimeLimit、280 动作或 1200 slot 是正常终止，可继续固定表下一项。
underflow、正确 deadline miss 和两条件 success 不同均是正常诊断结果。
第一个技术错误停止后续 episodes，保留 completed 部分和首错；剩余项为 not_run，不修后重跑。
发出 intent 但无正常 return 的调用明确标记未知/异常，不写成零动作，不重发，也不重启同 episode。
已实际返回但超过超时的动作仍保留真实 return 和已返回计数，同时技术失败。

控制停止后先 stop、使在途结果失效，确认 owner 退出、worker join、Graph 释放、原 sampler 恢复，随后关闭 Env。
accepted 但尚未进入 worker 的最后一个 pending 请求可被原 stop 取消：保留其 plan/观测来源，
单列 `cancelled_before_worker_at_stop`，不冒充模型调用或有 terminal 的已执行请求。
未确认 join 时不并发保存共享大数组，不报告通过。监督器收回子进程退出，外层命令另写独立 receipt。

## 证据与统计

逐 episode 保存原始输入、8D state、双图、独立 CPU policy/post/full chunks、noise 和全部 token 输入，
request/task/reset ID、真实观测 index/time/age、plan snapshot、采样/编码/replay/capture 计数、owner 标识，
发布时完成屏障和真实时间、stage/install/discard 原因、消费 slot/index/chunk ID/row offset、真实命令和 native return。
热路径仅用内存 CPU 数组及小型 flush 日志；大型 `arrays.pt` 在停止控制、join、释放和关闭后保存。
不在每次 native 后插 eager reference，不引入后台图像压缩或 GPU 存档服务。

动作来源通过原 install/stage/takeover 事件和消费 index 重建确定的 chunk ID/row offset；
随后检查该确定行与派发命令相同。两个浮点动作恰好相同不能作为来源身份依据。
报告每块消费行数、planned takeover、丢弃块/未消费行、underflow/其他无动作 slot、观测年龄、peak inflight。
模型 overlap 使用非 startup 的真实 `predict_action_chunk` 开始/返回与实际底层 native.step 开始/返回的交集。
serialized 必须零交集，async 按实际证据报告，不以线程存在代替 overlap。

所有延迟用同一 nearest-rank 经验分位数：有序 n 个值的 Pp 为第 `ceil(p*n)` 个，
输出 n、P50/P95/P99、max；同时保留原始样本。startup、请求、等待、Env、dispatch/观测年龄、
无动作与遗漏 slot、日志、最终序列化和整个 episode wall 分别报告。
每对只作任务结果、成功首步、动作数、wall 完成时间、无动作/最长间隔、重规划和块消费的描述性比较。

## 执行门与交付

E0：已读仓库 AGENTS、历史 D3-r1 结果所需字段、环境 metadata/固定快照浅层存在性、磁盘和 GPU 占用。
DevSpace 既有拦截记录按任务书事实追加并回读；未找到待追加附件，不能声称逐字粘贴了缺失文件。
本会话不重试不可用的 DevSpace 通道，也不把 FORBIDDEN session 边界推断成模型故障。

E1：实际入口 import-only/--help 在模型环境通过；CPU fixture 必须运行真实 engine/queue，
覆盖两条件、多行/row0 接管、late1/2/3/early/on-time/stale、None、无锁等待、慢 Env、终止优先级、
首错/未知返回、buffer 生命周期、身份追溯、预算、固定 manifest、退出与资源未确认；仅跑必要原回归。
ruff/format、命令、退出码与首个开发失败保存在 TESTS 文档和 preparation 原件，不预填运行结果。

E2：实现/测试/协议/manifest 先 commit/push；Issue #1 登记完整 40 位 HEAD、完整展开命令、
独占绝对输出路径、顺序/预算/等待/None/首错合同及旧未跟踪三文档名单。
用发布返回的实际 comment ID 只读回读一次，正文 exact 后才运行一次完整固定队列。
使用原已验证 offline/EGL/GLdispatch/GLX 环境变量；不提前创建正式输出目录。

E3：收回逐 episode 原件、总结果、子/监督退出与独立 receipt，生成
`SMOLVLA_GRAPH_IDENTITY_NATIVE_RESULT.md/.json`，提交推送结果并发布 Issue 评论；
按实际 ID 回读一次后提交发布回执。更新 HANDOVER 和 `SMOLVLA_ASYNC_NEXT_REVIEW.md`。
大数组留在 outputs，不入 Git；旧 A/B/C/D 原始结果不覆盖。

五个新 E 字段运行前均为 false，按实际证据分别判定：

| 字段 | 成立条件 |
|---|---|
| `native_closed_loop_contract_passed` | 20/20 完成，无技术错误/超预算/未知动作，初态 exact，退出和账目完整 |
| `native_multirow_consumption_observed` | 至少一个 async episode 从同一块向 native 连续发送两行以上 |
| `native_replanning_takeover_observed` | async 出现合法计划接管 row0，且后续请求用了新 native 反馈 |
| `native_model_control_overlap_observed` | async 存在实际 model/native 交集且 serialized 全部无交集 |
| `paired_scheduling_comparison_complete` | 10 对初态 exact、共用合同且完整收回结果/实际时隙成本 |

覆盖缺失时相应字段保持 false，不扩样获取覆盖。

## 已知限制

这 10 对是开发工程诊断，不构成显著性检验或新的科学基线资格。
旧 `baseline_qualified=false`、`realtime_qualified=false`、`predictor_benefit_tested=false`、
`risk_thresholds=null`、`old_confirmation=not_started_untouched` 保持；真实机器人调用和训练均为 0。
D3-r1 历史三个 CUDA 合同通过标志保持，不用它们代替 E native 结论。
同机负载和同步记录开销影响实际时延，运行时如实报告。
