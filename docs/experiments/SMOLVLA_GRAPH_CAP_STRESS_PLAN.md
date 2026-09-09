# E-RCV2：受控主机暂停下的真实恢复机制对照

2026-09-09。依据附件E-RCV1完整报告及Issue #1评论5601832578继续。
E-RCV1四条合同已通过，但raw始终3、probe为0，不能声称native恢复。
本轮新建独立诊断，不重复旧队列，不改写E-RCV1或旧E任何结果/资格。
最新执行安排见已纳入仓库的[执行计划书](SMOLVLA_E_RCV2_CODEX_EXECUTION_PLAN_20260909.md)。
准备修订尚未用于登记或native执行，原四条设计和600ms干预不变。

## 问题与唯一改动

验证真实CUDA Graph/worker/native controller中，暂时变慢后，disabled是否仍因历史闭锁只bootstrap，
候选是否经有效丢弃探针回cap并恢复合法planned row0接管。
两臂均使用原graph_identity_async；差异只有recovery_policy disabled/既有same_path_discard_probe_v1。
在每条首个planned的原完整CPU chunks和设备完成屏障之后、返回原采样/接纳之前，
独立诊断子类执行一次time.sleep(0.6)，记录实际起止和时长。两臂相同。
这是明确施加的主机完成发布暂停，不是GPU算子时延、自然负载事件或伪造tracker样本。
不暂停startup/bootstrap/recovery probe，不改时钟，不直接写tracker，不添加CUDA同步/event。
模型/RNG照常运行，不回滚、不重播种；该暂停与全部探针成本保留在真实wall和原request总时延中。
生产代码、默认关闭、factory和原入口不变；不提高cap、不清窗口、不接纳普通bootstrap。

## 固定四行与模型身份

| ordinal | task/state | Env seed | policy seed | arm |
|---:|---|---:|---:|---|
| 0 | 0/41 | 940041 | 950041 | disabled |
| 1 | 0/41 | 940041 | 950041 | candidate |
| 2 | 2/41 | 940241 | 950241 | candidate |
| 3 | 2/41 | 940241 | 950241 | disabled |

MANIFEST source_ordinals 1/1/5/5引用冻结原E fixed_manifest的两条async身份；运行结果保存展开清单。
每条新Env/new engine，原reset/set_init_state+10 settling，每对第二臂在推理前双图、8Dstate、
raw quaternion/EEF/gripper exact；先验顺序不因结果修改。两任务是开发案例。
原RTX4070TiSUPER；policy6721902bc4d61e50a3bfdb11dfb4cb626f05d102、
VLM7b375e1b73b11138ff12fe22c8f2822d8fe03467、assets0b3ea86be5fe169d0fd036ae63d1070ec09e90f6。
strict loader，bf16/fp32、AMP=false、50/1/10；20Hz/threshold30/P90/window50/margin1/cap8/guard2；
any-late整块丢弃，identity，compile=false。原cold/probe/fresh，owner只播种一次，最多2 capture。

## 有界执行与首错

总Env4、settling40、measured1120（每条280）、主调用640（每条160）、capture8；
setup/warmup/capture内上限8/24/8。probe仅候选，每条50，总100，含于主调用。
每条ready后1200 slots/60秒；startup30秒、单model15秒、单native30秒；外层870/900秒。
原监督器同等call-intent/return超时处理，仅终止自己创建的进程组；超时非成功。
唯一attempt1/retry0，无resume/replacement、改600ms重试、第三任务或额外native臂。
缺触发/未恢复/TimeLimit为有效负结果，不因此追加样本。
模型、初态、动作来源、未知调用、清理不确认等技术首错停止后续条目，not_run如实保留。
默认不推进无动作Env，不补零/保持动作/赶时隙。原逐chunk/row native source audit复用。
确认worker join/Graph释放/sampler恢复/metrics关闭/Env关闭后才写大数组；不修改既有证据。

## 准备、发布与判读

新增定向CPU测试检测暂停是否仅一次且不污染startup/probe，真实worker的disabled闭锁与候选恢复，
固定两臂、初态配对、缺证据不宣称恢复。复用已通过的whole-discard/预算/失效CPU合同，不重跑全审计。
CPU仅既有smolvla-rtc解释器；模型入口--help用libero-reference-venv，均uv offline/no-project/no-download，
不安装/sync/升级。保留实际测试退出码及开发首错。登记前/退出后各一次GPU及模型包元数据快照。
先提交推送实现/计划/manifest/测试；Issue #1登记完整HEAD、展开精确命令、独占输出和预算。
按返回实际comment ID同步回读一次、正文exact后唯一执行。
入口libero_graph_cap_stress_native.py，公开参数只有--execution-head和--output；
输出outputs/smolvla_graph_cap_stress_<HEAD前8位>必须未存在；内部worker仍由监督器调用。

主要证据：同epoch的超cap→有效probe丢弃接纳→原P90回cap→新planned→原审计通过的native row0。
负对照还要求至少一个快bootstrap installed但不接纳、历史不变、没有后续planned；
“快”严格使用原`latency_to_steps(total_chunk_s,20)+1 <= 8`，保留整数容差，不用近似350ms阈值。
恢复链同时记录暂停请求ID及其历史、同epoch的probe/planned ID、source_request_id/source_row_offset、
takeover index和native起止。原source audit已逐值确认唯一staged请求与发送命令，取证显式保留该来源。
缺少实际native row0，或暂停请求与恢复链不属同epoch，均不报告完整恢复。
两个disabled闭锁和两个candidate恢复均观测且四条完整/两对初态exact才置stress_mechanism_contrast_passed。
缺一则该字段false；正常完成与机制成功是不同字段。
本轮字段为stress命名空间，natural_latency_recovery_demonstrated固定false。
旧E整体资格、baseline_qualified/realtime_qualified/predictor_benefit_tested仍false，risk_thresholds=null，
old_confirmation=not_started_untouched。不做性能显著性、成功率泛化或自然负载因果推断。
完整结果/原始证据保存，结果提交推送，发布实际回执并更新NEXT_REVIEW；本固定队列至此停止。
