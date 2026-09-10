# E-NAT1：无注入延迟的原生调度对照

2026-09-10。接续正式结果评论5611585394和回执HEAD `4ab3a03e`。
用户本轮明确要求继续实验。E-RCV3四条受控600ms暂停机制对照已经独立接纳，不重复核验或运行。

## 问题与唯一比较因素

在不注入暂停、不改变同机负载的条件下，完成新的10任务/20条原生闭环开发对照。
两个条件都是原Graph sampler、single-owner worker、生产queue/planner和identity context，
均启用已经接纳的 `same_path_discard_probe_v1`；只改变控制端是否等待刚接受请求完成。
serialized等待，包括recovery probe；async继续消费旧active。生产默认仍disabled。
这不是候选恢复机制对原算法的收益对照，也不是predictor实验或旧E剩余项补跑。

## 固定清单

libero_object、task_order_index=0、initial_state_id=41。每行独立新Env/engine。
任务全名、revision和全部控制值由冻结的 `e.fixed_manifest()` 返回，新增字段只启用同一recovery。

| ordinal | task | condition | Env seed | policy seed |
|---:|---:|---|---:|---:|
|0|0|graph_serialized|940041|950041|
|1|0|graph_identity_async|940041|950041|
|2|1|graph_identity_async|940141|950141|
|3|1|graph_serialized|940141|950141|
|4|2|graph_serialized|940241|950241|
|5|2|graph_identity_async|940241|950241|
|6|3|graph_identity_async|940341|950341|
|7|3|graph_serialized|940341|950341|
|8|4|graph_serialized|940441|950441|
|9|4|graph_identity_async|940441|950441|
|10|5|graph_identity_async|940541|950541|
|11|5|graph_serialized|940541|950541|
|12|6|graph_serialized|940641|950641|
|13|6|graph_identity_async|940641|950641|
|14|7|graph_identity_async|940741|950741|
|15|7|graph_serialized|940741|950741|
|16|8|graph_serialized|940841|950841|
|17|8|graph_identity_async|940841|950841|
|18|9|graph_identity_async|940941|950941|
|19|9|graph_serialized|940941|950941|

RTX4070TiSUPER，原固定模型Python及offline uv/EGL/strict loader。
policy `6721902bc4d61e50a3bfdb11dfb4cb626f05d102`，VLM `7b375e1b73b11138ff12fe22c8f2822d8fe03467`，
assets `0b3ea86be5fe169d0fd036ae63d1070ec09e90f6`。
50/1/10、bf16/fp32、AMP=false；20Hz、threshold30、原float32/linear P90/window50、
margin1/min0/cap8/guard2、max_late2、any-late whole-discard、identity/fallback、compile=false全部保持。
startup cold/probe/fresh、owner一次seed、最多2 capture和原cap8 gate不改，不增加warmup。
原工厂/seed/reset/set_init_state/10 settling保持；第二条件推理前检查双图/8Dstate/quaternion/EEF/gripper exact。

## 预算和停止

Env≤20；settling≤10/条、≤200总；measured≤280/条、≤5600总；主策略≤160/条、≤3200总。
probe≤50/条、≤1000总，包含在主策略预算；capture≤2/条、≤40总，setup/warmup/内部capture≤40/120/40。
ready后1200个50ms wall slots/60秒；startup30秒，model15秒，native30秒，外层3570秒TERM/3600秒KILL。
超时仅管理自建worker进程组，单调用超时TERM后5秒仍未退出才KILL。
attempt1/retry0；首个技术错误、startup拒绝、未知返回、初态差异、非法动作来源或清理失败即停止后续。
正常success/TimeLimit/wall或动作上限、未触发probe、未恢复和正确late丢弃均是有效结果，不追加样本。
None不推进Env、不发零/保持动作、不推进queue索引；不暂停wall、不补发旧slot。
不改变其他进程、不按GPU负载择时或重试。训练/predictor/reference/真实机器人调用均0。

## 实现、准备、执行

新增独立入口 `libero_graph_natural_native.py`，复用原episode/controller/budget/source audit。
继承已验收CPU phase/terminal/cleanup journal及首次模型前checkpoint，显式跳过stress pause分支。
无新增CUDA调用或计时同步。保存日志的wall成本保留，不把与旧E-RCV3的差异解释为性能收益。
只测试新清单、两种调度无注入路径、慢策略下原恢复仍有效、未知请求和缺worker回执不误判通过。
原通过测试不全量重跑；固定模型解释器仅import/--help且CUDA未初始化，无依赖修改。
准备日志/metadata/一次GPU与磁盘快照保存到 `outputs/smolvla_graph_natural_preparation_4ab3a03e/`。
源码/计划/实际测试报告提交推送后，Issue #1发布完整HEAD和唯一output/精确命令，实际ID一次GET exact后运行。
唯一输出 `outputs/smolvla_graph_natural_<execution_head前8位>/`；拒绝已存在输出，不能resume。

## 预先固定的结果解释

仅使用新 `nat_*` 字段；旧E整体false、E-RCV2整体false、E-RCV3受控机制true保持。
20条全部完成、10对初态exact、原source audit/预算/清理/exit通过且error/unknown0，才接纳新整体合同和完整配对。
完整对比较success、实测wall、动作数、无动作slots、模型/native真实重叠、接管次数与时延nearest-rank分位数。
部分队列只比较完整配对，保留技术失败与not_run；不把不完整分母伪装成20条。
若自然观测到超cap→有效discard probe→原P90回cap→同epoch新planned→native row0，报告恢复observed；
无该链则false，不因此注入延迟或加样本，不推导持续恢复概率。
退出后仅CPU回读本轮checkpoint/arrays/journal，逐值复核动作来源和恢复row0；清理前不归档大数组。
baseline_qualified/realtime_qualified/predictor_benefit_tested仍false，risk_thresholds=null，old_confirmation不动。
这是已使用开发任务上的工程调度对照，不做显著性/泛化或真机硬实时声明。
