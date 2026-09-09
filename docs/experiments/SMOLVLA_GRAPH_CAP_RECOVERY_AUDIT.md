# E-L1：稳态 planned 恢复闭锁的最小核验

2026-09-09；交接HEAD `223156f3d41dd37bc982bd691123f8490ed93445`。
授权文件为《SmolVLA_ES1接纳_Cap恢复闭锁诊断与有界实验_Codex任务书_20260909.md》，
同时读取《SmolVLA_223156f3_独立核验与恢复闭锁分析_20260909.md》及实际AGENTS.md。
相关目录没有嵌套AGENTS.md/CLAUDE.md；三个旧未跟踪文档保留。
实际拦截记录中的“旧实验JSON只读明细查询被拦截”条目已存在，本轮只读核对，没有重复追加或重试原DevSpace操作。

**结论：原默认路径的 planned 恢复闭锁已由真实 worker/queue 的 CPU 实验复现。**
这是新的稳态问题；已接纳的E-S1 startup取整审计结论保持。

## 仅两个旧案例的独立复算

从旧E报告archives字段定位ordinal1/task0/async和ordinal5/task2/async的`result.json`。
按实际finished_at通知顺序重建接纳样本，调用原LatencyTracker、compute_delay_plan和latency_to_steps，
逐字段比较这两个案例的82+31个planner事件。没有重做315条总审计或读取任何arrays。
原件执行HEAD `e00b8731b44b80a9e80dc0af91e4434a9af3d7dd`。

| 旧案例 | 接纳request IDs | 最终linear P90秒 | raw | 超cap机会 | 分支 |
|---|---|---:|---:|---:|---|
| ordinal1/task0/A | 1,3,4,5,6 | 0.714749813079834 | 16 | 67 | cap_wait65、bootstrap2 |
| ordinal5/task2/A | 1,3,4,5,6,7,8,9,10,11,12,13 | 0.5179850459098816 | 12 | 4 | cap_wait4 |

两条均保持稳态reset_epoch1/task_epoch0，已记录主请求无重叠、peak_inflight1；
所有超cap机会均没有尚在途的planned。首次超cap后没有新planned。
ordinal1首/末超cap事件时间为1045879.419912789 / 1045905.076903700，
ordinal5为1045982.874619886 / 1045983.263572664（原host monotonic秒）。

task0接纳样本：0.2039953419007361、0.3028119750088081、0.21727405302226543、
0.3505812670337036、0.957529005012475秒。原float32/linear P90为0.714749813079834秒，
经原20Hz取整与margin1得raw16，planned仍按cap/available截断，未用planned值代替raw判据。
request7/8是普通bootstrap，完成0.34502341505140066 / 0.6467783220577985秒，
均installed、复用capture2、未接纳，之后历史不变。各自单样本换算为8和14，不能将二者都称为预算内快样本。

小型机器证据见`SMOLVLA_GRAPH_CAP_RECOVERY_AUDIT.json`；完整此次两案例读取结果、脚本和退出回执保存在
`outputs/smolvla_graph_cap_recovery_preparation_223156f3/old_cap_audit.json`、`audit_old_cap.py`、`old_cap_audit_receipt.json`。

## 性能共现的边界

task0 A：54.583567秒/210次动作，slot账目dispatch210、env_busy453、scheduler_miss408、underflow21。
最长派发间隔10.476126秒、单次native最大9.220309秒。首次cap之后相交或后续的106个native区间，最大1.909865秒。
task2 A：21.041416秒/280动作，dispatch280、env_busy87、scheduler_miss54；
native最大0.280218秒，首次cap之后相关native最大0.082794秒。
这些是时段共现和独立类别账目，不把等待、调度遗漏、Env耗时或44.21秒的旧配对差额分摊给闭锁。
旧三个完整配对仍按首错截断；task1的两条件轨迹/工作量不同，task3单边不加入配对。

## 源码机制与边界修正

原`_make_request_locked`在raw超cap且active非空时cap_wait，空时identity bootstrap；
原startup complete后的普通bootstrap不进tracker，而符合条件的planned完成会接纳。
窗口按接纳样本滚动，不随墙钟自动过期。因此同上下文且无在途planned时，
超cap阻止新planned，bootstrap不能更新历史；worker仍可补块和执行，闭锁对象是planned重规划。

实际reset和set_task**并不清空该tracker**。Graph适配器reset清队列并安排owner资源reset；
set_task使旧plan/staged失效，两者保留时延窗口。每个native episode新建engine才构造空tracker。
CPU边界测试如实记录这些行为，没有预填“reset清历史”或声称在所有重置情形下必然持续闭锁。

## E-L1真实CPU判别

候选改动前，4项characterization首次通过（0.51秒）：

1. 原NativeEngine实际完成cold/probe/fresh；一次稳态planned慢完成触发超cap。
   后续两次较快bootstrap均合法installed，但tracker不变、无新planned，单在途且同epoch。
2. 无慢样本正控制连续产生3次planned并合法接管，证明夹具没有卡住worker。
3. 上述真实五样本调用原实现得P90/16；两个bootstrap继续列为排除样本。
4. reset/task保留历史，新engine为空窗口；不调用主线程`_run_request`，不手改startup状态。

复用原CPU fake policy/Graph/Env、真实NativeEngine/worker/queue与受控时钟。
新代码加入后，四项默认关闭characterization也通过；后续恢复准备结果见TESTS。

## 同路径复用判定及候选边界

identity planned的观测/pre、独立vision_encode、future_image_tokens/masks、完整policy/post/CPU输出路径
不依赖takeover plan；plan仅在predicted输入构造或后续queue stage处使用。
因此可增加独立recovery_probe request kind，走同一identity计算分支，在原完成屏障后只更新有效历史并返回，
跳过queue install/stage。此处不存在迫使改成“接纳bootstrap”的准备阻塞。

候选`same_path_discard_probe_v1`是新的显式采样/调度协议，默认disabled；原P90/window50/cap8及bootstrap接纳规则不改。
有效且同task/reset epoch的probe只接纳一次；失效或stop后不接纳，预算最多50且reset/task不重置。
同时补上原实验归档的清理边界：Env关闭或Graph/processor/metrics清理未确认时不保存大数组。
CPU定向测试覆盖该实际close错误路径，正常关闭路径保持原行为。

## 保持与限制

E-S1旧9>8与新3≤8都使用当前startup probe，不用P90，结论保持；本轮未新做startup-only native。
CPU受控延迟不代表native时延或收益。新的四条难例诊断须经准备门、提交登记和单次回读，结果另记。
旧E7 completed/1失败/12 not_run、4 success/3 TimeLimit、3完整配对及两个整体false不变；
D历史true、旧E三个observed、科学资格false和旧confirmation untouched均保持。
