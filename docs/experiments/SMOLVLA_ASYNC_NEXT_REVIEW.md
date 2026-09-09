# SmolVLA 异步接入：下一轮审阅差异

2026-09-09。A 连续模型合同通过；B 在 `5d45353ff98fc448bc514b284c1a19609405585b`
完成 40/40 条原生运行、20 对逐步 exact。C 在 `290c1a3dbaa90d4e3d900c4c4eb83aa6840b2ce8`
完成全部6,318个CPU请求trace。原contract_gap来自任务书与正式strict-deadline裁决冲突，
现有whole-discard实现正确。新版本仅纠正验收；严格deadline新回放在
`3ae1dc569007e612bca459ba6680dd94204268a6`完成40条/6,318请求、7项边界和完整账目，正常exit0。
[新C结果](SMOLVLA_ASYNC_STRICT_DEADLINE_RESULT.md)及[勘误版协议](SMOLVLA_ASYNC_TIMING_REPLAY_PLAN.md)。
[完整结果](LIBERO_GRAPH_NATIVE_EQUIVALENCE_RESULT.md)及[机器记录](LIBERO_GRAPH_NATIVE_EQUIVALENCE_RESULT.json)。

## D3-r1固定CUDA模型与worker合同通过

D3-r1执行源码`04902a527d684dd43bb55098c8b5f39ae2a96fa2`，
正常包入口的可选datasets边界修复完成；原环境不变，39项CPU和模型环境导入门通过。
唯一一次新D3-r1完成20/20对token和12/12个真实worker事件，三个CUDA通过标志均true。
52主调用+12reference，15 capture，单列setup15/warmup45/capture内调用15；
37.979632秒，child/supervisor均exit0，worker join、图释放和sampler恢复确认。
[D3-r1结果](SMOLVLA_GRAPH_TOKEN_WORKER_R1_RESULT.md)及[固定协议](SMOLVLA_GRAPH_TOKEN_WORKER_R1_PLAN.md)。

旧D执行`4f171a4b04dbf3111167dc6437658cdd2f1df4cb`的datasets导入失败及三个false继续保留，
没有改写旧报告或结果目录。[旧D结果](SMOLVLA_GRAPH_TOKEN_WORKER_RESULT.md)。

| 接口 | 当前实现/证据 | 尚需完成 |
|---|---|---|
| 输入入口 | 固定十帧20对noise/完整/去pad/post均exact；每主请求编码与采样各一次，token Graph内部无额外视觉编码，持有输出独立 | D合同已完成 |
| 模型与stream所有权 | 真实worker的102条模型/processor调用及capture/reset/退出归于同一owner，join/释放/恢复全部确认 | D合同已完成 |
| reset/task | 真实0→3→0序列和ready后在途reset完成，CPU epoch及时失效；reset、两次task变化和stop的四个旧返回均stale | D合同已完成 |
| 输出与发布 | 12次主请求与显式noise reference全部exact；12次独立CPU chunks及完成屏障，186个CPU夹具消费动作和退出后档案收回 | D合同已完成 |
| 无动作行为 | 新C严格deadline回放有669启动无动作ticks，ready后underflow0，eager迟到丢弃4次 | native协议仍需另行固定wall tick、消费数量与无动作处置 |

导入边界阻塞已闭合。D3-r1达到固定合同的停止点；下一轮native的动作消费、wall tick和无动作处置
仍须另行固定。当前没有启动native或predictor新实验，原科学资格不变。

## 已确认的接管合同与历史勘误

[正式裁决5471357164](https://github.com/Lebron-233/lerobot/issues/1#issuecomment-5471357164)及
[复核5554561794](https://github.com/Lebron-233/lerobot/issues/1#issuecomment-5554561794)先于旧C实验：
提前仅staging，准时从新块第0行接管，任何late>0整块丢弃并继续旧active；耗尽返回None并计underflow。
stale旧返回不得清掉更新plan。max_late_steps=2用于guard sizing/诊断，不授权无补偿裁剪。

旧C的takeover_index=3、next_action_index=4首例正确结果就是deadline_miss及旧动作4。
其原expected=101、exit2和contract_gap永久保留于
`outputs/smolvla_async_timing_5d45353f_290c1a3d/result.json`，由此次审阅解释验收错误。
当前队列不需要为这个首例修复。连续性补偿/残差RTC属于另行研究范围。

上述接口已在D3-r1通过固定真实CUDA验证，新的native队列仍未启动。

## 动作转换链

当前固定 LIBERO checkpoint 的 preprocessor 是 rename、batch、task newline、tokenizer、device、normalizer；
postprocessor 是 device、unnormalizer。完整严格载入清单保存在 B 的 `policy_load.json`。
backend 实际拒绝的是 **preprocessor 中 enabled 的 RelativeActionsProcessorStep**，理由是观测 anchor/rebase
尚未定义；它没有据 native controller 的 relative OSC 模式作这个判断。现有校验保留。

| 层级 | 当前含义 |
|---|---|
| policy-normalized action | padded 50×32 中有效的前7维；当前原生 selector 每次仅消费第1行 |
| post-policy action | 用固定 checkpoint 统计反归一化后的7维控制量 |
| controller command | 送给 Panda relative OSC 的平移/旋转增量及夹爪指令；有限超界值按既有 native 控制器处理 |
| native env.step | 控制器推进动力学并产生下一真实观测；实际发送值与准备值已在 B 逐步核对 |

队列中的 post-policy action 仍是控制命令，不是已实现的 EEF 位姿变化。
OSC 命令不能直接积分成下一8D state；normalized 动作、axis-angle 与双指 qpos 也不采用 delta_sum。
下一轮先保持 identity context/state。已启用 RelativeActionsProcessorStep 的其他策略仍需独立 anchor 合同。

## 下一轮固定比较

1. **eager-sync 与 graph-sync**：保持现有50/1/10和同一配对任务/seed，隔离计算实现；本轮工程结果已取得。
2. **同一 sampler 的同步与 identity 异步**：协议先写死50步预测块的消费数量、重规划与接管规则、delay cap、
   guard 和 wall-tick underflow处理。同步对照使用同一消费合同。当前每次只消费1行的基线，
   不能直接为异步连续消费多行提供资格结论。
3. **identity 与 predicted context**：只有接口、时序和动作消费合同一致后，才固定同一 sampler 比较 context。
   记录输入上的 oracle 仅作独立离线诊断，不进入 native 执行。

每一轮的完整 manifest、seed、实现 HEAD、输出目录和停止门槛均需重新固定后执行。
旧基线资格仍未通过（开发185/200，但 task5为14/20，低于16/20）。下一协议必须明确继续只作工程诊断，
或先建立满足科学资格要求的基线；工程 exact flag 不能代替资格门槛。旧确认队列仍未启动。

## 适用限制

C 使用 CPU 索引标记和 B 的已完成延迟，不包含真实视觉策略或环境反馈，也没有验证 CUDA capture
与真实异步线程并发。不同延迟 trace 导致消费的标记数不同，不能据此推断原生任务成功率。
本轮没有训练 predictor、校准 risk 或启动新的异步原生队列；三个原资格 flag 为false，risk_thresholds为null。
