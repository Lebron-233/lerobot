# SmolVLA 异步接入：下一轮审阅差异

2026-09-09。A 连续模型合同通过；B 在 `5d45353ff98fc448bc514b284c1a19609405585b`
完成 40/40 条原生运行、20 对逐步 exact。C 在 `290c1a3dbaa90d4e3d900c4c4eb83aa6840b2ce8`
完成全部6,318个CPU请求trace。原contract_gap来自任务书与正式strict-deadline裁决冲突，
现有whole-discard实现正确。新版本仅纠正验收；严格deadline新回放在
`3ae1dc569007e612bca459ba6680dd94204268a6`完成40条/6,318请求、7项边界和完整账目，正常exit0。
[新C结果](SMOLVLA_ASYNC_STRICT_DEADLINE_RESULT.md)及[勘误版协议](SMOLVLA_ASYNC_TIMING_REPLAY_PLAN.md)。
[完整结果](LIBERO_GRAPH_NATIVE_EQUIVALENCE_RESULT.md)及[机器记录](LIBERO_GRAPH_NATIVE_EQUIVALENCE_RESULT.json)。

## 尚缺的接口与合同

| 接口 | 已确认现状 | 下一轮需要定下的具体差异 |
|---|---|---|
| 输入入口 | 新 graph helper 接收 raw images/masks；异步 planned/startup_probe 已先编码图像，并用 future_image_tokens/masks 调用 predict_action_chunk | 增加明确的 token 输入入口，把当次已编码的两路 tokens、masks、语言、state、noise 交给同一 graph core。identity 必须来自冻结的当前观测；不能对 tokens 再调用图像编码，也不能隐式 fallback |
| 模型和 stream 所有权 | helper 在创建线程记录 owner；现有异步模型推理在后台 worker；graph capture 要求同进程没有并行 CUDA 工作 | 在 worker 内创建、准备、replay、同步和释放 graph；task 变化先建立新 capture，再接受相应请求。控制线程不得在 capture 期间对 GPU 队列/prefix/processor 发起 CUDA 工作 |
| reset | queue/task epoch 已能拒绝旧结果；CPU 线程用例通过。engine.reset 当前从调用线程直接 reset policy/pre/post | 将 GPU 模型/processor reset 和图释放交回 owner，并明确 in-flight 完成或取消的边界。单靠丢弃旧 epoch 结果不足以证明 capture/reset 的资源顺序 |
| 输出与发布 | helper 的完整输出 clone 已通过连续 replay 持有测试；queue 已复制 policy/post-policy chunks；engine 已在 publication 前等待 device completion | 保留这三个现有承诺。token 入口接通后仍须在新路径证明独立输出和完成屏障，随后重新登记有限模型合同 |
| 无动作行为 | 新C严格deadline回放启动期间有 669 个明确无动作 tick，ready 后 underflow=0；eager trace 有4次迟到丢弃 | 原生调度器明确每个 wall tick 的 startup/underflow 处置与停止条件，记录缺动作。不能暂停时钟或 hold-last 来计作连续新控制 |

## 已确认的接管合同与历史勘误

[正式裁决5471357164](https://github.com/Lebron-233/lerobot/issues/1#issuecomment-5471357164)及
[复核5554561794](https://github.com/Lebron-233/lerobot/issues/1#issuecomment-5554561794)先于旧C实验：
提前仅staging，准时从新块第0行接管，任何late>0整块丢弃并继续旧active；耗尽返回None并计underflow。
stale旧返回不得清掉更新plan。max_late_steps=2用于guard sizing/诊断，不授权无补偿裁剪。

旧C的takeover_index=3、next_action_index=4首例正确结果就是deadline_miss及旧动作4。
其原expected=101、exit2和contract_gap永久保留于
`outputs/smolvla_async_timing_5d45353f_290c1a3d/result.json`，由此次审阅解释验收错误。
当前队列不需要为这个首例修复。连续性补偿/残差RTC属于另行研究范围。

真正的下一工程任务是上表的token入口、同一worker的模型/reset所有权与完成后发布。
本次勘误不启动这些GPU/模型任务，也不运行新的native队列。

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
