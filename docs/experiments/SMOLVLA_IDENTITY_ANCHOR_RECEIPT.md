# F-IAR1 执行回执：identity 基底置换通过开发推进条件

2026-09-17（Asia/Tokyo）。完整GPU配对执行及独立CPU审计通过；这是新开发发现，不是F-ACQ1-R1资格阴性的翻转。

## 身份、授权范围及记录

- execution HEAD：`1f74c04d2ce9d02562d12e7dff0207f6bd9c24ba`；分支 `codex/smolvla-graph-native-equivalence`。代码、独立审计器、测试和计划先提交推送。
- 真实预登记：[5705835108](https://github.com/Lebron-233/lerobot/issues/1#issuecomment-5705835108)，唯一标识F-IAR1-REGISTER:1f74c04d2ce9d02562d12e7dff0207f6bd9c24ba。
- 先只读GET确认无同标识，再结构化gh POST一次，随后按返回真实ID独立GET。POST/GET均成功；本轮未请求返回HTTP响应头，因此不虚构具体HTTP状态码。
- preparation SHA256 `e1b769dda168cf40a3e7d4d254a33d5ba85f0380dacad5774f82247500b25849`；正文SHA256 `5a3c57e82a67c382136c7b3e63d6aeebf8ed6431d438d8b6664d4604f5859f6b`。
- 本地POST/GET为实际工具返回六字段的规范化投影，正文通过stdin原文传输并与registration.md逐字相等；不是原始HTTP字节捕获，来源保存在registration_receipt.json。
- 唯一原始目录：`/home/rp/Workspace/SmolVLA_RTC/lerobot/outputs/smolvla_identity_anchor_1f74c04d`。
- WebCodex Job `22cd58d4-b534-41d1-be35-5e57dce92a24`：completed、exit0，duration_ms52375。
- worker PID605582，exit0、wait收回；后续/proc/605582不存在。监督UTC 2026-09-16T23:11:47.478042+00:00至23:12:35.601446+00:00，wall48.12344381600269秒。
- 本轮attempt1/retry0；此前ACQ原失败与R1两个执行均原样封存。没有重跑旧worker、删除旧目录或重新登记旧尝试。

## 实验做了什么

只读取原72训练/18episode和16开发验证/4episode；错配训练69、验证16，共85。原三个delay4单例不替换、不删例。
把冻结F-ACR1第72步同一已保存残差delta=h(a)-h(0)，从原B+delta移到当前视觉token I+delta。幅度固定1、FP32先减后加、BF16往返，不重新训练/选点/扫alpha。
每个样本先重放I、B、原B+delta的完整50x32输出逐值exact，再计算I+delta的true/zero/mismatched。
所有动作、mask、同split/task/state/delay循环供体来自原冻结开发记录；新零动作raw和输出均exact回I。
R1的32资格样本与其张量未加载；既有R1/其他pending文档只做字节保护，不用于新指标计算。task8/9标签和confirmation未读。

## 技术证据

| 项目 | 实际 |
|---|---:|
| CPU测试 | 68 passed in 1.33s；Ruff通过 |
| 新科学来源与前次准备哈希一致 | 7份 |
| 正式decoder | 525=264旧控制重放+88新true+88新zero+85新mismatch |
| VLA加载 | 1 |
| predictor模型加载/前向 | 0/0，使用已保存delta |
| I/B/原中心化完整输出exact | 各88 |
| 新zero raw/完整输出exact | 88 |
| Graph captures | 8 |
| Graph内部setup/warmup/capture | 8/24/8，另计于525 |
| Graph setup图像编码 | 0 |
| phase started/returned | 616/616 |
| 独立NumPy数值比较 | 1575项通过 |
| 新Env/图像编码/训练/反传/真机 | 全部0 |

独立CPU审计一次，exit0，内部2.059603782981867秒；CUDA未初始化，审计模型前向0。
独立指标/推进判据与运行器一致，来源、选样、供体、组合和精度、全部对照重放、预算及phase闭合获接纳。
first_failure=null，stop_reason=null，无强制终止、pending或active，VLA冻结、Graph释放。
全部旧13份未提交文档仍原字节保留。没有覆盖旧R1报告或旧原始输出。

## 科学结果与限制

```text
independent_contract_accepted = true
development_followup_supported = true
independent_qualification_claimed = false
```

开发验证16/4、episode等权：I首动作MSE0.031709702433，原B+delta 0.013862918700，新I+delta 0.012048372598。
新方案相对I降低62.004145%，相对原中心化降低13.089207%；相对I为12改善/4恶化、4/4episode改善，四个留一episode平均收益都为正。
相对原中心化则只有7样本改善/9恶化、3/4episode改善；不能说逐样本全面更好。
相对新错配，true首动作macro为0.012048372598 vs 0.044078063496，4/4episode改善，但仅6样本改善/10恶化。这是多数样本方向与均值方向不一致，不能只摘宏均值或宣称普遍动作机制成立。

动作块存在真实权衡：新chunk0.026728824581，相对I0.053248021229改善，但比原中心化0.021462120248增加24.539534%。
新token MSE2672.575387909，仍高于I2670.828930525；不是视觉预测整体变准。
新方案相对I的验证净收益仍集中：单episode7/49占70.223571%，单样本7/49/5占63.872976%。这两项为退出后描述性汇总，不参与推进条件或删例。
最不利验证样本6/48/6相对I增加0.039780288237，其他不利样本完整保留。

训练72/18：新首动作MSE0.053600007189，I0.075968549259、原中心化0.055629032020；相对I降低29.444477%，50样本改善/22恶化、14/18episode改善。
相对原中心化首动作仅降低3.647421%，32样本改善/40恶化、7/18episode改善，chunk也增加4.348678%。
训练样本4/46/5相对I增加0.642751500463，旧严重局部退化并未消失。
训练错配比较严格同69例，新true配对MSE可由0.117012616613-0.059756622511得到约0.057255994102；不拿72例均值直接与69例比较。

## 误差分解和合理解释

独立审计对每个配对的七维保存输出验证：MSE变化=2*mean((control-oracle)*delta_output)+mean(delta_output^2)。
验证新方案相对I：交叉项-0.036363069966、扰动能量+0.016701740132，相加为-0.019661329835。
验证新方案相对原中心化：交叉项-0.005482376171、扰动能量+0.003667830070，相加为-0.001814546101。
全部逐例/逐维项保存于independent_audit.json；DESCRIPTIVE_SUMMARY.json另存episode等权汇总。
该恒等式描述输出误差如何变化，不是视觉Jacobian分析或物理因果识别。

本轮支持的窄结论：在旧开发分布上，同一固定动作残差不必叠加到学习基底B上才产生平均首动作收益，identity基底构成值得后续研究的候选。
本轮不能证明B是R1失败的唯一根因：B在这份旧开发集本来就优于I，且没有在新资格集评估I+delta。
不能宣称已修复跨初态泛化、降低部署延迟、提高闭环成功率或取得安全资格。缓存delta省略预测器计算，不是公平部署FLOPs/时延比较。

## 交付及接续

新完整报告、机器结果、独立审计及本回执显式整理为docs/experiments/SMOLVLA_IDENTITY_ANCHOR_{RESULT.md,RESULT.json,AUDIT.json,RECEIPT.md}。
开发条件通过允许发布新结果，但必须附本回执中的全部不利证据；结果commit和实际评论ID在独立发布回执记录，不提前伪造。
没有自动启动后继训练、额外资格采集、alpha扫描、时延或闭环；R1资格数据继续冻结。
后续应在原开发集检验identity基底下直接训练及动作块/单样本退化约束，首先解释并限制4/46/5这样的严重误差，而不是直接部署这个均值更好的候选。
本次明确安全拦截0；一次早期读取events.jsonl时文件尚未创建返回not_found，随后观察同一Job正常完成，不是安全拒绝或worker重试。
生产默认、baseline_qualified/realtime_qualified/predictor_benefit_tested仍false，risk_thresholds=null。
