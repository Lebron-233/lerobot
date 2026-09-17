# F-IQ1执行回执与路线决策：新初态上整块改善，首动作退化

2026-09-17（Asia/Tokyo）。本轮完成一次新的冻结候选资格实验，不是继续在旧开发集上调参。结果为技术接纳、科学门未过；不能宣称完成预测式异步推理。

## 实际身份

execution/code HEAD：71e5510c0463bca8e0a992d75015209445dde70a。
工作区：/home/rp/Workspace/SmolVLA_RTC/lerobot；分支codex/smolvla-graph-native-equivalence。
实际预登记：https://github.com/Lebron-233/lerobot/issues/1#issuecomment-5707461303。
唯一OUT：outputs/smolvla_identity_qualification_71e5510c。
PREP：outputs/smolvla_identity_qualification_preparation_71e5510c。
preparation SHA256：af814718fa816ab9e7a968fb4797242917b926b6a6ca25673dfce0a7ba811b9c。
登记正文SHA256：34da7163fe4d5d1bc42ec9b87ab2e700677b69b085c7e35d4f8fd349a8120142。
result.json SHA256：bc65b85406d250d7fb834e55668440980108a8320829b686840c4213cb12f632。
independent_audit.json SHA256：5d818beb531148c4eb0835429810a8fc27a1b7690c700fdbc265b79eb913a9e6。
登记按唯一标识GET查重、gh literal argv单次POST、真实ID独立GET；正文exact。保存实际返回的规范化六字段投影及来源，不冒称原始HTTP字节捕获。

## 为什么选择这一实验

TUA只保留3/72更新，开发验证仍劣于冻结IAR；反复修改目标不能替代泛化检验。原ACQ1-R1验证的是B+(h(a)-h(0))，而旧开发集表现最好的冻结IAR使用I+(h(a)-h(0))。后者的新初态证据此前尚缺。
本轮模型仍为原centered第72步、69680参数、幅度1、FP32先减后加再BF16，真实两次预测器前向，不用缓存delta。不训练、不搜索系数、不选检查点。
新初态固定task6/7 state4..7，八个身份在本仓库317条既有started记录中无冲突。该独立性仅覆盖本仓库可见记录，不覆盖未知外部运行，也不是未见任务benchmark。
旧ACQ1-R1的32个样本、task8/9标签和confirmation21..40未读取。历史只读取身份元数据；原16开发验证例仅exact锚点。

## 已审计结果

independent_contract_accepted=true；heldout_primary_gate_passed=false；heldout_robustness_gate_passed=false。
新Env/episode=8/8，N=32，M=30。delay3有30例、delay4有2例；两例在对应同task/state/delay组是单例，依法不生成错配，没有删除真实/零动作样本。

| 条件 | N/episode | 首动作7D MSE | 50x7D chunk MSE | 有效token MSE |
|---|---:|---:|---:|---:|
| Identity | 32/8 | 0.035956297515 | 0.052467653136 | 2536.974969105 |
| IAR true | 32/8 | 0.044241214108 | 0.032694949373 | 2539.016751494 |
| IAR zero | 32/8 | 0.035956297515 | 0.052467653136 | 2536.974969105 |
| IAR mismatch | 30/8 | 0.051547574324 | 0.082069500374 | 2597.443959518 |

相对Identity：首动作误差增加23.041629%，chunk误差降低37.685512%，token误差增加0.080481%。首动作18/32样本改善、14恶化；4/8episode改善，未达到6/8；平均收益(Identity-IAR)=-0.008284916594。
相对错配必须使用同30个接收者：真实IAR macro=0.045323464277，错配macro=0.051547574324，误差降低12.074496%；21/30样本、7/8episode改善。不能用32例真实均值和30例错配均值直接计算配对收益。

最坏样本6/5/6：Identity首动作0.012364432766，IAR0.186868129171，增加0.174503696405。
次坏6/5/5：Identity0.016392578543，IAR0.156615877618，增加0.140223299074。
最不利episode6/5：Identity0.010261292808，IAR0.088214360116。8种留一episode中，只有删除6/5后的平均收益为正(+0.001667676365)，其余7种均为负；实际没有删掉此episode或调整结论。
主门失败于“首动作均值优于Identity”及“至少6/8episode优于Identity”；稳健要求的全部留一正收益也不满足。对照覆盖、错配比较、chunk和两项样本多数要求均如实保留。

意义：真实承诺动作信息在配对错配检验和chunk指标上有信号，但当前冻结候选不能可靠改善接管时的首动作。不是证明整个预测方法族无效；不以chunk阳性掩盖预登记主指标阴性，不把动作MSE等同闭环成功率或真实安全。

## 执行、预算与独立性

正式Job：f6666f03-ba5a-436e-a111-adf4ff844ce7，completed/exit0，duration_ms154783。
worker667044已收回，后续/proc/667044不存在。
监督UTC2026-09-17T02:21:10.625672至02:23:41.119078，wall150.493437725秒。
first_failure=null，stop_reason=null，pending/active为空；无强制终止、attempt1/retry0。

| 项目 | 实际 |
|---|---:|
| VLA/预测器加载 | 1/1 |
| 正式离线decoder/预测器前向 | 190/220 |
| 旧锚点exact | 16 |
| 新双相机编码/current exact | 40/8 |
| 新identity/zero完整输出exact | 32/32 |
| 新Env/settling/measured | 8/80/1717 |
| native main/capture | 96/16 |
| native Graph内部setup/warmup/capture | 16/48/16 |
| offline capture及内部setup/warmup/capture | 4及4/12/4 |
| phase started/returned | 283/283 |
| 新训练/反传/真机/旧资格标签读取 | 全部0 |

Identity采集本身6/8成功，只能归属Identity；预测器没有控制任何Env，不是IAR闭环成功率。
独立CPU审计Job be768385-d9b5-4585-8710-3566c420c6b1只运行一次，exit0；1717个native动作和70次接管重新核验，378项数值比较、283阶段闭合，8份模型前初态checkpoint及全部选样输入/供体/残差/预算通过。
审计内部wall2.538503372秒，CUDA=false，模型前向/新Env0。数值归约使用NumPy、门独立实现；native来源与选样复用原冻结实现。未重新执行未来图像编码或模型前向，不能称CPU从头验证模型输出真实性。
23份科学来源、620份相关已追踪源码及21份原pending绑定并核验。旧实验与全部旧消耗原样保留；本轮不把历史训练成本写成0。
62项CPU测试最终通过1.28秒；Ruff通过。原native审计适配在允许的旧E-NAT1 task0/state41归档上只读核验200动作/8接管，不重跑该仿真。一次unused import开发错误已在冻结前修正。

## 后继工程路线

按运行前计划选择分支B：本批IQ1资格数据冻结，停止围绕同一小预测器与同16开发验证例反复训练；不是把阴性改名为成功。
目标分为两层：可复现的异步动作执行与更优的学习预测补偿。前者不逻辑依赖后者。E-NAT1已有单轮10对调度证据；此轮没有重算历史墙钟或把旧收益冒充新收益。
新的工程主线优先检查Identity异步与guided RTC的时间语义、真实额外计算和小规模受控闭环。当地源码已有RTC接口，但不等于当前自研Graph/队列可直接启用：RTC ActionQueue明确跳过real_delay行；当前预测式takeover从row0开始，两者不能直接拼接。
执行步骤和硬边界详见SMOLVLA_ASYNC_NEXT_EXECUTION.md。该后继计划尚未启动模型/Env，不自动发布下一次预登记或承诺后台运行。所有生产/实时/闭环补偿资格仍false，risk_thresholds=null。

## 工具拦截与发布

本次一次git ls-remote远端分支只读查询被拒绝，原文：因 OpenAI 无法确定请求的安全状态，已拦截此工具调用。
未重发或换通道重做该查询，不计为已核实远端状态。之后独立的普通非强制git push成功返回90c54a5f..71e5510c；实际预登记POST/GET、正式实验及独立审计均成功。
一个已知路径拼写错误的rg搜索exit2属于普通查找错误，与安全拒绝分开记录；未据此重跑实验。
本轮完整阴性及正面子结果按“有意义资格结果”范围整理发布，必须同时披露未过门和不利样本；实际结果提交与评论ID以发布回执为准，不提前伪造。
