# E-RGC1：完整 RTC 引导的 Graph 等价与成本结果

日期：2026-09-22。执行 HEAD `a43a31d8c4416a27f826a996dae98d2910f9b2e5`。
预登记真实评论 ID `5769675534`，运行前单次 POST、独立实际 ID GET 及正文逐字核验完成。
原件：`outputs/smolvla_rtc_graph_a43a31d8`。

## 主要结果

固定288次请求全部完成并通过独立CPU审计。三臂完整50x32、归一化50x7和后处理50x7输出逐值一致；完整原RTC公式、十步解码与全部Jacobian/VJP均保留。
有前缀Graph的完整请求均值108.184ms、P99/最大值115.793ms；同期原生eager均值660.163ms、P99/最大值804.444ms。平均时延降低83.61%，P99降低85.61%，平均时延比6.10倍；80/80个相同输入与重复编号配对Graph更快。Graph所有80个计时请求低于350ms，达到原20Hz/cap8/余量1的计算预算。
仅缓存权重的eager均值663.416ms，并未显示速度改善。因此，缓存是解除捕获限制的实现条件，本轮主要加速来自完整Graph路径，不能把收益全部归给权重缓存。

这是固定delay3、30x7前缀及相同输入签名下的完整RTC计算候选，不是实时环境并发、动态延迟支持、任务保持或部署资格。旧无RTC候选、E-SEQ1及所有阴性不改写。

## 为什么转向这一实验

E-SEQ1全程保序恢复了已知7/19，但新增2/10、7/10回退，不能继续把保留row0当作一般修复。完整RTC可以让动作生成考虑旧承诺前缀，但此前RTC-FULLPATH1的eager尾延迟超预算。本轮只解决这项计算前置条件，不同时修改接管规则或搜索新的算法。

最初合成捕获确实失败：原RTC在捕获中把普通CPU前缀权重复制到CUDA，报“Cannot copy between CPU and CUDA tensors during CUDA graph capture unless the CPU tensor is pinned”。DevicePrefixRTC只在捕获前调用原CPU公式并缓存其精确设备值；原RTCProcessor.denoise_step、euler_integrate及生产源码没有改动。仍对完整denoiser映射求VJP，不省略Jacobian，不把缺失25维视作零目标，不改变BF16/FP32边界、guidance或步数。

## 固定设计与分母

原RTC-FULLPATH1的16个已见开发输入：task6/7、state48/49、request3..6。来自原准备data.pt的当前原始图像/状态、语言、噪声和来源锁定的归一化旧前缀，不读未来标签选择方案；不是新的独立资格数据。
三臂同模型与同数据：native_eager原生完整RTC；cached_eager仅缓存相同权重；graph使用同cached处理器并捕获完整采样。EXP、horizon10、guidance10、delay3、chunk50、10步固定。
每输入每臂6次，轮转三臂次序，共288。repeat0三臂都无前缀，共48校准请求；repeat1..5都有真实前缀，共240，其中每臂80次用于下表。没有删慢请求、追加预热、追加样本或改容差。
一个owner/runtime跨16例，只准备无前缀/有前缀两个Graph。每次请求重新预处理并编码双相机；视觉编码不在Graph内，仍计入完整时延。静态缓冲中的视觉tokens、mask、语言、state、noise和真实前缀每次全部刷新。不同输入未触发再次捕获。

## 完整请求时延

| 指标 | 原生 eager | 仅缓存权重 eager | 缓存权重＋Graph |
|---|---:|---:|---:|
| 有前缀计时请求 | 80 | 80 | 80 |
| 平均/ms | 660.163 | 663.416 | 108.184 |
| P50/ms | 656.171 | 657.806 | 106.888 |
| P95/ms | 694.814 | 689.524 | 113.118 |
| P99及最大值/ms | 804.444 | 858.555 | 115.793 |
| 超过350ms请求 | 80 | 80 | 0 |
| ceil(P99秒×20)+1 | 18 | 19 | 4 |

计时从预处理和搬运至真实视觉编码、前缀预填充、完整十步采样/VJP、后处理、GPU同步与CPU证据复制结束；磁盘写入在外。Graph不是只测decoder，也不是用旧缓存图像特征跳过新编码。
相对cached_eager，Graph平均时延降低83.69%，P99降低86.51%；同样80/80配对更快。cached_eager均值比native_eager高0.49%，不据此宣称缓存显著变慢，只能说本轮没有速度收益证据。
旧731.916ms是另一轮的尾延迟，不能作为本轮加速比的分母。本轮直接使用同期native_eager的804.444ms。
80次请求来自16个重复输入，nearest-rank P99恰为最大值，不是80个独立场景或长期尾概率保证。当前没有Env负载，不能继承为并发闭环时延通过。6.10倍是平均请求时延比，不是机器人任务速度或成功率提升。

## 输出一致性与完整VJP证据

独立审计核验288份实际动态输入、48份无前缀原完整输出、96组三臂输出（共576项数组比较）、192项重复输出稳定性。各组中的完整、归一化和后处理动作逐值一致，不仅首动作或均方误差接近。旧前缀和原CPU权重值也逐值一致。
原生eager和cached_eager各实际执行960次RTC包装步，其中各800次带前缀VJP。Graph正式重放96次：无前缀16次、有前缀80次；有前缀捕获本身记录完整10投影/10VJP，80次重放对应800步已捕获引导计算。没有把这800步误报为800次新的Python autograd调用。
全部参数requires_grad=false且grad=None。推理期对噪声动作输入求导是RTC的一部分，不是训练或优化器更新。
CPU审计没有重新运行模型、视觉编码、VJP、后处理或GPU计时；它核验保存的比较数据、源码与来源锁、独立时延归约、捕获账目和退出证据。完整计算等价结论限于此次固定16输入与配置。

## 必须保留的启动代价

第一个Graph完整请求4.695499秒，包含两个图的准备；无前缀图准备1.320912秒、有前缀图准备3.297201秒，合计4.618113秒。runtime初始化另记0.000437秒。
每图setup1/warmup3/capture1，共10次额外sampler调用、100次额外解码步，其中50次带前缀VJP，独立计账，不混入288正式请求。
repeat0的16次无前缀调用是校准分组，不是16次进程冷启动。其平均完整时延分别294.335/264.093/352.985ms；Graph均值混合一次捕获和15次复用，不能据此描述冷启动。全部校准值已保存在DESCRIPTIVE.json。
不能用约108ms稳态速度替代约4.70秒首次准备；未来控制器必须在ready之前准备Graph，签名变更不得在控制期间静默再捕获。当前实现只支持delay3/前缀30x7，尚不支持在线动态delay。

## 执行与完整性

| 项目 | 实际值 |
|---|---:|
| VLA加载 | 1 |
| 正式完整请求/真实双相机编码 | 288/288 |
| Graph准备/正式重放 | 2/96 |
| 调用意图与返回闭合 | 288/288 |
| phase开始/返回 | 17/17 |
| 正式attempt/retry | 1/0 |
| 新Env/训练更新/真机 | 0/0/0 |

单owner退出时恢复原sampler及processor、释放两图；正式worker1635334、外层和独立审计均exit0，进程已不存在，无首错、stop_reason、pending、active或强制终止。
监督UTC2026-09-22T00:52:32.204645至00:54:53.945435，141.740814秒。正式Job9a5181c5-1221-41d8-a6f4-28f7cdc0751b；审计Job7d8a9f80-c57f-4ff0-a3de-dc1819a04a1d。
717份来源/源码hash及原21份pending内容和状态保持。CPU回归36通过/2个CUDA用例跳过；独立启用CUDA后本文件12项通过，含10项重复CPU和2项CUDA，不能将两个数量相加宣称48个不同用例。Ruff及暂存差异检查通过。原始合成捕获首错与冻结前格式修正均保留，不是正式worker重试。
本机4070Ti SUPER，driver580.178.04，PyTorch2.11.0+cu128；没有依赖或驱动升级、干预其他负载。独立统计归档没有新增模型前向或Env。

## 决策与后继边界

independent_contract_accepted=true；full_rtc_equivalence_passed=true；graph_budget_passed=true；graph_faster_mean=true；rtc_graph_cost_candidate_supported=true。
live_feedback_tested=false；task_retention_tested=false；deployment_qualified=false。

本轮得到可继续接线的完整RTC计算候选，不再因先前eager超预算而排除完整前缀约束路线。下一步先验证动态delay和真实旧前缀的刷新、单owner及队列时间身份，再在固定开发身份上做无RTC/RTC的真实异步并发与任务对照。该步骤尚未执行，不覆盖旧任务保持阴性，不改生产默认。
GitHub连接器读取上一轮评论返回FORBIDDEN，该读取没有换通道重复。新实验登记通过原生gh单次POST及真实ID独立GET成功；目前没有新的写入/运行安全拦截。结果发布状态另见publication_receipt.json。
