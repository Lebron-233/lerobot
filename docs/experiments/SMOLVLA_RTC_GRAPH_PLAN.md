# E-RGC1：完整 RTC 引导的 CUDA Graph 等价与成本验证

2026-09-22。接续 E-SEQ1（e8206030）：无条件保序没有保住同期任务成功，旧时间对齐候选 a5e9fdde 及全部阴性保持冻结。本轮转回完整前缀条件生成；不修改旧控制器、权重或 src/lerobot。

## 本轮唯一问题

在保留原 RTCProcessor.denoise_step 的完整 Jacobian/VJP、7D 有效维度掩码、EXP、horizon10、guidance10、delay3 和十步解码的条件下，能否消除主机调度开销，降低整个真实推理请求的成本？先验证这一前置条件，不直接启动新闭环或宣称修复任务回退。

原始 RTC 合成 Graph 捕获实际失败在 get_prefix_weights(...).to(cuda)：非 pinned CPU 到 GPU 复制在捕获期间被拒。本轮 DevicePrefixRTC 仅在捕获前调用原 CPU 权重公式并将其精确值缓存到 GPU；get_prefix_weights 返回同一设备缓冲。denoise_step 和 euler_integrate 均直接继承/调用既有实现，不删除 Jacobian，不改精度或系数。合成 CUDA 测试已验证动态前缀更新与原 eager 完整输出逐值一致；这不是模型验证。

## 固定数据和三臂

只读复用 outputs/smolvla_rtc_fullpath_preparation_92c39812/data.pt 的原 16 个开发输入：task6/7、state48/49、request3..6。它们包含当前原始观测、来源锁定的归一化旧动作 30x7 前缀、原噪声和无引导输出。不读新资格标签，不新增 Env，不训练，不调用真机。

三臂固定为 native_eager（原 CPU 生成权重的完整 RTC）、cached_eager（只缓存相同权重，仍 eager）、graph（相同缓存处理器，完整采样/VJP CUDA Graph）。三臂使用同一模型、同预处理/视觉编码/后处理、同噪声和前缀。权重缓存不改变算法；cached_eager 对照分离缓存本身和 Graph 的收益。
每个输入每臂 6 次，总 288 请求。repeat0 三臂均无前缀，48 请求用于原完整输出校准；repeat1..5 均带真实前缀，每臂 80 请求，全部用于稳态统计。按(sample_index+repeat)%3轮转三臂顺序。不得增加调用、删除慢值或选负载时机。
一个模型 owner、一个 runtime 跨全部16输入，共2个固定形状 Graph（无前缀/有前缀）。第一个无前缀 Graph 请求同时捕获二者，有引导的准备只使用形状匹配的零前缀，不输出给控制器；后续真实前缀必须复制到静态输入。新图像每请求重新编码，不用缓存视觉 token 冒充全路径。签名变化显式失败，不静默重新捕获；本轮只支持delay3/horizon10/30x7，不能外推为动态延迟已接通。

## 预算和预先判据

一次 VLA 加载；288 真实双相机编码及正式完整请求；2次 Graph 捕获，各 setup1/warmup3/capture1，总10次额外 sampler调用独立计账。capture本身各含10投影；有前缀 Graph 必须完整记录10次原 VJP。Graph 正式 replay96（无前缀16/有前缀80），捕获重放的800 VJP与实际 Python 包装调用计数分开报告。无优化器更新、无参数梯度，推理期对输入求导不是训练。

技术门：三臂288请求的8项动态输入与各自原记录exact；48无前缀完整50x32与原输出exact；96组跨三臂的完整/归一化/后处理输出exact；每样本所有有前缀重复输出一致；不调整容差。前缀本身和设备缓存权重值exact，全部有限。每请求计时覆盖预处理、数据搬运、重新视觉编码、完整采样、后处理、同步、CPU证据复制；磁盘写入在外。校准48请求单独全报，第一次捕获及runtime初始化/释放成本保留；不能用稳态值冒充冷启动。

效率门：仅在技术门及独立审计通过时，graph 80请求P99满足ceil(20*P99)+1<=8（350ms）且全部80请求<=350ms，graph平均时延小于同期native_eager。其他两臂尾延迟原样报告，不要求未优化臂为了成为对照也满足实时门。nearest-rank 80个数的P99=最大值，不作总体尾概率保证。前缀贴合只作描述，不是专家动作、任务成功或部署资格。

模型阶段90秒；每输入阶段90秒；每请求30秒；使用原共享监督soft1500/hard1530/工具1620秒，attempt1/retry0。技术首错、非有限、来源改变、预算或输出exact失败即封存，不重跑该正式合同；正常成本失败全部保留。只终止本轮自有进程组，TERM后5秒才KILL。

## 冻结、登记和审计

新增3个实验源文件、测试和本计划，不改旧源码。保留原21份pending。普通提交/非强制推送至codex/smolvla-graph-native-equivalence，唯一CPU prepare绑定HEAD、源码/模型/环境/旧16例哈希及历史清单。Issue#1 E-RGC1-REGISTER:<HEAD>只读查重、单次结构化POST、真实ID独立GET/body exact后启动。普通网络错误核实后原样修正；明确安全/权限拒绝停止对应操作，不换通道规避。
退出后一次独立CPU审计：数据身份、输出exact、三臂顺序、分母、重复稳定性、捕获和VJP账目、调用与17阶段闭合、sampler/processor恢复、独立分位数及判读。审计不重新执行模型、编码、VJP或GPU计时。结果无论方向据实归档。成功只意味着该固定输入签名下 RTC 全路径计算候选可继续进入动态前缀接线与并发闭环，不意味着任务保持或原三阶段部署完成。

参考（只查阅文档，不安装/升级）：PyTorch CUDA semantics https://docs.pytorch.org/docs/main/notes/cuda.html；NVIDIA CUDA Graph integration https://docs.nvidia.com/dl-cuda-graph/torch-cuda-graph/torch-integration.html。固定缓冲与禁止捕获中主机同步是实现约束，不以文档取代本机测试。
