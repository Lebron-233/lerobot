# 冻结Graph异步候选：使用、证据与停止边界

2026-09-17。候选a5e9fddefc0cfa03ea1509493e8b304ec7c06d92（实现8112f8d7）；独立确认E-GFC1执行fc5da9d2634c7a43bba05802128efa92b81779b2。
状态：工程链与本轮线上时延通过，任务保持统计门未通过（串行32/32、异步31/32；回退概率95%上界16.67%>10%）。候选保持冻结，不提升为生产默认。

## 固定实现与参数

核心实现GraphFeedbackPredictor位于examples/advanced/predictive_async/libero_graph_feedback.py，复用FullPath/SmolVLAGraphRuntime和原InferenceOwner/control_loop/RTCExecutionQueue。具体源码及模型文件SHA256见SMOLVLA_GRAPH_CANDIDATE_LOCK.json。已有src/lerobot与实验候选源码不因确认结果而改变。
唯一验证硬件为本机RTX4070Ti SUPER、驱动580.178.04、专用libero-reference-venv及固定LIBERO资产。其他硬件/库/图像布局未获相同资格。
Graph、chunk50、denoising10、20Hz、cap8、margin1、guard2、threshold30、P90/window50、initial-delay7；RTC与学习视觉预测关闭。单owner、单在途，使用真实当前观测和专用CUDA Generator。

## 必须保持的生命周期

1. 用原native工厂创建/reset到登记初态，10次settling；重置policy/pre/post及队列身份。
2. owner线程首次bootstrap才创建Graph并完成捕获；记录setup/warmup/capture和完整启动成本，结果安装后才进入control t0。首次可能接近2秒，不能当成约84ms稳态请求。
3. 每个请求重新处理当前双相机/状态，生成新noise；所有动态Graph输入更新，返回独立CPU动作副本。控制中禁止额外capture，输入签名变化或runtime丢失应显式失败。
4. 请求提交与结果安装只在Env返回边界。新块row i对应请求观测索引+i；根据实际已消费行数trim一次。空等不推进动作索引，旧epoch/已关闭/超cap结果不安装。
5. episode结束先关闭队列阻止发布，再收回已提交请求；同owner恢复sampler并释放Graph，join后才能关闭Env。完成但未安装的请求照常计数和计时。

不要在原确认目录重新启动worker、覆盖结果、隐藏失败或丢弃慢请求。该目录的一次正式尝试已经消耗；没有隐式重试或后台工作。
原运行入口为libero_graph_confirmation.py，必须同时通过独立HEAD、prepare、真实预登记ID/body、原pending及来源哈希验证。结果提交之后HEAD不同，不应通过放宽校验强行重跑旧合同。

## 当前证据的位置

原件：outputs/smolvla_graph_confirmation_fc5da9d2。
完整64episode数据和每步数组保留本地。仓库docs/experiments下SMOLVLA_GRAPH_CONFIRMATION_RESULT.md/.json、AUDIT.json、DESCRIPTIVE.json、RECEIPT.json对应本轮报告、独立审计、事后诊断与退出记录。
失败案例7/19已是看过的确认数据，后继诊断必须明确标注；不能调参后重标独立验证。原先导8对不拼入32对。

## 已知未解决问题及后继约束

唯一回退并无超预算请求（该episode最长94.354ms）、underflow或过期安装；首次动作分歧在index20，request2输入/noise/full output在两臂相同，但串行等待、异步执行两步旧动作。这个起点不足以证明唯一因果机制。
后续可以另立不改权重的仿真反事实诊断研究等待/接管时序，但那是新合同，不是重跑本轮去找成功。任何改过调度策略或参数的候选都必须另版本、另未用确认队列；不以更换置信界、加旧先导或继续扩样挽救本轮。
这里未批准真机、用户环境长期无人看管运行或更改生产默认；没有硬实时和全任务泛化保证。明确权限/安全拒绝应停止对应动作，不用其他工具规避。原21份pending保留，不reset/clean/stash/强推。
