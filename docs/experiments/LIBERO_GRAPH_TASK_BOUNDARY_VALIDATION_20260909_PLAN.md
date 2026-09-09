# 全十步graph的十任务真实记录输入验证：2026-09-09固定执行协议

用户要求先收回已启动实验的真实结果并继续实验。本轮已只读闭合原90组匹配结果：
单步64/90，十步80/90；不重跑两者、不修改旧资格或确认。

本协议只执行已提交的任务边界修复，不修改模型数值代码。修复实现为bfaeb0de，
当前文档提交后的exact HEAD将在执行前Issue #1评论中登记。只有评论发布并回读正文
一致之后才执行首个模型调用。此前没有拿到发布ID的旧注册不视为本次启动授权证据。

## 唯一计算实验

入口：`examples/advanced/predictive_async/profile_libero_graph_recorded.py`。
新输出：`outputs/libero_graph_recorded_task_boundary_20260909`，必须不存在。
输入：已关闭的`outputs/libero_single_step_native_ee273bce`，原全部90条轨迹。
每条初始、floor(n/2)、末帧三个观测，总270条，按原task/state/stage顺序全部保留。
不新增原生任务；不重跑失败；不读取旧确认、SO101保留或匹配控制的轨迹图像。

保持checkpoint6721902b、VLM7b375e1b、原处理器/精度/图像和状态约定，50/1/10。
每次都重新编码当前两路图像，prefix及全部十步flow才进入graph。
不训练、不编译、不更改attention、TF32或去噪步数、不复用上次图像token。

唯一修复是明确task边界重新capture一次，不截断或填充语言、不改变原预处理。
每task记录八个输入shape、十次[1,50,32]投影、setup成本和实际graph/编码计数。
每task首先eager setup调用seed989900+task，再三次内部side-stream预热和一次capture；
第一task另有五对初始warmup（989990..989994）。正式seed990000+index；
eager/graph先后交替。所有270个输入重新验证，原5953905f的81通过+技术停止完整保留。

三项逐元素比较：完整[1,50,32]chunk、所选[1,7]归一化动作、反归一化动作。
必须finite、消费后动作队列为空，重建8D state等于保存state。
首个不相等或task内部shape不兼容立即停止，保存错误和此前结果，不放宽容差、不换输入。
保存全部配对NPZ、时序和结果JSON。300秒命令界限；工具会话在本次交互中收回退出状态。

主时序包含新noise、完整新视觉编码、输入copy、selector、post及CUDA完成，
不含selector前processor、磁盘读图、环境推进和单列task setup。
报告n/均值/P50/经验P95/范围、速度比、超过50ms个数和全部十次task setup。
预先按全270个成功完成且三项全部exact定义工程验证通过，不按速度筛选样本。

## 结论范围

这是记录输入上的计算等价性和时序验证，不生成闭环成功率；
不把同一批观测上的重复工程验证写成独立统计确认。
固定形状只在一个task内复用，task切换成本必须另计。

不自动开放新原生任务或预测器训练。
`baseline_qualified=false`，`realtime_qualified=false`，
`predictor_benefit_tested=false`，`risk_thresholds=null`。
