# F-DTC1：演示原始行与物理控制时间的有界诊断

接续7f12fa6d，2026-09-22。保留原SmolVLA/Graph/控制器及全部阴性。此次前置实验不训练、不生成策略动作，不重跑旧模型实验。

## 来源与已知差异

固定上游openvla/modified_libero_rlds@6ce6aaaaabdbe590b1eef5cd29c0d33f14a08551的object首4分片作为有界数据候选。首分片record7与上一轮转换后episode814的175行state/action逐值相同，双相机decoded RGB不exact，最大差17/20。不得声称JPEG差异已经证明无影响。

clip-rt/modified_libero_hdf5@6a6659f8ac7d580fd594173a0e3abf880c843130的milk demo_32所有175行动作float32后与RLDS相同，首25行state相同，之后最大state差约0.004132。模拟states[:,0]在第1行之后连续差0.05秒；第0行是原始初始化state，不能将其到第1行0.55秒差当作控制周期。该镜像不是RLDS所有状态逐值相同的替代品。

转换脚本固定fps10但逐行写入；上游再生成代码逐保留动作step且默认控制20Hz。F-DTC1用实际回放检验两种时间解释，不改存档timestamp，不作插值、动作重复或重新标注。

## 固定两臂

相同milk demo_32的原始full simulator state初始化，seed0，固定10次初始化空动作。分别控制频率20和10Hz，每臂按顺序执行同一175行float32 RLDS动作一次；即使原生success提前出现仍完成此数据回放，不作为策略episode成功率。每行记录动作前8维状态和sim.time，终局单独记录。50Hz? 不允许；不搜索其他频率、种子或初始化方式。2Env，20settling，350measurement动作，无模型/优化器/Graph捕获。

固定比较：来源命令必须逐值一致；每臂实际sim.time差应为1/hz（atol1e-9）；比较175行state MSE和末端位置RMSE，保留两者全部不利结果。20Hz比10Hz更贴近数据是数据时间诊断，不是新策略加速或总体频率结论；不同机器物理/图像差异不放宽为exact。

独立CPU审计重新从保存数组计算动作匹配、时钟差、状态误差与退出，不运行仿真。若任何来源/退出异常或20Hz没有更符合原始状态，数据物理时间合同保持未接纳，不进入训练。即便此诊断支持20Hz，也只证明这条相同命令示例与上游生成规则的一致性，四任务整套数据完整性/划分及同预算训练仍是单独步骤。

## 数据准备与完整性

首4分片固定、CRC32C逐记录核验。只选本地task0/2/6/7对应alphabet soup/salad dressing/butter/milk语言，不混用转换后数字ID。完整轨迹以来源路径+state/action内容哈希标识；排除已见175行milk接口诊断。每任务哈希排序首条dev、第二条sealed、其余train，少于3轨迹即停止不追加分片；sealed仅结构与哈希核验，不解码其图像或算模型指标。该小数据划分仅相对本次微调，不宣称基座未见。

代码冻结、一次prepare、唯一预登记及真实ID/正文回读后才运行2Env；attempt1/retry0，总180秒，工具240秒，只运行自有进程；失败保存不覆盖不换种子。旧21份pending保持。明确安全拒绝不绕过；普通网络失败先核实状态。无生产默认修改、真机或无限后台任务。
