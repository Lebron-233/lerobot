# E-GCR1：完整输出不变，固定原生并发负载下Graph通过时延预算

2026-09-17。execution HEAD：b293f66133429969984efc39f3c4f49eae9ecf92。
实际预登记：https://github.com/Lebron-233/lerobot/issues/1#issuecomment-5711657045 。
原件：outputs/smolvla_graph_replay_b293f661；准备：outputs/smolvla_graph_replay_preparation_b293f661。
preparation SHA256：9e5a77f7357a540918729aee5791675b5f77420e4db6bca902527b93879a2f34。
result.json SHA256：33a3e45005118ce6e587c540fa6e0cfdfa516a995587abba2bfbe0f16468a3dd。

## 结论

status=completed；independent_contract_accepted=true；fixed_output_exact=true；graph_concurrent_budget_passed=true。
全部206个完整50x32模型输出逐值等于各自原存档，原生重放142个后处理50x7输出亦exact。Graph在真实native/渲染并发负载下P99=122.374ms，eager=400.613ms，前者满足原350ms上界，后者不满足。两臂同模型、输入、噪声、完整路径与固定原生命令，不用跨轮延迟相减作为主要证据。
这是保持输出的执行优化与受控并发预算成果，不是新策略闭环成功率、任务非劣、总体尾部估计或部署完成。realtime_qualified仍false。

## 设计与分母

先使用原16开发样本，每例每臂2次，64请求。每4例独立Graph生命周期，初次捕获留在repeat0，repeat1每臂16次仅用于进入并发阶段的前置预算检查；64份完整输入token/mask/语言/state/noise和输出均exact，不放宽容差。
然后重放E-OBS1全部8条async轨迹，每条eager/graph各一次，共16Env实例。成功和失败轨迹均包含，同初始化、种子、同原始1410命令各重放两遍，总2820动作；模型每次用相同来源观测和noise，不接管Env。每臂71请求，bootstrap8、非bootstrap63。新建Env不等于新独立任务样本。
固定原生动作以20Hz推进，模型在原请求观测索引发起，只有一个owner与一个在途请求；图捕获完成后才开始原生measurement。没有减少十步解码、改变50行动作块、引入RTC/预测器或更改权重。两臂使用同公开API与同SmolVLAGraphRuntime的模式开关。

## 实际完整请求时延

| 并发负载非bootstrap指标 | eager | Graph |
|---|---:|---:|
| 请求数 | 63 | 63 |
| 平均/ms | 339.193 | 93.463 |
| P50/ms | 336.453 | 93.375 |
| P95/ms | 372.124 | 109.506 |
| P99及最大值/ms | 400.613 | 122.374 |
| ceil(P99秒*20)+1 | 10 | 4 |

平均耗时下降72.4454%，P99下降69.4533%；相应耗时比3.6292和3.2737。相同63个输入配对全部Graph更快。该配对统计是正式审计后的保存记录描述，不是新增判据。Graph距350ms上界约227.626ms。
原预算保持20Hz/cap8/一行余量，没有删慢请求、降频、扩大cap或等待低负载重跑。63个观测的nearest-rank P99就是最大值；请求来自8个轨迹，不是63个独立场景。

固定输入阶段repeat1的每臂16次：eager平均276.117ms/P99=291.047ms，Graph平均69.453ms/P99=70.834ms。并发确实增加了Graph耗时，因此以并发122.374ms而不是离线70.834ms判断下一步预算。
模型/native真实区间交集：eager450、Graph143；这证明两臂都有实际并发。Graph调用更短自然可能对应更少交集，交集数不是独立任务数或“重叠越多越好”。
计时涵盖预处理/输入搬运、每次真实双相机编码、prefill、原十步解码、后处理、GPU同步和CPU证据复制；磁盘保存不在计时内。未测真实线上队列等待。不同原生命令的执行时隙和GPU调度不可能逐纳秒相同；这里控制的是初始化、命令序列、模型输入和固定20Hz规则。

## 必须保留的代价：首次Graph调用更慢

| 原生重放bootstrap/每臂8次 | eager | Graph |
|---|---:|---:|
| 平均完整时延/s | 0.276313 | 1.375259 |
| 最大/s | 0.294062 | 1.403783 |

首次Graph请求含捕获，明显比eager慢，不能将稳态数字写成冷启动数字。12次总捕获准备耗时15.771194秒，最大单次1.382794秒；包含固定阶段4次及重放阶段8次。每次setup1/warmup3/capture1，总12/36/12内部sampler调用（60次）独立计账，不冒充正式206次之一或计为零。
本轮所有捕获都发生在各自native measurement之前。后续真实闭环必须在ready前完成捕获并显式记录startup；控制过程中不得因形状变化静默重捕获。

## 独立审计与真实消耗

| 项目 | 实际 |
|---|---:|
| VLA加载/新训练/RTC/预测器 | 1 / 0 / 0 / 0 |
| 正式完整请求与真实双相机编码 | 206 / 206 |
| 完整50x32输出exact | 206 |
| 重放后处理50x7输出exact | 142 |
| 原生Env重放/初态exact | 16 / 16 |
| measurement/settling | 2820 / 160 |
| Graph捕获 | 12 |
| owner/runtime释放与sampler恢复 | 20 / 20 |
| 独立关闭调用 | 6118 |
| phase started/returned | 223 / 223 |

固定请求64加原生重放142；底层native共2980，外层2820个environment_step只是相同measurement的包装，不重复算动作。
独立CPU审计Job200bd776-d349-4736-9d86-4c03e50d132a exit0；核验来源输出/动态输入、原生命令、初态、采样次序、单owner/单在途、Graph成本与退出、独立分位数和预算。CPU审计不重新运行模型、视觉编码、后处理或物理仿真；也不独立测量GPU时间。固定64份token输入有原存档逐值对照，重放142份不在CPU重新从RGB编码，而以相同原图/noise路径和完整原输出一致性约束。
正式Job2e7ab876-4067-490e-9b9c-0d82230a7a12，worker718733、外层及审计均exit0，worker已收回。监督UTC08:53:36.373179至08:57:09.330786，wall212.957628秒。first_failure/stop_reason为空，pending/active空，无强制终止，attempt1/retry0。
48项CPU回归（3.96秒）、Ruff与暂存diff检查通过；725份来源/代码哈希与原21份pending状态/字节保持。正式运行中无源码修改，无补样或参数搜索。本轮工具调用没有新的安全/权限拒绝。

## 面向最终目标的接续

本轮补上“相同输出且有真实并发负载下的时延预算”证据，支持将Graph接入观测原点共享owner/controller，再另立新的任务保持试验。不得用原存档动作重放的结果代替模型真正控制Env的成功率。
Graph虽不改相同输入的输出，但完成更早会改变实际接管时刻、被选中的动作行和下一次观测；不能据此保证任务轨迹或成败不变。E-OBS1串行7/8、异步6/8及其不利配对仍有效，没有被本轮“修复”。
新任务试验的准确身份、次序、预算、判据须单独冻结并登记，不能把本次重放或旧pilot重标独立确认。旧IQ1/ACQ资格继续封存；guided RTC加速不属于本结果，本轮无新策略闭环或真机。
结果与计划正常推送后，实际结果评论ID另记出版回执，不提前虚构。
