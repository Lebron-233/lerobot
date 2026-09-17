# E-GCR1：相同输入与相同原生负载下的Graph全路径核验

2026-09-17。接续E-OBS1/568b494b：等待消除，但异步6/8成功对串行7/8，在线P99=375.470ms。不能把不同轨迹的耗时差异归为Graph加速。本轮暂停新任务资格，仅验证保持算法的执行优化及并发预算。

## 范围与固定次序

不修改VLA、RTC公式、SmolVLAGraphRuntime或旧E-OBS控制器。使用同一公开predict_action_chunk及同一Graph runtime的eager/graph模式，原50行动作块、10步、20Hz、cap8/一行动作余量，无RTC/未来预测器/训练。
先用FULLPATH1准备存档中的16个旧开发输入（task6/7,state48/49,request3..6），每例2臂2次，64完整请求。每4例独立生命周期，Graph首次捕获成本计入repeat0，repeat1仅作进入并发负载的前置检查。每一请求的全部输入、完整50x32输出必须逐值等于原存档；不只看前7维，不放宽exact。
两臂通过同一个已安装runtime切换模式，不嵌套sampler。每请求重新从原图编码、刷新所有静态输入，保留CPU独立输出。4次捕获，每次setup1/warmup3/capture1，20个内部完整sampler调用另列。

若固定输入Graph P99按ceil(20*P99)+1<=8通过，才进行原生负载重放；否则64次后结束并封存，不进入Env。

## 相同原生负载的并发测量

来源为E-OBS1全部8条async轨迹，不按成功/失败挑选。原动作总1410、模型请求71（其中bootstrap8，非bootstrap63）。每条分别eager/graph重放，两臂次序按原pair_index奇偶交替，16个新建Env实例但全部是已使用身份的重放，不是新的独立策略rollout。
Env与来源的task/state/seed/relative OSC/256双相机/10settling一致；复核原始初态checkpoint exact。每20Hz时隙最多重放一个原命令，固定动作数和原请求观测索引；不让新模型输出控制Env，不按当前结果改变命令。实际相机/物理推进用于构造真实并发负载，但模型输入使用原存档观测和同noise，故可逐值比较所有输出而不混入新轨迹差异。
全部142请求均重新预处理、双相机编码、解码和后处理；完整50x32和后处理50x7输出exact对原71份输出。bootstrap完成、Graph捕获结束后才开始native measurement，不在并发native推进中捕获。
单model owner线程，单在途；下一个固定观测索引到达时上一个请求仍未结束则首错停止，不排队改变固定负载。关闭在owner线程释放Graph并恢复sampler，join后才关闭Env。固定早终止差异、输入/输出/来源/预算/超时均停止，不重试、不补轨迹。
16次重放总2820 measurement/160 settling；最多16*280 measurement预算保护。8个Graph bootstrap捕获，40内部sampler调用另列。无新成功率、任务非劣、策略修复、环境反馈控制或部署结论。

## 测量和判据

206正式完整请求=64固定+142原生重放。每请求计时涵盖原图预处理/搬运、实际视觉编码、prefix prefill、十步动作解码、动作后处理、GPU同步和CPU证据复制；磁盘存档在计时外。模型区间单独用于与真实native.step区间求交。排队不混入算子时间；此受控重放不是在线动作队列性能证据。
并发时延报告每臂全部63个非bootstrap请求，bootstrap16个全部单列，不删慢请求。nearest-rank P50/P95/P99（63下P99等于最大值），配对相同输入，报告两臂与Graph/eager比值；不能称63个独立场景。
Graph并发预算仍为ceil(P99秒*20)+1<=8（350ms）。技术exact/退出通过、真实并发区间成立、并发Graph预算通过，才支持随后另立新配对任务保持合同；本轮不自动读取新确认标签。任务保持不会由等价或时延直接推断。

## 准备、审计和停止

已有21份pending与所有旧阴性保留，IQ1/ACQ资格不读取。源码/测试/计划先正常提交推送，CPU prepare绑定HEAD、全部相关源码、权重/配置和原始来源SHA256、准备data.pt及原专用环境。唯一输出按HEAD命名。
E-GCR1-REGISTER:<HEAD>在Issue #1单次GET查重，单次gh literal argv POST、实际ID独立GET/body exact落地后启动监督入口。普通接口错误可透明修复；明确权限/安全拒绝停止对应动作，不改工具绕过。禁止reset/clean/stash/force。
模型90秒、每模型请求15秒、native30秒、每重放episode100秒，总soft1200/hard1230/工具1320秒；attempt1/retry0。仅对自有worker组TERM后等5秒才KILL，结果和阶段账本不可覆盖。
退出后一次独立CPU审计所有206请求、来源输入/输出、相同原生命令、初态、capture账目、owner/Graph释放、并发时间区间、调用退出与独立分位数。完整执行应223个phase闭合、12capture及60内部sampler另计。CPU审计不重新编码、不执行模型或Env，不能称独立验证GPU运算本身。
PyTorch官方CUDA语义仅用于核对静态地址更新与捕获约束：https://docs.pytorch.org/docs/main/notes/cuda.html 。不升级本地依赖或下载执行外部代码。
