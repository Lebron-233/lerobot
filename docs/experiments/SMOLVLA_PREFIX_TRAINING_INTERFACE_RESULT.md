# F-PTI1：前缀条件训练接口与真实演示身份核验结果

2026-09-22。执行HEAD：`30e495c32fb49740bb487b90920d18e4f7656395`。预登记实际ID：`5771229301`。本轮没有重跑旧RTC/Graph闭环，而是按接续路线新增默认不接入生产的训练接口、真实演示核验与五次真实检查点前向/反向诊断。

## 核心结论

**训练接口的技术可行性已通过真实检查点验证。** 原生C=0和新接口C=0的完整输出、逐元素损失、标量损失、直接输出梯度，以及8个投影参数张量的梯度逐值相同。非零前缀C=3、C=8及终端窗口C=1均完成完整模型前向与autograd.grad。独立CPU审计通过，原模型权重、Graph、控制器和原时间对齐规则保持不变。

**公开演示数据不能只相信元数据文件名。** 锁定的HuggingFaceVLA/libero快照将目标episode814指向file037，但文件实际只有episode90..92；通过修正索引和实际Parquet footer定位file213，再核对两来源的整175行数值，取得身份正确的原始双相机演示片段。没有把错误数据重标为目标任务。

本轮仍没有训练出新策略，没有新增任务成功率或独立资格结果。物理控制步长与存档10FPS的转换链、全数据训练/开发/封存划分尚未完成，不能把接口通过写成数据与训练全部就绪。

## 1. 新接口实际改变了什么

`smolvla_prefix_training.py`提供独立函数，不修改原`src/lerobot`、sampler、Graph执行器或控制队列：

- 标量时间直接调用原`embed_suffix`；逐动作时间使用原动作投影、时间sin/cos与MLP，不再把[B,H,D]错误扩成[B,1,H,D]。
- 已承诺前缀使用干净动作、flow time=0；未执行后缀按原flow matching加噪。观测仍位于动作窗口起点，前缀是该观测之后承诺执行的动作，不是观测之前的历史。
- 仅合法后缀的7个实际动作坐标贡献损失；内部32维其余坐标以及episode终端padding均排除。终端padding还从attention keys中排除，窗口不跨episode。
- 每个样本必须至少一个合法后缀动作。前缀没有直接输出损失，不代表前缀embedding不能获得来自后缀的间接梯度。
- C=0且完整有效窗口保留原strided mean，确保标量和梯度的逐值等价。终端/非零C采用明确有效分母；未来普通微调与前缀微调必须共享该损失归约，而非混用不同padding分母。

初测16项CPU测试为15通过、1失败：elementwise已exact，但masked_select展平后求均值改变浮点归约顺序。修复为原归约，没有放宽C0容差。最终相关新旧CPU回归154通过、1个旧CUDA测试按条件跳过，Ruff通过。测试中的随机微型Transformer执行了合成优化器一步；这不是预训练SmolVLA权重更新，不计作正式训练。

## 2. 实际数据核验：发现错误索引并恢复正确身份

### 2.1 原始快照中的不一致

锁定：`HuggingFaceVLA/libero@86958911c0f959db2bbbdb107eb3e17c5f9c798e`。

元数据：episode814、task25、语言“pick up the milk and place it in the basket”，175行、全局索引154492:154667，标注位于`data/chunk-000/file-037.parquet`。

实际下载file037：103342449字节，SHA256 `053a7c57ce09d15802a7747503b4e8825d59871297d2cc3a2928176e78ee6654`。仅有761行，episode90/91/92分别223/258/280行，task8/7/5；没有814。这是本次实际文件核验，不是根据网页猜测。仅证明该锁定快照中的此项映射错误，不推断所有LIBERO版本都有该问题。

### 2.2 正确的定位与交叉核验

另一官方来源：`lerobot/libero@a1aaacb7f6cd6ee5fb43120f673cebb0cfea7dd4`，将814映射到file213。下载其数值文件54292字节并检查实际episode/frame/global/task，与元数据匹配。

对原始内嵌图像仓库进行有界HTTP Range footer二分查找：9次文件尾探针定位file213实际含episode813..817。随后只下载该分片102411765字节，SHA256 `badabde0650deb7b356aa99d5cf98cd62a5a3a354b874acb4a081a6c4669990d`。

原图像来源与修正数值来源，对814的全部175行在以下7个字段逐值相同：episode_index、task_index、frame_index、index、timestamp、observation.state、action。状态175×8、动作175×7且有限；帧0..174、全局154492..154666。使用原内嵌图像，直接解码frame20和frame172的两路256×256 RGB，没有二次翻转或重采样。

数据包：`outputs/smolvla_prefix_training_interface_20260922/demo_packet.npz`，SHA256 `b134a3efe14f2d745c4ed7dddb0319394d36e4ae0afba5e8a51825259b6d3276`。

这是一条演示的数值接口探针，不是完整训练集，不声明基础模型没见过该演示。公共数据仓库episode814也不能等同于本地仿真initial_state814或旧7/19身份。该演示已经用于开发，不应随后纳入声称未见的微调验证集。

### 2.3 时间语义仍需进一步核实

实际存档元数据FPS=10，时间戳相邻约0.1秒。公开openpi转换示例也写入fps10并逐条保留源episode步骤；但这个例子并不能证明上述快照的完整物理来源链。当前20Hz仿真控制频率未修改，也没有将数据标签改成20或插值动作。正式训练前需核对原始演示/转换与控制步长，而不是只根据FPS字符串认定兼容或不兼容。

## 3. 五次真实检查点前向与梯度检查

同一冻结SmolVLA严格加载一次。固定noise seed20260922、flow time0.5，使用原检查点pre/post处理器而非重拟合归一化。仅临时开启action_in_proj、action_time_mlp_in/out、action_out_proj的8个参数张量求导；通过完整冻结Transformer传播梯度，不等于验证了所有专家参数的训练优化器。

|固定案例|观测帧|C|有效动作行|计入损失坐标|实际标量loss|
|---|---:|---:|---:|---:|---:|
|原生C0|20|0|50|350|0.3913222253|
|新接口C0|20|0|50|350|0.3913222253|
|新接口C3|20|3|50|329|0.4097996056|
|新接口C8|20|8|50|294|0.5240527987|
|终端窗口C1|172|1|3|14|0.2013812214|

不同C对应不同条件和分母，上表不是训练前后改善或优劣排序。没有优化器更新，没有训练收敛曲线。

检查通过：
- 原生/接口C0的完整速度输出50×32、elementwise loss、标量loss、直接预测梯度，4项数组exact；8个参数梯度张量exact。
- 其他3个案例的模型梯度有限且8参数张量均非零；直接预测梯度在前缀、终端padding和无效坐标严格为零。
- 干净前缀和逐动作时间正确，terminal窗口只计2行×7坐标，不跨episode。
- 原处理器的动作归一化/逆变换在预定rtol=atol=1e-5通过；仅该roundtrip使用此容差，C0等价仍是exact。

正式进程exit0、五次前向started/returned=5/5、autograd.grad=5、attempt1/retry0，进程内wall约7.771秒。PyTorch CUDA峰值allocated为1557575680字节（约1.45GiB），不是整卡总占用或完整专家训练显存估计。没有调用sampler、Graph、Env或真机；这些训练forward时间不能作为异步推理时延。

## 4. 独立审计与完整性

退出后CPU独立审计exit0：重算5个case的mask、有效分母、loss和直接输出梯度算术，检查C0四数组与八参数梯度exact。独立NumPy归约使用预定2e-6算术容差，不改变两实现之间的exact要求。

审计没有重新运行Transformer或推导模型参数梯度，也没有重新编码视觉、仿真或测量GPU时间。参数梯度证据来自真实GPU执行并在CPU核对保存数组，不把它称为第二套自动微分实现。

33份本轮绑定的源码/数据/检查点来源最终哈希未变；另外回读数据合同中的4份原始数据文件哈希一致。21份原有pending保持，所有原冻结core保持。临时requires_grad恢复，8个投影参数前后哈希不变，预训练optimizer updates=0。已完成工具Job，不遗留训练任务。

## 5. 当前里程碑及下一关

已完成：独立前缀训练函数、逐动作时间接口、终端感知后缀损失、CPU回归、身份正确的真实演示、真实检查点前向/梯度与独立保存证据审计。

未完成：全训练数据时间/相机语义核验、按完整轨迹封存划分、普通与前缀条件同预算微调、新checkpoint的条件采样和Graph等价、真实异步任务保持及新独立确认。不能将本次接口通过升级为任务保持通过。旧所有阴性结果不改写。

具体接续见`SMOLVLA_PREFIX_TRAINING_INTERFACE_NEXT.md`。本报告生成时新结果提交/评论尚未发布，以原输出publication_receipt.json为实际发布状态。本轮没有新的安全权限拦截；原生Git提交/推送与必要新登记成功。普通格式/环境探测问题在冻结前修正并记录，没有改动依赖或为消除工作区风险提示而清理旧文件。

## 原始证据

- 数据与开发探针：`outputs/smolvla_prefix_training_interface_20260922`
- 正式检查与审计：`outputs/smolvla_prefix_training_real_30e495c3`
- 运行Job：`1662f75a-5c32-4785-954e-e563f2b2cb99`
- 预登记：Issue1 comment5771229301（单次POST，独立GET/body exact）

公开来源：
https://huggingface.co/datasets/HuggingFaceVLA/libero/tree/86958911c0f959db2bbbdb107eb3e17c5f9c798e
https://huggingface.co/datasets/lerobot/libero/tree/a1aaacb7f6cd6ee5fb43120f673cebb0cfea7dd4
https://raw.githubusercontent.com/Physical-Intelligence/openpi/main/examples/libero/convert_libero_data_to_lerobot.py
转换示例只用于理解存档字段，不作为本快照物理步长已证明的证据。
