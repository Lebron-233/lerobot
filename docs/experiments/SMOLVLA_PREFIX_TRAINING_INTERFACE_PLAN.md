# F-PTI1：独立前缀训练接口与真实检查点梯度可行性

接续 dfc7ccec，2026-09-22。只修复训练前置接口和数据合同；不扫描RTC权重、不更改原Graph/控制器/起始行、不改变生产默认。

## 已确定的开发检查

smolvla_prefix_training.py新增默认不接入生产的函数。标量时间调用原embed_suffix；逐动作时间使用原MLP与共享sin/cos，前缀time=0、后缀原flow time。合法动作坐标7/内部32；末尾右padding从attention keys和损失排除；每例必须至少一个合法后缀。前缀不贡献直接预测损失，但仍作为上下文影响后缀。C=0、无终端padding保留原MSE的strided reduction，不以改变容差掩盖浮点求和顺序差异。所有未来比较臂必须共享掩码与损失分母。

CPU测试：原标量/均匀逐动作时间embedding、C=0完整elementwise与所有微型模型参数梯度exact；C=0/1/3/8/49，混合C、终端padding、不跨episode、空后缀拒绝、前缀上下文依赖，以及一次微型随机Transformer的合成更新。该更新不属于预训练SmolVLA训练。

## 数据诊断与边界

实际下载锁定HuggingFaceVLA/libero@86958911c0f959db2bbbdb107eb3e17c5f9c798e。其meta把episode814/milk指向file037，但实际文件含90..92且仅761行，不能直接训练。另一官方lerobot/libero@a1aaacb7f6cd6ee5fb43120f673cebb0cfea7dd4将814正确指向file213；通过原始图像分片footer二分定位独立核实，不重写上游元数据。prepare_prefix_training_demo.py要求两来源的episode/frame/index/task/timestamp/state/action整175行exact后才导出两个图像anchor(20,172)。不根据终局结果选样。

数据只用于数值接口诊断。10FPS存档时间与20Hz物理控制步长的完整转换链、全数据训练/开发/封存划分尚未建立；不得宣称已具备正式任务微调数据。只导出同episode行序，不重采样或将FPS改写为20。原图像像素直接解码，不再次翻转。使用基线检查点原preprocessor归一化，不拟合新统计。没有未来环境观测或旧策略自蒸馏标签。

## 正式真实检查点检查（一个新登记、一次attempt/retry0）

一个冻结checkpoint加载。使用真实episode814的anchor20，完整50行窗口；固定noise seed=20260922，flow time=0.5。执行原生训练forward(C0)及新增接口C0/C3/C8；再以anchor172的终端3行窗口、C1执行一次。共5个完整视觉/Transformer训练forward和5次autograd.grad，不调用sampler或Graph，不运行Env，不作优化器更新，不保存新checkpoint。

只临时开启action_in_proj、action_time_mlp_in/out、action_out_proj这四模块的8个参数tensor求导，用完整冻结Transformer传播梯度；不声称已经验证全专家参数训练。原生C0与接口C0的所有elementwise输出、标量loss和8个参数梯度均必须exact；C3/C8/terminal的有效梯度有限且非零，直接预测值梯度在前缀/终端/无效坐标为零。保留输出、梯度、输入、mask和参数摘要供退出后CPU审计。预处理逆变换与原动作在atol=rtol=1e-5核验，只此roundtrip使用该容差；C0不放宽。

模型加载最多90秒，每次forward/grad最多60秒，总alarm300秒，工具360秒；CUDA峰值内存单报。技术首错立即停止并封存，不追加重跑；恢复临时requires_grad标志。只读核验权重文件与原core哈希，旧21pending不变。登记须绑定执行HEAD、接口/运行脚本和data contract/packet哈希，真实comment ID独立GET/body exact后才加载模型。必要提交/推送/Issue1发布沿用已有用户授权。

退出后CPU独立核查保存数组、C0 exact、掩码/分母、5次调用和退出记录，不重新运行模型或物理仿真。完成只支持训练接口技术可行性，不支持收敛、成功率、实时部署或20Hz数据资格。下一步仍须补齐完整演示时间语义及按轨迹封存划分，才能预登记同预算普通微调与前缀条件微调。
