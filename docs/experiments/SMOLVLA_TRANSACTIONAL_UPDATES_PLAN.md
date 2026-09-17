# F-TUA1：全训练集有限步接受与完整优化器回退

2026-09-17（Asia/Tokyo）。接续已审计F-UDI1结果5706792143；用户要求以完成SmolVLA异步推理为目标持续实验。
本轮直接检验实际有限步接受/回退是否保住已有候选收益，不再次盲调loss，不重跑旧实验。

## 固定设计

起点仍F-ACR1 centered第72步（69680参数、seed20260912），使用I+(h(a)-h(0))，幅度1、FP32先减后加再BF16。
保持BRP1的目标、训练参照、scales、样本权重、72例multi_conditioned顺序、batch1和AdamW设置：
lr1e-4/wd1e-4/betas0.9,0.999/eps1e-8/foreach=false/clip1。每例仅提出一次更新，总72次提案，不做回溯步长搜索。
拒绝是本轮算法中的事务回退，不是失败worker重试。拒绝时恢复提案前的完整模型state_dict与optimizer.state_dict，
包括moment、step及参数组；逐值核对恢复，清空梯度。下一样本仍按原顺序，优化器累计步数只累计被接受提案。

所有72训练样本/18episode（每episode4例）在每个候选提案后都做真实预测器+原Graph decoder前向，不提前短路。
每次计算原BRP完整训练目标均值、首动作7D MSE均值、完整50x7D chunk均值、相对Identity最大正超额首动作误差。
仅当目标严格下降，且其他三量相对当前保留模型均不增加（方向容差1e-7+1e-6*abs(control)），才接受。
其他三量还必须不劣于初始模型，防止多步累积容差。约束针对训练集聚合量，不保证每个样本不恶化或物理安全。
所有提议/接受/拒绝、全72例预测、参数和优化器状态都保存；不删除坏提案，不选训练外最好检查点。

16个反复使用的开发验证样本仅参与初始已知输出exact重放及全部提案结束后的评估，不参与接受规则、停止或选点。
原R1资格32样本、task8/9标签、confirmation均不读取。无新Env、图像编码、VLA训练、真机或部署。
原plain/guarded/BRP为已审计历史对照，不冒称同期重训。oracle仍为future视觉+当前state+同language/noise冻结策略输出，不是专家。

## 评估与推进

初始88例各重放Identity、旧B+delta、冻结I+delta完整50x32输出exact；所有初始残差raw/BF16也exact。
每次提案训练前走原eager可微decoder；候选训练扫描与最终评估使用同一原Graph路径。
最终88例保持true/zero/mismatched，训练69、验证16可错配，原三单例保留true/zero；88份zero raw及完整输出exact回I。
最终训练输出必须逐值等于最后接受的训练扫描输出（零接受时等于初始）。不把不同参数时刻的提案均值合成模型性能。

开发推进沿用BRP全12项条件，另要求非零接受：验证首动作优于I/自身错配且不劣冻结IAR；
验证chunk优于冻结IAR/guarded/plain；训练最大正超额优于冻结/guarded/plain；相对I至少3/4episode、超过8/16样本改善；
验证最大正超额不劣冻结。复用旧判据算术但输出候选名称统一为transaction，历史BRP单独命名brp，不混淆。
训练门通过是规则约束的直接结果，不单独作为科学成功。有效且非退化的回退机制与最终验证结果分开报告。
即使开发推进通过也不是新独立资格，后续仍需新初态、完整预测器路径时延和受控闭环。

## 冻结预算与退出

VLA1、预测器实例1；72次AdamW提案、72次反传，实际保留更新0..72；拒绝不得把已消耗前向/反传记为0。
正式decoder5781=264初始+72训练+5184全训练扫描+261最终；predictor11210=176+144+10368+522。
Graph capture<=448（初始8+72*6+最终8），每次内部setup/warmup/capture=1/3/1另计；内部调用不混入正式decoder。
模型90秒、初始180秒、72提案阶段1200秒、单提案60秒、最终180秒、单decoder30秒；soft1500/hard1530、工具1620秒。
仅自有worker组TERM后5秒才KILL；attempt1/retry0，无自动第二worker、修改判据重跑或扩大训练预算。
继承固定offline/EGL/LIBERO_CONFIG_PATH/LD_PRELOAD环境、移除PYTHONPATH，固定uv与Python；不改依赖/驱动/配置。

代码、计划、合成CPU回归测试提交推送到原分支后绑定真实HEAD，新PREP/OUT按HEAD唯一命名。
21份原pending/阴性文档完整保留并绑定哈希；新准备只执行一次。先只读查F-TUA1-REGISTER:<HEAD>，不存在才
结构化gh literal argv单次POST，真实ID独立GET正文exact、保存实际规范化返回和来源后监督运行。
未知POST状态不重发；明确安全/权限拒绝停止对应动作，不换通道绕过。普通错误据实处理。

退出后一次CPU/NumPy审计全部控制/5184扫描/最终指标、目标、提案门、梯度裁剪和AdamW算术、接受或拒绝时
完整参数/moment/step状态、最终状态及5857个阶段闭合。科学rtol1e-6/atol1e-7；优化器rtol1e-5/atol1e-7；exact不放宽。
审计不重新求导或执行模型，不能声称独立验证梯度真实性。所有原始负结果不覆盖。
技术失败只本地报告；有意义的已审计机制/验证发现可向Issue反馈，必须同时写出未通过条件和限制，不称部署成功。
本合同无自动追加训练、新资格采集、时延或闭环；所有生产资格保持false，risk_thresholds=null。

API实现依据：PyTorch官方AdamW文档明确optimizer.state_dict不包含模型参数，load_state_dict按参数顺序关联状态；因此两者分别保存和恢复。
https://docs.pytorch.org/docs/main/generated/torch.optim.AdamW.html
