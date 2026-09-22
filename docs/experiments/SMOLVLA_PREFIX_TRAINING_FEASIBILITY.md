# F-PTF0：前缀条件训练的前置核查与精确接续

日期：2026-09-22。核查基线 HEAD `c6bc33ec9f6b057d44f16e6f1ca182b674abc9c9`。本次新增工作是一个可复跑的 CPU 能力探针与有界数据清点，不是新增模型训练、任务 rollout 或独立资格实验。

## 1. 先完成既有结果核验，不重复科学实验

接续时 E-RPI1 / E-RPF1 已完成：482 个动态接口请求、50 个真实反馈 episode，两个独立审计和结果发布均已存在。结果提交 c6bc33ec，实际结果评论 5770860427。核查了原 750 份来源文件、21 份旧 pending、原始评论回读正文及两个 worker 的退出状态；均与保存证据一致。本次没有重复 prepare、登记、模型调用、episode 或既有独立审计。

五臂成功数依次为串行9/10、原对齐异步9/10、承诺无引导8/10、承诺EXP7/10、承诺前缀限定7/10。前缀限定恢复7/18但丢失EXP成功的7/19，同时未完成0/10和2/11。原报告 SMOLVLA_RTC_PREFIX_SCOPE_RESULT.md 及全部不利结果保持，不能把当前推进路线写成已获得更好的部署候选。

## 2. 本次真实 CPU 探针

脚本：`examples/advanced/predictive_async/probe_smolvla_prefix_training.py`。
原始回执：`outputs/smolvla_prefix_training_feasibility_20260922/probe.json`。
执行 Job：`c6ebf53b-9ba3-44e5-982b-36eef4c9a363`，exit 0。Ruff 检查通过。

探针直接调用当前 VLAFlowMatching.embed_suffix，但仅提供微型随机 CPU Linear 投影，没有构造或加载预训练 VLA。scalar time [2] 正常返回 [2,50,8]，对合成动作输入的梯度有限且非零。共享时间编码函数已支持 per-action time [2,50] 并返回 [2,50,8]；然而当前 SmolVLA embed_suffix 随后的统一 unsqueeze 会把时间嵌入变成4维，再试图扩为3维，实际报错：

```
expand(torch.FloatTensor{[2, 1, 50, 8]}, size=[2, 50, 8]): the number of sizes provided (3) must be greater or equal to the number of dimensions in the tensor (4)
```

源码依据：`src/lerobot/policies/smolvla/modeling_smolvla.py` 的 embed_suffix（849行起，`time_emb[:, None, :].expand_as(action_emb)`）；共享helper在 `src/lerobot/policies/common/vla_utils.py` 41行起。

这是拟议训练时前缀条件接口的能力缺口，不是现有 guided RTC 或既有 Graph 路径的数值错误。没有修改原方法，也没有假装切换 `mode=trained` 就能解决。

另外在 C=0、3、8、49 四个固定合成案例核验：保留干净动作前缀，后缀按流匹配加噪，损失只落在未执行后缀的7个有效坐标；前缀和25个填充坐标的预测值梯度为0，后缀预测值梯度非零，分母分别为700、658、588、14。这只验证损失掩码算术，不是实际Transformer反传、训练收敛或任务效果证据，也没有检查完整真实episode的终端padding。

本探针预训练模型加载0、真实策略前向0、Env0、优化器更新0、网络请求0，CUDA未初始化；有微型合成投影调用和autograd检查，不能将它描述为完全没有张量计算。

## 3. 当前训练数据不能直接复用

本地 LIBERO 配置的 datasets 指向 `libero-reference-cache/no-datasets-downloaded`，该目录不存在。在专用实验缓存及标准Hugging Face缓存的有界扫描中，未找到当前LIBERO演示数据。扫描不覆盖整台机器或Local233，不声称其他位置也没有数据。

唯一读取到的演示元数据是 `lerobot/svla_so100_pickplace` 缓存：robot_type=so100，50 episodes、19631 frames、30 Hz，state/action都是6维关节坐标。这不是当前8维state、7维relative OSC、20 Hz的LIBERO实验数据；不得补一列零就当成同本体训练集，也不能用旧策略rollout伪装成专家演示。

公开 `lerobot/libero` 数据卡列出panda、8维state、7维action、双256x256相机，但其当前卡片元数据fps为10.0。它是下一步可核查的来源，不是已经确认的20 Hz训练集。下一次必须锁定仓库revision并核实实际转换/采样/控制步长、动作单位与归一化来源；不得擅自把10 Hz标为20 Hz或把元数据差别认定为旧失败的根因。本次未下载该语料或确认其物理控制频率。

## 4. 下一份实验应做什么

第一关是数据合同：选来源明确且许可适用的LIBERO演示，固定revision，核实相机键、state/action语义、时间步、预处理和基线检查点的归一化。按完整轨迹划分train/dev/sealed，不随机按帧泄漏；已用十个任务初态和旧32对不能重标为独立验证。新划分只声明相对本次微调独立，不擅自声称基座预训练从未见过这些演示。

第二关是独立、默认关闭的SmolVLA前缀训练接口：支持逐动作流时间，前缀time=0、干净动作作为条件，后缀加噪且只计算后缀损失；先证明C=0与原训练路径/输出一致，正确处理C边界、末尾padding、有效动作维度、梯度和checkpoint配置。保留旧控制器、Graph与所有冻结文件，不直接改生产默认。当前探针没有实现这一新训练接口。

第三关才是同预算普通微调与前缀条件微调的受控对照。输入为观测o_t以及在该观测之后已声明将执行的a[t:t+C]，目标是a[t+C:t+H]；线上前缀必须来自实际提交快照，不能读取未来演示或环境状态。干净演示前缀不能自动覆盖旧策略的错误前缀，必须报告这种分布差异，不能承诺仅行为克隆就恢复所有失败。

真正训练前另行固定可训练模块、C分布、更新数、数据/评价预算和接纳门。随后对新权重重新验证eager/Graph等价、完整并发请求预算与真实闭环任务保持；不能继承旧权重的等价资格。开发未通过就保留结果，不继续扫描到阳性；通过后才另立未使用身份的独立确认。

## 5. 官方资料和未完成范围

2026-09-22查阅：
- https://huggingface.co/docs/lerobot/main/rtc
- https://huggingface.co/docs/lerobot/main/en/pi05
- https://huggingface.co/datasets/lerobot/libero （Dataset Structure中的meta/info.json展示）

官方main文档的训练式RTC要求兼容、经过对应前缀训练的Pi05检查点；这不等于当前SmolVLA自动支持该模式。其训练范式可以参考，但新SmolVLA接口、真实数据获取/时间语义核验、微调、重新Graph验证和新闭环确认均未完成。读取数据卡两个raw/resolve JSON链接失败，频率信息来自成功读取的官方数据卡正文，并非已下载验证的快照。

本次GitHub连接器读取旧结果提交返回FORBIDDEN，没有重做该拒绝读取。既有结果已发布的事实来自本地真实发布回读证据；新CPU探针及其归档独立于该请求。用户必要发布授权沿用，但明确安全拒绝不绕过。未清理21份旧文件、未改变权重/原core/生产默认，没有后台训练。
