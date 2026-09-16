# F-ACQ1-R1：仅修复启动环境的单次恢复执行

2026-09-17（Asia/Tokyo）。用户在收到 F-ACQ1 真实失败回执后要求继续实验。
本增补只授权一次独立登记的环境恢复执行；不是原 attempt1 的重启、覆写或隐性重试。

## 旧记录和范围

旧 execution HEAD `c1d1f65bfe3a102510529f0c500c58c046961665`、预登记5701187626，
输出 `outputs/smolvla_action_qualification_c1d1f65b` 封存不变。
旧 result.json SHA256 `ee396d6c2f49d04fbc7247ffeecb3d26708c939cffd8f70ca61c7d0c4b16acad`。
其状态 technical_failure，首错 LIBERO 首次配置 input 的 EOFError；16/16旧锚点通过，
48 decoder、64 predictor、VLA1、预测器3、offline capture2，内部setup/warmup/capture2/6/2。
旧新Env/采样/训练均0；没有episode目录、aligned_cache或新指标，worker已收回。
系列层面已消耗1次正式尝试；R1另有1次机会，成功或失败后系列总尝试为2、环境恢复执行为1。
不得把系列总尝试仍报告为1；每个独立输出保留自己的attempt1/retry0。

R1 的科学问题、模型权重、采集身份/种子、采集与选样次序、供体、oracle、精度、
三项指标、主门、稳健门、预算、停止条件均沿用 SMOLVLA_ACTION_QUALIFICATION_PLAN.md。
这包括8个原固定初态和N≤32，不更换或追加样本。旧失败没有消费这些初态，
但新CPU准备仍必须复核本仓库outputs的完整身份索引；有冲突即停，不换state。
不读task8/9标签或confirmation21–40，不训练、不选点、不改残差、不做闭环/时延实验。

## 唯一代码修复

恢复原 LIBERO_REFERENCE_PREPARATION_SUMMARY.json 的已有进程环境：

```text
删除继承的 PYTHONPATH
HF_HUB_OFFLINE=1
TRANSFORMERS_OFFLINE=1
MUJOCO_GL=egl
PYOPENGL_PLATFORM=egl
LIBERO_CONFIG_PATH=/home/rp/Workspace/SmolVLA_RTC/libero-reference-cache/config
LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libGLdispatch.so.0:/usr/lib/x86_64-linux-gnu/libGLX.so.0
```

采用 WebCodex run_process executable=/usr/bin/env 和literal argv传递上述设置，
再调用原uv与固定Python；不使用shell包装、不改全局环境/配置/依赖/驱动。
CPU准备、登记验证、worker均检查这些值、专用config.yaml SHA256
`1acadaf137bd6e0e9cc1c5130d4904f418f336aa912c417a74aa35be9335dc8b`、已有GL库，
以及package-local assets链接指向原冻结snapshot。不自动新建 ~/.libero 配置。

原 native factory 构建移动到worker模型加载前的environment_preflight阶段，限30秒；
仅构建原工厂/任务清单，不创建Env，不编码，不执行策略，不读取初态张量或标签。
该阶段要求CUDA未初始化；随后仍先重放原16锚点再创建8个identity Env。
新阶段计入原soft1500/hard1530秒，不扩展其他原有阶段预算。
CPU准备保存环境证据，正式worker保存同样证据，独立CPU审计核对两者与文件哈希。
不改任何预测、采集、对齐、损失或判门算法。

## 执行次序

1. 阅读原失败回执；保护旧9份pending及4份失败报告字节。仅显式提交4份失败报告、
   本增补、最小源代码/审计修改及CPU回归测试，不提交tensor/权重/凭据，不全量git add。
2. 固定uv `run --offline --no-project --no-python-downloads --python <PY> python -m pytest`
   执行原qualification/centered及新增environment测试；Ruff独立记录退出码。
   在正确env下做一次真实CPU配置及原factory构建检查，Env/CUDA/模型前向均0；首错停止。
3. 修复提交推送后绑定实际execution HEAD。R1仍使用原按HEAD分目录的入口，
   因新HEAD产生全新PREP和OUT；仅对该新HEAD执行一次CPU --prepare，绝不重跑旧prepare。
4. 预登记标识 `F-ACQ1-R1-REGISTER:<execution_HEAD>`，同时含 `F-ACQ1-REGISTER:<execution_HEAD>`
   兼容原入口身份检查。正文必须包含本增补、旧失败/系列尝试数、实际HEAD、唯一OUT、
   新preparation SHA256、原固定manifest/权重来源/预算、完整启动环境。
   先单次只读GET查新标识；不存在才用gh literal argv独立POST一次；返回实际ID后独立GET。
   保存投影JSON及来源回执，正文exact且源/历史/环境不变后才能启动。
5. 单次正式supervise入口，不直接调用--worker。外层WebCodex超时1620秒，原内部监督不变；
   跟踪同一Job到终态，不把timeout当无效果或另启替代worker。失败也保存首错和调用闭合。
6. 退出后一次独立CPU审计，再一次生成完整报告。R1报告以独立命名
   SMOLVLA_ACTION_QUALIFICATION_R1_RESULT.md/.json、R1_AUDIT.json、R1_RECEIPT.md保存，
   不覆盖旧4份报告。保留逐样本不利结果、真实预算/Graph内部次数/phase/退出和系列总消耗。

本轮仍遵守此前用户的反馈渠道限制：必要预登记可发布；实验未成功时不发布结果评论，
在本地保留完整失败/阴性证据并在对话汇报；达到成功条件后才提交推送R1结果并回复Issue #1。
预登记必须披露原失败，不能把失败包装为从未尝试。无论阳性/阴性/技术失败均向用户如实报告。
明确安全或权限拒绝停止对应动作，不换通道绕过；普通命令参数错误可做透明的最小修正。
R1正式worker首错后不再自动重启或新建R2；不追加训练、时延或闭环。
生产默认、baseline_qualified/realtime_qualified/predictor_benefit_tested仍false，risk_thresholds=null。
