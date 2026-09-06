# M5.4-L1：LeIsaac SO101 最小适配计划与源码裁决

日期：2026-09-06。状态：**规划完成；原样同进程直连 NO-GO；建议采用明确标识的 SO101 transfer profile + 本机双进程最小桥。适配尚未实现，环境／模型／闭环均未执行。**

## 1. 本轮决定

用户已选择已有 LeIsaac SO101，并要求本次继续工作不再交另一个 Pro 对话审核，关键问题和提交仍在 GitHub 留痕。本轮在本会话完成源码调查和技术判断；不再等待旧侧栏裁决。恢复记录见 [Issue #1 的本轮范围](https://github.com/Lebron-233/lerobot/issues/1#issuecomment-5560402567)。

交接授权的实质工作是最小适配规划。它不自动授权安装、环境启动、权重加载、闭环运行、训练或改换 checkpoint。本计划将实施和运行所需的新增范围具体化，不将它们写成已执行。

**判断不是“SO101 没有环境”，而是“不能原样复用冻结 SO100 的部署合同”。** 已有 PickOrange 的关节、RGB、任务终止接口足以设计有界适配，但至少存在三项需要明示的变化：

1. SO101 的模拟关节坐标、gripper 与 front 相机，不等于冻结 SO100 的机械标定和 top 视角。转换可定义，跨 embodiment 等价性没有证据。
2. LeIsaac 核实版本依赖 Isaac Lab 2.3.0；其 Isaac Sim 5.1 部署要求 Python 3.11，而本 LeRobot checkout 要求 Python >=3.12。所核实组合不能按包声明装进同一解释器。
3. 当前 production rollout 没有 Gym episode/outcome 桥；直接运行同步 eval 不能建立 pending inference 期间旧队列驱动环境的证据。

最窄后续方案：**不换模型、不训练，明确接受一次 SO100 权重到 SO101 模拟任务的迁移评测；模拟器使用独立 Python 3.11 进程，当前 Python 3.12 进程继续独占 production engine 和唯一动作队列。** 这需要新增部署范围，不是现有有限 runtime GO 的自动延伸。

## 2. 绑定的源码与证据

本轮开始时，工作区干净，本地 HEAD 与 `git ls-remote` 远端分支均为 `0702d2d58c3d085bbedc2371d9822f86bf35df4f`。本提交仅增加规划文档；下列 LeRobot 源码证据均绑定这一代码版本。

| 对象 | 本轮实际核实的版本／入口 |
| --- | --- |
| LeRobot | `Lebron-233/lerobot@0702d2d58c3d085bbedc2371d9822f86bf35df4f`，分支 `codex/smolvla-future-latent-m3` |
| LeIsaac | `LightwheelAI/leisaac@24d3bcd3f1e4585740fc79921782c41617237812`，本轮 GitHub main 查询结果 |
| EnvHub 入口／资源仓库 | `LightwheelAI/leisaac_env@6c35af0af55506eb75c5592930134d4af44e8341`；Hub API 本轮返回同一 revision |
| Isaac Lab | LeIsaac extra 声明 `isaaclab[isaacsim,all]==2.3.0`；tag `v2.3.0` 指向 `3c6e67bb5c7ada942a6d1884ab69338f57596f77` |
| 首个规划任务 | `LeIsaac-SO101-PickOrange-v0`，管理器环境；是本轮技术提案，不是用户指定的唯一任务 |

必要源码入口：

- [Hub PickOrange loader](https://huggingface.co/LightwheelAI/leisaac_env/blob/6c35af0af55506eb75c5592930134d4af44e8341/envs/so101_pick_orange.py)：调用未固定 revision 的 `snapshot_download`，设置资源目录，启动 `AppLauncher`，硬编码 `cuda:0`，再调用 `export_env`。本轮未执行这一入口。
- [export_env](https://github.com/LightwheelAI/leisaac/blob/24d3bcd3f1e4585740fc79921782c41617237812/source/leisaac/leisaac/utils/envhub_utils.py)：`parse_env_cfg`、`use_teleop_device('so101leader')`、关闭 recorder、`gym.make`。
- [任务注册](https://github.com/LightwheelAI/leisaac/blob/24d3bcd3f1e4585740fc79921782c41617237812/source/leisaac/leisaac/tasks/pick_orange/__init__.py)：本任务直接使用 `isaaclab.envs:ManagerBasedRLEnv`，不是 Direct 或 Mimic 版本。
- [单臂模板](https://github.com/LightwheelAI/leisaac/blob/24d3bcd3f1e4585740fc79921782c41617237812/source/leisaac/leisaac/tasks/template/single_arm_env_cfg.py)：相机、观测组、decimation、25 s episode、空 reward 配置及 timeout。
- [动作配置／遥操作转换](https://github.com/LightwheelAI/leisaac/blob/24d3bcd3f1e4585740fc79921782c41617237812/source/leisaac/leisaac/devices/action_process.py) 与 [双向数值转换](https://github.com/LightwheelAI/leisaac/blob/24d3bcd3f1e4585740fc79921782c41617237812/source/leisaac/leisaac/utils/robot_utils.py)。
- [SO101 资产配置、joint limits 与 rest pose](https://github.com/LightwheelAI/leisaac/blob/24d3bcd3f1e4585740fc79921782c41617237812/source/leisaac/leisaac/assets/robots/lerobot.py)。
- [PickOrange 配置及随机化](https://github.com/LightwheelAI/leisaac/blob/24d3bcd3f1e4585740fc79921782c41617237812/source/leisaac/leisaac/tasks/pick_orange/pick_orange_env_cfg.py) 与 [真实 success 谓词](https://github.com/LightwheelAI/leisaac/blob/24d3bcd3f1e4585740fc79921782c41617237812/source/leisaac/leisaac/tasks/pick_orange/mdp/terminations.py)。
- [Isaac Lab step／自动 reset](https://github.com/isaac-sim/IsaacLab/blob/3c6e67bb5c7ada942a6d1884ab69338f57596f77/source/isaaclab/isaaclab/envs/manager_based_rl_env.py#L154) 与 [JointPositionActionCfg 默认值](https://github.com/isaac-sim/IsaacLab/blob/3c6e67bb5c7ada942a6d1884ab69338f57596f77/source/isaaclab/isaaclab/envs/mdp/actions/actions_cfg.py#L26)。
- [LeIsaac 依赖声明](https://github.com/LightwheelAI/leisaac/blob/24d3bcd3f1e4585740fc79921782c41617237812/source/leisaac/pyproject.toml)、[本仓库 Python 要求](https://github.com/Lebron-233/lerobot/blob/0702d2d58c3d085bbedc2371d9822f86bf35df4f/pyproject.toml#L32)、[Isaac Sim 5.1 包的 Python ==3.11.* 声明](https://pypi.org/project/isaacsim/5.1.0.0/)。
- [冻结 runtime loader／公开 rollout 限制](https://github.com/Lebron-233/lerobot/blob/0702d2d58c3d085bbedc2371d9822f86bf35df4f/src/lerobot/rollout/context.py#L320)、[production engine](https://github.com/Lebron-233/lerobot/blob/0702d2d58c3d085bbedc2371d9822f86bf35df4f/src/lerobot/rollout/inference/predictive_async.py#L123) 与 [metrics sink 协议](https://github.com/Lebron-233/lerobot/blob/0702d2d58c3d085bbedc2371d9822f86bf35df4f/src/lerobot/rollout/inference/metrics.py)。

## 3. 必要的语义映射

### 3.1 三层动作空间必须分开

```text
policy-normalized chunk / committed prefix
    -> 同一冻结 postprocessor
    -> physical degrees[5] + gripper RANGE_0_100
    -> 显式 SO101 transfer 数值转换
    -> 模拟器 JointPositionAction 的 radian targets[6]
```

predictor 的前缀只能来自第一层。不能把模拟器弧度、遥操作归一化值或已经 postprocess 的动作送回 predictor，也不能在队列之后增加未声明的动作裁剪或插值。

固定顺序保持：`shoulder_pan.pos, shoulder_lift.pos, elbow_flex.pos, wrist_flex.pos, wrist_roll.pos, gripper.pos`。读取模拟器时按实际 joint names 获取对应索引，不假定 USD 内部排列等于字典排列。动作配置显式设 `preserve_order=True`；使用 position action，而非 relative、EEF 或 IK action。

LeIsaac 源码中的 USD 角度范围为：

| joint | USD degrees | 官方遥操作 motor range |
| --- | --- | --- |
| shoulder_pan | [-110, 110] | [-100, 100] |
| shoulder_lift | [-100, 100] | [-100, 100] |
| elbow_flex | [-100, 90] | [-100, 100] |
| wrist_flex | [-95, 95] | [-100, 100] |
| wrist_roll | [-160, 160] | [-100, 100] |
| gripper | [-10, 100] | [0, 100] |

**官方 `convert_lerobot_action_to_leisaac` / `preprocess_device_action` 不适合直接接当前 physical-degree 输出。** 它们会将前五维按 [-100,100] 再做一次区间变换。例如 shoulder_pan 的输入 90 会映射成 99 degrees，而不是 90 degrees。

本提案在 SO101 模拟坐标中定义以下显式转换；这是 transfer 合同，而非对 SO100 标定等价的断言：

```text
action to sim:
    q_target[j] = degree_action[j] * pi / 180                 (j = 0..4)
    q_target[5] = (-10 + 110 * gripper_action / 100) * pi / 180

state from sim:
    state[j] = measured_q[j] * 180 / pi                     (j = 0..4)
    state[5] = (measured_q[5] * 180 / pi + 10) * 100 / 110
```

state 使用 measured `joint_pos`，不是 `joint_pos_rel`、速度、last_action 或 target。六个值保留后由原 processor pad 到 32，不采用 LeIsaac 默认导出器的前五维 [-100,100] state。

官方 JointPositionAction 默认会加 initial joint offset；当前 SO101 资产的六个初始值均为零。本提案将 `use_default_offset=False` 显式冻结，使输入就是绝对模拟目标，不依赖以后资产的默认 offset。若实际解析出的 joint 名称、方向／零点说明或资产限位不支持上述 transfer 定义，先修改并重新记录映射合同；不能用维数通过代替语义成立。

执行前须明确超限目标的处理。最小能力验证采用记录并终止该次技术验证，不静默 clip、scale 或给 predictor 伪造已执行 prefix；这类终止与任务失败分开。观测到的实际关节状态不裁剪成目标状态。

### 3.2 图像、状态与 processors

已有观测为嵌套 `obs['policy']`，含 `joint_pos`、`wrist`、`front` 等字段。两相机均为 RGB、640x480、30 FPS，观测 term 设置 `normalize=False`；必须在第一次有界运行中核实实际返回的 dtype、batch 维和帧更新，不把配置当成运行证据。

提案映射为 `front -> top -> camera1`，`wrist -> wrist -> camera2`。这里 `top` 仅是冻结模型输入接口的别名，结果包必须保留原始相机名称与位姿，不能声称 front 真的是原 top。官方 front 相机位于 Robot/base，相对位置 `(0.0,-0.5,0.6)`、ROS quaternion wxyz `(0.1650476,-0.9862856,0,0)`；wrist 位于 Robot/gripper，相对位置 `(-0.001,0.1,-0.04)`、quaternion `(-0.404379,-0.912179,-0.0451242,0.0486914)`。

PickOrange 默认对物体／plate 位置和 front 相机位姿做 reset 随机化。本提案保留这些环境配置，通过预先指定相同 episode seed 配对；不静默关闭随机化或将已改相机称为原训练视角。

模型／VLM／predictor 权重与 pre/postprocessor 实例均保持交接身份：

```text
policy: lerobot/smolvla_base@c83c3163b8ca9b7e67c509fffd9121e66cb96205
VLM: HuggingFaceTB/SmolVLM2-500M-Video-Instruct@7b375e1b73b11138ff12fe22c8f2822d8fe03467
predictor: m52a_1eb1adf7_portable_checkpoint_v2/runtime_best.pt
predictor: token_dim=960, action_dim=6, state_dim=32, horizon=8
risk thresholds: null
```

保留 checkpoint 的六维 `so100.buffer.action` statistics，不替换成 SO101 数据统计。新部署名拟为 `leisaac_so101_transfer_v1`，真实 embodiment 为 SO101。旧 `robot.type='so100_follower'` 的公开门禁继续有效；新入口明确校验 transfer profile，不能伪装 robot.type、猴子补丁旧 validator，或手工注入同维数据后声称原部署通过。

## 4. 最小部署与 30 Hz 控制

### 4.1 两进程，而不是更换当前模型环境

```text
Python 3.12：现有 LeRobot checkout / 固定模型栈
  observation snapshot -> notify_observation -> get_action
             |            production worker / sole ScheduledActionQueue
             |            identity OR predicted
             v
  显式 state/action/camera 映射 + 单一控制循环
             |
             | 本机 IPC；一次 reset/step/close；无第二动作队列
             v
Python 3.11：LeIsaac + Isaac Lab 2.3.0 + 兼容 Isaac Sim
  已有 PickOrange -> real physics / RGB / reward / terminated / truncated
```

使用标准库提供的本机 IPC，不创建通用 RPC 服务、远端调度器或恢复框架。消息只需 episode 标识、逻辑 step、reset seed、六维动作和 observation/outcome；RGB 使用原始 uint8 字节，状态使用固定 float32 数值。跨进程不传 torch/CUDA 对象，也不依赖两个 NumPy 版本能反序列化彼此的 ndarray pickle。

只有控制线程调用环境 step。worker 通过已有 Robot wrapper 所需的快照接口读取最近完成的观测副本，不直接操作 Isaac 或争抢 IPC。快照接口真实标识 SO101，不注册成 SO100。

模拟器 launcher 复用已核实 `parse_env_cfg -> use_teleop_device -> gym.make` 路径。资源下载在之后独立冻结的准备阶段按上述 Hub revision 进行；运行时使用固定本地资源目录，不执行内嵌 `revision=None` 下载的一行 loader。当前项目 Python 要求、依赖锁、已接受的模型 runtime 不降级。若改用支持 Python 3.12 的新 Isaac 版本，则属于另一项 runtime 迁移，不能当成这里已核实的组合。

### 4.2 时间与动作消费

拟显式配置 `sim.dt=1/60 s`、`decimation=2`，于是一个 env step 为 1/30 s；相机 update period 为 1/30 s，render cadence 对齐该间隔。这是本提案的环境配置变化：上游单臂模板原设 `decimation=1`，不能无说明地把原始 env.step 计为一个候选控制 step。

复用现有控制计时工具，不新造 clock／queue。每个测量 tick 按以下顺序推进：

```text
读取该 tick 的完整观测 snapshot（保留 sim step／capture／receive 时间）
    -> notify_observation(snapshot)
    -> get_action(None)                 # 每 tick 恰好一次
    -> 将 post-policy 动作转换为 sim target
    -> env.step(target)                # 推进 1/30 s 的真实模拟动力学
    -> 保存下一 snapshot 与真实 outcome
```

pending inference 不阻塞 get/step：旧 committed queue 继续驱动环境；按生产规则新 chunk 接管，late chunk 整块丢弃，合法 d0 不调用 predictor。延迟控制保持 `q=0.9 / margin=1 / guard=2 / max_late_steps=2 / horizon<=8`。需要的 delay 超过 cap 时遵循既有处理，不为使实验继续而扩大 cap。

bootstrap／真实 queue underflow 时，模拟器保持上一次已经发送的物理目标并继续 step，记录该行为；不自行往队列填充“成功动作”。启动预热不计为测量 episode，并在正式 episode 前执行已有 reset/epoch 流程。

推理延迟、IPC/RGB 搬运、env.step 和 GPU 竞争全部保留真实 wall time。若控制循环错过完整 slot，或 snapshot 的 sim step 与消费 tick 不一致，记为技术时间对齐失败，停止本次有界验证；不快进 get、暂停计时或在模拟器慢运行时仍声称 real-time 30 Hz。ordinary late inference 与控制 slot 丢失是不同事件，前者仍按 whole-discard 正常记录。

## 5. Episode 与真实任务 outcome

PickOrange success 同时要求三个 orange 相对 plate 满足 x/y 在 (-0.10,0.10) m、z 在 (-0.07,0.07) m 内，以及机器人回到官方 rest-pose ranges。它不是“夹住一个橘子”或单纯到达目标位置。默认 episode 为 25 s；在提案的 30 Hz 配置下对应 750 个 env steps，属于环境时间上限，不是本轮指定的实验样本量。

已核实配置只有 `success` 这一项非超时 termination，另有 time_out。Isaac Lab step 在返回前会自动 reset 终止环境，之后才构造 observation。因此该精确配置下应立即复制返回的 `terminated` 作为 episode success、返回的 `truncated` 作为 timeout 标志；两者同时为真时分别保留。不能对返回的 reset observation 重新调用 success 谓词，也不能假设通用 `info['is_success']` 存在。

每次终止先冻结该 episode 的 reward/termination/ticks，调用已有 engine reset 失效旧请求和队列，再接收下一 episode 观测。为指定下一个 seed 进行显式 reset 时，先处理上述边界，绝不让自动 reset 后的新观测被旧 task/reset epoch 使用。旧结果是否失效继续使用现有机制，不新增恢复协议。

该模板 reward 配置为空；保留环境返回 reward，但不把它当作有辨识力的 dense progress reward。成功主指标来自上述环境谓词；默认 subtask observations 只能作为原生诊断，不取代 success。

任务级陈旧性证据的因果链是：`当前观测 -> 后台推理期间旧动作推进真实环境 -> identity/predicted 接管差异 -> 真实轨迹和 success/timeout`。identity/predicted 配对只改变 context mode，预先分别固定环境 seed 与 policy CPU/CUDA 随机种子，保持相同部署 profile、权重、初始配置和调度参数；相同 seed 本身不证明 GPU 物理逐位确定性。同步 policy 运行可用于能力诊断，但不能替代这个比较。

## 6. 后续有界实施面与验证

只有明确扩展至上述 transfer profile／双进程部署后，才实施下列文件；此处列的是提案，不是已存在代码。

| 拟文件／复用点 | 具体工作 |
| --- | --- |
| `examples/advanced/predictive_async/leisaac_so101_contract.py`（新增） | 六关节映射、raw observation 适配、固定 transfer profile 与真实 joint-name 解析；不加载模型或 Isaac |
| `examples/advanced/predictive_async/leisaac_so101_env_server.py`（新增） | 仅在 Python 3.11 环境运行；固定 task/资源、配置 dt/decimation/action order、单环境 reset/step/close 和 outcome 复制 |
| `examples/advanced/predictive_async/eval_leisaac_so101.py`（新增） | 当前 Python 3.12 中的专用入口；显式 transfer 身份、IPC snapshot、唯一控制循环、episode accounting 与停止后物化 |
| 现有 `_load_frozen_future_latent_runtime`、predictor loader、`PredictiveAsyncInferenceEngine`、`ThreadSafeRobot`、计时工具 | 复用权重加载及推理／队列实现；不重写 queue、full RTC、风险 gating 或 robot 公共 factory |
| `tests/rollout/test_leisaac_so101_contract.py`（新增） | 只覆盖下表新增接口；旧科学阶段和不受改动影响的通过测试不重跑 |

每个新增检查对应一个具体失败及其决策后果：

| 检查 | 要检出的失败 | 失败后的行动 |
| --- | --- | --- |
| CPU 角度／gripper 往返与端点、打乱 joint order | 二次归一化、gripper 符号／比例错误、依赖 USD 顺序 | 修复映射；不加载 policy／环境 |
| 两相机 observation fixture 与相机来源字段 | RGB/channel/batch 次序错误、将 front 冒充原 top、state 使用了 target | 修复 contract；不声称跨 embodiment 等价 |
| 已有 engine + 有界接口 fixture | notify/get 顺序错、额外队列、postprocess 值混入 normalized prefix、late 部分接管 | 修复桥；测试本身只算工程证据，不当作新实验 |
| termination/reset 边界 fixture | 自动 reset 后误判 success、双重 episode 记账、旧结果进入新 episode | 修复 outcome/epoch 接线；不改变官方任务定义 |
| 后续单独冻结的 env/IPC 小规模 smoke | 实际 joint order／帧更新不符、资源／版本不兼容、控制 slot 丢失 | 停止技术验证，记录具体不符；不静默升级依赖或增加 horizon |
| 后续同权重任务能力诊断，再进入配对闭环 | base policy 无可辨识任务行为、全部技术终止或明显任务 floor | 保留原始失败，报告证据不足；更换权重／训练另开研究范围 |

CPU fixtures 是定向接口测试，不新建研究用 simulator，不产出科学成功率。样本数、seeds、mode 顺序、run 目录与每次执行上限，须在实际运行前单独固定；本计划不编造成功率、统计功效或未执行的命令。

telemetry 沿用已接受方式：hot path 只将 events/ticks/outcome 存入内存。控制停止、worker 确认 join、sink close 后才序列化到新的独立 artifacts 目录。IPC 中的 observation transport 不等于文件遥测；不恢复 public file-backed 热路径。安装资源、首次环境启动、首次模型调用与闭环样本执行分别保持可识别的边界，不混入旧 M5.3 runs。

## 7. 本轮完成情况与剩余边界

本轮完成了官方源码只读核实、明确的否决／推荐、上述文件级计划以及 GitHub 规划记录。没有安装包、import LeIsaac/Isaac/模型、启动环境、读取 checkpoint／dataset／test cache、运行 CPU 测试或新 episode。当前 `.venv` 的目录级元数据查询未发现 Isaac/LeIsaac dist-info；这不证明整台机器没有别的安装，也不是本裁决的依据。

原 [M5.3h 有限 GO](https://github.com/Lebron-233/lerobot/issues/1#issuecomment-5554561794) 保留，M5.3d/h PASS、M5.3f/g FAIL 均不重跑或改写。SO101 共用 GPU／IPC／环境步进属于新运行负载，不继承 synthetic timing PASS。原 B4 test 终态保留，risk thresholds 仍为 null。

**最窄新增范围是：接受具名 SO101 transfer profile 和本机双 Python 进程的部署适配，实施定向接口测试；之后另行冻结一次有界环境／模型能力验证，再决定是否有条件做 identity/predicted 任务配对。** 不需要先换 checkpoint 或训练；也没有证据保证冻结 base policy 在该任务成功。若希望证明原 SO100 部署完全等价，仍缺机械标定／相机外参证据，不能由上述数值往返测试替代。

本轮不再依赖其他 Pro 对话。当前剩余的是实质部署范围和运行准备条件，不是外部审阅等待。
