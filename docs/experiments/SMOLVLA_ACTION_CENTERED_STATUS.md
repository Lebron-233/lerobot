# F-ACR1：实现与CPU准备完成，GPU驱动版本不一致阻止正式启动

2026-09-11，接续Issue #1评论5627262353。原F-PFX1独立审计783项接纳及主门false保持，未重跑。

## 本次完成

新增独立PLAN、libero_action_centered.py、audit_libero_action_centered.py和test_libero_action_centered.py。
实现固定无动作FP32缓存基底，加中心化h(a)-h(0)或普通h(a)的两组实验。
组合先减后加，零动作不变量有专门浮点回归测试；不改旧模型、case权重、数据或判据。
两组同参数/初始化/72更新预算，中心化两次h前向、普通一次；不宣称相同FLOPs。

原固定uv offline/no-project/no-python-downloads解释器下，12项合成CPU测试通过，exit0，1.65秒。
覆盖零动作exact、先减后加、普通对照、动作梯度/共享bias抵消、主门每一必要条件及调用预算。
Ruff check三份新Python文件通过，exit0。
CPU准备命令exit0，工具外层3.752秒；固定72训练/16验证、原基底88例、供体及case权重来源核对通过。
准备目录：outputs/smolvla_action_centered_preparation_11c81ae1；manifest.json、weights.json、preparation.json已保存。
这些是实现/准备结果，不是F-ACR1模型结果；独立审计器尚未在本轮真实输出上运行。

## 新的实际阻塞：不是工具安全拦截

pytest启动阶段的既有CUDA可用性检查发出Error804，测试本身以CPU完成。
随后nvidia-smi实际exit18，返回：

```
Failed to initialize NVML: Driver/library version mismatch
NVML library version: 580.178
```

只读定位得到：

| 项目 | 实际值 | 读取方式 |
|---|---|---|
| 当前已加载NVIDIA内核模块 | 580.173.02 | /proc/driver/nvidia/version |
| 当前内核对应磁盘模块 | 580.178.04 | /sbin/modinfo -F version nvidia |
| 64位NVML库 | libnvidia-ml.so.580.178.04 | ldconfig目录及readlink |
| 64位CUDA驱动库 | libcuda.so.580.178.04 | ldconfig目录及readlink |

裸modinfo最初因PATH未找到命令；使用系统已有/sbin/modinfo后查询成功。未安装任何包。
目前能确认加载模块与磁盘模块/用户态库不同，不能确认是谁、何时、通过何种机制更新了驱动。
这不是F-ACR1科学阴性结果，也不是安全检查拒绝。没有尝试改LD_LIBRARY_PATH、替换驱动库、卸载模块、
停止桌面/其他进程、安装驱动或重启主机。驱动维护涉及整机状态，需要操作者安排。

## 接续

先维护主机使实际加载的NVIDIA模块与磁盘驱动/CUDA/NVML库一致；由于磁盘三项已经一致，
保存工作后重启是优先考虑的维护途径，但本次没有执行或验证重启后的结果。
维护后检查nvidia-smi及原固定解释器CUDA初始化，再核对原88份基底缓存重放门。
不更改解释器或依赖来伪造原环境恢复；不把其他CPU方案当作此次冻结GPU实验的等价执行。

新实验代码提交推送后，GPU恢复时按当时实际HEAD及唯一未使用输出路径登记并回读正文exact，
之后才启动一次正式worker。准备已完成，不重复--prepare，不重跑F-PFX1；历史三份pending结果文档仍保持。
计划850次完整decoder、1143次支路前向、144更新/反传是上限/应完成数，不是本轮已使用计算。
本次F-ACR1新增VLA加载/支路真实前向/训练/Env/native/test读取均0；未预登记或启动正式实验，
未创建正式运行输出，没有自有后台实验。生产默认与闭环资格false、risk_thresholds=null、confirmation untouched。
