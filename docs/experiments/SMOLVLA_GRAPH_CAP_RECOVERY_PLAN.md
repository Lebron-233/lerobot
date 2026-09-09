# E-RCV1：同路径丢弃探针的固定四条native协议

2026-09-09。E-L1已证实原默认路径的planned恢复闭锁，见AUDIT。
本协议按《SmolVLA_ES1接纳_Cap恢复闭锁诊断与有界实验_Codex任务书_20260909.md》冻结。
实现、协议、manifest、实际测试先提交推送；执行HEAD在新Issue #1预登记中给出。

## 唯一候选及生产默认

`recovery_policy=same_path_discard_probe_v1`；生产构造默认`disabled`，factory/default配置不变。
新增request kind和最小处理分支，复用原NativeEngine/Graph owner、worker-loop、queue及原controller。
没有第二队列、主线程模型调用、新GPU服务或改用其他恢复策略。

仅在startup complete、identity context/fallback、原raw>8且active非空、原单在途允许、
本episode probe计数小于50的原cap_wait机会派发。queue threshold仍30。
读取正常notify的最新独立CPU反馈及当前task/reset epoch，保留原请求创建时间为完成样本起点。
探针没有takeover plan，不改active/staged/committed prefix/index；与identity planned一样执行
观测/pre、单次vision编码、十步flow、完整50行输出、post、独立CPU chunks及既有设备完成屏障。
完成后同task/epoch且未stop才接纳一次真实时延，明确记录`discarded_recovery_probe`；所有输出丢弃，不install/stage。
失效记录`stale_recovery_probe`且不接纳；模型/有限性等失败沿原Graph fatal路径退出。
同路径指模型输入及完成计算路径；探针按合同不做planned专属的queue plan构造及committed-prefix证据拷贝。
起止定义仍为requested_at至原完成屏障，实际发生的CPU审计/控制成本不扣除，不声称两种请求耗时应相等。

P90保持窗口50、float32成员、numpy linear插值及原latency_to_steps整数容差，margin1/min0/max8不改。
不清历史、不删慢样本、不接纳普通bootstrap，不按“尝试过恢复”强行clamp。
raw回到8以内后自动回到原planned，实际接管还须满足原whole-discard/索引合同。
50次预算在请求派发前计数，reset/task不清预算；native每episode新engine。
预算耗尽返回原cap_wait，active空仍走原bootstrap。underflow不推进Env、不补零或保持动作、不追赶旧时隙。

两个条件均启用同一候选；serialized对探针也等待该请求的原CPU完成通知，async保持不等待。
探针真实消耗主策略调用、RNG和wall，不重新播种或回滚RNG，不要求不同条件逐动作exact。
全部新主机审计与小型日志成本保留在wall；不新增CUDA同步/event/额外GPU取值或热路径大数组写盘。

## 固定四行与初态

| ordinal | pair | task/state | Env seed | policy seed | condition |
|---:|---:|---|---:|---:|---|
| 0 | 0 | 0/41 | 940041 | 950041 | graph_serialized |
| 1 | 0 | 0/41 | 940041 | 950041 | graph_identity_async |
| 2 | 1 | 2/41 | 940241 | 950241 | graph_identity_async |
| 3 | 1 | 2/41 | 940241 | 950241 | graph_serialized |

TASK_NAMES/task_order_index0沿原libero_object工厂；task0为alphabet_soup，task2为salad_dressing。
manifest由原rows0/1/5/4复制identity，重编号并增加明确候选字段；逐字段固定，不提供任意筛选或resume参数。
每条原工厂创建1个Env，必要ensure/隐含reset，原seed/reset/set_init_state(row41)，恰好10 settling。
每对第二条件在推理前核对新双相机、8Dstate、raw quaternion/EEF/gripper exact；首个差异立即停止。
每条cold→probe→fresh原3阶段、owner播种一次、最多2 capture，无额外启动请求或跨episode缓存。
原startup cap拒绝是本轮首错，不能以正在测稳态为由忽略。

## 模型、环境和数值身份

原GPU：NVIDIA GeForce RTX 4070 Ti SUPER。
policy revision `6721902bc4d61e50a3bfdb11dfb4cb626f05d102`；
VLM `7b375e1b73b11138ff12fe22c8f2822d8fe03467`；assets `0b3ea86be5fe169d0fd036ae63d1070ec09e90f6`。
原严格loader、processor、相机/state/action转换、bf16/fp32混合精度、AMP=false、50/1/10。
20Hz、threshold30、P90/window50、margin1/min0/max8、guard2/max_late2、any-late整块丢弃、identity、compile=false。

模型Python：`/home/rp/Workspace/SmolVLA_RTC/libero-reference-venv/bin/python`；
CPU Python：`/home/rp/miniconda3/envs/smolvla-rtc/bin/python`；
uv：`/home/rp/miniconda3/envs/smolvla-rtc/bin/uv`。
仅`uv run --no-config --no-project --offline --no-python-downloads --python <既有解释器>`，清除PYTHONPATH。
无安装、升级、sync或环境替换；模型Python/version/140项metadata前后直接比较。
GPU在登记前、退出后各保存一次只读快照及其他进程背景，不干预其他任务或等待低负载反复探测。

## 实际上限和停止

| 资源 | 每条 | 总计 |
|---|---:|---:|
| Env | 1 | 4 |
| settling | 10 | 40 |
| measured native step | 280 | 1120 |
| ready后slots | 1200（60秒） | 固定4条各自窗口 |
| startup | 30秒 | 原3阶段 |
| 主策略 | 160 | 640 |
| 其中recovery probe | 50 | 200，含在主调用内 |
| capture | 2 | 8 |
| setup/warmup/capture内 | 每capture 1/3/1 | 最多8/24/8 |
| reference/predictor/训练/真机 | 0 | 0 |

单模型请求15秒、单native调用30秒。外层870秒请求退出、900秒硬停止，只处理自建worker进程组；
单调用超时则TERM后5秒仍未退KILL。实际监督器使用新四行manifest、总预算与这两个外层时限。
队列最多执行一次，retry/resume/replacement=0；不运行第三对、第三条件或原算法native对照臂。

success、TimeLimit、动作或wall上限、正确underflow/late丢弃、预算内未触发探针或未恢复均是有效结果。
技术错误、初态不一致、非法索引/CPU输出、未知调用或清理未确认则停止后续条目，余项not_run。
未知native不重发。stop/join/Graph释放/sampler恢复/metrics关闭/Env关闭确认后才保存CPU数组，
关闭失败保留小型结果和首错，不保存未确认清理的大数组。

## 准备门、登记和命令

E-L1 4项原路径characterization先通过后才实施候选。候选准备覆盖默认关闭、staged保留、有效接纳一次、
满窗口45个有效probe自然恢复、持续慢50个probe不恢复且预算封顶、失效/stop/首错、原controller等待差异、
固定表/预算及关闭失败不存数组；原startup/late/完成屏障/join必要回归及模型环境新入口通过。
实际运行拆分与首个开发失败见TESTS；准备门文件保存真实回执，监督器在启动前要求全部通过。

入口：`examples/advanced/predictive_async/libero_graph_cap_recovery_native.py`。
公开参数只有`--execution-head <40位HEAD> --output <独占绝对路径>`；内部worker由监督器启动。
新输出固定为`outputs/smolvla_graph_cap_recovery_<HEAD前8位>`，启动前不得存在。

环境/解释器组合从旧实际launch_command逐项复制：HF_HUB_OFFLINE=1、TRANSFORMERS_OFFLINE=1、
MUJOCO_GL=egl、PYOPENGL_PLATFORM=egl、LIBERO_CONFIG_PATH原config，
LD_PRELOAD为原libGLdispatch.so.0:libGLX.so.0；用上述uv和模型Python，`python -u -X faulthandler`运行新入口。
提交推送后新Issue #1登记完整HEAD、展开的精确命令、独占目录、候选差异、默认关闭、旧资格保持、预算和首错合同。
按返回实际comment ID同步回读一次，正文exact后直接执行唯一队列。

## 新证据与解释

小型calls intent/return保留真实调用；逐请求保存kind/epoch/frame、trigger raw、tracker前后成员数/估计、
capture/replay、真实完成时长、discarded/stale outcome。大数组仅清理退出后保存。
分别计cap_wait、probe、steady bootstrap、underflow、env_busy、scheduler_miss、serialized_wait；probe不是接管。
native恢复链必须在同epoch具备raw>8→有效probe接纳→raw≤8→新planned→实际native消费合法接管row0；
源chunk逐值核对沿用原source audit。仅bootstrap installed或probe dispatched不算恢复。

字段采用新recovery命名空间：started/probe observed/estimator within/planned takeover/four episode contract/two pair comparison。
未触发则observed=false并报告覆盖不足；不把四条完成替代旧E20条。
旧两个整体字段保持false、D历史true/旧E三个observed保留；科学资格false、risk_thresholds=null、旧confirmation untouched。

这两对是有意选择的困难开发案例；仅比较候选下S/A。旧native只作历史背景，没有候选对原算法的native效应估计。
两种条件请求数可不同、RNG与轨迹可分化，GPU快照不建立因果，Env长尾不全部归因于闭锁。
结果收回后分开提交RESULT.md/json、发布/回读结果评论、提交独立RECEIPT并更新HANDOVER/NEXT_REVIEW，到此停止。
