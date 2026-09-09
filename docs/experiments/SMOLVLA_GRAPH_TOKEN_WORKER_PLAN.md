# D：Graph token 入口与真实 worker 模型合同

2026-09-09。依据《SmolVLA_C审阅通过_D阶段续接任务书_20260909.md》，从3751f2d9继续。
A/B及新的严格deadline C已接受，本轮执行D1→D2→D3。`whole_discard_any_late`保持。

## 实现与线程边界

`SmolVLAGraphRuntime`的RGB入口先编码一次，token入口直接调用同一TokenGraph核心。
调用原sample_actions的token override API，保持相机顺序、sqrt(D) scaling、mask、语言、state与噪声。
沿用模型自身token校验，缺配对、shape/device不符及非空state/RTC/timing参数明确失败；未知参数由签名拒绝。
原采样点每主请求采样一次，显式noise不采样；capture外fork_rng保护setup对随机序列的影响。
Graph输出在owner stream上clone。每次capture保留实际setup/warmup/capture调用计数（含失败中的已发起调用）、
十次[1,50,32]投影形状及真实replay计数，setup视觉编码为0。

具名`SmolVLAGraphIdentityEngine`复用原worker-loop、planner、epoch和ScheduledActionQueue。
共享engine仅增加默认无操作的worker资源上下文/请求终点，以及保留默认错误策略的判定接点。
factory、sync/RTC和默认reset顺序不变。实验适配器在worker创建/安装/释放helper，建立inference_mode和
原配置autocast，所有processor与模型reset也在worker执行。首个模型错误为fatal。

控制线程reset立即增加reset_epoch并清CPU active/plan/staged；在途GPU工作正常完成后成为stale。
owner在下一请求前执行待处理policy/pre/post reset和图释放。启动期间reset/task仍按原startup失败合同结束。
图按(request.reset_epoch, request.task_epoch, task)识别世代；startup probe自身的epoch推进也会触发重建。
发布前policy/post分别复制为独立、有限的CPU chunks并完成设备屏障。stop先使CPU世代失效，
owner完成在途请求、恢复sampler和同步释放后join；超时不标记joined/released。

固定地址、同进程capture并发及输出复用约束依据已读的
[PyTorch 2.11 CUDA说明](https://docs.pytorch.org/docs/2.11/notes/cuda.html#cuda-graphs)与
[CUDAGraph API](https://docs.pytorch.org/docs/2.11/generated/torch.cuda.CUDAGraph.html)。
保持global capture检查，API及依赖使用现有环境。

## 固定输入、环境与预算

唯一来源为`outputs/libero_single_step_native_ee273bce/`中task0–9、state41、observation0，
目录ordinal=task_id×9；读取events到observation0即停止，复用recorded_batch的原双相机/四元数/8D重建。
控制线程的uint8双相机及8个CPU标量由同一帧构造，使用原build_dataset_frame/prepare_observation API，
核对重建state和两路图像exact。其余帧、旧确认0–40、SO101和predictor数据不读取。

专用模型Python：`/home/rp/Workspace/SmolVLA_RTC/libero-reference-venv/bin/python`。
policy revision `6721902bc4d61e50a3bfdb11dfb4cb626f05d102`，VLM revision
`7b375e1b73b11138ff12fe22c8f2822d8fe03467`；snapshot均沿用`libero-reference-cache/hub`和原strict loader。
固定RTX4070TiSUPER、chunk50/denoising10、原混合精度与处理器，context=identity、20Hz planner、
queue threshold30、quantile0.9/window50/margin1/min-max delay0–8、guard2/max_late2。

uv使用`--no-project --offline --no-python-downloads --python <既有解释器>`，无with或依赖同步。
CPU定向测试使用已有smolvla-rtc环境；模型环境的旧测试模块因缺datasets跳过，未安装软件补齐。
新增native episode=0、机器人动作=0，无模拟器/真机/predictor/训练/compile/量化或新attention。

总上限120次主策略调用（含reference）和30次capture。实际有限表为52次主调用+12次reference=64次，
预计token10+worker5=15次capture；每次一次eager setup、三次side-stream warmup、一次capture调用，单列。
整个模型子进程总上限300秒；监督进程295秒发送终止以保留清理/落盘时间，300秒仍未退出则强制终止。

## Token连续等价性：40次主调用

eager序列与token-Graph序列各只在开头设seed1019001，均按task0–9、每任务连续2请求。
每次由外部prepare_images/encode_image_tokens一次，然后从policy batch中移除RGB，以token-only kwargs调用。
比较20对实际noise、完整[1,50,32]、去pad[1,50,7]及完整post-policy chunk，要求dtype/shape一致、exact且finite。
主请求的prepare_images、视觉编码与noise均记录真实API计数；Graph内部零额外视觉编码。
每任务第一完整输出持有至后续请求及所有后续任务后核对，setup不计入40个正式请求。

## 真实worker固定表：12次主调用+12次reference

真实start/resume，控制线程只用notify_observation/get_action与CPU Event；主线程不手调_run_request。
worker仅初始设seed1019002，reset不重新播种。A=task0（语言长度12），B=task3（长度13）。

| 顺序/标签 | 当前task | 请求 | 明确交错与接受结果 |
|---|---|---|---|
| 0 cold | A | bootstrap/cold_temporary | 临时安装，不ready |
| 1 probe | A | startup_probe/probe | 真实d8 prefix；原startup gate，probe_discarded |
| 2 fresh | A | bootstrap/fresh_warmed | 新epoch图，安装后ready |
| 3 planned | A | planned | staging后按原takeover_index消费新第0行 |
| 4 reset_inflight | A | planned | CPU chunks完成、发布前暂停；控制线程reset；旧结果stale |
| 5 new_epoch | A | bootstrap | owner处理reset，重建图；安装新epoch结果 |
| 6 new_epoch_planned | A | planned | 接管并消费第0行 |
| 7 a_to_b | A | planned | 发布前set_task(B)，旧active仍标A，旧返回stale |
| 8 b_bootstrap | B | bootstrap | 消费剩余旧A active至空；B新图及安装 |
| 9 b_to_a | B | planned | 发布前set_task(A)，旧active仍标B，旧返回stale |
| 10 a_return | A | bootstrap | 消费剩余旧B active至空；A新世代重建并安装 |
| 11 stop_inflight | A | planned | 发布前另一个CPU控制线程stop；世代失效后释放Event；stale并join |

每个planned请求前仅通过get_action把available降至30；正常staging后通过get_action消费至takeover第0行。
任务变化后的两个bootstrap前消费完旧active。这些有限CPU消费是生命周期夹具的一部分。
每个主请求终结后，owner以同一已编码输入和显式noise运行原eager方法一次作参考，比较完整/padded/post输出。
reference位于原请求发布和完成时长采样之后，单列耗时；下一notify要等reference完成。
它不写入tracker。原主请求latency包含该请求真实准备、CPU复制和注入等待，不修改tracker/时间/索引。

Event注入点为“模型结果与CPU复制/屏障已完成、尚未queue publication”，不是取消GPU kernel。
注入Event最多等待10秒，单请求终点最多等待15秒。遇首个不等、错误epoch、额外编码、owner错误、
非有限/未完成CPU结果、startup gate失败或退出未确认就停止，保留已完成项及首例，不重试/补样本。
CPU测试另覆盖capture构造尚未完成时reset、join超时不得声称释放和默认engine旧错误重试策略。

## 执行、输出与判定

本轮源码/协议先提交推送，Issue #1登记exact HEAD、下列展开后的命令和输出目录，按实际ID回读后执行一次。
新目录`outputs/smolvla_graph_async_contract_<execution_HEAD前8位>/`，禁止覆盖。
启动命令由uv运行专用Python监督脚本；其模型子进程使用同一解释器执行：

```bash
env -u PYTHONPATH HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 /home/rp/Workspace/SmolVLA_RTC/libero-reference-venv/bin/python -u examples/advanced/predictive_async/validate_smolvla_graph_worker.py --execution-head NEW_EXACT_HEAD --output /home/rp/Workspace/SmolVLA_RTC/lerobot/outputs/smolvla_graph_async_contract_NEW_HEAD8
```

hot path只留内存；消费停止、worker join/资源关闭后保存CPU request_arrays.pt及result.json。
保留输入/core tokens/prefix/noise/完整及post chunks、request/task/reset/capture标识、owner/API计数、metrics终点、
控制注入和CPU消费、退出回执。无法join时不并发序列化worker仍在写的数据。大数组不提交Git。

分别判定token_graph_equivalence_passed、graph_worker_lifecycle_passed、graph_identity_engine_integration_passed；
完整固定用例、对应数值/生命周期条件及正常退出均满足才可声明D通过。结果/回执另提交推送并发布评论回读。
当前三个未跟踪文档原样保留并在登记中列明；执行代码及本协议必须无未提交修改，旧入口clean校验不改。

## 已知限制

D是固定帧、确定性交错的模型/线程合同；主线程允许等Event，快速CPU消费不是20Hz连续物理控制。
耗时含诊断开销，不作为新同步加速数字或predictor收益；不外推B每次消费1行结论至native多行消费。
`baseline_qualified=false`、`realtime_qualified=false`、`predictor_benefit_tested=false`、
`risk_thresholds=null`、`old_confirmation=not_started_untouched`保持。D结束后不自动开启新native队列。
