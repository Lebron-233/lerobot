# LIBERO 十步 Graph 原生等价性固定协议

日期：2026-09-09。起点：`08c0d98f021d888aaeb43becb5e60a0a879f9fb1`。
工作分支：`codex/smolvla-graph-native-equivalence`。
本文件与实现、40 条 manifest 和 CPU 定向测试结果一起提交推送；Issue #1 的新注册评论
固定该提交的完整 execution HEAD 与下面两个命令的实际展开值。回读确认后按 A → B 执行。
结果另作提交，不覆盖本协议或旧结果。输出目录必须不存在，禁止续跑、替换、重试或扩大样本。

## 既有事实与本轮用途

既有 90 对匹配诊断、270 个观测点的 graph 计算验证及旧资格失败结果作为既定输入。
本轮回答实验局部 runtime 的随机序列、输出所有权和原生逐步等价性。
`baseline_qualified=false`、`realtime_qualified=false`、`predictor_benefit_tested=false`、
`risk_thresholds=null`、`old_confirmation="not_started_untouched"` 保持不变。
不读取旧确认 state 21–40，不读取 SO-101 holdout，不训练预测器，不改变去噪步数、精度或容差。

## A：运行时与固定模型合同

新 helper `examples/advanced/predictive_async/smolvla_graph_runtime.py` 仅由新实验入口显式安装。
原有 checkpoint loader、默认 policy/select_action、sync、RTC 和 predictive engine 的默认路径保持原样。
仍通过原 production selector 生成 50 步、每请求消费 1 步、完整执行 10 次 flow 更新。
共用已验证的 GraphSampler：graph 只捕获 prefix 与十步 denoise；两路图像每请求重新编码。
两路视觉 tokens、两路 masks、语言 tokens/mask、model-ready state、实际 noise 全部刷新。
同任务 shape/dtype/device 不兼容直接失败；任务切换释放旧图再捕获，无任务图池。

一个 worker 的一个线程独占模型，capture 期间该进程没有并行 CUDA 推理或原生推进。
每次 replay 后、返回 selector 前 clone 完整 chunk，副本开销计入 selector；消费者可持有旧输出。
噪声仍在原 sampler 的采样位置调用原 `sample_noise` 一次。每 episode 在 settling 后设一次 seed；
不逐请求重新设 seed。graph 准备的先行完整 eager setup、三次 warmup 和 capture 全用显式 noise，
在 CPU/CUDA RNG 保存恢复上下文内执行；准备不是原生控制动作。
reset 清空 policy/pre/post 和 runtime 的旧输出引用；同任务可复用 graph。
退出（含异常）恢复 original sampler，等待 GPU 完成后释放图及缓冲。

eager 使用每个真实请求的 Python projection hook 检查十个 `[1,50,32]`。
graph 记录每个 capture 的十个真实 projection shape；请求只记录 capture_id、一次真实 replay、
完整输出形状和 finite。capture 的 hook 不计作每次 replay 的 Python 执行证据。
CPU 定向测试覆盖输出别名、连续 RNG、八输入刷新、shape/dtype/device、reset/任务切换、
非 finite 原生提交前失败、异常清理和首个失败后的停止派发；复用旧 eager episode 的边界测试。

正式 A 的输入来自 `outputs/libero_single_step_native_ee273bce/tuple_{9*t:03d}` 中
仅 task 0–9、state 41 的初始 observation 0：10 个既有输入，无 native 动作。
eager 与 graph 两条正式序列各在开始时设置 seed `1009001` 一次，按 task 递增、request 0/1 连续运行。
每 task 重置 policy/pre/post，序列中不逐请求重置 seed；总计 20 对，40 次正式请求。
依次 exact 比较实际 noise、完整 `[1,50,32]`、normalized `[1,7]`、postprocessed `[1,7]`。
每 task 保留 request 0 的返回 tensor，在 request 1 后检查旧内容未变。
另外使用同一批输入做 task `0 → 1 → 0`，eager/graph 各从 seed `1009001` 开始一次，
各 3 次请求，检查上述四数组 exact 且三个 graph capture_id 各异。全部模型合同合计 46 次控制采样。
发生第一不等或技术故障即停止 A，保存已完成数组、进度、首个失败和退出记录，不启动 B。

## B：固定 20 对 / 40 条新原生 episode

完整、机器执行的 manifest：[`LIBERO_GRAPH_NATIVE_EQUIVALENCE_MANIFEST.json`](LIBERO_GRAPH_NATIVE_EQUIVALENCE_MANIFEST.json)。
以下每行严格执行两个新 episode；所有 40 条 ordinal、tuple_id、模式、任务、state 和两个 seed
都在上述已提交 JSON 中。禁用任务、state、次数等筛选参数。

| pair | task | state | env seed | policy seed | 顺序 |
|---:|---:|---:|---:|---:|---|
| 0 | 0 | 41 | 940041 | 950041 | eager → graph |
| 1 | 0 | 49 | 940049 | 950049 | graph → eager |
| 2 | 1 | 41 | 940141 | 950141 | eager → graph |
| 3 | 1 | 49 | 940149 | 950149 | graph → eager |
| 4 | 2 | 41 | 940241 | 950241 | eager → graph |
| 5 | 2 | 49 | 940249 | 950249 | graph → eager |
| 6 | 3 | 41 | 940341 | 950341 | eager → graph |
| 7 | 3 | 49 | 940349 | 950349 | graph → eager |
| 8 | 4 | 41 | 940441 | 950441 | eager → graph |
| 9 | 4 | 49 | 940449 | 950449 | graph → eager |
| 10 | 5 | 41 | 940541 | 950541 | eager → graph |
| 11 | 5 | 49 | 940549 | 950549 | graph → eager |
| 12 | 6 | 41 | 940641 | 950641 | eager → graph |
| 13 | 6 | 49 | 940649 | 950649 | graph → eager |
| 14 | 7 | 41 | 940741 | 950741 | eager → graph |
| 15 | 7 | 49 | 940749 | 950749 | graph → eager |
| 16 | 8 | 41 | 940841 | 950841 | eager → graph |
| 17 | 8 | 49 | 940849 | 950849 | graph → eager |
| 18 | 9 | 41 | 940941 | 950941 | eager → graph |
| 19 | 9 | 49 | 940949 | 950949 | graph → eager |

两条件均为 eager10 / graph10，50/1/10、同 execution HEAD、同精度与相同原始配置。
固定 policy `6721902bc4d61e50a3bfdb11dfb4cb626f05d102`，
VLM `7b375e1b73b11138ff12fe22c8f2822d8fe03467`，
native assets `0b3ea86be5fe169d0fd036ae63d1070ec09e90f6`，strict loader 零不匹配。
使用现有完整 LIBERO-Object task order 0、Panda relative OSC、20 Hz、256×256 双相机、
hard_reset 固定 row、10 settling、每 episode 最多 280 measured actions、`info.is_success`。
成功优先于同时到达的 TimeLimit；普通超时继续配对流程。最多 11,200 measured 和 400 settling。
不调用默认一步的 `single.run_worker`；共用原有原生工厂和 `run_episode(...denoising_steps=10)`。

监督进程只启动一个自有 worker/process group，每阶段 7,200 秒上限；超时 SIGTERM 触发
Python 异常清理，30 秒仍未退出才终止该进程组。已经 started 却无 final 的条目标技术失败，
其未返回 native call 标 unknown；未 started 的条目保持 not_run。不会重发未知的 native 动作。

每步保留原始双相机、转换前 EEF/gripper、8D state、四元数、normalized state、发送动作、
实际 native step 的开始/返回、成功/终止及 cleanup。新增每请求 noise/full chunk/normalized/post
独立 NPZ 快照，以及模式/capture_id/replay_count。大数组只保存在 outputs。
每对完成后按 observation index 比较初始及每步输入，随后比较实际 noise、全 chunk、
normalized/post 动作、native outcome、首次成功、总步数及 terminal observation。
第一个可判定分歧即停止后续 pair。当前对已经实际执行的所有步骤完整保留。
输入相同且 noise 相同而输出不同分类 runtime equivalence failure；相同动作后首先出现观测差异，
分类 environment reproducibility difference，不直接归因于 graph，不改容差、不重试。
必要诊断仅允许单独标注的既有失败输入离线重放，不产生新原生动作。

主工程延迟严格计观测返回后到 CPU 动作可用的 wall 时间，包括这一间隔内的观测存档。
另记录 processing（观测转换 + 预处理 + selector/post/GPU完成）、预处理、selector 到 CPU 动作、
观测存档和每请求证据写入、环境 step 以及完整 episode wall；日志成本明确单列。
每次新任务 graph 的 preparation_seconds 包括先行完整 eager setup、视觉编码、三次 warmup、capture
和完成同步；另单列 eager_setup_seconds。模型 strict load 成本单列；不把准备费用藏在稳态数字外。
每对、每模式总体报告均值、经验 P50/P95/P99（排序后 ceil(q*n)）、最大值和 >50 ms 数量，
以及首次请求和其余请求。总体同时给逐请求统计、每 episode 平均延迟及配对差，保留成功/终止对照。

只有 40/40 completed、技术失败 0、20 对所有比较 exact、所有环境关闭、original sampler 恢复、
graph 释放及 worker exit 0，才设置 `native_graph_equivalence_passed=true`。
双方同样 timeout 仅算行为一致，不算成功。不计算成功率提升 p 值，不更新旧科学资格。

## 固定启动与登记

仓库：`/home/rp/Workspace/SmolVLA_RTC/lerobot`。
专用 Python：`/home/rp/Workspace/SmolVLA_RTC/libero-reference-venv/bin/python`。
依赖和 EGL 组合保持原样，不安装升级包。登记时把下列 `EXECUTION_HEAD` 和 `HEAD8` 展开为
刚提交推送的完整值，并分别登记 A、B 的完整命令及绝对输出目录；A 的 result 和 supervisor
必须在同一 execution HEAD 下 passed / exit 0，B 入口才放行。

```bash
env -u PYTHONPATH HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 MUJOCO_GL=egl PYOPENGL_PLATFORM=egl LIBERO_CONFIG_PATH=/home/rp/Workspace/SmolVLA_RTC/libero-reference-cache/config LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libGLdispatch.so.0:/usr/lib/x86_64-linux-gnu/libGLX.so.0 /home/rp/Workspace/SmolVLA_RTC/libero-reference-venv/bin/python -u -X faulthandler examples/advanced/predictive_async/libero_graph_native_equivalence.py --phase model --execution-head EXECUTION_HEAD --output /home/rp/Workspace/SmolVLA_RTC/lerobot/outputs/libero_graph_model_HEAD8
```

```bash
env -u PYTHONPATH HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 MUJOCO_GL=egl PYOPENGL_PLATFORM=egl LIBERO_CONFIG_PATH=/home/rp/Workspace/SmolVLA_RTC/libero-reference-cache/config LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libGLdispatch.so.0:/usr/lib/x86_64-linux-gnu/libGLX.so.0 /home/rp/Workspace/SmolVLA_RTC/libero-reference-venv/bin/python -u -X faulthandler examples/advanced/predictive_async/libero_graph_native_equivalence.py --phase native --execution-head EXECUTION_HEAD --output /home/rp/Workspace/SmolVLA_RTC/lerobot/outputs/libero_graph_native_HEAD8
```

监督进程保存 exact HEAD、实际命令、manifest、包版本、GPU 状态、worker PID 和退出证据。
源码检查要求 tracked 文件干净；用户开工前的未跟踪任务书不参与执行，不为清理状态而提交它。

## C 与停止后的交付

仅 B 通过才执行无 native 动作的 identity async 时序回放。复用现有
`PredictiveAsyncInferenceEngine`、`ScheduledActionQueue`；用 B 的实际完成延迟固定顺序输入 CPU fake
policy，以 20 Hz 递增 wall ticks。缺动作如实记录 underflow，不暂停时间、不保持旧动作冒充输出。
检查观测和 next_action_index 冻结、committed prefix、提前完成、迟到处理、reset/task epoch、
动作绝对索引和一个 in-flight；planner 只见已经完成的历史延迟。不得借此启动新的环境或预测器实验。
若复用需要改变动作含义或队列现有承诺，先形成下一轮审阅差异，不改掉 relative-action 校验来通关。
下轮文档须区分 LIBERO relative OSC 与 RelativeActionsProcessorStep 的重锚定，固定 identity state/context，
说明异步消费多动作 chunk 相对于本轮每请求一动作是另一个实验变量。
若 A/B 未过，则保留首个失败、全部实际记录和 not_run，C 记录未执行原因及具体接线缺口。

## 参考

固定地址与静态 shape、side-stream warmup、capture 期间单进程 CUDA 工作约束、CPU hook 不会被 replay，
依照 [PyTorch 2.11 CUDA Graphs 文档](https://docs.pytorch.org/docs/2.11/notes/cuda.html#cuda-graphs)。
图对象释放与 replay API 依照 [PyTorch 2.11 CUDAGraph](https://docs.pytorch.org/docs/2.11/generated/torch.cuda.CUDAGraph.html)。
