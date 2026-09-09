# D Graph/token/worker 结果：导入阶段阻塞

2026-09-09。**D1/D2代码与36项CPU定向测试完成；唯一一次D3在加载模型前失败，三个真实CUDA通过标志均为false。**

执行源码：`4f171a4b04dbf3111167dc6437658cdd2f1df4cb`，分支`codex/smolvla-graph-native-equivalence`，实现和协议已先提交推送。
[登记评论5595549440](https://github.com/Lebron-233/lerobot/issues/1#issuecomment-5595549440)在启动前发布并按实际ID回读一致。
[固定协议](SMOLVLA_GRAPH_TOKEN_WORKER_PLAN.md)、[定向测试](SMOLVLA_GRAPH_TOKEN_WORKER_TESTS.md)、
[机器结果](SMOLVLA_GRAPH_TOKEN_WORKER_RESULT.json)。
[结果评论5595618423](https://github.com/Lebron-233/lerobot/issues/1#issuecomment-5595618423)已发布并回读一致，
[发布回执](SMOLVLA_GRAPH_TOKEN_WORKER_RECEIPT.json)记录提交、失败状态及退出确认。

## 首个真实失败与退出

指定模型解释器为`/home/rp/Workspace/SmolVLA_RTC/libero-reference-venv/bin/python`。
入口导入实验适配器时，`src/lerobot/rollout/__init__.py:27`执行
`require_package("datasets", extra="dataset")`，抛出：

```text
ImportError: 'datasets' is required but not installed.
```

失败发生在`validate_smolvla_graph_worker.py`的module import，尚未进入main、严格loader或任一模型请求。
实际模型主调用0、reference0、capture0、运行中读取记录帧0；这些零值由失败位置确定，模型内计数器尚未创建。
模型worker没有启动，join/graph release/sampler restore均记为不适用，不能写成已通过。

只启动一次，墙时 **4.924667秒**；child exit **1**、supervisor exit **2**，均已收回。
295秒清理/300秒硬上限均未触发，未超时。按已登记停止条件结束，没有重试、补样本、安装依赖或换环境。

| 判定 | 标志 | 实际覆盖 |
|---|---|---|
| token_graph_equivalence_passed | false | 真实20对未运行 |
| graph_worker_lifecycle_passed | false | 真实CUDA worker未启动 |
| graph_identity_engine_integration_passed | false | 真实Graph/CPU发布集成未运行 |

## 已完成的代码与定向测试

helper抽取TokenGraph核心，RGB编码一次后委托；已编码token直接进入同一十步路径。
配对/shape/device校验、非空state/RTC等参数拒绝、原noise采样点、capture外RNG保护、独立输出clone与
实际replay/投影/setup计数均已实现。capture中途失败也保留已发起的setup计数。

实验适配器复用原队列、planner和worker-loop。新增三个小型默认接点，默认engine、factory和sync/RTC行为保留。
适配器把helper与processor/reset/释放放在owner，reset立即失效CPU世代，下一请求前处理owner reset；
policy/post复制为独立有限CPU chunks，stop失效在途返回，退出同步完成后才标记释放。生产队列deadline规则未改。

最终定向测试 **36 passed in 0.81s，exit0**；六个Python文件的ruff/format通过。
包含同一12事件控制脚本的真实CPU worker运行、完整启动、在途reset、A→B→A、在途stop、独立CPU输出、
capture未完成时reset、join超时和首错fatal，并复用默认engine旧错误策略/epoch/屏障用例。
CPU测试使用已有smolvla-rtc环境；模型专用环境缺datasets的初次收集日志及开发期失败也已保留。

## 证据

原始目录：`/home/rp/Workspace/SmolVLA_RTC/lerobot/outputs/smolvla_graph_async_contract_4f171a4b/`。
`model.log`是完整首个ImportError栈；`execution.json`记录实际命令、解释器、UTC起止、PID、一次运行和退出；
`registration.md`/`registration_readback.json`保存登记；`supervise.py`保留限时逻辑；定向测试日志一并保存。

模型入口尚未创建自己的`result.json`或`request_arrays.pt`。本报告及`failure_collection.json`是退出后整理的诊断，
没有伪造原模型结果或空的逐请求档案。旧A/B、两个C结果目录与原未跟踪三份任务文档均保留。

## 已知问题与下一停止点

冻结模型环境和生产rollout包的必需依赖不一致，是当前确定的执行阻塞。
本轮准备漏检了指定模型环境的完整rollout导入链；初次测试收集已显示datasets缺失，
切换CPU测试环境没有解决D3入口的依赖。这个入口准备缺口需要在新D登记前收敛。
D的token数值等价、GPU owner/reset/stop与CPU发布合同仍未获得真实模型证据；已实现代码不能替代这些验证。
下一次D需要先另行固定依赖/导入入口的解决范围，再登记新的有界验证。本轮不改环境或绕过包入口继续运行。

`baseline_qualified=false`、`realtime_qualified=false`、`predictor_benefit_tested=false`、
`risk_thresholds=null`、`old_confirmation=not_started_untouched`保持。
新native episode/机器人动作均0；A/B/C不重跑，未训练predictor、校准risk或开启新的native队列。
