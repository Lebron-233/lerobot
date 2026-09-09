# D3-r1：无环境变更的导入边界修复与原固定 CUDA 合同

日期：2026-09-09。起点 `da6ecd7bf109c558fbcce0ff27e0605e1ed7a37f`，
分支 `codex/smolvla-graph-native-equivalence`。旧 D3 的导入失败证据和三个 false 原样保留。
本轮承接用户继续实验的指令；在新登记前明确唯一准备修复，不重开 D 数值/生命周期方案。

## 唯一准备修复

`src/lerobot/rollout/__init__.py` 不再让独立 inference 导入强制依赖 datasets。
复用 `_datasets_available`，数据集存在时保留完整 rollout 的原导入顺序和全部公开导出；
数据集不存在时仍从正常包入口导入原 inference/factory/sync/RTC/predictive_async 类，
访问完整 rollout 的数据集相关导出仍由 `require_package` 抛出明确缺依赖错误。
不伪造 sys.modules，不直接加载被绕过的生产文件，不复制 engine，不修改 package availability。

业务修复只涉及包入口和定向测试；另将旧监督逻辑固定为具名CLI脚本。
模型环境完全不变：不安装、升级、降级或链接依赖；
不切换 D3 解释器，不用 PYTHONPATH 注入另一环境。不改权重、处理器、精度、sampler、
队列、planner、worker、capture 模式、容差、D3 请求表或任何科学资格。

## 启动前门禁

1. 保存两个现有环境的版本清单，核对专用 Python、HEAD/branch、GPU与三个旧未跟踪文档。
2. 新定向测试验证缺 datasets 时 canonical inference 类/factory、缺 extra 的明确错误、
   dataset-enabled 环境全部旧导出，以及真实 D3 入口的 import-only（无 main/loader/输入读取，CUDA未初始化）。
3. 既有 36 项定向测试在已有 smolvla-rtc 环境重跑；新导入测试还在指定模型环境执行。
   dataset-enabled 分支在模型环境因真实缺 datasets 可以显式 skip，不能用 skip 替代入口导入门。
4. `validate_smolvla_graph_worker.py --help` 必须在指定模型 Python 的原正常脚本入口成功退出。
   import-only/--help 不作为模型尝试，不运行 D3 main，不新增 episode，不读记录帧。
5. 新代码、协议、测试结果先提交/推送；实际执行 HEAD、完整展开命令、新输出目录和旧未跟踪
   文档名单登记 Issue #1，按发布返回的实际 ID 同步回读正文一致后，仅执行一次 D3-r1。

## 固定模型执行

完整沿用 `SMOLVLA_GRAPH_TOKEN_WORKER_PLAN.md` 的 20 对 token + 12 个真实 worker事件，
两序列 seed1019001、worker seed1019002；52主请求+12reference、预计15 capture，
硬上限总120请求/30 capture/300秒，295秒请求退出。固定十个 task/state41/observation0，
同一 policy/VLM revisions、RTX4070TiSUPER、chunk50/denoising10。

模型解释器仍为 `/home/rp/Workspace/SmolVLA_RTC/libero-reference-venv/bin/python`。
原模型子命令不变，只展开新 execution HEAD 和新的不可覆盖输出目录：

```bash
env -u PYTHONPATH HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 /home/rp/Workspace/SmolVLA_RTC/libero-reference-venv/bin/python -u examples/advanced/predictive_async/validate_smolvla_graph_worker.py --execution-head NEW_EXACT_HEAD --output /home/rp/Workspace/SmolVLA_RTC/lerobot/outputs/smolvla_graph_async_contract_NEW_HEAD8
```

监督使用已提交的 `examples/advanced/predictive_async/supervise_smolvla_graph_worker.py`，
同一专用解释器，经现有 uv 的 `--no-config --no-project --offline --no-python-downloads`
运行，不触发安装或依赖同步。它复用旧 `outputs/smolvla_graph_async_contract_4f171a4b/supervise.py`
的单次运行及295/300秒限时逻辑，保存本次命令/进程退出。
模型入口会自己新建结果目录；登记和监督日志先写到独立准备目录，不能提前占用该目录。

## 停止与交付

准备失败保留真实首例，不以模型运行试探导入；权限/平台安全拒绝不换通道重做被拒绝操作。
D3-r1 首个真实错误/不等/不正确发布/无法退出立即停，不追加输入、调用、capture或重试。
本轮结果使用新 R1 文件；旧失败报告、A/B/C与所有旧目录只读保留。

分别报告 token_graph_equivalence_passed、graph_worker_lifecycle_passed、
graph_identity_engine_integration_passed；不得把准备门通过写成 CUDA通过。
结果和回执分开提交推送，Issue #1结果发布后按实际 ID 同步回读。

`baseline_qualified=false`、`realtime_qualified=false`、`predictor_benefit_tested=false`、
`risk_thresholds=null`、`old_confirmation=not_started_untouched` 保持。
新增native episode/机器人动作=0，不重跑A/B/C、不打开旧确认/predictor数据，不训练、校准或启新native队列。
D合同完整通过即为本轮重大成果和停止汇报点；D失败则如实交付已覆盖部分和首个技术缺口。
