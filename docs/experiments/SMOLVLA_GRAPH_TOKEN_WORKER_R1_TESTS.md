# D3-r1 导入边界与 CPU 定向测试

2026-09-09。仅准备/CPU证据，不代表真实 CUDA通过。

已有 dataset-enabled `smolvla-rtc` 环境：**39 passed in 12.15s，exit0**。
其中36项是原 `SMOLVLA_GRAPH_TOKEN_WORKER_TESTS.md` 的同一节点集合，
另加 `tests/test_rollout_optional_imports.py` 三项。

指定模型环境 `libero-reference-venv`：新导入测试 **2 passed, 1 skipped in 6.93s，exit0**。
唯一 skip 是 dataset-enabled 的全部公开导出用例；该用例已经在上面的39项中实际通过。
模型环境实际通过缺 datasets 的 canonical inference/错误边界和真实 D3 import-only 用例；
import-only 使用正常包入口，没有执行 main/loader/fixed_inputs，子进程CUDA未初始化。

同一模型解释器正常运行 `validate_smolvla_graph_worker.py --help`，exit0，
导入链不再被 datasets 门阻断。没有模型调用/capture或输入帧读取。
根包缺 datasets 时访问完整 RolloutContext/InteractiveSession/create_strategy 仍明确报缺 extra；
dataset-enabled 根包保留所有 `__all__` 导出及原类对象。

命令统一使用既有 `/home/rp/miniconda3/envs/smolvla-rtc/bin/uv run --no-config --no-project
--offline --no-python-downloads --python <各自既有解释器> python ...`，清除PYTHONPATH、
HF_HUB_OFFLINE=1、TRANSFORMERS_OFFLINE=1，不安装或同步依赖。

原始日志和两个环境的执行前版本清单位于：
`outputs/smolvla_graph_async_contract_r1_preparation_da6ecd7b/`。
保留开发期首个ruff缺空行日志；补一个空行后ruff check通过，format检查两文件不需改动。

新增监督入口是旧D监督逻辑的具名CLI版本：相同单次子进程、295秒SIGTERM/300秒SIGKILL，
正常回收后写执行回执；输出目录和模型日志独占，旧模型输入/用例脚本不改。
它不创建模型、不导入torch、不启动后台写盘，不重复分发失败请求。

## Codex 接续完成的门禁

2026-09-09 接管时分支与完整 HEAD `da6ecd7bf109c558fbcce0ff27e0605e1ed7a37f`
一致；包入口及三项导入测试未再修改，沿用上述39项、模型环境2通过/1跳过和真实D3入口帮助日志。
固定GPU为NVIDIA GeForce RTX 4070 Ti SUPER，未发现并行模型实验。

最终三文件检查首次发现监督脚本的两处 `timezone.utc` 不符合现有ruff的UP017规则，
仅改为等价的 `UTC` 别名。原失败保留在 `final_ruff_check.log`，没有执行模型试运行。
修正后包入口、导入测试和监督脚本三文件的ruff check全部通过，format检查为
`3 files already formatted`；监督脚本 `--help` 和 `git diff --check` 均exit0。
完整命令及退出记录见准备目录 `final_preparation_checks_2.json`，对应四份日志均已保存。

专用模型解释器经既有uv离线运行，使用importlib.metadata枚举版本；Python路径、Python版本、
全部140项包版本记录与原 `model_environment_before.json` 逐项一致。
执行前清单及比较分别为 `model_environment_execution_before.json` 和
`model_environment_execution_before_comparison.json`，没有安装、同步或变更模型环境。

监督脚本已只读核对：仓库根定位正确；使用专用解释器和已回读一致的登记，限制HEAD及独占输出路径；
独占打开model.log后只创建一个子进程，以295秒SIGTERM/300秒SIGKILL限制本次进程组，
wait确认退出后才写执行回执。D3请求表、模型数值、worker/queue及原停止规则未改。
