# E-S1 定向测试实际记录

2026-09-09；实现、协议和测试尚未提交时执行。本轮无开发测试或lint失败，以下均为首次运行结果。
原始日志和独立命令退出回执位于：
`outputs/smolvla_graph_startup_boundary_preparation_9038ba38/`。

本轮修改只有独立startup入口、定向测试及新协议/审计文档，没有修改生产算法或旧E入口。
运行前已明确各门的失败用途：CPU合同失败则修正新入口并保留首错，模型环境入口失败则阻止登记，
lint/format失败则修正新文件。没有重跑旧46项，也没有在测试门调用真实模型推理或native Env。

## CPU 合同与必要回归

环境：清除PYTHONPATH，`HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1`。
工作目录：`/home/rp/Workspace/SmolVLA_RTC/lerobot`。

```bash
/home/rp/miniconda3/envs/smolvla-rtc/bin/uv run --no-config --no-project --offline --no-python-downloads \
  --python /home/rp/miniconda3/envs/smolvla-rtc/bin/python python -m pytest -q --maxfail=1 \
  tests/test_smolvla_graph_startup_boundary.py \
  tests/test_smolvla_graph_identity.py::test_join_timeout_is_not_reported_as_release \
  tests/test_smolvla_graph_identity.py::test_reset_before_capture_finishes_preserves_startup_failure \
  tests/rollout/inference/test_predictive_async.py::test_chunk_completion_barrier_precedes_queue_publication
```

实际：**19 passed in 8.33s，exit 0**；外层命令wall 13.319557秒。
证据：`cpu_first.log`、`cpu_first_receipt.json`。
其中新文件16项（含边界参数化），必要原回归3项。

测试复用实际gate、LatencyTracker、planner、worker和queue，以CPU fake policy/Env/Graph及受控时钟驱动。
覆盖旧实际样本触发9>8且未进入tracker；350ms附近原取整容差及margin一次；
float32输入、linear P90、窗口滚动、清空、单位、原始required与cap/available截断的区别；
cold/probe/fresh顺序、epoch0/0/1及仅probe进入tracker、原owner播种一次；
通过ready立即停止且get/measured均0；gate、模型和初态首错后不补调用；
主策略与零measured预算在派发前生效；固定manifest拒绝扩样；
stop/join/释放/恢复/关闭、在途失效和未确认清理时不保存数组。

## 固定模型环境的新入口

同样使用上述offline环境变量并清除PYTHONPATH。

```bash
/home/rp/miniconda3/envs/smolvla-rtc/bin/uv run --no-config --no-project --offline --no-python-downloads \
  --python /home/rp/Workspace/SmolVLA_RTC/libero-reference-venv/bin/python python -m pytest -q --maxfail=1 \
  tests/test_smolvla_graph_startup_boundary.py::test_actual_entry_import_only_and_help_without_env_or_model
```

实际：**1 passed in 8.23s，exit 0**；外层命令wall 11.319655秒。
证据：`model_entry.log`、`model_entry_receipt.json`。
该测试对子进程中的实际新入口执行import-only及`--help`，确认导入不初始化CUDA、不创建Env或推理模型。
无安装、sync、版本调整或混入另一环境包。140项distribution metadata与旧E结束快照直接比较相同，
见 `model_environment_before.json` 和 `model_environment_vs_old_e.json`。

## 格式与静态检查

以下命令的文件参数均为：
`examples/advanced/predictive_async/libero_graph_startup_boundary.py tests/test_smolvla_graph_startup_boundary.py`。

```bash
/home/rp/Workspace/SmolVLA_RTC/libero-reference-venv/bin/ruff check examples/advanced/predictive_async/libero_graph_startup_boundary.py tests/test_smolvla_graph_startup_boundary.py
/home/rp/Workspace/SmolVLA_RTC/libero-reference-venv/bin/ruff format --check examples/advanced/predictive_async/libero_graph_startup_boundary.py tests/test_smolvla_graph_startup_boundary.py
```

实际分别为 `All checks passed!`、`2 files already formatted`，均exit 0。
证据：`ruff_first.log` / `ruff_first_receipt.json`、`format_check.log` / `format_check_receipt.json`。
测试前已对这两个新文件执行ruff format，2 files reformatted；没有因测试失败改变生产计时或参数。

## 与真实诊断分开

这些是准备门，不是GPU/native结果。旧证据的23个startup、315个planner判据及初态CPU原件复算
属于E-S0，见独立AUDIT。E-S1真实启动必须在本文件、实现、PLAN/MANIFEST提交推送并完成新登记回读后进行，
其实际gate结果、调用预算、清理和退出写入新的RESULT。
