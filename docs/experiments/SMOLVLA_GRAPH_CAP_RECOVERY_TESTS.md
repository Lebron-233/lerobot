# E-L1 / E-RCV1 实际准备测试

2026-09-09。工作目录`/home/rp/Workspace/SmolVLA_RTC/lerobot`。
原始日志、独立退出回执及pytest evidence目录：`outputs/smolvla_graph_cap_recovery_preparation_223156f3/`。
全部Python命令清除PYTHONPATH，设置HF_HUB_OFFLINE=1、TRANSFORMERS_OFFLINE=1；无依赖安装或sync。

## 修改前的判别门

```bash
/home/rp/miniconda3/envs/smolvla-rtc/bin/uv run --no-config --no-project --offline --no-python-downloads \
  --python /home/rp/miniconda3/envs/smolvla-rtc/bin/python python -m pytest -q --maxfail=1 \
  --basetemp=/home/rp/Workspace/SmolVLA_RTC/lerobot/outputs/smolvla_graph_cap_recovery_preparation_223156f3/cpu_l1_tmp \
  tests/test_smolvla_cap_recovery.py
```

当时该文件只有4项旧行为characterization；实际 **4 passed in 0.51s，exit0**。
`cpu_l1_first.log` / `cpu_l1_first_receipt.json`保留。
证明原路径闭锁、正控制、旧真实样本raw16及reset/task/new engine边界；没有新模型/native。

## 候选准备与必要回归

```bash
/home/rp/miniconda3/envs/smolvla-rtc/bin/uv run --no-config --no-project --offline --no-python-downloads \
  --python /home/rp/miniconda3/envs/smolvla-rtc/bin/python python -m pytest -q --maxfail=1 \
  --basetemp=/home/rp/Workspace/SmolVLA_RTC/lerobot/outputs/smolvla_graph_cap_recovery_preparation_223156f3/cpu_recovery_tmp \
  tests/test_smolvla_cap_recovery.py \
  tests/test_smolvla_graph_startup_boundary.py::test_real_startup_passes_then_stops_without_get_or_measured_dispatch \
  tests/test_smolvla_graph_startup_boundary.py::test_first_failure_has_no_followup_model_native_or_seed \
  tests/policies/rtc/test_scheduled_action_queue.py::test_any_late_result_is_dropped_without_skipping_its_prefix \
  tests/rollout/inference/test_predictive_async.py::test_chunk_completion_barrier_precedes_queue_publication \
  tests/test_smolvla_graph_identity_native.py::test_unconfirmed_join_never_serializes_shared_worker_arrays
```

首次实际 **4 passed、1 failed in 0.71s，exit1**，其余未运行。
已通过的4项确认默认关闭仍保持旧characterization。首错在测试的最终动作比对：
`torch.from_numpy(native_steps[-1]["action"])`收到原journal保存的list而不是ndarray。
该测试之前的probe丢弃、接纳一次、queue对象/索引/staged不变及原接管断言均已通过，
失败不构成恢复机制的反例。只将读取改为`torch.tensor(..., dtype=torch.float32)`，期望值与候选算法未变。
`cpu_recovery_first.log`及独立receipt、当次evidence原样保留。

随后执行相同节点，basetemp换为`cpu_recovery_remaining_tmp`，仅追加：

```text
-k 'not (original_cap_latch_survives or positive_control_continues or old_samples_recompute or reset_and_task_boundaries)'
```

实际 **21 passed、4 deselected in 9.79s，exit0**；`cpu_recovery_remaining.log` / `_receipt.json`保留精确argv。
四项已通过characterization没有重复执行。**当前代码的25项独立定向覆盖由4+21组成**，不是声称某一条命令25 passed。
新文件17项（含参数化），必要原回归8项。早先修改前的4项不再累加进25的分母。

关键候选证据：真实worker完成49次340ms planned填满50成员窗口，再完成6次400ms planned导致超cap；
原50窗口自然接纳45个10ms恢复probe后回到cap内，此时仍保留5个400ms慢样本，再发生真实planned和合法row0接管。
没有手工向tracker添加恢复样本或清空历史。CPU fake Env建历史最多2000步，实际测试全部在此范围内；
这是受控夹具上限，不改变native280/1120预算。

持续慢500ms probe跑到50上限仍超cap，第51个机会在派发前cap_wait；active耗尽的bootstrap继续排除，reset/task不重置probe预算。
另覆盖原staged/active/plan/prefix/index保留，有限CPU chunks/完成屏障、同路径vision/noise/capture次数，
reset/task/stop后失效不接纳，policy/nonfinite首错fatal，原serialized/async对probe的等待差异，
四行清单/预算及Env close失败不序列化数组。原startup、late整块丢弃、完成屏障和join回归通过。

## 模型环境入口

```bash
/home/rp/miniconda3/envs/smolvla-rtc/bin/uv run --no-config --no-project --offline --no-python-downloads \
  --python /home/rp/Workspace/SmolVLA_RTC/libero-reference-venv/bin/python python -m pytest -q --maxfail=1 \
  tests/test_smolvla_cap_recovery.py::test_actual_recovery_entry_import_and_help_without_inference
```

实际 **1 passed in 8.60s，exit0**；`model_entry.log` / `_receipt.json`。
测试真正导入新入口并执行--help，断言CUDA未初始化，无模型加载/推理或Env。
固定解释器/version/140项metadata与E-S1结束快照exact，`model_environment_before.json` / `model_environment_vs_e_s1.json`留证。

## 静态检查与首错

检查文件：`predictive_async.py`、`libero_graph_identity_native.py`、`libero_graph_cap_recovery_native.py`及新测试。
Ruff使用`/home/rp/Workspace/SmolVLA_RTC/libero-reference-venv/bin/ruff`。
首次check为I001（测试import顺序），exit1，保留`ruff_first.log` / `_receipt.json`。
先显式加入仓库examples导入路径，再整理导入顺序；没有混入其他环境包。
最终四文件`ruff check`为All checks passed、`ruff format --check`为4 files already formatted，均exit0，
见`ruff_final` / `format_final`日志和receipt。格式化只涉及本轮4个Python文件。

每类运行都有具体门：characterization反例将停止候选；队列/计时/预算/失效失败阻止native登记；
入口或lint失败阻止冻结。此次两个开发首错是import排序和夹具list读取，修正原件与期望均可审阅。
没有重跑旧19项、46项或旧GPU/native队列。上述CPU证据不被命名为native恢复结果。
