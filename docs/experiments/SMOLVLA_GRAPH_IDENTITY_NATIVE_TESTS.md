# E 原生闭环：准备与定向测试记录

2026-09-09。以下为本轮实际执行记录；E 模型/native 队列尚未运行。

最终 CPU 定向集合 **46 passed in 9.56s，exit 0**（新增 E 的26项与必要原回归20项）。
固定模型环境实际入口测试 **1 passed in 9.48s，exit 0**，测试内分别运行正常导入和 `--help`。
导入子进程断言 CUDA 未初始化；准备结果不代表 E 的 CUDA/native 验收。

CPU fixture 运行原 `SmolVLAGraphIdentityEngine` worker、planner 与 `ScheduledActionQueue`，
模型、Graph capture 和 Env 使用 CPU fixture。可控时钟/事件让 serialized 等待跨 slot，
async 在旧块可用时与 fixture Env 的实际调用重叠。

覆盖多行/row0接管、early/on-time/late1/2/3/stale、None不推进、慢Env不补发、成功优先于TimeLimit、
首个模型/非有限/native错误与迟返回、CPU双图buffer复用、配对初态不同时模型零调用、
动作数值相同时依然按request/activation追溯来源、独立post chunk、不二次反归一化、
各预算及实际dispatch前拦截、固定20行、未确认join不保存共享数组、原stop/reset/Graph资源回归。

## 保留的开发失败

首个ruff失败为fixture导入名与参数同名触发F811（`ruff_first.log`及对应receipt，exit1）。
改为本文件的CPU fixture；随后新增测试导入顺序触发I001（`ruff_final.log`及对应receipt，exit1）。
整理导入后最终ruff check与format --check均exit0。旧失败日志保持原件。
首轮新增测试为21 passed in 8.87s；扩充并加入capture/setup/owner账目后运行上面的46项。

## 环境与原件

CPU：`/home/rp/miniconda3/envs/smolvla-rtc/bin/python`。
模型：`/home/rp/Workspace/SmolVLA_RTC/libero-reference-venv/bin/python`。
uv仅离线选择上述已有解释器，清除PYTHONPATH，HF_HUB_OFFLINE=1 / TRANSFORMERS_OFFLINE=1。
无安装、同步、升级、降级或其他环境包路径注入。

模型环境140项metadata、Python路径和版本与已接纳D3-r1原件直接一致；固定快照目录浅层存在。
准备原件：`outputs/smolvla_graph_identity_native_preparation_a6966f38/`，正式运行复制这些小文件。
大型D3-r1数组未重读，没有重跑D3 CUDA或C的40条回放。

DevSpace原拦截记录已按本任务书中的明确事实追加并回读。待追加附件未找到，
追加段注明来源是任务书而非缺失附件的逐字内容；FORBIDDEN session边界与旧find_spec拦截区分。
未重试DevSpace，未由不可用通道推断新的模型根因。

## 实际命令与退出

### 首轮新增CPU用例

原始日志：`outputs/smolvla_graph_identity_native_preparation_a6966f38/cpu_tests_first.log`。

```bash
/home/rp/miniconda3/envs/smolvla-rtc/bin/python -m pytest -q --maxfail=1 tests/test_smolvla_graph_identity_native.py
```

exit `0`。

```text
Testing with DEVICE='cuda'
.....................                                                    [100%]
21 passed in 8.87s
```

### 最终CPU定向用例及原回归

原始日志：`outputs/smolvla_graph_identity_native_preparation_a6966f38/cpu_targeted_final.log`。

```bash
/home/rp/miniconda3/envs/smolvla-rtc/bin/uv run --no-config --no-project --offline --no-python-downloads --python /home/rp/miniconda3/envs/smolvla-rtc/bin/python python -m pytest -q --maxfail=1 tests/test_smolvla_graph_identity_native.py tests/policies/rtc/test_scheduled_action_queue.py::test_early_result_waits_for_exact_takeover_index tests/policies/rtc/test_scheduled_action_queue.py::test_exactly_on_time_stage_switches_on_next_get tests/policies/rtc/test_scheduled_action_queue.py::test_any_late_result_is_dropped_without_skipping_its_prefix tests/policies/rtc/test_scheduled_action_queue.py::test_stale_result_cannot_clear_a_newer_plan_across_a_b_a_tasks tests/policies/rtc/test_scheduled_action_queue.py::test_takeover_plan_freezes_padded_private_committed_prefix tests/test_smolvla_graph_identity.py::test_join_timeout_is_not_reported_as_release tests/test_smolvla_graph_identity.py::test_reset_before_capture_finishes_preserves_startup_failure tests/test_smolvla_graph_identity.py::test_real_loop_owner_cpu_publication_and_ready_reset tests/test_smolvla_graph_native.py::test_feature_off_never_installs_runtime tests/test_smolvla_graph_native.py::test_consecutive_noise_output_lifetime_and_all_eight_inputs tests/test_smolvla_graph_native.py::test_graph_audit_requires_capture_and_actual_replay_not_python_hooks tests/rollout/inference/test_predictive_async.py::test_chunk_completion_barrier_precedes_queue_publication tests/rollout/inference/test_predictive_async.py::test_predictive_metrics_request_error_preserves_error_policy tests/rollout/inference/test_predictive_async.py::test_predictive_metrics_sink_closes_after_worker_finishes
```

exit `0`。外层实测 `13.870686` 秒。

```text
Testing with DEVICE='cuda'
..............................................                           [100%]
46 passed in 9.56s
```

### 模型环境真实入口导入及帮助

原始日志：`outputs/smolvla_graph_identity_native_preparation_a6966f38/model_import_entry.log`。

```bash
/home/rp/miniconda3/envs/smolvla-rtc/bin/uv run --no-config --no-project --offline --no-python-downloads --python /home/rp/Workspace/SmolVLA_RTC/libero-reference-venv/bin/python python -m pytest -q --maxfail=1 tests/test_smolvla_graph_identity_native.py::test_actual_entry_import_only_and_help_without_cuda_or_model_work
```

exit `0`。外层实测 `12.907570` 秒。

```text
Testing with DEVICE='cuda'
.                                                                        [100%]
1 passed in 9.48s
```

### 最终ruff与format

原始日志：`outputs/smolvla_graph_identity_native_preparation_a6966f38/ruff_passed.log`。

```bash
/home/rp/Workspace/SmolVLA_RTC/libero-reference-venv/bin/ruff check examples/advanced/predictive_async/libero_graph_identity_native.py tests/test_smolvla_graph_identity_native.py
```

exit `0`。外层实测 `0.032987` 秒。

```bash
/home/rp/Workspace/SmolVLA_RTC/libero-reference-venv/bin/ruff format --check examples/advanced/predictive_async/libero_graph_identity_native.py tests/test_smolvla_graph_identity_native.py
```

exit `0`。外层实测 `0.011629` 秒。

```text
All checks passed!
2 files already formatted
```
