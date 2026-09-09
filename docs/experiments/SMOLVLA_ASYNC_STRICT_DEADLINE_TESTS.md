# C 严格 deadline 勘误：定向测试记录

2026-09-09。测试使用既有 smolvla-rtc 环境；未安装或升级依赖。

结果：**14 passed in 0.55s，exit 0**。含导入的进程墙时为 6.738675 秒。
测试发现的具体失败将阻止固定 trace 实测：错误接受晚到块、发送了错误的新/旧动作、
清掉较新计划、错误推进空队列索引，或当前未完成延迟泄漏到 planner 历史。

新增验收覆盖 late=1/2/3 的完整 whole-discard 条件，下一旧动作分别为4/5/6。
复用现有提前与准时接管、私有 prefix、单在途/guard、耗尽 underflow、A→B→A stale、
reset/task provenance 和只取消匹配计划用例。队列及推理生产实现未修改。

完整命令（仓库根目录执行）：

```bash
env -u PYTHONPATH HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 /home/rp/miniconda3/envs/smolvla-rtc/bin/python -m pytest -q tests/test_smolvla_async_timing_replay.py tests/policies/rtc/test_scheduled_action_queue.py::test_bootstrap_get_tracks_absolute_index_and_underflow tests/policies/rtc/test_scheduled_action_queue.py::test_takeover_plan_freezes_padded_private_committed_prefix tests/policies/rtc/test_scheduled_action_queue.py::test_plan_creation_enforces_single_in_flight_and_available_guard tests/policies/rtc/test_scheduled_action_queue.py::test_early_result_waits_for_exact_takeover_index tests/policies/rtc/test_scheduled_action_queue.py::test_exactly_on_time_stage_switches_on_next_get tests/policies/rtc/test_scheduled_action_queue.py::test_any_late_result_is_dropped_without_skipping_its_prefix tests/policies/rtc/test_scheduled_action_queue.py::test_late_result_with_exhausted_active_explicitly_underflows tests/policies/rtc/test_scheduled_action_queue.py::test_stale_result_cannot_clear_a_newer_plan_across_a_b_a_tasks tests/policies/rtc/test_scheduled_action_queue.py::test_task_change_invalidates_plan_and_stage_but_preserves_active_provenance tests/policies/rtc/test_scheduled_action_queue.py::test_reset_clears_all_state_without_reusing_action_indexes tests/policies/rtc/test_scheduled_action_queue.py::test_cancel_only_clears_the_matching_plan
```

原始 pytest 输出：

```text
Testing with DEVICE='cuda'
..............                                                           [100%]
14 passed in 0.55s
```

公共 conftest 的 DEVICE 标签来自环境能力探测；所选队列测试和 VirtualControl 使用 CPU tensor/device，未加载模型。

两个改动 Python 文件的定向静态检查：

```bash
/home/rp/Workspace/SmolVLA_RTC/libero-reference-venv/bin/ruff check examples/advanced/predictive_async/replay_smolvla_async_timing.py tests/test_smolvla_async_timing_replay.py
/home/rp/Workspace/SmolVLA_RTC/libero-reference-venv/bin/ruff format --check examples/advanced/predictive_async/replay_smolvla_async_timing.py tests/test_smolvla_async_timing_replay.py
```

返回 `All checks passed!`、`2 files already formatted`。本轮仅运行上述定向用例；A/B采用已接受的记录。
