# C 定向测试记录

使用既有 smolvla-rtc Python，CPU tensor/fake policy，无权重、无 native env。

首次运行：新驱动误用了不存在的 `engine.is_ready`，导致一个测试失败，其他 **31 passed**。
首个失败为 `AttributeError: 'PredictiveAsyncInferenceEngine' object has no attribute 'is_ready'`。
改为既有 `ready` 属性并修正完成计数后，仅重跑被改动驱动的两项：**2 passed，0.33 s**。
已有的 30 项队列/线程/默认关闭检查不受驱动修改影响，无重复重跑。
Ruff check / format 通过。

新测试明确检查时钟在无动作时仍推进、planner 仅使用已完成历史，以及边界报告会如实指出
轻微迟到裁剪缺失。测试通过代表检测器正确；不代表轻微迟到合同已经实现。
已有线程测试覆盖真实单 worker staging、完成 barrier、late discard、reset、task epoch 和索引不复用。
已有默认配置及 metrics 默认关闭测试在此环境通过，补齐参考环境中因 datasets 缺失而跳过的两项检查。

```bash
env -u PYTHONPATH HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 /home/rp/miniconda3/envs/smolvla-rtc/bin/python -m pytest -q tests/test_smolvla_async_timing_replay.py tests/policies/rtc/test_scheduled_action_queue.py tests/rollout/inference/test_predictive_async.py::test_identity_request_stages_without_early_switch_or_second_vision_pass tests/rollout/inference/test_predictive_async.py::test_chunk_completion_barrier_precedes_queue_publication tests/rollout/inference/test_predictive_async.py::test_late_predictive_chunk_is_discarded_without_skipping_its_prefix tests/rollout/inference/test_predictive_async.py::test_task_change_invalidates_inflight_result_but_preserves_active_provenance tests/rollout/inference/test_predictive_async.py::test_reset_discards_inflight_result_without_reusing_action_index tests/rollout/inference/test_predictive_async_config.py::test_predictive_async_config_defaults tests/rollout/inference/test_predictive_async_config.py::test_predictive_metrics_default_off_does_not_construct_sink
```

修复后：

```bash
env -u PYTHONPATH HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 /home/rp/miniconda3/envs/smolvla-rtc/bin/python -m pytest -q tests/test_smolvla_async_timing_replay.py
```
