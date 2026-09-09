# D Graph/token/worker 定向测试

2026-09-09。最终 **36 passed in 0.81s，exit0**。使用既有smolvla-rtc环境；无安装/升级。

覆盖token-only无RGB、显式noise不额外采样、缺配对/shape/device/非空state与RTC/未知参数拒绝；
连续RNG、完整输出独立性、task世代重建与capture失败恢复；真实CPU worker的ready后在途reset、
CPU双chunk独立、capture构造未完成时reset保留startup失败、A→B→A stale、join超时、首个错误fatal。
同一D3控制脚本的12事件表也在真实CPU worker上完成，12主调用+12reference、5次fake capture、4次stale。
复用旧engine的屏障、reset/task失效、错误重试、sink关闭和startup中断测试，检验默认接点行为。

最终命令（仓库根目录）：

```bash
env -u PYTHONPATH HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 /home/rp/miniconda3/envs/smolvla-rtc/bin/uv run --no-config --no-project --offline --no-python-downloads --python /home/rp/miniconda3/envs/smolvla-rtc/bin/python python -m pytest -q tests/test_smolvla_graph_identity.py tests/test_smolvla_graph_native.py::test_feature_off_never_installs_runtime tests/test_smolvla_graph_native.py::test_consecutive_noise_output_lifetime_and_all_eight_inputs tests/test_smolvla_graph_native.py::test_reset_reuses_same_task_but_a_b_a_recaptures tests/test_smolvla_graph_native.py::test_incompatible_same_task_inputs_fail_without_replay_and_restore tests/test_smolvla_graph_native.py::test_preparation_exception_restores_rng_sampler_and_buffers tests/test_smolvla_graph_native.py::test_graph_audit_requires_capture_and_actual_replay_not_python_hooks tests/rollout/inference/test_predictive_async.py::test_chunk_completion_barrier_precedes_queue_publication tests/rollout/inference/test_predictive_async.py::test_reset_discards_inflight_result_without_reusing_action_index tests/rollout/inference/test_predictive_async.py::test_task_change_invalidates_inflight_result_but_preserves_active_provenance tests/rollout/inference/test_predictive_async.py::test_predictive_metrics_request_error_preserves_error_policy tests/rollout/inference/test_predictive_async.py::test_predictive_metrics_sink_closes_after_worker_finishes tests/rollout/inference/test_predictive_async.py::test_identity_startup_reset_and_task_boundaries
```

原始输出：

```text
Testing with DEVICE='cuda'
....................................                                     [100%]
36 passed in 0.81s
```

六个改动Python文件的ruff check返回All checks passed，ruff format --check返回6 files already formatted。
日志中的公共DEVICE标签来自conftest能力探测，所选测试实际使用CPU/fake tensor和worker。
CPU测试不生成真实CUDA通过标志；D3必须单独执行。

## 开发期失败证据

模型专用环境的旧测试模块缺datasets，首次收集exit4/1 skipped；随后改用已有CPU环境。
新测试导入路径、在inference_mode外修改inference tensor的测试操作，以及控制脚本误用staged枚举名均已修正。
首次日志与中间失败保留于/tmp/smolvla_d_targeted_tests_initial.log、_1.log、_2.log及
/tmp/smolvla_d_identity_tests_3.log；D3运行后随新输出目录保存。这些开发测试没有执行真实模型或native。
