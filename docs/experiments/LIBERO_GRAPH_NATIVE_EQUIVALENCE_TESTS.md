# Graph runtime 定向测试记录

日期：2026-09-09。所有检查在真实模型/原生协议执行之前完成。

## 结果

- Ruff check：5 个新增/修改 Python 文件通过；ruff format 已应用。
- `tests/test_smolvla_graph_native.py`、`tests/test_libero_cuda_graph_compute.py`、
  `tests/rollout/test_libero_reference_qualification.py`、`tests/rollout/test_libero_reference_smoke.py`：
  **59 passed，2.33 s**。这些是 CPU/fake model、fake env 的合同测试，未加载模型权重，未派发真实 native episode。
- 最后修改汇总输出后重跑新合同测试，并尝试默认配置检查：**14 passed，1 skipped，1.26 s**。
  `tests/rollout/inference/test_predictive_async_config.py` 整个可选模块因 `datasets` 未安装而跳过。
  未为此安装依赖。默认 sync/RTC factory 和 policy 源文件未修改。
- 专用 pytest conftest 打印 `Testing with DEVICE='cuda'`，上述具体用例均显式构造 CPU tensor 或 fake env；
  这条全局提示不构成真实 GPU 模型验证证据。

## 检测的具体失败

输出生命周期用例持有第一次返回值，第二次 replay 写固定缓冲后比较第一次内容。
连续 RNG 用例让 fake capture 真正消耗随机数，并比较两个连续请求的实际 noise 及完整输出。
八输入刷新用例同时更新双路视觉、双 masks、语言 tokens/mask、state、noise；不同 shape/dtype/device
在第二次 replay 前失败。任务 A→B→A 必须出现三个不同 capture，episode reset 清旧输出但保留同任务图。
非 finite 输出用例执行 10 个 fake settling 后失败，证明 measured actions=0、环境关闭、sampler 恢复且保留数组。
graph 审计拒绝用十次 Python hook 冒充 replay，并要求有效 capture 与实际 replay_count=1。
worker 技术错误用例只派发第一条；配对分歧用例只派发当前两条；之后无派发且 sampler 恢复。
缺 final 的已开始 native call 标 unknown，未启动的其余 39 条保持 not_run。
旧用例覆盖 280 步边界、success/timeout 同时发生、每 episode reset/seed 顺序、计数一致及 cleanup 失败撤销成功。

## 首次失败及处理

第一次 pytest 收集命令漏传已固定的 `LIBERO_CONFIG_PATH`。
导入 `lerobot.envs.libero` 时，LIBERO 提示 `Do you want to specify a custom path for the dataset folder? (Y/N):`，
随后 pytest 以 `OSError: pytest: reading from stdin while output is captured!` 退出 2；尚未执行测试。
补回既有配置路径和 EGL/offline 启动变量后，同一测试范围通过。
没有改 LIBERO 配置、依赖版本、样本或实验条件。

## 命令

```bash
env -u PYTHONPATH DEVICE=cpu HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 MUJOCO_GL=egl PYOPENGL_PLATFORM=egl LIBERO_CONFIG_PATH=/home/rp/Workspace/SmolVLA_RTC/libero-reference-cache/config LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libGLdispatch.so.0:/usr/lib/x86_64-linux-gnu/libGLX.so.0 /home/rp/Workspace/SmolVLA_RTC/libero-reference-venv/bin/python -m pytest -q tests/test_smolvla_graph_native.py tests/test_libero_cuda_graph_compute.py tests/rollout/test_libero_reference_qualification.py tests/rollout/test_libero_reference_smoke.py
```

```bash
env -u PYTHONPATH HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 LIBERO_CONFIG_PATH=/home/rp/Workspace/SmolVLA_RTC/libero-reference-cache/config /home/rp/Workspace/SmolVLA_RTC/libero-reference-venv/bin/python -m pytest -q tests/test_smolvla_graph_native.py tests/rollout/inference/test_predictive_async_config.py -k 'test_smolvla_graph_native or defaults or default_off'
```
