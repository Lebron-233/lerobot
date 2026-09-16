# F-ACQ1 执行回执：旧锚点通过，LIBERO 配置导入失败

日期：2026-09-17（Asia/Tokyo）。本轮已真实执行一次正式监督入口，不是准备检查，也没有完整完成新初态资格实验。

## 身份、登记与退出

- execution HEAD：`c1d1f65bfe3a102510529f0c500c58c046961665`；分支 `codex/smolvla-graph-native-equivalence`。
- 工作区：`/home/rp/Workspace/SmolVLA_RTC/lerobot`。
- 真实预登记：[5701187626](https://github.com/Lebron-233/lerobot/issues/1#issuecomment-5701187626)。上一轮结构化 POST 返回 HTTP201；本轮只 GET 该实际 ID，正文与本地 registration.md exact，未再次 POST。
- preparation.json SHA256：`2530c70f3f8760a1cd3c7305024da73cbc358180e166ffeac612af0fdfb58fb3`。
- 登记正文 SHA256：`805b361c33d7e85767325c4b3272e6a3d6a06b54d257bb3b22c8b4ee2e2bba8b`，8331 UTF-8 bytes。
- 原 POST 的六字段 JSON 从此前实际工具响应恢复，GET 六字段 JSON 从本轮实际响应保存；这是规范化 JSON，不是原始 HTTP 字节录制。来源说明已存于准备目录 registration_persistence_receipt_20260917.json；未把 GET 伪称为本轮 POST。
- 唯一输出：`/home/rp/Workspace/SmolVLA_RTC/lerobot/outputs/smolvla_action_qualification_c1d1f65b`。
- WebCodex Job：`cce337ea-677b-499e-aa1d-9bf9669f33f7`，terminal/failed，外层 exit2，duration_ms=20634。
- worker PID：567236，exit2，已 wait 收回；后续 `/proc/567236` 不存在。
- 监督阶段 UTC：2026-09-16T16:55:01.319990+00:00 至 2026-09-16T16:55:17.558956+00:00；wall=16.239000335001037 秒。
- attempt1/retry0；无强制终止，无 pending calls，无 active phases，Graph 已释放。

## 真实完成范围

| 项目 | 实际值 |
|---|---:|
| VLA 加载 / 预测器加载 | 1 / 3 |
| 原16个旧验证锚点 exact | 16 / 16（运行器记录） |
| 正式 decoder / 预测器前向 | 48 / 64 |
| 离线 Graph captures | 2 |
| capture 内部 setup / warmup / capture 调用 | 2 / 6 / 2，另计于正式 decoder |
| phase started / returned | 66 / 66 |
| 新 Env / 新 episode / 新资格样本 | 0 / 0 / 0 |
| native / settling / measured / 新图像编码 | 全部0 |
| 训练 / 反传 / 真机 / 旧 test 标签读取 | 全部0 |

`anchor_replay.pt` 已保存（71206435 bytes）；`aligned_cache.pt`、`metric_rows.jsonl`、`selections.json` 均未生成，也没有 episode_* 目录。
锚点通过说明本轮运行器成功重构并重放了冻结旧样本；它不是新初态收益，也没有通过后续完整独立合同审计。
退出后的只读哈希检查确认原10份来源文件均未改变。`frozen_weights_after.pt` 未生成，不能宣称完整运行末尾的内存权重前后 exact 审计已通过。

## 原始首错与根因

```text
libero_action_qualification.py:471
  factory = natural.trace.checkpoint_factory(e.reference.make_native_env_factory(), calls)
libero_reference_qualification.py:370
  from libero.libero import benchmark, get_assets_path
libero/libero/__init__.py:104
  answer = input(...)
EOFError: EOF when reading a line
```

model.log 的最后一条提示：

```text
Do you want to specify a custom path for the dataset folder? (Y/N):
```

本次执行调用沿用了新执行提示词中的 uv 命令，但没有附带此前 native 已登记的环境变量。新入口也没有在模型加载前检查这些变量。只读检查发现 LIBERO_CONFIG_PATH、MUJOCO_GL、PYOPENGL_PLATFORM、HF_HUB_OFFLINE、TRANSFORMERS_OFFLINE、LD_PRELOAD 均未设置，PYTHONPATH 未设置。

安装包的 __init__.py 第5–8行在缺少 LIBERO_CONFIG_PATH 时选择 `~/.libero/config.yaml`；第100–106行在配置文件不存在时调用 input。本机 `/home/rp/.libero/config.yaml` 不存在，因此非交互 worker 在该提示处 EOF。这不是工具安全拦截，不是 GPU 锚点失败，也不是新初态模型阴性。

原有专用配置实际上存在：
`/home/rp/Workspace/SmolVLA_RTC/libero-reference-cache/config/config.yaml`。
SHA256：`1acadaf137bd6e0e9cc1c5130d4904f418f336aa912c417a74aa35be9335dc8b`。
此前固定启动设置见 docs/experiments/LIBERO_REFERENCE_PREPARATION_SUMMARY.json 与 LIBERO_REFERENCE_QUALIFICATION_PLAN.md。无需新建默认配置、安装依赖或变更驱动。

## 已验证的最小修正范围

单独使用 `/usr/bin/env` 的 literal argv，恢复原有进程级环境，执行一次只读 LIBERO 配置导入检查；没有调用正式入口、环境工厂、模拟器、VLA 或预测器：

```text
-u PYTHONPATH
HF_HUB_OFFLINE=1
TRANSFORMERS_OFFLINE=1
MUJOCO_GL=egl
PYOPENGL_PLATFORM=egl
LIBERO_CONFIG_PATH=/home/rp/Workspace/SmolVLA_RTC/libero-reference-cache/config
LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libGLdispatch.so.0:/usr/lib/x86_64-linux-gnu/libGLX.so.0
```

其后仍使用既有 uv、offline/no-project/no-python-downloads 与固定 Python。
诊断结果：无需 input 即导入成功，CUDA 未初始化，新 Env/模型前向均0，配置 SHA256 未变，资产链接 exact 指向 `0b3ea86be5fe169d0fd036ae63d1070ec09e90f6`。
完整证据：environment_diagnostic.json。该检查只证明本次配置导入错误可由上述环境恢复消除，不证明完整环境工厂、渲染或episode已验证。
这不是第二次正式实验，不计为追加科学样本；原 attempt1 失败保持不变。

## 独立审计与科学结论

退出后按合同只运行一次 audit_libero_action_qualification.py，exit2，原文 `ValueError: Run not completed`；independent_audit.json 原样保留，没有覆盖或重审。审计在 CPU 入口确认 CUDA 未初始化后，于运行状态门停止，未进入数组/指标审计，模型前向及 Env 为0。
随后报告生成器完成，生成 REPORT.md/REPORT.json。自动 REPORT.md 优先显示的是审计未接纳错误；运行的真正首错是上文 LIBERO EOFError，原始 result.json/worker_result.json/model.log 均保存。

- independent_contract_accepted=false：本轮技术执行未完整完成。
- heldout_primary_gate_passed=false、heldout_robustness_gate_passed=false 是失败退出的机器标志；科学上均为未评估，而不是负结果。
- 新样本数实际0；没有可用的首动作/chunk/token MSE，没有改善率、配对分母或闭环收益可报告。
- baseline_qualified/realtime_qualified/predictor_benefit_tested 仍 false，risk_thresholds=null，旧 confirmation untouched。

## 交付与后继边界

原失败输出、日志、登记和一次执行记录不删除、不覆盖，不在该输出目录重新启动 worker。本轮没有重跑 --prepare、43项既有测试或 Ruff，没有修改实验源码、原始模型、依赖、驱动或旧 pending 文件。

本轮结果、独立审计及本回执整理为 docs/experiments/SMOLVLA_ACTION_QUALIFICATION_RESULT.md、SMOLVLA_ACTION_QUALIFICATION_RESULT.json、SMOLVLA_ACTION_QUALIFICATION_AUDIT.json、SMOLVLA_ACTION_QUALIFICATION_RECEIPT.md；仅本地保存。实验未成功，未新增 GitHub 结果评论，未提交或推送。

后继技术修复应仅补齐固定启动环境并增加加载前只读环境门：在模型加载前检查专用配置、原资源路径、offline/EGL 设置及无交互导入。样本、种子、模型、权重、容差和科学判据不因本次失败改变。再次正式执行需要独立、明确的新执行记录和预登记，不能伪装为继续原 attempt1，也不能将原目录清空重用。本回执没有启动后续训练、时延实验或闭环。
