# E-RCV2 实际准备测试

2026-09-09。接续HEAD `5e2679a95adefa0de57e10b05506d20f35c7cac2`。
工作目录`/home/rp/Workspace/SmolVLA_RTC/lerobot`，原件位于
`outputs/smolvla_graph_cap_stress_preparation_5e2679a9/`。
本文件根据现存日志及命令退出回执整理；本次接续没有重跑这些已通过的检查。

## 实际结果

| 检查 | 原始输出 | 命令退出码 | 独立外层秒 |
|---|---|---:|---:|
| 首轮CPU | 2 passed、1 failed in 0.60s，首错停止 | 1 | 6.692047 |
| 剩余CPU | 3 passed in 0.41s | 0 | 4.620351 |
| 固定模型解释器入口import/--help | 正常help；CUDA initialized: False | 0 | 4.407288 |
| 首轮Ruff | F811，clock fixture导入与测试参数同名 | 1 | 0.017091 |
| 最终Ruff | All checks passed! | 0 | 未记录 |
| 首轮格式差异 | 2 files would be reformatted | 1 | 未记录 |
| 最终格式 | 2 files already formatted | 0 | 未记录 |
| 当时git diff --check | 空输出 | 0 | 未记录 |

共有5项不同CPU测试通过，来自首轮已通过的2项和第二轮3项；没有一条命令报告“5 passed”。
模型入口是独立命令，不计为第6项pytest测试。日志中的pytest耗时与命令外层耗时分别保留。

首轮命令：

```bash
/home/rp/miniconda3/envs/smolvla-rtc/bin/uv run --no-config --no-project --offline --no-python-downloads \
  --python /home/rp/miniconda3/envs/smolvla-rtc/bin/python python -m pytest -q --maxfail=1 \
  --basetemp=/home/rp/Workspace/SmolVLA_RTC/lerobot/outputs/smolvla_graph_cap_stress_preparation_5e2679a9/cpu_first_tmp \
  tests/test_smolvla_cap_stress.py
```

第二轮仅执行以下三项，保留前两项已通过的结果：

```bash
/home/rp/miniconda3/envs/smolvla-rtc/bin/uv run --no-config --no-project --offline --no-python-downloads \
  --python /home/rp/miniconda3/envs/smolvla-rtc/bin/python python -m pytest -q --maxfail=1 \
  --basetemp=/home/rp/Workspace/SmolVLA_RTC/lerobot/outputs/smolvla_graph_cap_stress_preparation_5e2679a9/cpu_remaining_tmp \
  'tests/test_smolvla_cap_stress.py::test_one_host_pause_with_real_worker_history_and_recovery[candidate]' \
  tests/test_smolvla_cap_stress.py::test_empty_or_unstarted_queue_cannot_claim_recovery \
  tests/test_smolvla_cap_stress.py::test_missing_intervention_and_probes_cannot_supply_stress_evidence
```

`cpu_first_receipt.json`与`cpu_remaining_receipt.json`保存精确argv、exit和外层耗时；对应.log保留原始pytest输出。
现存`finish_preparation.py`记录第二轮命令清除PYTHONPATH，设置HF_HUB_OFFLINE=1、TRANSFORMERS_OFFLINE=1、DEVICE=cpu。
该辅助脚本首次运行停在Ruff失败，不是可以自动继续的runner；后续检查由独立日志和receipt确认。

## 五项覆盖及首错

1. 固定清单：四条均graph_identity_async，task0/2、原state/seeds，disabled/candidate/candidate/disabled顺序及600ms干预；候选总probe预算100。
2. disabled真实CPU worker：startup未暂停，首个planned只暂停一次；实际请求样本约610ms使历史超cap；快bootstrap installed但历史不变，后续planned仍被阻止。
3. candidate真实CPU worker：同一首个planned干预，有效同路径probe逐次丢弃并接纳；原窗口自然回cap，仍保留慢样本，随后planned并从合法row0接管。
4. 全部not_run时，四条合同、两个disabled闭锁与两个candidate恢复字段均不能成立，原科学资格仍false。
5. 缺干预、缺probe的记录不能提供stress恢复证据。

首轮candidate在恢复断言之后调用不存在的`ScheduledActionQueue.next_index`而失败。
只将测试消费步数改为真实request plan的`planned_delay_steps + 1`；第二轮恢复后的takeover断言通过。
候选算法、600ms干预和数值期望没有为此改变。首错日志与当次CPU evidence保留。

首轮Ruff的F811通过模块级`clock = base.clock`复用同一个fixture解决；没有改fixture实现。
原名`ruff_final.log`和`ruff_final_receipt.json`仍表示首次失败，最终通过原件是`ruff_pass.log`/receipt。
`format_first.diff`及receipt保留初次差异，`format_pass.log`/receipt记录最终通过。

夹具明确使用CPU NativeEngine/worker/queue、FakeGraph和受控时钟。
框架打印DEVICE='cuda'不能据此计为CUDA推理；CPU接管断言不能替代尚未运行的native source audit。
原E-RCV1已通过的whole-discard、预算、失效、CPU发布/完成屏障和清理合同继续复用，本次未重复执行。

## 固定模型入口与静态检查

`model_entry_receipt.json`保存完整uv命令：指定
`/home/rp/Workspace/SmolVLA_RTC/libero-reference-venv/bin/python`，
使用runpy实际执行`libero_graph_cap_stress_native.py --help`，执行前后断言CUDA未初始化。
`model_entry.log`实际列出公开参数`--execution-head`和`--output`，exit0；没有加载策略或创建Env。
本轮没有安装、升级、sync或替换解释器。

Ruff使用既有`/home/rp/Workspace/SmolVLA_RTC/libero-reference-venv/bin/ruff`，
范围仅`libero_graph_cap_stress_native.py`与`test_smolvla_cap_stress.py`。
当时`git diff --check`不覆盖未跟踪文件，不能把这一空diff的exit0当作四个新增文件的完整审阅。
接续时已逐一读取新入口、测试、PLAN和MANIFEST；没有发现需要修改算法、暂停时长或冻结条件的问题。
本次按明确路径暂存全部6个E-RCV2新增文件及NEXT_REVIEW后，`git diff --cached --check`实际exit0，
覆盖本次提交的新增文件；三个旧未跟踪文档保持未暂存。

源码确认：`StressEngine._prepare_queue_actions`先返回原Graph适配器的独立CPU chunks，
原设备完成屏障与finite检查已经执行，随后才在首个planned暂停。
继承路径的原最终完成屏障、实际时延采样、tracker接纳及whole-discard处置继续执行；没有新增GPU调用。

## 尚未执行的准备与实测

现有`preparation_gates.json`只表示CPU、入口和lint/格式准备通过，不能代表资源准备或实测放行。
模型版本/包元数据、GPU和磁盘准备查询在前一工具会话被平台安全审查拒绝，未返回查询结果、PID或exit code。
本次接续尚未收到维护者解除记录，没有重试该查询、启动模型/Env、预登记或新native队列。
本轮实测恢复条件与剩余交付见[接续状态](SMOLVLA_GRAPH_CAP_STRESS_PREPARATION.md)。
