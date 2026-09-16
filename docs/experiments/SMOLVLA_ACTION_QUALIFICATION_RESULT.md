# F-ACQ1：冻结中心化候选的新初态资格实验

execution HEAD：`c1d1f65bfe3a102510529f0c500c58c046961665`。
运行状态：`technical_failure`；独立合同接纳：`False`；主门：`False`；稳健门：`False`。

本轮是开发中已出现的 task 6/7 的新初态资格评估，不是全新任务 benchmark。只由 identity async 采集；预测器没有控制环境。原 test/confirmation 不读取。
oracle 是未来视觉、当前 state、相同语言/noise 下冻结策略的输出，不是专家动作或成功率上界。

**技术证据未被独立接纳：不得把以下状态表述为模型阴性或资格通过。**

```text
Traceback (most recent call last):
  File "/home/rp/Workspace/SmolVLA_RTC/lerobot/examples/advanced/predictive_async/audit_libero_action_qualification.py", line 285, in main
    result = audit(output)
             ^^^^^^^^^^^^^
  File "/home/rp/Workspace/SmolVLA_RTC/lerobot/examples/advanced/predictive_async/audit_libero_action_qualification.py", line 108, in audit
    require(result["status"] == "completed" and result["first_failure"] is None, "Run not completed")
  File "/home/rp/Workspace/SmolVLA_RTC/lerobot/examples/advanced/predictive_async/audit_libero_action_qualification.py", line 17, in require
    raise ValueError(message)
ValueError: Run not completed

```
## 执行与退出

```json
{
  "counts": {
    "predictor_loads": 3,
    "vla_loads": 1,
    "predictor": 64,
    "decoder": 48,
    "anchor_exact": 16,
    "offline_captures": 2
  },
  "native_budget": {},
  "episodes_completed": 0,
  "attempts": 1,
  "retries": 0,
  "execution": {
    "command": [
      "/home/rp/Workspace/SmolVLA_RTC/libero-reference-venv/bin/python",
      "-u",
      "-X",
      "faulthandler",
      "/home/rp/Workspace/SmolVLA_RTC/lerobot/examples/advanced/predictive_async/libero_action_qualification.py",
      "--execution-head",
      "c1d1f65bfe3a102510529f0c500c58c046961665",
      "--output",
      "/home/rp/Workspace/SmolVLA_RTC/lerobot/outputs/smolvla_action_qualification_c1d1f65b",
      "--worker"
    ],
    "child_pid": 567236,
    "child_exit_code": 2,
    "exit_confirmed": true,
    "started_at_utc": "2026-09-16T16:55:01.319990+00:00",
    "finished_at_utc": "2026-09-16T16:55:17.558956+00:00",
    "wall_seconds": 16.239000335001037,
    "stop_reason": null,
    "forced_termination": false,
    "pending_calls_at_exit": [],
    "active_phases_at_exit": {}
  }
}
```

本轮没有重新训练、选择检查点、调残差幅度或训练风险门；动作指标改善不代表视觉 token 整体更准确。wall 包含审计重放/采集/离线评估，不是部署延迟比较。闭环成功率、自然负载恢复、真机安全与统计显著性均不成立。baseline_qualified/realtime_qualified/predictor_benefit_tested 仍 false，risk_thresholds=null。

下一步：仅在合同接纳且主门、稳健门均通过后，另立真实完整计算路径的配对时延合同；若主门通过但稳健门未过，回开发集研究残差收缩/训练目标，禁止用本轮新初态集调参。若主门未过，保留负结果并分析跨初态失败，不追加样本直到通过。
