# 给 Codex 的 F-ACQ1 执行提示词

请完成一次 F-ACQ1 冻结候选新初态资格实验，并把可审核结果回复到
`Lebron-233/lerobot` 的 Issue #1。执行代码已开发，不需要重新设计实验或训练模型。
代码身份以发布本提示词的评论中给出的 **CODE_HEAD** 为准。

## 必读与边界

工作区：`/home/rp/Workspace/SmolVLA_RTC/lerobot`。
分支：`codex/smolvla-graph-native-equivalence`。
先读AGENTS.md、原评论5683274000、SMOLVLA_ACTION_CENTERED_RESULT.md、
SMOLVLA_ACTION_CENTERED_NEXT.md和SMOLVLA_ACTION_QUALIFICATION_PLAN.md。
只执行新合同。保留旧pending/未提交文档，禁止reset/clean/stash、全量git add、force push。
若HEAD不是评论指定CODE_HEAD，不自行切换/覆盖工作树；先报告真实差异。
不要重跑F-ACR1/F-PFX1或旧prepare，不安装依赖、不改变驱动，不修改门/容差/权重。
遇到安全拦截停止相关动作并报告，不换工具绕过。技术错误只保存首错，不自动修后重跑正式worker。

## A. CPU准备和预登记

使用固定解释器与已有uv，所有Python命令均用下面的uv前缀；不下载或安装：

```bash
cd /home/rp/Workspace/SmolVLA_RTC/lerobot
UV=/home/rp/miniconda3/envs/smolvla-rtc/bin/uv
PY=/home/rp/Workspace/SmolVLA_RTC/libero-reference-venv/bin/python
HEAD=$(git rev-parse HEAD)
# 将HEAD与发布评论内CODE_HEAD逐字核对；分支/工作树也必须符合上述边界。
PREP="$PWD/outputs/smolvla_action_qualification_preparation_${HEAD:0:8}"
OUT="$PWD/outputs/smolvla_action_qualification_${HEAD:0:8}"

"$UV" run --offline --no-project --no-python-downloads --python "$PY" -m pytest \
  tests/test_libero_action_qualification.py tests/test_libero_action_centered.py -q
"$UV" run --offline --no-project --no-python-downloads --python "$PY" -m ruff check \
  examples/advanced/predictive_async/libero_action_qualification.py \
  examples/advanced/predictive_async/audit_libero_action_qualification.py \
  examples/advanced/predictive_async/report_libero_action_qualification.py \
  tests/test_libero_action_qualification.py
"$UV" run --offline --no-project --no-python-downloads --python "$PY" \
  examples/advanced/predictive_async/libero_action_qualification.py --execution-head "$HEAD" --prepare
```

每一步分别记录命令/退出码；任一步非0即停，不让最后一个命令exit0掩盖前面的失败。
上面是合成CPU测试，控制台既有“DEVICE=cuda”提示不是已经进行了真实GPU实验。
本次开发未执行正式`--prepare`；初态0–3是否未使用，必须由该CPU身份门确认。
若目标已出现、历史格式未知或目录已存在，停止并反馈，不能自动替换state或删除目录。
允许准备失败；不能为了“完成实验”越过资格检查。

通过后核对现有GPU/驱动可用性，不进行系统维护。保存环境metadata及一次nvidia-smi输出。
准备正文`$PREP/registration.md`，至少包含：
`F-ACQ1-REGISTER:<完整HEAD>`、完整execution HEAD、唯一OUT绝对路径、
preparation.json SHA256、计划文件路径、固定8个身份/seed、权重来源、0训练、各预算、
attempt1/retry0、全部停止规则，以及“结果无论阳性阴性均报告，不扩大范围”。

只POST一次，然后按POST返回的实际数字ID单次GET，不轮询整个Issue：

```bash
gh api --method POST repos/Lebron-233/lerobot/issues/1/comments \
  -F body=@"$PREP/registration.md" > "$PREP/registration_post.json"
# 从返回JSON的id字段读取实际ID；禁止猜测、沿用历史ID或重复POST。
ID=$("$UV" run --offline --no-project --no-python-downloads --python "$PY" -c \
  'import json,sys; print(json.load(open(sys.argv[1]))["id"])' "$PREP/registration_post.json")
gh api "repos/Lebron-233/lerobot/issues/comments/$ID" > "$PREP/registration_readback.json"
```

另外保存POST/GET退出码。正文exact和身份/源哈希门由正式入口再次验证。
若POST状态不明，不自动重发；没有可靠实际ID时停止并说明发布状态。

## B. 唯一正式worker

```bash
"$UV" run --offline --no-project --no-python-downloads --python "$PY" \
  examples/advanced/predictive_async/libero_action_qualification.py \
  --execution-head "$HEAD" --output "$OUT"
```

使用前台监督入口，不能直接调用隐藏的--worker；记录外层起止/退出码。
入口依次完成16个旧验证锚点exact检查、8个identity新初态采集、N≤32样本离线配对评估。
新采集不会由预测器控制。首错即终止后续；只管理自己的进程，不重试、不补episode、
不增加训练或更换普通组等对照。worker退出后确认自有子进程已收回。
正式调用非0时仍须保留日志并反馈；不要因为shell的set -e丢失失败报告。

## C. 退出后独立CPU审计与报告

成功完成或产生result.json后，单次执行审计；失败也保存审计结果，不能覆盖重审：

```bash
"$UV" run --offline --no-project --no-python-downloads --python "$PY" \
  examples/advanced/predictive_async/audit_libero_action_qualification.py --output "$OUT"
"$UV" run --offline --no-project --no-python-downloads --python "$PY" \
  examples/advanced/predictive_async/report_libero_action_qualification.py --output "$OUT"
```

审计和报告分别记录退出码。没有result.json的准备/启动失败，改写明确的失败报告，
不要伪造“模型失败”或完整调用数；不为得到报告重新启动实验。
审计模型前向/新Env/CUDA初始化必须为0。技术合同接纳和两个科学门分别报告，
绝不把“脚本exit0”解释为模型假设通过。

## D. 提交并评论反馈

将本轮REPORT.md、REPORT.json、独立审计JSON、执行回执分别整理为
`docs/experiments/SMOLVLA_ACTION_QUALIFICATION_RESULT.md`、
`SMOLVLA_ACTION_QUALIFICATION_RESULT.json`、`SMOLVLA_ACTION_QUALIFICATION_AUDIT.json`、
`SMOLVLA_ACTION_QUALIFICATION_RECEIPT.md`。仅显式暂存这些本轮结果文件并提交推送；
大型pt数组保留在唯一OUT，不提交Git；旧pending不动。

在Issue #1新增一条结果评论，唯一标识`F-ACQ1-RESULT:<execution_HEAD>`。
内容必须包括：execution/result HEAD、原预登记实际ID、输出目录、固定commit报告链接、
真实N/M/episode数、全部对照的首动作/chunk/token MSE、配对分母、样本与episode改善/恶化、
最坏样本、最大单episode/单样本净贡献、留一episode、delay分层、零动作/锚点exact计数、
权重冻结、实际native/decoder/predictor/encoding/capture预算及内部setup/warmup/capture、
phase started/returned、worker和外层退出码、first_failure、pending/active、强制终止与重试数，
以及独立合同接纳、主门、稳健门三个不同结论。

可引用自动REPORT.md，不能只摘好看的均值。阴性与技术失败都必须发布。
POST一次后按实际ID单次GET核对正文exact，保存独立发布回执；若无法确认发布，明确说明。
报告生产默认和闭环资格仍false，risk_thresholds=null，旧confirmation untouched。
最后根据新计划的分支规则提出下一轮建议，但**不要自动启动额外训练、时延实验或闭环**。
最终回复用户：真实完成内容、未完成/阻塞、结果commit、结果评论URL和唯一输出路径。
