"""Render F-ACQ1 results for the issue thread, including adverse sample outcomes."""

import argparse
import json
import math
from collections import Counter
from pathlib import Path


def render(result, audited):
    accepted = audited.get("independent_contract_accepted") is True
    primary = accepted and audited.get("heldout_primary_gate_passed") is True
    robust = accepted and audited.get("heldout_robustness_gate_passed") is True
    lines = ["# F-ACQ1：冻结中心化候选的新初态资格实验", "",
             f"execution HEAD：`{result.get('execution_head', 'unknown')}`。",
             f"运行状态：`{result.get('status', 'unknown')}`；独立合同接纳：`{accepted}`；"
             f"主门：`{primary}`；稳健门：`{robust}`。", "",
             "本轮是开发中已出现的 task 6/7 的新初态资格评估，不是全新任务 benchmark。"
             "只由 identity async 采集；预测器没有控制环境。原 test/confirmation 不读取。",
             "oracle 是未来视觉、当前 state、相同语言/noise 下冻结策略的输出，不是专家动作或成功率上界。", ""]
    if not accepted:
        lines += ["**技术证据未被独立接纳：不得把以下状态表述为模型阴性或资格通过。**", "",
                  "```text", str(audited.get("first_failure") or result.get("first_failure") or "Audit missing"), "```"]
    summary = audited.get("summary", result)
    if summary.get("metrics"):
        lines += ["## 新初态结果（episode 等权）", "",
                  "| 条件 | 样本 / episode | 首动作7D MSE | 50×7D MSE | token MSE |",
                  "|---|---:|---:|---:|---:|"]
        for arm, values in summary["metrics"].items():
            m = values["macro"]
            if m:
                lines.append(f"| {arm} | {values['samples']} / {len(values['episodes'])} | "
                             f"{m['row0']:.12f} | {m['chunk']:.12f} | {m['latent']:.9f} |")
            else:
                lines.append(f"| {arm} | 0 / 0 | NA | NA | NA |")
        lines += ["", "## 配对、集中度和不利结果", ""]
        for name, c in summary["contrasts"].items():
            lines.append(f"相对 `{name}`：配对 {c['paired_samples']} 样本 / {c['paired_episodes']} episode；"
                         f"严格改善 {c['sample_improved']} 样本、{c['episode_improved']} episode；"
                         f"配对均值收益（对照−中心化） `{c['macro_benefit']}`。")
            lines.append("")
        c = summary["contrasts"]["base"]
        per_sample = c["per_sample"]
        sizes = Counter(tuple(r["key"][:2]) for r in per_sample)
        contributions = [{**r, "macro_contribution": r["control_minus_centered"] / sizes[tuple(r["key"][:2])] / len(sizes)}
                         for r in per_sample]
        total = math.fsum(r["macro_contribution"] for r in contributions)
        top = max(contributions, key=lambda r: r["macro_contribution"]) if contributions else None
        share = top["macro_contribution"] / total if top and total > 0 else None
        lines += [f"最大单 episode 净收益份额：`{c['largest_episode_net_share']}`；"
                  f"最大单样本：`{top['key'] if top else None}`，净收益份额：`{share}`。"
                  "份额可能大于1（其他样本抵消收益）；非正净收益时不计算份额。", "",
                  f"留一 episode 后全部配对均值收益：`{json.dumps(c['leave_one_episode_out'], ensure_ascii=False)}`。", "",
                  "相对基底差值最不利的至多10个样本（正数改善、负数恶化；不删例）：", "",
                  "| task / state / request | 基底−中心化首动作MSE |", "|---|---:|"]
        for r in sorted(per_sample, key=lambda r: r["control_minus_centered"])[:10]:
            lines.append(f"| {'/'.join(map(str, r['key']))} | {r['control_minus_centered']:.12f} |")
        lines += ["", "## 冻结判据", "", "```json", json.dumps({
            "primary_checks": summary["primary_checks"], "robustness_checks": summary["robustness_checks"],
            "delay_descriptive": summary["delay_descriptive"]}, ensure_ascii=False, indent=2), "```", ""]
    lines += ["## 执行与退出", "", "```json", json.dumps({k: result.get(k) for k in
              ("counts", "native_budget", "episodes_completed", "attempts", "retries", "execution")},
              ensure_ascii=False, indent=2), "```", "",
              "本轮没有重新训练、选择检查点、调残差幅度或训练风险门；动作指标改善不代表视觉 token 整体更准确。"
              "wall 包含审计重放/采集/离线评估，不是部署延迟比较。闭环成功率、自然负载恢复、真机安全与统计显著性均不成立。"
              "baseline_qualified/realtime_qualified/predictor_benefit_tested 仍 false，risk_thresholds=null。", "",
              "下一步：仅在合同接纳且主门、稳健门均通过后，另立真实完整计算路径的配对时延合同；"
              "若主门通过但稳健门未过，回开发集研究残差收缩/训练目标，禁止用本轮新初态集调参。"
              "若主门未过，保留负结果并分析跨初态失败，不追加样本直到通过。"]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = json.loads((args.output / "result.json").read_text())
    audit_path = args.output / "independent_audit.json"
    audited = json.loads(audit_path.read_text()) if audit_path.exists() else {}
    report = render(result, audited)
    metric_path = args.output / "metric_rows.jsonl"
    per_sample = [json.loads(line) for line in metric_path.read_text().splitlines()] if metric_path.exists() else []
    with (args.output / "REPORT.md").open("x") as stream:
        stream.write(report)
    with (args.output / "REPORT.json").open("x") as stream:
        json.dump({"result": result, "independent_audit": audited, "per_sample_metrics": per_sample},
                  stream, ensure_ascii=False, indent=2)
    print(json.dumps({"report": str(args.output / "REPORT.md"), "bytes": len(report.encode())}))


if __name__ == "__main__":
    raise SystemExit(main())
