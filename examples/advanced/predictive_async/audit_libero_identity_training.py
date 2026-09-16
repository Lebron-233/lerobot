"""F-ITC1 CPU evidence audit: NumPy metrics, soft objectives, AdamW arithmetic.

Saved gradients are not independently re-differentiated; no model forward occurs.
"""

import argparse
import json
import math
import time
import traceback
from collections import Counter, defaultdict
from pathlib import Path

import audit_libero_identity_anchor_probe as old
import libero_identity_training as r
import numpy as np
import torch

require, compare, array = r.require, old.compare, old.array


def load(path):
    return torch.load(path, map_location="cpu", weights_only=False)


def loss_value(metrics, identity, scales, weight, arm):
    require(arm in r.ARMS, "Unknown audited objective")
    value = metrics["latent"]/scales["latent"] + weight*metrics["row0"]/scales["row0"]
    if arm == "guarded":
        value += metrics["chunk"]/scales["chunk"]
        value += max(0.0, metrics["row0"]-identity["row0"])/scales["row0"]
        value += max(0.0, metrics["chunk"]-identity["chunk"])/scales["chunk"]
    return value


def adamw_step(previous, gradients, state):
    """Independent float64 arithmetic; state is local audit scratch, not a model."""
    result = {}
    for name, value in previous.items():
        if name not in gradients:
            result[name] = array(value)
            continue
        g = array(gradients[name])
        m, v, step = state.get(name, (np.zeros_like(g), np.zeros_like(g), 0))
        m, v, step = 0.9*m + 0.1*g, 0.999*v + 0.001*g*g, step+1
        state[name] = (m, v, step)
        result[name] = array(value)*(1-r.LR*r.WD) - r.LR*(m/(1-0.9**step))/(np.sqrt(v/(1-0.999**step))+1e-8)
    return result


def evidence_check(ev, sample, actions, zero=False):
    r.pilot.require_equal(ev["actions"], actions, "Actual actions differ")
    r.pilot.require_equal(ev["mask"], sample["mask"], "Receiver mask differs")
    for i in range(2):
        a, z = ev["action_delta"][i], ev["zero_delta"][i]
        require(a.shape == z.shape == sample["inputs"][i].shape and torch.isfinite(a).all()
                and torch.isfinite(z).all(), "Invalid saved branch outputs")
        raw = sample["inputs"][i].float() + (a.float()-z.float())
        r.pilot.require_equal(ev["raw"][i], raw, "Subtract-before-add differs")
        r.pilot.require_equal(ev["visual"][i], raw.to(torch.bfloat16).float(), "BF16 boundary differs")
        if zero:
            r.pilot.require_equal(a, z, "Zero branch equality failed")
            r.pilot.require_equal(raw, sample["inputs"][i].float(), "Zero raw differs")


def decision(rows):
    """Decision independent of runner statistics/tolerance implementations."""
    val = [x for x in rows if x["split"] == "validation"]
    train = [x for x in rows if x["split"] == "train"]
    require(len(val) == 16 and len(train) == 72, "Decision coverage differs")

    def episodes(subset, arm, metric):
        groups = defaultdict(list)
        for x in subset:
            groups[tuple(x["key"][:2])].append(x["metrics"][arm][metric])
        return {k: float(np.mean(v)) for k, v in groups.items()}

    def mean(arm, metric, subset=val):
        return float(np.mean(list(episodes(subset, arm, metric).values())))

    def tol(x):
        return 1e-7+1e-6*abs(x)

    def harm(subset, arm):
        return max([0.0]+[x["metrics"][arm]["row0"]-x["metrics"]["identity"]["row0"] for x in subset])

    checks = []
    for control in ("identity", "guarded_mismatched"):
        subset = [x for x in val if control in x["metrics"]]
        require(subset, "Empty validation pairing")
        a, b = mean("guarded_true", "row0", subset), mean(control, "row0", subset)
        checks.append(a < b-tol(b))
    a, b = mean("guarded_true", "row0"), mean("frozen", "row0")
    checks.append(a <= b+tol(b))
    for control in ("frozen", "plain_true"):
        a, b = mean("guarded_true", "chunk"), mean(control, "chunk")
        checks.append(a < b-tol(b))
        a, b = harm(train, "guarded_true"), harm(train, control)
        checks.append(a < b-tol(b))
    a, b = episodes(val, "guarded_true", "row0"), episodes(val, "identity", "row0")
    checks.append(sum(a[k] < b[k]-tol(b[k]) for k in a) >= 3)
    checks.append(sum(x["metrics"]["guarded_true"]["row0"] < x["metrics"]["identity"]["row0"]
                      - tol(x["metrics"]["identity"]["row0"]) for x in val) > 8)
    a, b = harm(val, "guarded_true"), harm(val, "frozen")
    checks.append(a <= b+tol(b))
    return bool(all(checks))


def audit(output):
    torch.set_num_threads(1)
    require(not torch.cuda.is_initialized(), "Audit initialized CUDA")
    result = json.loads((output / "result.json").read_text())
    prep = r.validate(result["execution_head"])
    require(output == r.paths(result["execution_head"])[1], "Output identity differs")
    require(result["status"] == "completed" and result["first_failure"] is None, "Run not completed")
    ex = result["execution"]
    require(ex["child_exit_code"] == 0 and ex["exit_confirmed"] is True and not ex["stop_reason"]
            and not ex["forced_termination"] and not ex["pending"] and not ex["active"], "Exit not closed")
    require(result["attempts"] == 1 and result["retries"] == 0 and result["vla_frozen"]
            and result["graph_released"], "Freeze or attempt differs")
    compare(result["specification"], r.specification())
    for key in ("new_env", "image_encodings", "qualification_reads", "real_robot"):
        require(result[key] == 0, f"Forbidden action: {key}")
    for key in ("baseline_qualified", "realtime_qualified", "predictor_benefit_tested"):
        require(result[key] is False, "Qualification flag changed")
    require(result["risk_thresholds"] is None and result["old_confirmation"] == "untouched", "Safety state changed")
    samples, _, residuals, donors, manifest, weights, scales, refs = r.load_data()
    compare(manifest, prep["manifest"])
    compare(weights, prep["weights"])
    compare(scales, prep["scales"])
    by_key = {r.q.pfx.key(s): s for s in samples}
    train = [s for s in samples if s["split"] == "train"]
    order = r.q.cov.schedule(train, "multi_conditioned")
    names = {"_".join(map(str, k))+".pt" for k in by_key}
    require({x.name for x in (output / "controls").glob("*.pt")} == names, "Control coverage differs")
    require({x.name for x in (output / "predictions").glob("*.pt")} ==
            {a+"_"+n for a in r.ARMS for n in names}, "Final prediction coverage differs")
    require({x.name for x in (output / "training").glob("*.pt")} ==
            {f"{a}_{i:03d}.pt" for a in r.ARMS for i in range(1, 73)}, "Training coverage differs")
    controls, final, numerical = {}, {}, 0
    for key, s in by_key.items():
        c = load(output / "controls" / ("_".join(map(str, key))+".pt"))
        require(c["key"] == list(key), "Control key differs")
        evidence_check(c["evidence"], s, s["actions"])
        for field in ("action_delta", "zero_delta"):
            for a, b in zip(c["evidence"][field], residuals[key, "true"][field], strict=True):
                r.pilot.require_equal(a, b, "Warm-start predictor differs")
        for name, old_name in (("identity", "identity"), ("old_centered", "centered"), ("frozen", "identity_true")):
            r.pilot.require_equal(c["outputs"][name], refs[key]["outputs"][old_name], "Initial control replay differs")
            score = old.score(refs[key]["visual"][old_name], c["outputs"][name], s)
            compare(score, c["metrics"][name])
            numerical += 3
        controls[key] = c
    initial = load(r.p.ARCHIVE / "centered.pt")["state_dict"]
    log_rows, optimizer_arrays, training_stats = [], 0, {}
    for arm in r.ARMS:
        current = load(output / f"initial_{arm}.pt")
        require(current.keys() == initial.keys(), "Initial parameter names differ")
        for name in initial:
            r.pilot.require_equal(current[name], initial[name], "Matched warm start differs")
        state, losses = {}, []
        for step, index in enumerate(order, 1):
            s, k = train[index], r.q.pfx.key(train[index])
            saved = load(output / "training" / f"{arm}_{step:03d}.pt")
            require(saved["arm"] == arm and saved["step"] == step and saved["key"] == list(k), "Training order differs")
            compare(saved["scales"], scales)
            compare(saved["weight"], weights["rows"][index]["case_weight"])
            evidence_check(saved["evidence"], s, s["actions"])
            score = old.score(saved["evidence"]["visual"], saved["output"], s)
            compare(score, saved["metrics"])
            expected_loss = loss_value(score, controls[k]["metrics"]["identity"], scales, saved["weight"], arm)
            compare(expected_loss, saved["objective"])
            numerical += 4
            gradients = saved["clipped_gradients"]
            require(gradients and set(gradients) <= set(current), "Invalid gradient names")
            require(all(torch.isfinite(v).all() for v in gradients.values()), "Nonfinite gradients")
            norm = float(np.sqrt(sum(np.square(array(v)).sum() for v in gradients.values())))
            before = saved["gradient_norm_before_clip"]
            expected_norm = before*min(1.0, 1.0/(before+1e-6))
            require(math.isfinite(before) and math.isclose(norm, expected_norm, rel_tol=1e-5, abs_tol=1e-7)
                    and norm <= 1.00001, "Gradient clipping differs")
            expected = adamw_step(current, gradients, state)
            after = saved["weights_after"]
            require(after.keys() == current.keys(), "Post-update weight names differ")
            for name in after:
                require(np.isfinite(array(after[name])).all() and
                        np.allclose(expected[name], array(after[name]), rtol=1e-5, atol=1e-7),
                        f"AdamW arithmetic differs: {arm}/{step}/{name}")
                optimizer_arrays += 1
            current = after
            losses.append(saved["objective"])
            log_rows.append({k: saved[k] for k in ("arm", "step", "key", "metrics", "objective", "gradient_norm_before_clip")})
        checkpoint = load(output / f"{arm}.pt")
        require(checkpoint["arm"] == arm and checkpoint["step"] == 72 and checkpoint["source_step"] == 72,
                "Final checkpoint selection differs")
        compare(checkpoint["specification"], r.specification())
        for name, value in current.items():
            r.pilot.require_equal(value, checkpoint["state_dict"][name], "Final weights differ from last update")
        require(any(not torch.equal(current[n], initial[n]) for n in initial), "No weight update occurred")
        training_stats[arm] = {"updates": 72, "mean_recorded_loss": float(np.mean(losses)),
                               "first_loss": losses[0], "last_loss": losses[-1]}
        for k, s in by_key.items():
            saved = load(output / "predictions" / (arm+"_"+"_".join(map(str, k))+".pt"))
            require(saved["key"] == list(k) and saved["arm"] == arm and
                    saved["donor"] == (list(donors[k]) if donors[k] else None), "Final identity or donor differs")
            contexts = {"true", "zero", "mismatched"} if donors[k] else {"true", "zero"}
            require(set(saved["evidence"]) == set(saved["outputs"]) == set(saved["metrics"]) == contexts,
                    "Final context coverage differs")
            scores = {}
            for context in contexts:
                actions = (torch.zeros_like(s["actions"]) if context == "zero" else
                           by_key[donors[k]]["actions"] if context == "mismatched" else s["actions"])
                evidence_check(saved["evidence"][context], s, actions, context == "zero")
                value = saved["outputs"][context]
                require(value.shape == (1, 50, 32) and torch.isfinite(value).all(), "Invalid final output")
                if context == "zero":
                    r.pilot.require_equal(value, controls[k]["outputs"]["identity"], "Zero output differs")
                scores[context] = old.score(saved["evidence"][context]["visual"], value, s)
                compare(scores[context], saved["metrics"][context])
                numerical += 3
            final[arm, k] = scores
    compare(log_rows, list(map(json.loads, (output / "training.jsonl").read_text().splitlines())))
    rows = []
    for s in samples:
        k = r.q.pfx.key(s)
        scores = dict(controls[k]["metrics"])
        for arm in r.ARMS:
            scores.update({f"{arm}_{c}": v for c, v in final[arm, k].items()})
        rows.append({"key": list(k), "split": s["split"], "delay": s["delay"],
                     "donor": list(donors[k]) if donors[k] else None, "metrics": scores})
    compare(rows, list(map(json.loads, (output / "rows.jsonl").read_text().splitlines())))
    summary = r.statistics(rows)
    for name, value in summary.items():
        compare(value, result[name])
    require(decision(rows) == result["development_followup_supported"], "Independent decision differs")
    captures = json.loads((output / "captures.json").read_text())
    require(len(captures) <= 24 and all(c["status"] == "captured" and c["eager_setup_calls"] == 1
            and c["side_stream_warmup_calls"] == 3 and c["capture_calls"] == 1 for c in captures), "Graph budget differs")
    compare(result["counts"], {"vla_loads": 1, "predictor_loads": 2, "decoder": 930, "predictor": 1508,
        "identity_exact": 88, "old_centered_exact": 88, "frozen_exact": 88, "gradient_decoder": 144,
        "backward": 144, "updates": 144, "zero_exact": 176, "captures": len(captures)})
    active, events = {}, Counter()
    for event in map(json.loads, (output / "events.jsonl").read_text().splitlines()):
        events[event["event"]] += 1
        name = event["phase"]
        if event["event"] == "started":
            require(name not in active, "Duplicate phase")
            active[name] = event["limit"]
        else:
            require(event["event"] == "returned" and name in active, "Failed or unmatched phase")
            require(event["seconds"] <= active.pop(name), "Phase exceeded budget")
    require(not active and events["started"] == events["returned"] == 1081, "Phase accounting differs")
    require(r.source_hashes() == prep["source_hashes"] and not torch.cuda.is_initialized(), "Sources or CPU audit changed")
    return {"independent_contract_accepted": True, "development_followup_supported": decision(rows),
            "numerical_comparisons": numerical, "optimizer_parameter_arrays_checked": optimizer_arrays,
            "optimizer_arithmetic_steps": 144, "gradients_independently_redifferentiated": False,
            "training": training_stats, "phases": dict(events), "cuda_initialized": False,
            "audit_model_forwards": 0, "new_env": 0, "qualification_reads": 0,
            "summary": summary, "per_sample": rows, "first_failure": None}


def render(result, audited):
    lines = ["# F-ITC1：Identity中心化微调与软退化惩罚", "",
             f"execution HEAD: `{result.get('execution_head')}`。",
             f"运行 `{result.get('status')}`；独立接纳 `{audited['independent_contract_accepted']}`；"
             f"开发推进条件 `{audited.get('development_followup_supported')}`。", "",
             "固定旧72训练/16开发验证；同一旧第72步初始化、各72更新，lr1e-4；不选点。",
             "guarded是在plain上同时增加chunk和相对identity的两项超额软惩罚，不是硬安全约束。",
             "没有读取R1资格样本、创建Env或新图像编码；oracle是未来视觉条件下的冻结策略，不是专家。", ""]
    if not audited["independent_contract_accepted"]:
        return "\n".join(lines+["```text", str(result.get("first_failure")), str(audited.get("first_failure")), "```", ""])
    for split, s in audited["summary"]["splits"].items():
        lines += [f"## {split}（episode等权）", "", "| 条件 | 样本/episode | 首动作MSE | chunk MSE | token MSE |",
                  "|---|---:|---:|---:|---:|"]
        for name, row in s["metrics"].items():
            m = row["macro"]
            lines.append(f"| {name} | {row['samples']}/{len(row['episodes'])} | {m['row0']:.12f} | {m['chunk']:.12f} | {m['latent']:.9f} |")
        lines += ["", "全部配对收益为对照减干预；错配采用同一子集。", ""]
        for name, c in s["contrasts"].items():
            lines.append(f"{name}: N={c['samples']}/{len(c['episodes'])}episode；均值收益{c['macro_benefit']:.12f}；"
                         f"样本{c['sample_directions']}；episode改善{c['episode_improved']}。")
        lines += ["", f"最大正超额首动作误差（相对I）：`{json.dumps(s['worst_row0_excess_vs_identity'])}`。", ""]
        for arm in r.ARMS:
            c = s["contrasts"][f"{arm}_true_vs_identity"]
            lines += [f"### {arm}最不利样本（不删例）", ""]
            for row in sorted(c["per_sample"], key=lambda x: x["benefit"])[:5]:
                lines.append(f"`{row['key']}`: {row['benefit']:.12f} ({row['direction']})")
            lines += ["", f"留一episode平均收益: `{json.dumps(c['leave_one_out'])}`", ""]
    lines += ["## 固定判据及执行", "", "```json", json.dumps({"checks": audited["summary"]["development_checks"],
        "counts": result["counts"], "phases": audited["phases"], "execution": result["execution"],
        "training": audited["training"]}, indent=2), "```", "",
        "CPU审计重算全部保存指标/目标函数及AdamW参数更新算术；没有重新求导或执行模型。",
        "固定guarded候选条件未满足时不改选plain、不改目标系数或追加训练。开发阳性不等于独立资格。",
        "R1阴性和新资格集封存不变；baseline_qualified/realtime_qualified/predictor_benefit_tested均false，risk_thresholds=null。", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    require(not (output / "independent_audit.json").exists(), "Audit exists; do not overwrite")
    start = time.perf_counter()
    try:
        audited = audit(output)
    except BaseException:
        audited = {"independent_contract_accepted": False, "first_failure": traceback.format_exc()}
    audited["wall_seconds"] = time.perf_counter()-start
    r.write(output / "independent_audit.json", audited)
    result = json.loads((output / "result.json").read_text())
    with (output / "REPORT.md").open("x") as stream:
        stream.write(render(result, audited))
    r.write(output / "REPORT.json", {"result": result, "audit": audited})
    print(json.dumps({k: v for k, v in audited.items() if k not in ("summary", "per_sample")}), flush=True)
    return 0 if audited["independent_contract_accepted"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
