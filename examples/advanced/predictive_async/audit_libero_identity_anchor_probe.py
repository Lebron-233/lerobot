"""Independent CPU/NumPy reduction for F-IAR1; no models or new environments."""

import argparse
import json
import math
import time
import traceback
from collections import Counter, defaultdict
from pathlib import Path

import libero_identity_anchor_probe as p
import numpy as np
import torch


def array(value):
    return value.detach().cpu().float().numpy().astype(np.float64)


def score(visual, output, sample):
    error = array(output)[0, :, :7] - array(sample["oracle"])[0, :, :7]
    totals, sizes = [], []
    for i in range(2):
        mask = sample["inputs"][i + 2].numpy().astype(bool)
        delta = (array(visual[i]) - array(sample["future"][i]))[mask]
        totals.append(np.square(delta).sum())
        sizes.append(delta.size)
    p.require(sum(sizes) > 0, "Empty visual support")
    return {"row0": float(np.square(error[0]).mean()), "chunk": float(np.square(error).mean()),
            "latent": float(np.sum(totals) / sum(sizes))}


def decomposition(control, treatment, oracle):
    old = array(control)[0, :, :7] - array(oracle)[0, :, :7]
    delta = array(treatment)[0, :, :7] - array(control)[0, :, :7]
    cross = 2 * old * delta
    energy = delta * delta
    direct = (old + delta) ** 2 - old ** 2
    p.require(np.allclose(direct, cross + energy, rtol=1e-10, atol=1e-12), "Decomposition identity failed")
    return {"row0_change": float(direct[0].mean()), "cross_term": float(cross[0].mean()),
            "perturbation_energy": float(energy[0].mean()), "row0_dimension_contributions": (direct[0] / 7).tolist(),
            "chunk_change": float(direct.mean())}


def independent_decision(rows):
    selected = [r for r in rows if r["split"] == "validation"]
    p.require(len(selected) == 16, "Validation coverage changed")

    def episodes(arm, metric, subset):
        groups = defaultdict(list)
        for r in subset:
            groups[tuple(r["key"][:2])].append(r["metrics"][arm][metric])
        return {k: float(np.mean(v)) for k, v in groups.items()}

    checks = []
    for control in ("identity", "centered", "identity_mismatched"):
        subset = [r for r in selected if control in r["metrics"]]
        a, b = episodes("identity_true", "row0", subset), episodes(control, "row0", subset)
        av, bv = np.mean(list(a.values())), np.mean(list(b.values()))
        checks.append(bv - av > 1e-7 + 1e-6 * abs(bv))
    a, b = episodes("identity_true", "row0", selected), episodes("identity", "row0", selected)
    checks.append(sum(b[k] - a[k] > 1e-7 + 1e-6 * abs(b[k]) for k in a) >= 3)
    checks.append(sum(r["metrics"]["identity"]["row0"] - r["metrics"]["identity_true"]["row0"] >
                      1e-7 + 1e-6 * abs(r["metrics"]["identity"]["row0"]) for r in selected) > 8)
    av = np.mean(list(episodes("identity_true", "chunk", selected).values()))
    bv = np.mean(list(episodes("identity", "chunk", selected).values()))
    checks.append(av <= bv + 1e-7 + 1e-6 * abs(bv))
    return bool(all(checks))


def compare(a, b):
    if isinstance(a, dict):
        p.require(isinstance(b, dict) and a.keys() == b.keys(), "JSON keys differ")
        for k, v in a.items():
            compare(v, b[k])
    elif isinstance(a, list):
        p.require(isinstance(b, list) and len(a) == len(b), "JSON length differs")
        for x, y in zip(a, b, strict=True):
            compare(x, y)
    elif isinstance(a, float):
        p.require(math.isfinite(a) and math.isfinite(b) and math.isclose(a, b, rel_tol=1e-6, abs_tol=1e-7),
                  "Numerical reduction differs")
    else:
        p.require(a == b, "JSON value differs")


def audit(output):
    torch.set_num_threads(1)
    p.require(not torch.cuda.is_initialized(), "Audit entered with CUDA initialized")
    result = json.loads((output / "result.json").read_text())
    head = result["execution_head"]
    prep = p.validate(head)
    p.require(output == p.paths(head)[1] and result["status"] == "completed" and result["first_failure"] is None,
              "Run incomplete or output identity differs")
    execution = result["execution"]
    p.require(execution["child_exit_code"] == 0 and execution["exit_confirmed"] is True
              and not execution["stop_reason"] and not execution["forced_termination"]
              and not execution["pending"] and not execution["active"], "Execution not closed")
    p.require(result["attempts"] == 1 and result["retries"] == 0 and result["vla_frozen"]
              and result["graph_released"], "Freeze/attempt differs")
    for name in ("predictor_forwards", "training_updates", "backward", "new_env", "image_encodings", "qualification_reads", "real_robot"):
        p.require(result[name] == 0, f"Forbidden operation: {name}")
    samples, bases, residuals, donors, contract = p.load_sources()
    compare(contract, prep["manifest"])
    expected_files = {"_".join(map(str, p.q.pfx.key(s))) + ".pt" for s in samples}
    p.require({f.name for f in (output / "predictions").glob("*.pt")} == expected_files, "Prediction coverage differs")
    rows, decompositions, numerical = [], [], 0
    for sample in samples:
        key = p.q.pfx.key(sample)
        saved = torch.load(output / "predictions" / ("_".join(map(str, key)) + ".pt"),
                           map_location="cpu", weights_only=False)
        p.require(saved["key"] == list(key) and saved["split"] == sample["split"] and saved["delay"] == sample["delay"],
                  "Saved identity differs")
        p.require(saved["donor"] == (list(donors[key]) if donors[key] else None), "Donor differs")
        expected = set(p.ARMS) - ({"identity_mismatched"} if donors[key] is None else set())
        p.require(set(saved["outputs"]) == set(saved["visual"]) == set(saved["metrics"]) == expected, "Arms differ")
        for arm, ref in (("identity", sample["archived_full_chunk"]), ("base", bases[key]["output"]),
                         ("centered", residuals[key, "true"]["output"]),
                         ("identity_zero", sample["archived_full_chunk"])):
            p.pilot.require_equal(saved["outputs"][arm], ref, "Control/zero full output differs")
        for context in ("true", "zero", "mismatched"):
            arm = f"identity_{context}"
            if arm not in expected:
                continue
            r = residuals[key, context]
            for i in range(2):
                delta = r["action_delta"][i].float() - r["zero_delta"][i].float()
                raw = sample["inputs"][i].float() + delta
                p.pilot.require_equal(saved["raw_new"][arm][i], raw, "Transplant raw differs")
                p.pilot.require_equal(saved["visual"][arm][i], raw.to(torch.bfloat16).float(), "Transplant BF16 differs")
                if context == "zero":
                    p.pilot.require_equal(raw, sample["inputs"][i].float(), "Zero raw differs")
        for arm, ref in (("identity", sample["inputs"][:2]), ("base", bases[key]["visual"]),
                         ("centered", residuals[key, "true"]["visual"])):
            for i in range(2):
                p.pilot.require_equal(saved["visual"][arm][i].float(), ref[i].float(), "Control visual differs")
        scores = {}
        for arm, value in saved["outputs"].items():
            p.require(value.shape == (1, 50, 32) and torch.isfinite(value).all(), "Invalid output")
            scores[arm] = score(saved["visual"][arm], value, sample)
            compare(scores[arm], saved["metrics"][arm])
            numerical += 3
        rows.append({"key": list(key), "split": sample["split"], "delay": sample["delay"],
                     "donor": saved["donor"], "metrics": scores})
        for treatment, control in p.CONTRASTS:
            if control in expected:
                decompositions.append({"key": list(key), "split": sample["split"],
                    "contrast": f"{treatment}_vs_{control}",
                    **decomposition(saved["outputs"][control], saved["outputs"][treatment], sample["oracle"])})
    compare(rows, [json.loads(line) for line in (output / "rows.jsonl").read_text().splitlines()])
    summary = p.summarize(rows)
    for k, value in summary.items():
        compare(value, result[k])
    p.require(independent_decision(rows) == result["development_followup_supported"], "Independent decision differs")
    captures = json.loads((output / "captures.json").read_text())
    compare(result["counts"], {"decoder": 525, "vla_loads": 1, "identity_exact": 88, "base_exact": 88,
                              "centered_exact": 88, "zero_exact": 88, "captures": len(captures)})
    p.require(len(captures) <= 8 and all(c["status"] == "captured" and c["eager_setup_calls"] == 1
              and c["side_stream_warmup_calls"] == 3 and c["capture_calls"] == 1 for c in captures), "Graph accounting differs")
    active, events = {}, Counter()
    for event in map(json.loads, (output / "events.jsonl").read_text().splitlines()):
        name = event["phase"]
        events[event["event"]] += 1
        if event["event"] == "started":
            p.require(name not in active, "Duplicate phase")
            active[name] = event["limit"]
        else:
            p.require(event["event"] == "returned" and name in active, "Unmatched/failed phase")
            p.require(event["seconds"] <= active.pop(name), "Phase exceeded deadline")
    p.require(not active and events["started"] == events["returned"], "Unclosed phase")
    p.require(p.source_hashes() == prep["source_hashes"] and not torch.cuda.is_initialized(), "Sources/CUDA changed")
    return {"independent_contract_accepted": True, "development_followup_supported": independent_decision(rows),
            "numerical_comparisons": numerical, "phases": dict(events), "cuda_initialized": False,
            "audit_model_forwards": 0, "summary": summary, "decompositions": decompositions,
            "first_failure": None}


def render(result, audited):
    lines = ["# F-IAR1：固定残差的 identity 基底置换", "",
             f"execution HEAD: `{result['execution_head']}`。",
             f"运行状态 `{result['status']}`；独立接纳 `{audited['independent_contract_accepted']}`；"
             f"开发后继条件 `{audited.get('development_followup_supported')}`。", "",
             "只使用旧开发72训练/16验证样本；R1资格32样本未载入。不是新独立资格或部署收益。",
             "唯一新干预I+(h(a)-h(0))，幅度1；delta来自旧已接受归档，未训练或执行预测器。",
             "oracle为同当前state/language/noise的未来视觉冻结策略输出，不是专家。", ""]
    if not audited["independent_contract_accepted"]:
        return "\n".join(lines + ["```text", audited["first_failure"], "```", ""])
    for split, data in audited["summary"]["splits"].items():
        lines += [f"## {split}（episode等权）", "", "| 条件 | 样本/episode | 首动作MSE | chunk MSE | token MSE |",
                  "|---|---:|---:|---:|---:|"]
        for name, a in data["metrics"].items():
            m = a["macro"]
            lines.append(f"| {name} | {a['samples']}/{len(a['episodes'])} | {m['row0']:.12f} | {m['chunk']:.12f} | {m['latent']:.9f} |")
        lines += ["", "差值均为对照−干预，正数代表改善；错配采用相同子集。", ""]
        for name, c in data["contrasts"].items():
            lines.append(f"{name}: N={c['paired_samples']}, episodes={c['paired_episodes']}, "
                         f"平均收益={c['macro_benefit']:.12f}, 样本方向={c['sample_directions']}, episode改善={c['episode_improved']}。")
        c = data["contrasts"]["identity_true_vs_identity"]
        lines += ["", "新true相对I最不利样本（不删例）：", ""]
        for r in sorted(c["per_sample"], key=lambda r: r["benefit"])[:5]:
            lines.append(f"`{r['key']}`: {r['benefit']:.12f} ({r['direction']})")
        lines += ["", f"留一episode收益: `{json.dumps(c['leave_one_out'])}`", ""]
    lines += ["## 判据与执行", "", "```json", json.dumps({"checks": audited["summary"]["development_checks"],
        "counts": result["counts"], "execution": result["execution"], "phases": audited["phases"]}, indent=2), "```", "",
        "全部七维输出误差的交叉项/扰动能量分解保存于independent_audit.json；它不是视觉Jacobian或物理因果识别。",
        "本轮不搜索alpha、不自动训练或重读资格集。旧R1阴性结论不变；闭环/时延/安全资格均未建立。", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    p.require(not (output / "independent_audit.json").exists(), "Audit exists; do not overwrite")
    start = time.perf_counter()
    try:
        audited = audit(output)
    except BaseException:
        audited = {"independent_contract_accepted": False, "first_failure": traceback.format_exc()}
    audited["wall_seconds"] = time.perf_counter() - start
    p.write(output / "independent_audit.json", audited)
    result = json.loads((output / "result.json").read_text())
    with (output / "REPORT.md").open("x") as stream:
        stream.write(render(result, audited))
    p.write(output / "REPORT.json", {"result": result, "audit": audited})
    print(json.dumps({k: v for k, v in audited.items() if k not in ("summary", "decompositions")}), flush=True)
    return 0 if audited["independent_contract_accepted"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
