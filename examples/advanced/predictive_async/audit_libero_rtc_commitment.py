"""CPU evidence audit: original dynamic RTC checks plus independently realized commitments."""

import argparse
import json
import math
import traceback
from collections import Counter
from pathlib import Path

import audit_libero_rtc_dynamic as dynamic
import libero_rtc_commitment as r
import torch

check, exact, quantiles = r.check, dynamic.exact, dynamic.base.quantiles


def inspect_episode(spec, record, arrays, initial):
    row = dynamic.inspect_episode(spec, record, arrays, initial)
    row["variant"] = spec["variant"]
    durations, commitments, submission_times = [], [], []
    dispatches, observations = arrays["control"]["dispatches"], arrays["observations"]
    for req in record["requests"]:
        stamp, decision, c = req["stamp"], req["decision"], req["commitment"]
        rid = stamp["request_id"]
        cover = 7 if not durations else math.ceil(20 * sorted(durations[-50:])[math.ceil(.9 * min(len(durations), 50))-1]) + 1
        if req["prefix_rows"]:
            check(stamp["expected_delay"] == cover, "Historical coverage estimate changed")
            check(cover + 2 <= req["prefix_rows"] <= 30, "Submission outside original coverage guard")
        else:
            check(stamp["expected_delay"] == 0, "Empty queue requires delay0")
        check(c["compute_cover_estimate"] == stamp["expected_delay"], "Coverage evidence differs")
        enabled = spec["variant"] == "committed_rtc" and rid > 1
        check(c["enabled"] == enabled and c["declared_steps"] == (stamp["expected_delay"] if enabled else None),
              "Wrong declared commitment")
        conditioned = stamp["expected_delay"] if spec["arm"] == "rtc_async" and req["prefix_rows"] else 0
        check(c["conditioned_steps"] == conditioned, "RTC condition declaration differs")
        if enabled:
            target = c["declared_steps"]
            if decision["accepted"]:
                b = c["boundary_at"]
                check(c["status"] == "installed" and c["consumed"] == target == conditioned ==
                    decision["actual_delay"] == decision["source_row"], "Commitment/conditioning/trim mismatch")
                count = sum(d["dispatched_at"] < b for d in dispatches)
                check(count == stamp["observation_index"] + target and
                    observations[count]["returned_at"] <= b <= req["published_at"], "Commitment boundary differs")
                check(c["result_ready_at_boundary"] == (req["completed_at"] <= b), "Ready classification differs")
                for name, expected in (("ready_hold_s", max(0.0, b-req["completed_at"])),
                                       ("late_compute_wait_s", max(0.0, req["completed_at"]-b))):
                    check(math.isclose(c[name], expected, abs_tol=1e-9), "Early/late evidence differs")
                check(math.isfinite(c["receive_elapsed_s"]) and c["receive_elapsed_s"] >= c["late_compute_wait_s"],
                      "Hidden receive wait")
                commitments.append({"request": rid, "declared": target,
                    "actual": decision["actual_delay"], "ready_hold_s": c["ready_hold_s"],
                    "late_compute_wait_s": c["late_compute_wait_s"],
                    "receive_elapsed_s": c["receive_elapsed_s"],
                    "ready_at_boundary": c["result_ready_at_boundary"]})
            else:
                check(decision["reason"] == "stale_or_closed" and c["status"] == "cancelled_at_stop" and
                    c["consumed_at_stop"] == len(dispatches)-stamp["observation_index"] <= target,
                    "Invalid cancelled commitment")
        else:
            check(c["boundary_at"] is None, "Control arm unexpectedly gated")
        if rid > 1:
            durations.append(req["complete_s"])
            submission_times.append(req["completed_at"]-req["requested_at"])
    row.update(commitments=commitments, submitted_to_complete_s=submission_times,
        max_observation_age_s=max(d["dispatched_at"] - record["requests"][d["request_id"]-1]["observation_returned_at"]
                                  for d in dispatches))
    return row


def summarize(rows):
    arms, cohorts = {}, {}
    for variant in r.VARIANTS:
        selected = [v for v in rows if v["variant"] == variant]
        times = [t for v in selected for t in v["request_seconds"]]
        stats = quantiles(times)
        stats.update(above350ms=sum(t > .35 for t in times), required_delay_steps=math.ceil(stats["p99"]*20)+1)
        cs = [c for v in selected for c in v["commitments"]]
        arms[variant] = {"episodes": len(selected), "successes": sum(v["success"] for v in selected),
            **{k: sum(v[k] for v in selected) for k in ("actions", "wall_s", "no_action_slots", "underflows",
                "expired", "cancelled_at_stop", "request_count", "guided_requests", "capture_seconds")},
            "request_time_s": stats, "bootstrap_s": quantiles([v["bootstrap_s"] for v in selected]),
            "submitted_to_complete_s": quantiles([t for v in selected for t in v["submitted_to_complete_s"]]),
            "model_native_intersections": sum(len(v["overlaps"]) for v in selected),
            "expected_actual_prefix": dict(Counter(str(v) for row in selected for v in row["delay_pairs"])),
            "commitments_installed": len(cs), "ready_at_boundary": sum(c["ready_at_boundary"] for c in cs),
            "late_at_boundary": sum(not c["ready_at_boundary"] for c in cs),
            "ready_hold_total_s": math.fsum(c["ready_hold_s"] for c in cs),
            "late_compute_wait_total_s": math.fsum(c["late_compute_wait_s"] for c in cs),
            "receive_total_s": math.fsum(c["receive_elapsed_s"] for c in cs),
            "max_observation_age_s": max(v["max_observation_age_s"] for v in selected)}
    for cohort in ("pilot_development", "known_diagnostic"):
        cohorts[cohort] = {v: {"n": sum(x["cohort"] == cohort and x["variant"] == v for x in rows),
            "successes": sum(x["success"] for x in rows if x["cohort"] == cohort and x["variant"] == v)}
            for v in r.VARIANTS}
    lost = {v: [] for v in r.VARIANTS[:-1]}
    quartets = []
    for pair, identity in enumerate(r.prior.PAIRS):
        selected = {v["variant"]: v for v in rows if v["pair_index"] == pair}
        check(set(selected) == set(r.VARIANTS), "Quartet coverage differs")
        quartets.append({"identity": list(identity), "arms": {a: {k: v[k] for k in ("success", "actions", "wall_s")}
                                                              for a, v in selected.items()}})
        for variant in lost:
            if selected[variant]["success"] and not selected["committed_rtc"]["success"]:
                lost[variant].append(list(identity))
    candidate, serial = arms["committed_rtc"], arms["serialized"]
    gates = {**{f"no_lost_vs_{v}": not identities for v, identities in lost.items()},
        "all_request_budgets": all(v["request_time_s"]["above350ms"] == 0 and
                                   v["request_time_s"]["required_delay_steps"] <= 8 for v in arms.values()),
        "candidate_overlap": candidate["model_native_intersections"] > 0,
        "candidate_wait_reduced": candidate["no_action_slots"]/candidate["actions"] < serial["no_action_slots"]/serial["actions"],
        "candidate_no_queue_fault": candidate["underflows"] == candidate["expired"] == 0,
        "candidate_commitments_applied": candidate["commitments_installed"] > 0 and candidate["guided_requests"] > 0}
    return {"arms": arms, "cohorts": cohorts, "quartets": quartets, "lost_success_identities": lost,
            "development_gates": gates, "development_followup_supported": all(gates.values())}


def audit(output):
    check(not torch.cuda.is_initialized(), "CPU audit only")
    result = json.loads((output / "result.json").read_text())
    check(result["status"] == "completed" and result["first_failure"] is None, "Formal run incomplete")
    ex = result["execution"]
    check(ex["exit_confirmed"] and ex["exit_code"] == 0 and not ex["forced"] and not ex["active"] and
        not ex["pending"] and ex["stop_reason"] is None, "Execution not closed")
    saved = r.validate(result["execution_head"], after=True)
    check(output == r.paths(result["execution_head"])[1] and
          json.loads((output / "manifest.json").read_text()) == saved["manifest"], "Manifest identity differs")
    rows, initials, boots, total = [], {}, {}, Counter()
    for spec in saved["manifest"]["rows"]:
        folder = output / f"episode_{spec['ordinal']:03d}"
        rec = json.loads((folder / "result.json").read_text())
        check(json.loads((folder / "started.json").read_text())["spec"] == spec, "Started identity differs")
        arrays, initial = r.load(folder / "arrays.pt"), r.load(folder / "initial_checkpoint.pt")
        rows.append(inspect_episode(spec, rec, arrays, initial))
        pair, boot = spec["pair_index"], arrays["outputs"][1]
        if pair in initials:
            check(r.e.initial_difference(initials[pair], initial) is None, "Quartet initial differs")
            for field in ("full", "noise", "original", "processed"):
                exact(boot[field], boots[pair][field], "Quartet bootstrap differs")
            for j in range(8):
                exact(boot["inputs"][j], boots[pair]["inputs"][j], "Quartet input differs")
        else:
            initials[pair], boots[pair] = initial, boot
        total.update(rec["budget"])
        del arrays
    check(len(rows) == result["episodes_completed"] == 40 and len(initials) == 10 and
        dict(total) == result["native_budget"] and all(total[k] <= v[1] for k, v in r.LIMITS.items()), "Coverage/budget differs")
    check(result["graph_captures"] == 80 and result["capture_internal"] == {"setup": 80, "warmup": 240, "capture": 80},
          "Capture accounting differs")
    journal = dynamic.ledger(output, 42, 40)
    check(journal["kinds"]["dynamic_graph_request"] == total["model"] and
        journal["kinds"]["native_step"] == total["settling"]+total["measurement"], "Call ledger differs")
    check(result["attempts"] == 1 and result["retries"] == 0 and result["vla_frozen"] and
        all(result[k] == 0 for k in ("training_updates", "qualification_reads", "real_robot")), "Scope differs")
    check(not torch.cuda.is_initialized(), "CPU audit initialized CUDA")
    return {"independent_contract_accepted": True, "frozen_baseline_unchanged": True,
        "episodes": 40, "initial_quartets_exact": 10, "native_actions_checked": total["measurement"],
        "request_outputs_checked": total["model"], "graph_captures": 80, "runtime_closed": 40,
        **summarize(rows), **journal, "per_episode": rows, "first_failure": None,
        "new_qualification_claimed": False, "deployment_qualified": False, "audit_model_forwards": 0,
        "cuda_initialized": False, "limitations": ["Known development identities, not independent confirmation.",
        "Binding the estimated prefix changes takeover timing and later feedback; it is a new scheduling candidate.",
        "Soft RTC guidance beyond C is unchanged; no unique failure mechanism established.",
        "CPU audit does not rerun model, VJP, RGB encoding, postprocessing, physics or GPU timing."]}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    output = p.parse_args().output.resolve()
    torch.set_num_threads(1)
    check(not (output / "independent_audit.json").exists(), "Audit already exists")
    try:
        value = audit(output)
    except BaseException:
        value = {"independent_contract_accepted": False, "first_failure": traceback.format_exc()}
    r.write(output / "independent_audit.json", value)
    print(json.dumps({k: v for k, v in value.items() if k != "per_episode"}), flush=True)
    return 0 if value["independent_contract_accepted"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
