"""Independent saved-evidence checks for prefix-only input and live feedback stages."""

import argparse
import json
import math
import traceback
from collections import Counter
from pathlib import Path

import audit_libero_rtc_dynamic as dynamic
import libero_rtc_prefix_scope as r
import torch
from rtc_dynamic_runtime import config
from rtc_prefix_scope_runtime import COMMITTED, VARIANTS, prefix_config, schedule_for

from lerobot.policies.rtc.modeling_rtc import RTCProcessor

check, load, exact, quantiles = r.check, r.load, dynamic.exact, dynamic.base.quantiles


def inspect_input(spec, row, sample):
    check(all(row[k] == v for k, v in spec.items()) and row["constraint_schedule"] == "ZEROS", "Input spec/config differs")
    guided, length, delay = spec["length"] > 0, spec["length"], spec["delay"]
    check(len(row["inputs"]) == (9 if guided else 8), "Input arity")
    for j in range(8):
        exact(row["inputs"][j], sample["inputs"][j], "Input differs")
    check(row["full"].shape == (1, 50, 32) and row["processed"].shape == (1, 50, 7), "Output shape")
    exact(row["normalized"], row["full"][..., :7], "Unpadding")
    check(all(torch.isfinite(row[k]).all() for k in ("full", "normalized", "processed")), "Nonfinite output")
    m = row["metadata"]
    check(m["mode"] == spec["arm"] and m["guided"] == guided and m["expected_delay"] == delay and
          m["prefix_length"] == length and m["effective_horizon"] == min(length, 10), "Dynamic metadata")
    check(m["replays"] == int(spec["arm"] == "graph") and
          m["captured_vjps_per_replay"] == (10 if guided and spec["arm"] == "graph" else 0), "VJP/replay metadata")
    if guided:
        p = sample["prefix"][:length]
        exact(row["inputs"][8], p, "Wrong prefix")
        pad = torch.zeros((50, 7))
        pad[:length] = p
        exact(row["padded"], pad, "Wrong padding")
        w = RTCProcessor(prefix_config()).get_prefix_weights(delay, min(length, 10), 50)
        exact(row["weights"], w, "Wrong explicit prefix weights")
        exact(row["weights"], (torch.arange(50) < delay).float(), "Tail target not zero")
    else:
        exact(row["full"], sample["archived_full_chunk"], "No-prefix archived output")
        check(row["weights"] is None and row["padded"] is None, "Unexpected no-prefix condition")
    check(math.isfinite(row["complete_s"]) and 0 < row["complete_s"] < 30, "Bad duration")


def input_audit(output, result):
    samples = load(r.prior.previous.DATA)
    rows, pairs = [], {}
    for spec in r.prior.input_schedule():
        row = load(output / f"request_{spec['ordinal']:03d}.pt")
        inspect_input(spec, row, samples[spec["sample"]])
        if spec["pair"] in pairs:
            for key in ("full", "normalized", "processed"):
                exact(row[key], pairs[spec["pair"]][key], "New formula eager/Graph mismatch")
        else:
            pairs[spec["pair"]] = row
        rows.append(row)
    check(len(rows) == result["completed_requests"] == 482 and len(pairs) == 241, "Input coverage")
    rec = result["runtime_receipt"]
    dynamic.captures(rec, 2)
    check(rec["requests"] == rec["rgb_encodings"] == 482 and
          rec["replays"] == {"no_prefix": 16, "guided": 225}, "Runtime coverage")
    check(sum(v["metadata"]["capture_created"] for v in rows) == 1, "Unexpected recapture")
    timing = {}
    for arm in ("native_eager", "graph"):
        ts = [v["complete_s"] for v in rows if v["arm"] == arm and v["length"]]
        timing[arm] = {**quantiles(ts), "above350ms": sum(t > .35 for t in ts)}
        timing[arm]["required_delay_steps"] = math.ceil(timing[arm]["p99"]*20)+1
    journal = dynamic.ledger(output, 11, 0)
    check(journal["kinds"] == {"dynamic_request": 482}, "Input calls differ")
    return {"independent_contract_accepted": True, "prefix_scope_equivalence_passed": True,
        "requests": 482, "submit_feasible_combinations_exact": 225, "no_prefix_anchors_exact": 32,
        "cross_arm_arrays_exact": 723, "timing": timing,
        "graph_budget_passed": timing["graph"]["above350ms"] == 0 and timing["graph"]["required_delay_steps"] <= 8,
        "graph_faster_mean": timing["graph"]["mean"] < timing["native_eager"]["mean"],
        "captures": 2, "replayed_guided_vjps": 2250, "extra_sampler_calls": 10,
        "first_graph_request_s": next(v["complete_s"] for v in rows if v["metadata"]["capture_created"]),
        "capture_seconds": sum(v["seconds"] for v in rec["captures"]), **journal}


def scope_output(out, req, record):
    variant = record["spec"]["variant"]
    schedule = schedule_for(variant)
    check(out["constraint_schedule"] == record["spec"]["constraint_schedule"] == schedule, "Wrong constraint schedule")
    exact(out["context_prefix"], out["prefix"], "Owned normalized prefix differs")
    check(out["context_stamp"] == req["stamp"] and out["context_prefix_source"] == req["prefix_source"], "Wrong context source")
    used = record["spec"]["arm"] == "rtc_async" and out["prefix"] is not None and len(out["prefix"]) > 0
    delay = req["stamp"]["expected_delay"] if used else 0
    length = len(out["prefix"]) if used else 0
    m = out["rtc_metadata"]
    check(m["mode"] == "graph" and m["guided"] == used and m["replays"] == 1 and
          m["expected_delay"] == delay and m["prefix_length"] == length and
          m["captured_vjps_per_replay"] == (10 if used else 0), "Wrong RTC input path")
    check(m["capture_created"] == out["capture_created"] == (req["stamp"]["request_id"] == 1), "Control-time capture")
    check(out["vision_encodes"] == 1 and len(out["inputs"]) == (9 if used else 8), "Input/encoding coverage")
    exact(out["noise"], out["inputs"][7], "Noise changed")
    if used:
        exact(out["inputs"][8], out["prefix"], "Model prefix differs")
        pad = torch.zeros((50, 7))
        pad[:length] = out["prefix"]
        exact(out["rtc_padded_prefix"], pad, "Stale prefix buffer")
        cfg = prefix_config() if schedule == "ZEROS" else config()
        exact(out["rtc_weights"], RTCProcessor(cfg).get_prefix_weights(delay, min(length, 10), 50), "Wrong scope weights")
        if schedule == "ZEROS":
            exact(out["rtc_weights"], (torch.arange(50) < delay).float(), "Explicit target beyond commitment")
    else:
        check(out["rtc_weights"] is None and out["rtc_padded_prefix"] is None, "Unguided control received targets")
    check(out["owner_thread"] == record["predictor_cleanup"]["owner"], "Wrong model owner")


def inspect_episode(spec, record, arrays, initial):
    row = dynamic.base.audit_episode(spec, record, arrays, initial, output_checker=scope_output)
    rec, closed = record["predictor_cleanup"], record["owner_cleanup"]
    dynamic.captures(rec, 2)
    check(rec["requests"] == rec["rgb_encodings"] == rec["live_requests"] == rec["explicit_noise_draws"] == len(record["requests"]),
          "Live request count")
    check(rec["owner"] == closed["owner_thread"] != record["controller_thread"] and closed["first_failure"] is None, "Owner cleanup")
    check(arrays["control"]["ended_at"] <= closed["started_at"] <= rec["closed_at"] <= closed["completed_at"], "Close chronology")
    dispatches, observations = arrays["control"]["dispatches"], arrays["observations"]
    native = [v for v in record["native_steps"] if v["segment"] == "measurement"]
    commitments, durations, submission_times, delay_pairs, guided = [], [], [], [], 0
    for req in record["requests"]:
        stamp, decision, c = req["stamp"], req["decision"], req["commitment"]
        rid, oi = stamp["request_id"], stamp["observation_index"]
        out = arrays["outputs"][rid]
        check(out["input_fingerprint"] == r.prior.g.observation_fingerprint(observations[oi]), "Current observation differs")
        check(req["completed_at"] <= closed["started_at"], "Released during request")
        for when in (req["requested_at"], req["published_at"]):
            check(not any(v["started_at"] < when < v["returned_at"] for v in native), "Request/install inside native step")
        guided += int(out["rtc_metadata"]["guided"])
        cover = r.old.estimate_delay(durations)
        if req["prefix_rows"]:
            check(stamp["expected_delay"] == cover and cover+2 <= req["prefix_rows"] <= 30, "Coverage/submission changed")
        else:
            check(stamp["expected_delay"] == 0, "Empty prefix delay")
        enabled = spec["variant"] in COMMITTED and rid > 1
        check(c["enabled"] == enabled and c["declared_steps"] == (stamp["expected_delay"] if enabled else None) and
              c["compute_cover_estimate"] == stamp["expected_delay"] and
              c["constraint_schedule"] == schedule_for(spec["variant"]), "Commitment declaration")
        conditioned = stamp["expected_delay"] if spec["arm"] == "rtc_async" and req["prefix_rows"] else 0
        check(c["conditioned_steps"] == conditioned, "Conditioning declaration differs")
        if enabled:
            target = c["declared_steps"]
            if decision["accepted"]:
                b = c["boundary_at"]
                check(c["status"] == "installed" and c["consumed"] == target == decision["actual_delay"] == decision["source_row"],
                      "Commitment/consumption/trim mismatch")
                if spec["arm"] == "rtc_async":
                    check(conditioned == target, "Guidance differs from commitment")
                count = sum(d["dispatched_at"] < b for d in dispatches)
                check(count == oi+target and observations[count]["returned_at"] <= b <= req["published_at"], "Wrong commitment boundary")
                for offset in range(target):
                    d = dispatches[oi+offset]
                    check(d["request_id"] == req["prefix_source"], "Committed old plan not executed")
                    exact(d["original"], out["prefix"][offset], "Committed action differs from submitted prefix")
                check(c["result_ready_at_boundary"] == (req["completed_at"] <= b), "Ready classification")
                for key, value in (("ready_hold_s", max(0., b-req["completed_at"])),
                                   ("late_compute_wait_s", max(0., req["completed_at"]-b))):
                    check(math.isclose(c[key], value, abs_tol=1e-9), "Hidden early/late interval")
                check(math.isfinite(c["receive_elapsed_s"]) and c["receive_elapsed_s"] >= c["late_compute_wait_s"], "Hidden receive wait")
                commitments.append({"request": rid, "declared": target, "actual": decision["actual_delay"],
                    "conditioned": conditioned, "ready_hold_s": c["ready_hold_s"],
                    "late_compute_wait_s": c["late_compute_wait_s"], "receive_elapsed_s": c["receive_elapsed_s"],
                    "ready_at_boundary": c["result_ready_at_boundary"]})
            else:
                check(decision["reason"] == "stale_or_closed" and c["status"] == "cancelled_at_stop" and
                      0 <= c["consumed_at_stop"] == len(dispatches)-oi <= target, "Cancelled commitment mismatch")
        else:
            check(c["boundary_at"] is None, "Ungated baseline unexpectedly gated")
        if rid > 1:
            durations.append(req["complete_s"])
            submission_times.append(req["completed_at"]-req["requested_at"])
            if decision["accepted"]:
                delay_pairs.append([stamp["expected_delay"], decision["actual_delay"], req["prefix_rows"]])
    check(rec["replays"].get("guided", 0) == guided and sum(rec["replays"].values()) == len(record["requests"]), "Replay accounting")
    row.update(variant=spec["variant"], identity=[spec["task_id"], spec["initial_state_id"]], cohort=spec["cohort"],
        commitments=commitments, guided_requests=guided, delay_pairs=delay_pairs,
        capture_seconds=sum(c["seconds"] for c in rec["captures"]), submitted_to_complete_s=submission_times,
        max_observation_age_s=max(d["dispatched_at"]-record["requests"][d["request_id"]-1]["observation_returned_at"] for d in dispatches))
    return row


def summarize(rows):
    arms, cohorts, groups = {}, {}, []
    for variant in VARIANTS:
        selected = [v for v in rows if v["variant"] == variant]
        ts = [t for v in selected for t in v["request_seconds"]]
        stats = {**quantiles(ts), "above350ms": sum(t > .35 for t in ts)}
        stats["required_delay_steps"] = math.ceil(stats["p99"]*20)+1
        cs = [c for v in selected for c in v["commitments"]]
        arms[variant] = {"episodes": len(selected), "successes": sum(v["success"] for v in selected),
            **{k: sum(v[k] for v in selected) for k in ("actions", "wall_s", "no_action_slots", "underflows", "expired",
                "cancelled_at_stop", "request_count", "guided_requests", "capture_seconds")},
            "request_time_s": stats, "bootstrap_s": quantiles([v["bootstrap_s"] for v in selected]),
            "submitted_to_complete_s": quantiles([t for v in selected for t in v["submitted_to_complete_s"]]),
            "model_native_intersections": sum(len(v["overlaps"]) for v in selected),
            "expected_actual_prefix": dict(Counter(str(p) for v in selected for p in v["delay_pairs"])),
            "commitments_installed": len(cs), "ready_at_boundary": sum(c["ready_at_boundary"] for c in cs),
            "late_at_boundary": sum(not c["ready_at_boundary"] for c in cs),
            "ready_hold_total_s": math.fsum(c["ready_hold_s"] for c in cs),
            "late_compute_wait_total_s": math.fsum(c["late_compute_wait_s"] for c in cs),
            "receive_total_s": math.fsum(c["receive_elapsed_s"] for c in cs),
            "max_observation_age_s": max(v["max_observation_age_s"] for v in selected)}
    for cohort in ("pilot_development", "known_diagnostic"):
        cohorts[cohort] = {v: {"n": sum(x["cohort"] == cohort and x["variant"] == v for x in rows),
            "successes": sum(x["success"] for x in rows if x["cohort"] == cohort and x["variant"] == v)} for v in VARIANTS}
    lost = {v: [] for v in VARIANTS[:-1]}
    for pair, identity in enumerate(r.prior.PAIRS):
        selected = {v["variant"]: v for v in rows if v["pair_index"] == pair}
        check(set(selected) == set(VARIANTS), "Five-arm coverage mismatch")
        groups.append({"identity": list(identity), "arms": {a: {k: v[k] for k in ("success", "actions", "wall_s")} for a, v in selected.items()}})
        for v in lost:
            if selected[v]["success"] and not selected["committed_prefix"]["success"]:
                lost[v].append(list(identity))
    c, s = arms["committed_prefix"], arms["serialized"]
    gates = {**{f"no_lost_vs_{v}": not ids for v, ids in lost.items()},
        "all_request_budgets": all(v["request_time_s"]["above350ms"] == 0 and v["request_time_s"]["required_delay_steps"] <= 8 for v in arms.values()),
        "candidate_overlap": c["model_native_intersections"] > 0,
        "candidate_wait_reduced": c["no_action_slots"]/c["actions"] < s["no_action_slots"]/s["actions"],
        "candidate_no_queue_fault": c["underflows"] == c["expired"] == 0,
        "candidate_commitments_applied": c["commitments_installed"] > 0 and c["guided_requests"] > 0}
    return {"arms": arms, "cohorts": cohorts, "groups": groups, "lost_success_identities": lost,
            "development_gates": gates, "development_followup_supported": all(gates.values())}


def feedback_audit(output, result, saved):
    rows, initials, boots, total = [], {}, {}, Counter()
    for spec in saved["manifest"]["rows"]:
        folder = output / f"episode_{spec['ordinal']:03d}"
        record = json.loads((folder / "result.json").read_text())
        check(json.loads((folder / "started.json").read_text())["spec"] == spec, "Started identity differs")
        arrays, initial = load(folder / "arrays.pt"), load(folder / "initial_checkpoint.pt")
        rows.append(inspect_episode(spec, record, arrays, initial))
        pair, boot = spec["pair_index"], arrays["outputs"][1]
        if pair in initials:
            check(r.e.initial_difference(initials[pair], initial) is None, "Five-arm initial mismatch")
            for field in ("full", "noise", "original", "processed"):
                exact(boot[field], boots[pair][field], "Five-arm bootstrap mismatch")
            for j in range(8):
                exact(boot["inputs"][j], boots[pair]["inputs"][j], "Five-arm input mismatch")
        else:
            initials[pair], boots[pair] = initial, boot
        total.update(record["budget"])
        del arrays
    check(len(rows) == result["episodes_completed"] == 50 and len(initials) == 10 and
          dict(total) == result["native_budget"] and all(total[k] <= v[1] for k, v in r.LIMITS.items()), "Feedback coverage/budget")
    check(result["graph_captures"] == 100 and result["capture_internal"] == {"setup": 100, "warmup": 300, "capture": 100}, "Graph count")
    journal = dynamic.ledger(output, 52, 50)
    check(journal["kinds"]["dynamic_graph_request"] == total["model"] and
          journal["kinds"]["native_step"] == total["settling"]+total["measurement"], "Model/native ledger")
    return {"independent_contract_accepted": True, "frozen_baseline_unchanged": True, "episodes": 50,
        "initial_five_arm_groups_exact": 10, "native_actions_checked": total["measurement"],
        "request_outputs_checked": total["model"], "graph_captures": 100, "runtime_closed": 50,
        **summarize(rows), **journal, "per_episode": rows}


def audit(output):
    check(not torch.cuda.is_initialized(), "CPU audit only")
    result = json.loads((output / "result.json").read_text())
    check(result["status"] == "completed" and result["first_failure"] is None, "Formal run incomplete")
    ex = result["execution"]
    check(ex["exit_confirmed"] and ex["exit_code"] == 0 and not ex["forced"] and not ex["active"] and
          not ex["pending"] and ex["stop_reason"] is None, "Execution not closed")
    stage = result["stage"]
    saved = r.validate(result["execution_head"], stage, after=True)
    check(output == r.paths(result["execution_head"], stage)[1] and
          json.loads((output / "manifest.json").read_text()) == saved["manifest"], "Output identity changed")
    value = input_audit(output, result) if stage == "inputs" else feedback_audit(output, result, saved)
    check(result["attempts"] == 1 and result["retries"] == 0 and result["vla_frozen"] and
          result["training_updates"] == result["qualification_reads"] == 0, "Scope changed")
    check(not torch.cuda.is_initialized(), "CPU audit initialized CUDA")
    value.update(stage=stage, first_failure=None, audit_model_forwards=0, cuda_initialized=False,
        new_qualification_claimed=False, deployment_qualified=False,
        limitations=["Known development data, not independent confirmation.",
            "Same coverage estimation rule does not guarantee equal per-request C across arms.",
            "Zero explicit tail weights do not guarantee unchanged tail actions through the full VJP.",
            "CPU audit does not rerun models, VJP, vision, postprocessing, physics or GPU timing."])
    return value


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
