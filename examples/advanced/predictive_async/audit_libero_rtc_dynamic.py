"""Independent CPU reductions/provenance audit; never rerun the model or physics."""

import argparse
import json
import math
import traceback
from collections import Counter
from pathlib import Path

import audit_libero_observation_async as base
import libero_rtc_dynamic as r
import torch
from rtc_dynamic_runtime import config

from lerobot.policies.rtc.modeling_rtc import RTCProcessor

check, load = r.check, r.load


def exact(a, b, label):
    if a is None or b is None:
        check(a is b, label)
    elif isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor):
        check(a.shape == b.shape and a.dtype == b.dtype and torch.equal(a, b), label)
    else:
        base.exact(a, b, label)


def captures(receipt, count):
    check(
        receipt["sampler_restored"] and receipt["processor_restored"] and receipt["graphs_released"],
        "Sampler/processor/graphs not restored",
    )
    check(receipt["owner"] == receipt["close_thread"], "Wrong release owner")
    check(len(receipt["captures"]) == count, "Capture coverage")
    for c in receipt["captures"]:
        check(
            c["status"] == "captured"
            and c["owner"] == receipt["owner"]
            and c["setup"] == 1
            and c["warmup"] == 3
            and c["capture"] == 1
            and c["captured_steps"] == 10
            and c["captured_vjps"] == (10 if c["guided"] else 0)
            and c["projection_shapes"] == [[1, 50, 32]] * 10,
            "Capture formula/accounting differs",
        )


def ledger(output, phases_expected, close_expected):
    intents, returns, closed = {}, set(), {}
    for v in map(json.loads, (output / "calls.jsonl").read_text().splitlines()):
        cid = v.get("call_id")
        if v["event"] == "call_intent":
            check(cid not in intents, "Repeated intent")
            intents[cid] = v
        elif v["event"] == "call_return":
            check(
                cid in intents
                and cid not in returns
                and math.isfinite(v["elapsed"])
                and 0 <= v["elapsed"] <= intents[cid]["limit"],
                "Call return/deadline",
            )
            returns.add(cid)
        elif v["event"] == "owner_resources_closed":
            check(v["ordinal"] not in closed and v["first_failure"] is None, "Owner failed")
            closed[v["ordinal"]] = v
        else:
            check(v["event"] != "call_error", "Call error")
    check(set(intents) == returns and len(closed) == close_expected, "Unclosed calls/owners")
    for v in intents.values():
        if v["kind"] == "environment_close":
            check(closed[v["ordinal"]]["completed_at"] <= v["timestamp"], "Env closed before owner")
    events, active = Counter(), {}
    for v in map(json.loads, (output / "events.jsonl").read_text().splitlines()):
        events[v["event"]] += 1
        if v["event"] == "started":
            check(v["phase"] not in active, "Duplicate phase")
            active[v["phase"]] = v["limit"]
        else:
            check(
                v["event"] == "returned" and v["phase"] in active and v["seconds"] <= active.pop(v["phase"]),
                "Phase error/deadline",
            )
    check(
        not active and events == {"started": phases_expected, "returned": phases_expected}, "Unclosed phases"
    )
    return {
        "calls_closed": len(returns),
        "phases": dict(events),
        "kinds": dict(Counter(v["kind"] for v in intents.values())),
    }


def input_audit(output, result):
    data = load(r.previous.DATA)
    rows, groups = [], {}
    processor = RTCProcessor(config())
    for spec in r.input_schedule():
        row = load(output / f"request_{spec['ordinal']:03d}.pt")
        check(all(row[k] == v for k, v in spec.items()), "Input schedule differs")
        sample = data[spec["sample"]]
        check(len(row["inputs"]) == (9 if spec["length"] else 8), "Dynamic input arity")
        for j in range(8):
            exact(row["inputs"][j], sample["inputs"][j], f"Input{j} differs")
        exact(row["normalized"], row["full"][..., :7], "Unpadding differs")
        check(row["full"].shape == (1, 50, 32) and row["processed"].shape == (1, 50, 7), "Action shape")
        check(all(torch.isfinite(row[k]).all() for k in ("full", "normalized", "processed")), "Nonfinite")
        m = row["metadata"]
        guided = spec["length"] > 0
        check(
            m["mode"] == spec["arm"]
            and m["guided"] == guided
            and m["expected_delay"] == spec["delay"]
            and m["prefix_length"] == spec["length"]
            and m["effective_horizon"] == min(spec["length"], 10),
            "Wrong dynamic metadata",
        )
        if guided:
            p = sample["prefix"][: spec["length"]]
            exact(row["inputs"][8], p, "Input prefix differs")
            pad = torch.zeros((50, 7))
            pad[: len(p)] = p
            exact(row["padded"], pad, "Padding differs")
            exact(
                row["weights"],
                processor.get_prefix_weights(spec["delay"], min(len(p), 10), 50),
                "Exact weights differ",
            )
        else:
            exact(row["full"], sample["archived_full_chunk"], "No-prefix anchor differs")
            check(row["weights"] is None and row["padded"] is None, "Unexpected prefix")
        check(math.isfinite(row["complete_s"]) and 0 < row["complete_s"] < 30, "Request time")
        if spec["pair"] in groups:
            for key in ("full", "normalized", "processed"):
                exact(row[key], groups[spec["pair"]][key], "Full dynamic equivalence " + key)
        else:
            groups[spec["pair"]] = row
        rows.append(row)
    check(len(rows) == result["completed_requests"] == 482 and len(groups) == 241, "Input coverage")
    rec = result["runtime_receipt"]
    captures(rec, 2)
    check(
        rec["requests"] == rec["rgb_encodings"] == 482 and rec["replays"] == {"no_prefix": 16, "guided": 225},
        "Runtime count",
    )
    check(sum(v["metadata"]["capture_created"] for v in rows) == 1, "Unexpected recapture")
    timing = {
        arm: base.quantiles([v["complete_s"] for v in rows if v["arm"] == arm and v["length"]])
        for arm in ("native_eager", "graph")
    }
    for arm, value in timing.items():
        value["above350ms"] = sum(v["complete_s"] > 0.35 for v in rows if v["arm"] == arm and v["length"])
        value["required_delay_steps"] = math.ceil(value["p99"] * 20) + 1
    journal = ledger(output, 11, 0)
    check(journal["kinds"] == {"dynamic_request": 482}, "Call population differs")
    return {
        "independent_contract_accepted": True,
        "dynamic_equivalence_passed": True,
        "requests": 482,
        "submit_feasible_combinations_exact": 225,
        "no_prefix_anchors_exact": 32,
        "cross_arm_arrays_exact": 723,
        "timing": timing,
        "graph_budget_passed": timing["graph"]["above350ms"] == 0
        and timing["graph"]["required_delay_steps"] <= 8,
        "graph_faster_mean": timing["graph"]["mean"] < timing["native_eager"]["mean"],
        "captures": 2,
        "replayed_guided_vjps": 2250,
        "extra_sampler_calls": 10,
        **journal,
    }


def dynamic_output(out, req, record):
    exact(out["context_prefix"], out["prefix"], "Predictor did not receive owned normalized prefix")
    check(
        out["context_stamp"] == req["stamp"] and out["context_prefix_source"] == req["prefix_source"],
        "Context stamp/source changed",
    )
    used = record["spec"]["arm"] == "rtc_async" and out["prefix"] is not None and len(out["prefix"]) > 0
    delay = req["stamp"]["expected_delay"] if used else 0
    length = len(out["prefix"]) if used else 0
    metadata = out["rtc_metadata"]
    check(
        metadata["mode"] == "graph"
        and metadata["guided"] == used
        and metadata["replays"] == 1
        and metadata["expected_delay"] == delay
        and metadata["prefix_length"] == length
        and metadata["captured_vjps_per_replay"] == (10 if used else 0),
        "RTC request input path differs",
    )
    check(
        metadata["capture_created"] == out["capture_created"] == (req["stamp"]["request_id"] == 1),
        "Control-time recapture",
    )
    check(out["vision_encodes"] == 1 and len(out["inputs"]) == (9 if used else 8), "Vision/input population")
    exact(out["noise"], out["inputs"][7], "Noise differs")
    if used:
        exact(out["inputs"][8], out["prefix"], "RTC used wrong prefix")
        padded = torch.zeros((50, 7))
        padded[:length] = out["prefix"]
        exact(out["rtc_padded_prefix"], padded, "Stale padded prefix")
        weights = RTCProcessor(config()).get_prefix_weights(delay, min(10, length), 50)
        exact(out["rtc_weights"], weights, "Stale or wrong RTC weights")
    else:
        check(out["rtc_weights"] is None and out["rtc_padded_prefix"] is None, "Unexpected guidance")
    check(out["owner_thread"] == record["predictor_cleanup"]["owner"], "Wrong inference owner")


def inspect_episode(spec, record, arrays, checkpoint):
    row = base.audit_episode(spec, record, arrays, checkpoint, output_checker=dynamic_output)
    rec, closed = record["predictor_cleanup"], record["owner_cleanup"]
    captures(rec, 2)
    check(
        rec["requests"]
        == rec["rgb_encodings"]
        == rec["live_requests"]
        == rec["explicit_noise_draws"]
        == len(record["requests"]),
        "Request/RGB accounting",
    )
    check(
        rec["owner"] == closed["owner_thread"] != record["controller_thread"]
        and closed["first_failure"] is None,
        "Owner isolation/cleanup",
    )
    check(
        arrays["control"]["ended_at"] <= closed["started_at"] <= rec["closed_at"] <= closed["completed_at"],
        "Cleanup chronology",
    )
    native = [v for v in record["native_steps"] if v["segment"] == "measurement"]
    delay_pairs = []
    guided_count = 0
    for req in record["requests"]:
        out = arrays["outputs"][req["stamp"]["request_id"]]
        obs = arrays["observations"][req["stamp"]["observation_index"]]
        check(out["input_fingerprint"] == r.g.observation_fingerprint(obs), "Live pixels/state mismatch")
        check(req["completed_at"] <= closed["started_at"], "Released during request")
        for when in (req["requested_at"], req["published_at"]):
            check(not any(v["started_at"] < when < v["returned_at"] for v in native), "Inside native call")
        guided_count += int(out["rtc_metadata"]["guided"])
        if req["decision"]["accepted"] and req["stamp"]["request_id"] > 1:
            delay_pairs.append(
                [req["stamp"]["expected_delay"], req["decision"]["actual_delay"], req["prefix_rows"]]
            )
    check(
        rec["replays"].get("guided", 0) == guided_count
        and sum(rec["replays"].values()) == len(record["requests"]),
        "Replay guidance count",
    )
    row.update(
        arm=spec["arm"],
        identity=[spec["task_id"], spec["initial_state_id"]],
        cohort=spec["cohort"],
        delay_pairs=delay_pairs,
        guided_requests=guided_count,
        capture_seconds=sum(c["seconds"] for c in rec["captures"]),
    )
    return row


def feedback_audit(output, result):
    rows, initials, boots, total = [], {}, {}, Counter()
    for spec in r.manifest("feedback")["rows"]:
        folder = output / f"episode_{spec['ordinal']:03d}"
        record = json.loads((folder / "result.json").read_text())
        check(json.loads((folder / "started.json").read_text())["spec"] == spec, "Started identity")
        arrays, initial = load(folder / "arrays.pt"), load(folder / "initial_checkpoint.pt")
        row = inspect_episode(spec, record, arrays, initial)
        rows.append(row)
        pair = spec["pair_index"]
        boot = arrays["outputs"][1]
        if pair in initials:
            check(r.e.initial_difference(initials[pair], initial) is None, "Triple initial mismatch")
            for field in ("full", "noise"):
                exact(boot[field], boots[pair][field], "Triple " + field + " mismatch")
            for j in range(8):
                exact(boot["inputs"][j], boots[pair]["inputs"][j], "Triple input mismatch")
        else:
            initials[pair], boots[pair] = initial, boot
        total.update(record["budget"])
    check(
        len(rows) == result["episodes_completed"] == 30
        and len(initials) == 10
        and dict(total) == result["native_budget"],
        "Global episode budget differs",
    )
    check(
        result["graph_captures"] == 60
        and result["capture_internal"] == {"setup": 60, "warmup": 180, "capture": 60},
        "Capture budget",
    )
    journal = ledger(output, 32, 30)
    check(
        journal["kinds"]["dynamic_graph_request"] == total["model"]
        and journal["kinds"]["native_step"] == total["settling"] + total["measurement"],
        "Model/native ledger",
    )
    arms = {}
    for arm in r.ARMS:
        selected = [v for v in rows if v["arm"] == arm]
        times = [t for v in selected for t in v["request_seconds"]]
        stats = base.quantiles(times)
        stats.update(
            above350ms=sum(t > 0.35 for t in times), required_delay_steps=math.ceil(stats["p99"] * 20) + 1
        )
        arms[arm] = {
            "episodes": len(selected),
            "successes": sum(v["success"] for v in selected),
            **{
                k: sum(v[k] for v in selected)
                for k in (
                    "actions",
                    "wall_s",
                    "no_action_slots",
                    "underflows",
                    "expired",
                    "cancelled_at_stop",
                    "request_count",
                    "guided_requests",
                    "capture_seconds",
                )
            },
            "request_time_s": stats,
            "bootstrap_s": base.quantiles([v["bootstrap_s"] for v in selected]),
            "model_native_intersections": sum(len(v["overlaps"]) for v in selected),
            "expected_actual_prefix": dict(Counter(str(v) for row in selected for v in row["delay_pairs"])),
        }
    lost = {a: [] for a in ("serialized", "aligned_async")}
    triples = []
    for pair, identity in enumerate(r.PAIRS):
        selected = {v["arm"]: v for v in rows if v["pair_index"] == pair}
        triples.append(
            {
                "identity": list(identity),
                "arms": {a: {k: v[k] for k in ("success", "actions", "wall_s")} for a, v in selected.items()},
            }
        )
        for arm in lost:
            if selected[arm]["success"] and not selected["rtc_async"]["success"]:
                lost[arm].append(list(identity))
    rtc = arms["rtc_async"]
    serial = arms["serialized"]
    gates = {
        "no_lost_vs_serialized": not lost["serialized"],
        "no_lost_vs_aligned": not lost["aligned_async"],
        "all_requests_budget": all(
            v["request_time_s"]["above350ms"] == 0 and v["request_time_s"]["required_delay_steps"] <= 8
            for v in arms.values()
        ),
        "rtc_overlap": rtc["model_native_intersections"] > 0,
        "rtc_wait_reduced": rtc["no_action_slots"] / rtc["actions"]
        < serial["no_action_slots"] / serial["actions"],
        "rtc_no_queue_fault": rtc["underflows"] == rtc["expired"] == 0,
        "rtc_prefix_applied": rtc["guided_requests"] > 0,
    }
    return {
        "independent_contract_accepted": True,
        "frozen_baseline_unchanged": True,
        "episodes": 30,
        "initial_triples_exact": 10,
        "native_actions_checked": total["measurement"],
        "request_outputs_checked": total["model"],
        "arms": arms,
        "triples": triples,
        "lost_success_identities": lost,
        "development_gates": gates,
        "development_followup_supported": all(gates.values()),
        "per_episode": rows,
        **journal,
    }


def audit(output):
    check(not torch.cuda.is_initialized(), "CPU audit only")
    result = json.loads((output / "result.json").read_text())
    check(result["status"] == "completed" and result["first_failure"] is None, "Formal run incomplete")
    ex = result["execution"]
    check(
        ex["exit_confirmed"]
        and ex["exit_code"] == 0
        and not ex["forced"]
        and not ex["active"]
        and not ex["pending"]
        and ex["stop_reason"] is None,
        "Execution not closed",
    )
    stage = result["stage"]
    saved = r.validate(result["execution_head"], stage, after=True)
    check(
        output == r.paths(result["execution_head"], stage)[1]
        and json.loads((output / "manifest.json").read_text()) == saved["manifest"],
        "Output/manifest mismatch",
    )
    value = input_audit(output, result) if stage == "inputs" else feedback_audit(output, result)
    check(
        result["attempts"] == 1
        and result["retries"] == 0
        and result["vla_frozen"]
        and result["training_updates"] == 0,
        "Scope changed",
    )
    value.update(
        stage=stage,
        first_failure=None,
        audit_model_forwards=0,
        cuda_initialized=False,
        new_qualification_claimed=False,
        deployment_qualified=False,
        limitations=[
            "Old development data; no independent task qualification.",
            "CPU audit does not rerun model, VJP, RGB encoding, postprocessing, physics or GPU timing.",
        ],
    )
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
