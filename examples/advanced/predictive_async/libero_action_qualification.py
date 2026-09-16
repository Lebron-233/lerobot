"""F-ACQ1: frozen F-ACR1 candidates on eight previously unused initial states.

No optimizer, checkpoint selection, learned gate, or predictor-controlled Env.
Only --prepare is CPU-only. The default entry supervises one registered worker.
"""

import argparse
import copy
import faulthandler
import hashlib
import json
import math
import os
import signal
import subprocess
import sys
import time
import traceback
from collections import Counter, defaultdict
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

import libero_action_centered as acr
import torch

pfx, cov, e, pilot, opt = acr.pfx, acr.cov, acr.e, acr.pilot, acr.opt
natural = cov.natural
SOURCE = e.REPO / "outputs/smolvla_action_centered_2926678f"
CHECKPOINTS = {
    "base": pfx.SOURCE / "case_no_action.pt",
    "centered": SOURCE / "centered.pt",
    "ordinary": SOURCE / "ordinary.pt",
}
# The unregistered draft's states 42--45 collided with legacy `tuple` records.
# These new fixed identities must STILL pass --prepare; unused status is not assumed.
# Never replace identities after registration or inspect old confirmation states 21--40.
PAIRS = ((6, 0), (7, 0), (7, 1), (6, 1), (6, 2), (7, 2), (7, 3), (6, 3))
ARMS = ("identity", "base", "centered_true", "centered_zero", "centered_mismatched",
        "ordinary_true", "ordinary_zero", "ordinary_mismatched")
NATIVE_LIMITS = {"episodes": (1, 8), "settling": (10, 80), "measurement": (280, 2240),
                 "model": (160, 1280), "capture": (2, 16)}
# 16 old validation anchors: 48 decodes + 64 predictor forwards, exact replay only.
# New N<=32, M<=N: 7N+2M decodes; 7N+3M predictor forwards.
LIMITS = {"encoding": 40, "decoder": 336, "predictor": 384}
SOFT_SECONDS, HARD_SECONDS = 1500, 1530


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def preparation_path(head):
    return e.REPO / "outputs" / f"smolvla_action_qualification_preparation_{head[:8]}"


def output_path(head):
    return e.REPO / "outputs" / f"smolvla_action_qualification_{head[:8]}"


def manifest():
    original = e.fixed_manifest()
    rows = []
    for task, state in PAIRS:
        row = copy.deepcopy(next(r for r in original["rows"]
                                 if r["task_id"] == task and r["condition"] == "graph_identity_async"))
        row.update(ordinal=len(rows), pair_index=len(rows), initial_state_id=state,
                   environment_seed=1100000 + 100 * task + state,
                   policy_seed=1110000 + 100 * task + state, split="qualification",
                   recovery_policy=natural.recovery.RECOVERY_POLICY, max_recovery_probes_per_episode=50)
        rows.append(row)
    return {"experiment": "F-ACQ1", "rows": rows, "max_samples_per_episode": 4,
            "native_limits": NATIVE_LIMITS, "offline_limits": LIMITS,
            "checkpoints": {k: str(v) for k, v in CHECKPOINTS.items()},
            "training_updates": 0, "old_test_reads": 0, "predictor_controls_environment": False,
            **{k: original[k] for k in ("policy_revision", "vlm_revision", "assets_revision")}}


def history_inventory(root):
    """Read identities only, not old results, observations, or test labels."""
    rows = []
    for path in sorted(Path(root).rglob("started.json")):
        value = json.loads(path.read_text())
        spec = value.get("spec", value.get("tuple", value))
        if not all(k in spec for k in ("task_id", "initial_state_id")):
            raise ValueError(f"Unknown historical identity schema: {path}")
        rows.append({"path": str(path.relative_to(root)), "task": int(spec["task_id"]),
                     "state": int(spec["initial_state_id"]), "sha256": digest(path)})
    return rows


def check_unused(history):
    overlap = sorted(set(PAIRS) & {(r["task"], r["state"]) for r in history})
    if overlap:
        raise ValueError(f"Qualification identities already used; no replacement: {overlap}")


def checkpoint_valid(saved, arm):
    expected_arm, step = ("case_no_action", 36) if arm == "base" else (arm, 72)
    if (saved.get("arm") != expected_arm or saved.get("best_step") != step
            or saved.get("seed") != 20260912 or saved.get("config") != asdict(pilot.config())):
        raise ValueError(f"Frozen checkpoint metadata differs: {arm}")


def source_files():
    return [*CHECKPOINTS.values(), SOURCE / "independent_audit.json", acr.BASE / "independent_audit.json",
            acr.BASE / "predictions.pt", SOURCE / "assessment_selected_centered.pt",
            SOURCE / "assessment_selected_ordinary.pt", pfx.scale.OLD, pfx.scale.NEW]


def prepare(head):
    acr.source_gate(head)
    if str(Path(sys.executable)) != e.PYTHON:
        raise ValueError("Use the frozen reference interpreter")
    path = preparation_path(head)
    if path.exists() or output_path(head).exists():
        raise ValueError("Preparation/output already exists; inspect, do not overwrite")
    history = history_inventory(e.REPO / "outputs")
    check_unused(history)
    for arm, file in CHECKPOINTS.items():
        checkpoint_valid(torch.load(file, map_location="cpu", weights_only=False), arm)
    for file in (SOURCE / "independent_audit.json", acr.BASE / "independent_audit.json"):
        if json.loads(file.read_text())["independent_contract_accepted"] is not True:
            raise ValueError("Source independent audit not accepted")
    if torch.cuda.is_initialized():
        raise ValueError("Preparation must not initialize CUDA")
    data = {"experiment": "F-ACQ1", "execution_head": head, "manifest": manifest(),
            "source_hashes": {str(p): digest(p) for p in source_files()},
            "history_scope": str(e.REPO / "outputs"), "history": history,
            "cuda_initialized": False, "model_forwards": 0, "new_env": 0}
    path.mkdir()
    e.write_json(path / "preparation.json", data)
    print(json.dumps({"prepared": True, "path": str(path), "sha256": digest(path / "preparation.json"),
                      "historical_identities": len(history), "new_episodes": 8}), flush=True)


def validate_preparation(head, registration=False):
    prep = preparation_path(head)
    saved = json.loads((prep / "preparation.json").read_text())
    # Roundtrip makes tuple/list representation identical to persisted JSON.
    if saved["execution_head"] != head or saved["manifest"] != json.loads(json.dumps(manifest())):
        raise ValueError("Prepared contract differs")
    if saved["source_hashes"] != {str(p): digest(p) for p in source_files()}:
        raise ValueError("Frozen source bytes changed")
    current = history_inventory(e.REPO / "outputs")
    check_unused(current)
    if current != saved["history"]:
        raise ValueError("History changed after preparation; review before any dispatch")
    if registration:
        body = (prep / "registration.md").read_text()
        response = json.loads((prep / "registration_readback.json").read_text())
        required = ("F-ACQ1", head, str(output_path(head)), digest(prep / "preparation.json"))
        if (response.get("body") != body or not isinstance(response.get("id"), int)
                or response.get("issue_url") != "https://api.github.com/repos/Lebron-233/lerobot/issues/1"
                or not all(text in body for text in required)):
            raise ValueError("Actual-ID registration readback gate failed")


class Budget(e.Budget):
    def check(self, kind):
        local, total = NATIVE_LIMITS[kind]
        if self.episode[kind] >= local or self.total[kind] >= total:
            raise RuntimeError(f"F-ACQ1 native {kind} exhausted before dispatch")


class Runtime(pilot.SmolVLAGraphRuntime):
    def _capture(self, inputs):
        if len(self.captures) >= 8:
            raise RuntimeError("F-ACQ1 offline capture limit exhausted")
        return super()._capture(inputs)


def take(counts, name):
    if counts[name] >= LIMITS[name]:
        raise RuntimeError(f"F-ACQ1 {name} exhausted before dispatch")
    counts[name] += 1


def cpu(values):
    return tuple(v.detach().cpu().clone() for v in values)


def validate_prefix(sample):
    """Validate new native delays, not the old development-only delay-3/4 table."""
    delay = sample["delay"]
    if isinstance(delay, bool) or not isinstance(delay, int) or not 0 <= delay <= 8:
        raise ValueError("Native delay is outside the frozen cap")
    if sample["future_index"] != sample["current_index"] + delay:
        raise ValueError("Future index differs from the committed delay")
    actions, mask = sample["actions"], sample["mask"]
    if actions.shape != (1, 8, 7) or not actions.is_floating_point():
        raise ValueError("Committed prefix shape/type differs")
    if mask.dtype != torch.bool or not torch.equal(mask.cpu(), (torch.arange(8) < delay)[None]):
        raise ValueError("Committed mask differs from the native delay")
    if not torch.isfinite(actions).all() or torch.count_nonzero(actions[:, delay:]):
        raise ValueError("Committed actions are nonfinite or padding is nonzero")


def branch(model, sample, actions, counts):
    batch = pilot.batch_for([sample], [0], "cuda")
    take(counts, "predictor")
    return model(batch["tokens"], batch["masks"], actions.to("cuda"), batch["action_mask"],
                 batch["state"], batch["delay"]).delta_tokens


def base_tokens(model, sample, counts):
    delta = branch(model, sample, torch.zeros_like(sample["actions"]), counts)
    return tuple(z.cuda().float() + dz for z, dz in zip(sample["inputs"][:2], delta, strict=True))


def residual(model, sample, base, arm, actions, counts):
    a = branch(model, sample, actions, counts)
    z = branch(model, sample, torch.zeros_like(actions), counts) if arm == "centered" else tuple(
        torch.zeros_like(v) for v in a)
    raw = acr.compose(base, a, z, arm)
    return {"action_delta": cpu(a), "zero_delta": cpu(z), "raw_visual": cpu(raw),
            "visual": cpu(tuple(v.to(torch.bfloat16).float() for v in raw)),
            "actual_actions": actions.cpu().clone(), "actual_mask": sample["mask"].clone()}


def decode(runtime, sample, visual, output, counts, label):
    take(counts, "decoder")
    values = tuple(v.cuda() for v in sample["inputs"])
    with pilot.phase(output, f"decode_{counts['decoder']}_{label}", 30):
        runtime.begin_episode("graph", sample["task"])
        value = runtime(None, None, values[4], values[5], values[6], noise=values[7],
                        future_image_tokens=tuple(v.to(device="cuda", dtype=values[c].dtype)
                                                  for c, v in enumerate(visual)),
                        future_image_token_masks=tuple(values[2:4])).detach().cpu().clone()
        if value.shape != (1, 50, 32) or not torch.isfinite(value).all():
            raise ValueError("Invalid original decoder output")
        return value


def replay_anchors(models, runtime, output, counts):
    # Old validation is used only for equality, never new selection or metric tuning.
    _, validation = pfx.load_data()
    frozen_base = {tuple(r["key"]): r for r in torch.load(acr.BASE / "predictions.pt",
                  map_location="cpu", weights_only=False) if r["context"] == "no_action"}
    references = {arm: {tuple(r["key"]): r for r in torch.load(SOURCE / f"assessment_selected_{arm}.pt",
                  map_location="cpu", weights_only=False) if r["context"] == "true"}
                  for arm in ("centered", "ordinary")}
    if len(validation) != 16:
        raise ValueError("Anchor coverage changed")
    evidence = []
    for sample in sorted(validation, key=pfx.key):
        with pilot.phase(output, f"anchor_{pfx.key(sample)}", 90):
            raw = base_tokens(models["base"], sample, counts)
            base = frozen_base[pfx.key(sample)]
            for a, b in zip(cpu(raw), base["raw_visual"], strict=True):
                pilot.require_equal(a, b, "Reconstructed FP32 base differs from F-PFX1 cache")
            base_output = decode(runtime, sample, base["visual"], output, counts, "anchor_base")
            pilot.require_equal(base_output, base["output"], "Anchor base output differs")
            saved = {"key": list(pfx.key(sample)), "base_raw": cpu(raw), "base_output": base_output}
            for arm in ("centered", "ordinary"):
                r = residual(models[arm], sample, raw, arm, sample["actions"], counts)
                r["output"] = decode(runtime, sample, r["visual"], output, counts, f"anchor_{arm}")
                expected = references[arm][pfx.key(sample)]
                for field in ("raw_visual", "visual"):
                    for a, b in zip(r[field], expected[field], strict=True):
                        pilot.require_equal(a, b, f"Anchor {arm} {field} differs")
                pilot.require_equal(r["output"], expected["output"], f"Anchor {arm} output differs")
                saved[arm] = r
            evidence.append(saved)
            counts["anchor_exact"] += 1
    torch.save(evidence, output / "anchor_replay.pt")


def extract(records, policy, pre, output, counts):
    samples, selections = [], []
    for record in records:
        spec = record["spec"]
        arrays = torch.load(output / f"episode_{spec['ordinal']:03d}/arrays.pt",
                            map_location="cpu", weights_only=False)
        pairs, excluded = cov.selected_pairs(record, arrays)
        selections.append({"ordinal": spec["ordinal"], "requests": [p["request_id"] for p in pairs],
                           "excluded": excluded})
        for number, pair in enumerate(pairs):
            inputs = cpu(pair["cached"]["inputs"])
            for kind in ("current", "future") if number == 0 else ("future",):
                take(counts, "encoding")
                with pilot.phase(output, f"encode_{spec['ordinal']}_{pair['request_id']}_{kind}", 30):
                    pre.reset()
                    batch = pre(pilot.worker_batch(pair[f"{kind}_observation"],
                                                  spec["task_name"].replace("_", " "), "cuda"))
                    images, masks = policy.prepare_images(batch)
                    z, zm = policy.model.encode_image_tokens(images, masks)
                    z = cpu(z)
                    for c in range(2):
                        pilot.require_equal(zm[c], inputs[c + 2], "Camera mask differs")
                        if kind == "current":
                            pilot.require_equal(z[c], inputs[c], "Current token differs from native")
                    if kind == "current":
                        pilot.require_equal(policy.prepare_state(batch), inputs[6], "Current state differs")
                        counts["current_exact"] += 1
            samples.append({"task": spec["task_id"], "initial_state_id": spec["initial_state_id"],
                            "ordinal": spec["ordinal"], "split": "qualification", "inputs": inputs,
                            "future": z, "archived_full_chunk": pair["cached"]["full_chunk"].clone(),
                            **{k: pair[k] for k in ("request_id", "current_index", "future_index",
                                                   "delay", "actions", "mask")}})
    samples.sort(key=pfx.key)
    for sample in samples:
        validate_prefix(sample)
    torch.save(samples, output / "aligned_cache.pt")
    e.write_json(output / "selections.json", selections)
    return samples


def metrics(visual, value, sample, oracle):
    diff = (value.double()[..., :7] - oracle.double()[..., :7]).square()
    numerator, denominator = 0.0, 0
    for a, b, mask in zip(visual, sample["future"], sample["inputs"][2:4], strict=True):
        numerator += float(((a.double() - b.double()).square() * mask.unsqueeze(-1)).sum())
        denominator += int(mask.sum()) * a.shape[-1]
    if not denominator:
        raise ValueError("No valid visual token")
    result = {"row0": float(diff[:, 0].mean()), "chunk": float(diff.mean()),
              "latent": numerator / denominator}
    if not all(math.isfinite(v) for v in result.values()):
        raise ValueError("Nonfinite evaluation metric")
    return result


def tolerance(value):
    return 1e-7 + 1e-6 * abs(value)


def aggregate(rows):
    def mean(values):
        return math.fsum(values) / len(values)

    by_arm, comparisons = {}, {}
    for arm in ARMS:
        available = [r for r in rows if arm in r["metrics"]]
        groups = defaultdict(list)
        for r in available:
            groups[tuple(r["key"][:2])].append(r["metrics"][arm])
        episodes = {f"{t}/{s}": {m: mean([r[m] for r in rs]) for m in ("row0", "chunk", "latent")}
                    for (t, s), rs in sorted(groups.items())}
        by_arm[arm] = {"samples": len(available), "episodes": episodes,
                       "macro": {m: mean([v[m] for v in episodes.values()]) for m in
                                 ("row0", "chunk", "latent")} if episodes else {}}
    for control in ("base", "identity", "ordinary_true", "centered_mismatched"):
        paired = [r for r in rows if control in r["metrics"]]
        groups = defaultdict(list)
        differences, improved = [], 0
        for r in paired:
            a, b = r["metrics"]["centered_true"]["row0"], r["metrics"][control]["row0"]
            differences.append({"key": r["key"], "control_minus_centered": b - a})
            groups[tuple(r["key"][:2])].append((a, b))
            improved += a < b - tolerance(b)
        ep = {f"{t}/{s}": {"centered": mean([a for a, _ in v]), "control": mean([b for _, b in v])}
              for (t, s), v in sorted(groups.items())}
        for v in ep.values():
            v["benefit"] = v["control"] - v["centered"]
        total = math.fsum(v["benefit"] for v in ep.values())
        loo = {k: mean([v["benefit"] for j, v in ep.items() if j != k]) for k in ep} if len(ep) > 1 else {}
        comparisons[control] = {"paired_samples": len(paired), "paired_episodes": len(ep),
                                "episodes": ep, "sample_improved": improved,
                                "episode_improved": sum(v["benefit"] > tolerance(v["control"])
                                                        for v in ep.values()),
                                "macro_benefit": total / len(ep) if ep else None,
                                "leave_one_episode_out": loo,
                                "largest_episode_net_share": max(v["benefit"] for v in ep.values()) / total
                                if total > 0 else None,
                                "per_sample": differences}
    complete = {(r["key"][0], r["key"][1]) for r in rows} == set(PAIRS)
    centered = by_arm["centered_true"]["macro"]
    checks = {"eight_episodes": complete, "mismatch_eight_episodes": comparisons["centered_mismatched"]["paired_episodes"] == 8}
    for arm in ("base", "ordinary_true", "centered_mismatched"):
        c = comparisons[arm]
        checks[f"row0_better_{arm}"] = bool(c["episodes"]) and c["macro_benefit"] > tolerance(
            mean([v["control"] for v in c["episodes"].values()]))
    for arm in ("base", "identity"):
        checks[f"chunk_not_worse_{arm}"] = bool(centered) and centered["chunk"] <= (
            by_arm[arm]["macro"]["chunk"] + tolerance(by_arm[arm]["macro"]["chunk"]))
    for arm in ("base", "centered_mismatched"):
        checks[f"six_episodes_better_{arm}"] = comparisons[arm]["episode_improved"] >= 6
    primary = all(checks.values())
    b = comparisons["base"]
    robustness = {"strict_sample_majority": b["sample_improved"] > b["paired_samples"] / 2,
                  "all_leave_one_episode_out_positive": len(b["leave_one_episode_out"]) == 8 and all(
                      v > tolerance(by_arm["base"]["macro"]["row0"])
                      for v in b["leave_one_episode_out"].values())}
    # Descriptive strata, never selection or a replacement primary endpoint.
    strata = {}
    for delay in sorted({r["delay"] for r in rows}):
        subset = [r for r in rows if r["delay"] == delay]
        strata[str(delay)] = {"samples": len(subset), "mean_base_minus_centered": mean([
            r["metrics"]["base"]["row0"] - r["metrics"]["centered_true"]["row0"] for r in subset])}
    return {"metrics": by_arm, "contrasts": comparisons, "delay_descriptive": strata,
            "primary_checks": checks, "robustness_checks": robustness,
            "heldout_primary_gate_passed": primary,
            "heldout_robustness_gate_passed": primary and all(robustness.values()),
            "statistical_significance_claimed": False}


def evaluate(samples, models, runtime, output, counts):
    donors = pfx.donor_map(samples)
    by_key = {pfx.key(s): s for s in samples}
    rows = []
    (output / "predictions").mkdir()
    for sample in samples:
        k = pfx.key(sample)
        with pilot.phase(output, f"sample_{k}", 120):
            raw = base_tokens(models["base"], sample, counts)
            visual = {"identity": cpu(sample["inputs"][:2]), "oracle": cpu(sample["future"]),
                      "base": cpu(tuple(v.to(torch.bfloat16).float() for v in raw))}
            evidence = {}
            for arm in ("centered", "ordinary"):
                for context in ("true", "zero", "mismatched"):
                    if context == "mismatched" and donors[k] is None:
                        continue
                    used = pfx.with_prefix(sample, by_key[donors[k]]) if context == "mismatched" else sample
                    actions = torch.zeros_like(sample["actions"]) if context == "zero" else used["actions"]
                    name = f"{arm}_{context}"
                    evidence[name] = residual(models[arm], sample, raw, arm, actions, counts)
                    visual[name] = evidence[name]["visual"]
            outputs = {name: decode(runtime, sample, z, output, counts, name) for name, z in visual.items()}
            pilot.require_equal(outputs["identity"], sample["archived_full_chunk"], "Native identity replay differs")
            for a, b in zip(evidence["centered_zero"]["raw_visual"], cpu(raw), strict=True):
                pilot.require_equal(a, b, "Centered zero FP32 base invariant differs")
            pilot.require_equal(outputs["centered_zero"], outputs["base"], "Centered zero full output differs")
            counts["identity_exact"] += 1
            counts["zero_exact"] += 1
            scores = {a: metrics(z, outputs[a], sample, outputs["oracle"]) for a, z in visual.items() if a != "oracle"}
            row = {"key": list(k), "delay": sample["delay"],
                   "donor": list(donors[k]) if donors[k] else None, "metrics": scores}
            saved = {**row, "base_raw": cpu(raw), "visual": visual, "outputs": outputs, "residuals": evidence}
            torch.save(saved, output / "predictions" / f"{k[0]}_{k[1]}_{k[2]}.pt")
            rows.append(row)
            with (output / "metric_rows.jsonl").open("a") as stream:
                stream.write(json.dumps(row) + "\n")
    return rows


def worker(args):
    acr.source_gate(args.execution_head)
    validate_preparation(args.execution_head)
    faulthandler.register(signal.SIGUSR1, file=sys.stderr, all_threads=True)
    torch.set_num_threads(1)
    counts, budget, records, runtime = Counter(), Budget(), [], None
    calls = e.Calls(args.output / "calls.jsonl")
    result = {"experiment": "F-ACQ1", "status": "technical_failure", "first_failure": None}
    try:
        with pilot.phase(args.output, "load", 90):
            if torch.cuda.get_device_name() != "NVIDIA GeForce RTX 4070 Ti SUPER":
                raise ValueError("Registered GPU differs")
            models, frozen = {}, {}
            for arm, path in CHECKPOINTS.items():
                saved = torch.load(path, map_location="cpu", weights_only=False)
                checkpoint_valid(saved, arm)
                model = pilot.LightweightFutureLatentPredictor(pilot.config())
                model.load_state_dict(saved["state_dict"], strict=True)
                if sum(p.numel() for p in model.parameters()) != 69680:
                    raise ValueError("Predictor parameter count differs")
                models[arm] = model.eval().requires_grad_(False).cuda()
                frozen[arm] = saved
                counts["predictor_loads"] += 1
            torch.save(frozen, args.output / "frozen_checkpoints_before.pt")
            policy, pre, post, report = e.load_runtime(e.POLICY, e.VLM)
            counts["vla_loads"] += 1
            e.write_json(args.output / "policy_load.json", report)
            runtime = Runtime(policy.model)
        with torch.no_grad(), pilot.phase(args.output, "anchor_replay", 180):
            replay_anchors(models, runtime, args.output, counts)
        factory = natural.trace.checkpoint_factory(e.reference.make_native_env_factory(), calls)
        with pilot.phase(args.output, "native_collection", 960):
            for spec in manifest()["rows"]:
                print(f"F-ACQ1 START {spec['ordinal']} task={spec['task_id']} state={spec['initial_state_id']}", flush=True)
                record, _ = e.run_episode(spec, args.output / f"episode_{spec['ordinal']:03d}",
                                          policy, pre, post, factory, budget, calls,
                                          engine_class=natural.NaturalEngine)
                records.append(record)
                if record["status"] != "completed":
                    raise RuntimeError(record["first_failure"] or "Native collection incomplete")
        with torch.no_grad(), pilot.phase(args.output, "offline_evaluation", 240):
            samples = extract(records, policy, pre, args.output, counts)
            rows = evaluate(samples, models, runtime, args.output, counts)
        n, m = len(rows), sum(r["donor"] is not None for r in rows)
        expected = {"encoding": n + 8, "decoder": 48 + 7 * n + 2 * m,
                    "predictor": 64 + 7 * n + 3 * m, "anchor_exact": 16,
                    "current_exact": 8, "identity_exact": n, "zero_exact": n,
                    "vla_loads": 1, "predictor_loads": 3}
        if any(counts[k] != v for k, v in expected.items()):
            raise ValueError(f"Formal accounting differs: {dict(counts)} expected={expected}")
        after = {a: opt.freeze_state(model) for a, model in models.items()}
        for arm in models:
            for key, value in after[arm].items():
                pilot.require_equal(value, frozen[arm]["state_dict"][key], "Frozen predictor changed")
        torch.save(after, args.output / "frozen_weights_after.pt")
        if not all(not p.requires_grad and p.grad is None for p in policy.parameters()):
            raise ValueError("VLA was not frozen")
        result.update(status="completed", samples=n, mismatched_samples=m, frozen_weights_unchanged=True,
                      vla_frozen=True, expected_counts=expected, **aggregate(rows))
    except BaseException:
        result["first_failure"] = traceback.format_exc()
    finally:
        calls.close()
        if runtime is not None:
            e.write_json(args.output / "offline_captures.json", runtime.captures)
            counts["offline_captures"] = len(runtime.captures)
            runtime.release_graph()
            result["graph_released"] = runtime.graph is None
        result.update(execution_head=args.execution_head, counts=dict(counts), native_budget=dict(budget.total),
                      episodes_completed=sum(r["status"] == "completed" for r in records),
                      predictor_controls_environment=False, training_updates=0, backward=0, real_robot=0,
                      old_test_reads=0, baseline_qualified=False, realtime_qualified=False,
                      predictor_benefit_tested=False, risk_thresholds=None,
                      old_confirmation="not_started_untouched", attempts=1, retries=0)
        e.write_json(args.output / "worker_result.json", result)
    return 0 if result["status"] == "completed" else 2


def supervise(args):
    acr.source_gate(args.execution_head)
    if str(Path(sys.executable)) != e.PYTHON or args.output != output_path(args.execution_head) or args.output.exists():
        raise ValueError("Use fixed interpreter and unused registered output")
    validate_preparation(args.execution_head, registration=True)
    args.output.mkdir()
    e.write_json(args.output / "manifest.json", manifest())
    command = [sys.executable, "-u", "-X", "faulthandler", str(Path(__file__).resolve()),
               "--execution-head", args.execution_head, "--output", str(args.output), "--worker"]
    start, utc = time.perf_counter(), datetime.now(UTC).isoformat()
    cursors, pending, active = {}, {}, {}
    reason, terminated, forced = None, None, False
    with (args.output / "model.log").open("x") as log:
        child = subprocess.Popen(command, cwd=e.REPO, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        print(f"F-ACQ1 worker pid={child.pid}", flush=True)
        try:
            while child.poll() is None:
                cov.parent.previous.update_pending(args.output, cursors, pending, active)
                now = time.perf_counter()
                expired = [v for v in pending.values() if now - v["timestamp"] > v["limit"]]
                phases = [v for v in active.values() if now - v["at"] > v["limit"]]
                if terminated is None and (expired or phases or now - start >= SOFT_SECONDS):
                    reason = {"expired_calls": expired, "expired_phases": phases, "outer_soft": now - start >= SOFT_SECONDS}
                    os.kill(child.pid, signal.SIGUSR1)
                    os.killpg(child.pid, signal.SIGTERM)
                    terminated = now
                if now - start >= HARD_SECONDS or (terminated is not None and now - terminated >= 5):
                    os.killpg(child.pid, signal.SIGKILL)
                    forced = True
                    break
                time.sleep(0.1)
        except BaseException:
            reason = reason or {"supervisor_exception": traceback.format_exc()}
            if child.poll() is None:
                os.killpg(child.pid, signal.SIGTERM)
                try:
                    child.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL)
                    forced = True
        code = child.wait()
    cov.parent.previous.update_pending(args.output, cursors, pending, active)
    path = args.output / "worker_result.json"
    result = json.loads(path.read_text()) if path.exists() else {"status": "technical_failure", "first_failure": "Worker receipt absent"}
    result["execution"] = {"command": command, "child_pid": child.pid, "child_exit_code": code,
                           "exit_confirmed": True, "started_at_utc": utc, "finished_at_utc": datetime.now(UTC).isoformat(),
                           "wall_seconds": time.perf_counter() - start, "stop_reason": reason,
                           "forced_termination": forced, "pending_calls_at_exit": list(pending.values()),
                           "active_phases_at_exit": active}
    accounting = natural.trace.journal_accounting(natural.trace.read_events(args.output / "calls.jsonl"))
    result["accounting"] = accounting
    if code or reason or pending or active or accounting["call_errors"] or accounting["unknown_calls"] or not accounting["journal_consistent"]:
        result.update(status="technical_failure", heldout_primary_gate_passed=False, heldout_robustness_gate_passed=False)
        result["first_failure"] = result.get("first_failure") or result["execution"]
    e.write_json(args.output / "result.json", result)
    print(json.dumps({k: result.get(k) for k in ("status", "samples", "heldout_primary_gate_passed", "heldout_robustness_gate_passed", "first_failure")}), flush=True)
    return 0 if result["status"] == "completed" else 2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execution-head", required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    signal.signal(signal.SIGTERM, e.request_shutdown)
    if args.prepare:
        if args.worker or args.output is not None:
            parser.error("--prepare cannot be combined with --worker or --output")
        prepare(args.execution_head)
        return 0
    args.output = (args.output or output_path(args.execution_head)).resolve()
    return worker(args) if args.worker else supervise(args)


if __name__ == "__main__":
    raise SystemExit(main())
