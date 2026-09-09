"""D3: one bounded token/real-worker contract, using only ten recorded initial frames."""

import argparse
import json
import signal
import subprocess
import sys
import threading
import time
import traceback
from collections import Counter
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from libero_graph_native_equivalence import FLAGS, POLICY, REPO, SOURCE, VLM
from libero_reference_qualification import TASK_NAMES
from libero_reference_smoke import CAMERA_KEYS, load_runtime, write_json
from libero_single_step_matched_control import initial_observation
from profile_libero_cuda_graph_compute import invoke_graph
from profile_libero_graph_recorded import recorded_batch
from smolvla_graph_identity import SmolVLAGraphIdentityEngine
from smolvla_graph_runtime import SmolVLAGraphRuntime

from lerobot.policies.utils import prepare_observation_for_inference
from lerobot.utils.constants import OBS_STATE
from lerobot.utils.feature_utils import build_dataset_frame

# Every request is explicit; no run-until-success loop or replacement samples.
WORKER_EVENTS = (
    ("cold", 0, "bootstrap", "cold_temporary", None),
    ("probe", 0, "startup_probe", "probe", None),
    ("fresh", 0, "bootstrap", "fresh_warmed", None),
    ("planned", 0, "planned", None, None),
    ("reset_inflight", 0, "planned", None, "reset"),
    ("new_epoch", 0, "bootstrap", None, None),
    ("new_epoch_planned", 0, "planned", None, None),
    ("a_to_b", 0, "planned", None, "task3"),
    ("b_bootstrap", 3, "bootstrap", None, None),
    ("b_to_a", 3, "planned", None, "task0"),
    ("a_return", 0, "bootstrap", None, None),
    ("stop_inflight", 0, "planned", None, "stop"),
)
UNTRACKED_DOCS = {
    "docs/experiments/CODEX_D_GRAPH_TOKEN_WORKER_CONTRACT_20260909.md",
    "docs/experiments/CODEX_NEXT_TASK_GRAPH_NATIVE_ASYNC_20260909.md",
    "docs/experiments/SMOLVLA_GRAPH_AB_REVIEW_C_CONTRACT_CORRECTION_20260909.md",
}


def cpu(value):
    if isinstance(value, np.ndarray):
        return torch.from_numpy(value.copy())
    if isinstance(value, torch.Tensor):
        return value.detach().to("cpu", copy=True)
    if isinstance(value, (tuple, list)):
        return tuple(cpu(v) for v in value)
    if isinstance(value, dict):
        return {k: cpu(v) for k, v in value.items()}
    return value


def clone_batch(batch):
    return {k: v.clone() if isinstance(v, torch.Tensor) else list(v) for k, v in batch.items()}


def equal(left, right):
    return (
        left.shape == right.shape
        and left.dtype == right.dtype
        and bool(torch.isfinite(left).all())
        and bool(torch.isfinite(right).all())
        and torch.equal(left, right)
    )


def comparisons(a, b):
    return {k: equal(a[k], b[k]) for k in ("noise", "full_chunk", "normalized_chunk", "post_chunk")}


class Budget:
    def __init__(self):
        self.main_calls = 0
        self.reference_calls = 0
        self.capture_attempts = 0

    def call(self, reference=False):
        if self.main_calls + self.reference_calls >= 120:
            raise RuntimeError("The fixed 120-call model budget is exhausted")
        if reference:
            self.reference_calls += 1
        else:
            self.main_calls += 1

    def capture(self):
        if self.capture_attempts >= 30:
            raise RuntimeError("The fixed 30-capture budget is exhausted")
        self.capture_attempts += 1


class AuditedRuntime(SmolVLAGraphRuntime):
    def __init__(self, model, budget):
        super().__init__(model)
        self.budget = budget

    def _capture(self, inputs):
        self.budget.capture()
        super()._capture(inputs)


class CallAudit:
    """Count the real named APIs and reject cross-thread calls in this test process."""

    def __init__(self, policy, budget):
        self.policy, self.budget = policy, budget
        self.owner = None
        self.counts = Counter()
        self.events = []
        self.originals = []

    def record(self, name):
        thread = threading.get_ident()
        if thread != self.owner:
            raise RuntimeError(f"{name} called outside the model owner")
        self.counts[name] += 1
        self.events.append({"event": name, "thread": thread})

    def attach(self):
        self.owner = threading.get_ident()
        methods = (
            (self.policy.model, "encode_image_tokens", "vision"),
            (self.policy.model, "sample_noise", "noise"),
            (self.policy, "prepare_images", "prepare_images"),
            (self.policy, "predict_action_chunk", "policy"),
            (self.policy, "reset", "policy_reset"),
        )
        for obj, name, label in methods:
            original = getattr(obj, name)
            self.originals.append((obj, name, original))

            def wrapped(*args, _original=original, _label=label, **kwargs):
                self.record(_label)
                if _label == "policy":
                    self.budget.call()
                return _original(*args, **kwargs)

            setattr(obj, name, wrapped)

    def detach(self):
        for obj, name, original in reversed(self.originals):
            setattr(obj, name, original)
        self.originals.clear()


class AuditedProcessor:
    def __init__(self, processor, audit, name):
        self.processor, self.audit, self.name = processor, audit, name
        self.steps = processor.steps

    def __call__(self, value):
        self.audit.record(self.name)
        return self.processor(value)

    def reset(self):
        self.audit.record(self.name + "_reset")
        self.processor.reset()


def fixed_inputs():
    registration = json.loads((SOURCE / "registration.json").read_text())
    batches, observations, specs = [], [], []
    for task, name in enumerate(TASK_NAMES):
        spec = registration["tuples"][task * 9]
        if spec["task_id"] != task or spec["initial_state_id"] != 41:
            raise ValueError("Only task0-9/state41/observation0 is registered")
        directory = SOURCE / f"tuple_{task * 9:03d}"
        initial = initial_observation(directory)  # Stop reading at observation0.
        batch = recorded_batch(directory, initial, name)
        obs = {f"state_{i}": float(batch[OBS_STATE][0, i]) for i in range(8)}
        for key in CAMERA_KEYS:
            obs[key.rsplit(".", 1)[1]] = (
                (batch[key][0].permute(1, 2, 0) * 255).round().to(torch.uint8).numpy().copy()
            )
        features = {
            OBS_STATE: {"dtype": "float32", "shape": (8,), "names": [f"state_{i}" for i in range(8)]},
            **{k: {"dtype": "video", "shape": obs[k.rsplit(".", 1)[1]].shape} for k in CAMERA_KEYS},
        }
        rebuilt = prepare_observation_for_inference(
            build_dataset_frame(features, obs, "observation"), torch.device("cpu")
        )
        if not all(equal(rebuilt[k], batch[k]) for k in (OBS_STATE, *CAMERA_KEYS)):
            raise ValueError("Worker observation reconstruction changed the fixed input")
        batches.append(batch)
        observations.append(obs)
        specs.append(
            {
                "task_id": task,
                "initial_state_id": 41,
                "observation_index": 0,
                "directory": str(directory),
                "source_tuple": spec,
                "state_and_both_cameras_reconstructed_exact": True,
            }
        )
    return batches, observations, features, specs


def sampler_arrays(runtime, normalized, post):
    return {
        "inputs": cpu(runtime.latest_inputs),
        "noise": cpu(runtime.latest_noise),
        "full_chunk": cpu(runtime.latest),
        "normalized_chunk": cpu(normalized),
        "post_chunk": cpu(post),
    }


def token_sequence(policy, pre, post, batches, budget, result, arrays):
    audit = CallAudit(policy, budget)
    audit.attach()
    pre, post = AuditedProcessor(pre, audit, "pre"), AuditedProcessor(post, audit, "post")
    result["token_pairs"] = []
    runtime = None
    try:
        with torch.inference_mode(), AuditedRuntime(policy.model, budget) as runtime:
            for mode in ("eager", "graph"):
                torch.manual_seed(1019001)  # Once per sequence, never once per request/task.
                held = []
                for task in range(10):
                    policy.reset()
                    pre.reset()
                    post.reset()
                    runtime.begin_episode(mode, task)
                    for request in range(2):
                        before = audit.counts.copy()
                        processed = pre(clone_batch(batches[task]))
                        images, masks = policy.prepare_images(processed)
                        image_tokens, image_masks = policy.model.encode_image_tokens(images, masks)
                        token_only = {k: v for k, v in processed.items() if k not in CAMERA_KEYS}
                        normalized = policy.predict_action_chunk(
                            token_only, future_image_tokens=image_tokens, future_image_token_masks=image_masks
                        )
                        post_chunk = post(normalized)
                        key = f"token_{mode}_{task:02d}_{request}"
                        arrays[key] = sampler_arrays(runtime, normalized, post_chunk)
                        if (
                            audit.counts["vision"] - before["vision"] != 1
                            or audit.counts["prepare_images"] - before["prepare_images"] != 1
                            or runtime.rgb_encodings != 0
                        ):
                            raise ValueError("Token main request did not encode exactly once externally")
                        if audit.counts["noise"] - before["noise"] != 1:
                            raise ValueError("Control request did not sample fresh noise exactly once")
                        if normalized.shape != (1, 50, 7) or post_chunk.shape != (1, 50, 7):
                            raise ValueError("Unexpected complete policy/post chunk shape")
                        for original, saved in held:
                            if not torch.equal(original, saved):
                                raise ValueError("A held output changed on a later request/task")
                        if request == 0:
                            held.append((runtime.latest, runtime.latest.clone()))
                        if mode == "graph":
                            checks = comparisons(arrays[f"token_eager_{task:02d}_{request}"], arrays[key])
                            row = {
                                "task_id": task,
                                "request": request,
                                "exact_finite": checks,
                                "external_visual_encodings": 1,
                                "graph_internal_visual_encodings": 0,
                                "owner_thread": audit.owner,
                                **runtime.metadata,
                            }
                            result["token_pairs"].append(row)
                            if not all(checks.values()):
                                result["first_failure"] = row
                                raise ValueError("First token Graph numerical mismatch")
                held.clear()
            result["token_captures"] = list(runtime.captures)
        result["token_graph_equivalence_passed"] = (
            len(result["token_pairs"]) == 20
            and runtime.graph is None
            and policy.model.sample_actions == runtime.original
        )
    finally:
        result["token_captures"] = [] if runtime is None else list(runtime.captures)
        result["token_audit"] = {
            "owner_thread": audit.owner,
            "counts": dict(audit.counts),
            "events": audit.events,
            "sampler_restored": runtime is not None and policy.model.sample_actions == runtime.original,
            "graph_released": runtime is not None and runtime.graph is None,
        }
        audit.detach()


class MemoryMetrics:
    def __init__(self):
        self.events = []
        self.closed = False

    def emit(self, event):
        self.events.append(dict(event))

    def close(self):
        self.closed = True


class ContractEngine(SmolVLAGraphIdentityEngine):
    def __init__(self, policy, pre, post, features, budget, arrays, device="cuda"):
        self.audit = CallAudit(policy, budget)
        self.budget, self.arrays = budget, arrays
        self.metrics = MemoryMetrics()
        self.done = {label: threading.Event() for label, *_ in WORKER_EVENTS}
        self.produced = {label: threading.Event() for label, *_ in WORKER_EVENTS}
        self.release = {label: threading.Event() for label, *_ in WORKER_EVENTS}
        self.rows = []
        self.current = None
        self.held = None
        self.held_copy = None
        super().__init__(
            policy=policy,
            preprocessor=AuditedProcessor(pre, self.audit, "pre"),
            postprocessor=AuditedProcessor(post, self.audit, "post"),
            hw_features=features,
            robot_wrapper=SimpleNamespace(robot_type="libero"),
            task=TASK_NAMES[0].replace("_", " "),
            fps=20,
            device=device,
            metrics_sink=self.metrics,
        )

    def _make_graph_runtime(self):
        self.audit.attach()
        torch.manual_seed(1019002)  # Initial worker seed only; reset never reseeds.
        return AuditedRuntime(self._policy.model, self.budget)

    @contextmanager
    def _worker_resources(self):
        try:
            with super()._worker_resources():
                try:
                    yield
                finally:
                    self.held = self.held_copy = None  # Release GPU references on their owner.
        finally:
            self.audit.detach()

    def _run_request(self, request):
        label = request.observation["d_event"]
        self.current = {
            "label": label,
            "request_id": request.request_id,
            "kind": request.kind,
            "startup_phase": request.startup_phase,
            "reset_epoch": request.reset_epoch,
            "task_epoch": request.task_epoch,
            "task": request.task,
            "owner_thread": threading.get_ident(),
            "queue_prefix_cpu": request.plan is None
            or request.plan.committed_policy_actions.device.type == "cpu",
        }
        self.rows.append(self.current)
        self.arrays[f"observation_{label}"] = cpu(request.observation)
        if request.plan is not None:
            self.current["plan"] = {
                "next_action_index": request.plan.next_action_index,
                "takeover_index": request.plan.takeover_index,
                "planned_delay_steps": request.plan.planned_delay_steps,
            }
            self.arrays[f"prefix_{label}"] = cpu(
                {
                    "policy_actions": request.plan.committed_policy_actions,
                    "post_policy_actions": request.plan.committed_post_policy_actions,
                    "mask": request.plan.committed_mask,
                }
            )
        before = self.audit.counts.copy()
        try:
            super()._run_request(request)
            self.current["main_completed"] = True
        except Exception:
            self.current["exception"] = traceback.format_exc()
            raise
        finally:
            self.current["vision_calls"] = self.audit.counts["vision"] - before["vision"]
            self.current["prepare_images_calls"] = (
                self.audit.counts["prepare_images"] - before["prepare_images"]
            )
            self.current["noise_draws"] = self.audit.counts["noise"] - before["noise"]

    def _prepare_queue_actions(self, actions, metrics):
        policy_cpu, post_cpu = super()._prepare_queue_actions(actions, metrics)
        label = self.current["label"]
        if (
            policy_cpu.device.type != "cpu"
            or post_cpu.device.type != "cpu"
            or policy_cpu.data_ptr() == post_cpu.data_ptr()
        ):
            raise ValueError("Queue boundary must receive independent CPU chunks")
        self.arrays[f"worker_{label}"] = sampler_arrays(
            self.runtime, policy_cpu.unsqueeze(0), post_cpu.unsqueeze(0)
        )
        self.current.update(self.runtime.metadata)
        self.current.update(
            cpu_chunks=True,
            device_completion_barrier=True,
            rgb_encodings_total=self.runtime.rgb_encodings,
            model_inference_mode=torch.is_inference_mode_enabled(),
            language_length=self.runtime.latest_inputs[4].shape[1],
        )
        if self.held is not None and not torch.equal(self.held, self.held_copy):
            raise ValueError("A previous worker output was overwritten")
        if self.held is None:
            self.held, self.held_copy = self.runtime.latest, self.runtime.latest.clone()
        if label in ("reset_inflight", "a_to_b", "b_to_a", "stop_inflight"):
            self.current["injection_point"] = "cpu_chunks_complete_before_queue_publication"
            self.produced[label].set()
            if not self.release[label].wait(10):
                raise TimeoutError(f"Controller did not release {label}")
        return policy_cpu, post_cpu

    def _request_finished(self, request):
        label = request.observation["d_event"]
        try:
            if self.current.get("exception"):
                return
            if (
                self.current["vision_calls"] != 1
                or self.current["prepare_images_calls"] != 1
                or self.current["noise_draws"] != 1
            ):
                raise ValueError("Worker main request encoding/noise count differs")
            # A separate explicit-noise reference after the original publication
            # boundary, while this worker still owns the model. Not tracker input.
            self.budget.call(reference=True)
            self.audit.record("reference")
            started = time.perf_counter()
            full = invoke_graph(self.runtime.original, self.runtime.latest_inputs)
            norm = full[:, :, :7]
            post = self._postprocessor(norm)
            reference = {
                "noise": cpu(self.runtime.latest_noise),
                "full_chunk": cpu(full),
                "normalized_chunk": cpu(norm),
                "post_chunk": cpu(post),
            }
            self.arrays[f"reference_{label}"] = reference
            self.current["reference_after_publication_seconds"] = time.perf_counter() - started
            self.current["reference_exact_finite"] = comparisons(self.arrays[f"worker_{label}"], reference)
            if not all(self.current["reference_exact_finite"].values()):
                raise ValueError("Worker output differs from the explicit-noise eager reference")
        except Exception:
            self.current["exception"] = traceback.format_exc()
            raise
        finally:
            self.done[label].set()


def worker_contract(policy, pre, post, observations, features, budget, result, arrays):
    engine = ContractEngine(policy, pre, post, features, budget, arrays)
    result["worker_requests"] = engine.rows
    consumed = []
    controls = []
    stopper = None
    completed = False
    try:
        engine.start()
        engine.resume()
        for label, task, kind, phase, injection in WORKER_EVENTS:
            # Consume real queued CPU actions only. Emptying before a new-task
            # bootstrap keeps cold capture out of the pre-existing planned cohort.
            target = 0 if label in ("b_bootstrap", "a_return") else 30
            if (engine.ready and kind == "planned") or label in ("b_bootstrap", "a_return"):
                maximum = engine.queue.available_steps()
                for _ in range(maximum):
                    if engine.queue.available_steps() <= target:
                        break
                    index = engine.queue.next_action_index
                    action = engine.get_action(None)
                    if action is None or action.device.type != "cpu":
                        raise ValueError("Control thread received no CPU action while draining")
                    consumed.append(
                        {"index": index, "task": engine.dispatched_task, "action": action.clone()}
                    )
            if engine.task != TASK_NAMES[task].replace("_", " "):
                raise ValueError("Unexpected task at the registered request boundary")
            engine.notify_observation({**observations[task], "d_event": label})
            if injection:
                if not engine.produced[label].wait(10):
                    raise TimeoutError(
                        f"Missing completed-model barrier for {label}: {engine.failure_traceback}"
                    )
                before = {
                    "reset_epoch": engine.queue.reset_epoch,
                    "task_epoch": engine.queue.task_epoch,
                    "index": engine.queue.next_action_index,
                    "owner_reset_calls": engine.audit.counts["policy_reset"],
                }
                if injection == "reset":
                    engine.reset()
                    if (
                        engine.queue.available_steps() != 0
                        or engine.queue.reset_epoch != before["reset_epoch"] + 1
                    ):
                        raise ValueError("Reset did not immediately invalidate CPU queue state")
                    if engine.audit.counts["policy_reset"] != before["owner_reset_calls"]:
                        raise ValueError("Controller reset touched the in-flight model")
                elif injection.startswith("task"):
                    new_task = int(injection[4:])
                    engine.set_task(TASK_NAMES[new_task].replace("_", " "))
                    action = engine.get_action(None)
                    if action is None or engine.dispatched_task != TASK_NAMES[task].replace("_", " "):
                        raise ValueError("Task change lost the old active provenance")
                    consumed.append(
                        {"index": before["index"], "task": engine.dispatched_task, "action": action.clone()}
                    )
                else:
                    stopper = threading.Thread(target=engine.stop, name="DStopController")
                    stopper.start()
                    if not engine._shutdown_event.wait(2):
                        raise TimeoutError("Stop was not acknowledged")
                    # stop has set shutdown; acquire its CPU lock after invalidation.
                    with engine._request_lock:
                        if engine.queue.reset_epoch == before["reset_epoch"]:
                            raise ValueError("Stop did not invalidate the queue before publication")
                controls.append(
                    {
                        "label": label,
                        "injection": injection,
                        "before": before,
                        "reset_epoch_after": engine.queue.reset_epoch,
                        "task_epoch_after": engine.queue.task_epoch,
                    }
                )
                engine.release[label].set()
            if not engine.done[label].wait(15):
                raise TimeoutError(f"Worker did not finish {label}: {engine.failure_traceback}")
            row = engine.rows[-1]
            if row.get("exception"):
                raise RuntimeError(row["exception"])
            if row["kind"] != kind or row["startup_phase"] != phase:
                raise ValueError(f"Registered request kind/phase differs for {label}")
            terminal = [
                e
                for e in engine.metrics.events
                if e.get("request_id") == row["request_id"]
                and e.get("event") in ("chunk_request", "request_error")
            ]
            if len(terminal) != 1:
                raise ValueError(f"Request {label} has no unique terminal accounting")
            row["terminal"] = terminal[0]
            if injection and terminal[0]["outcome"] != "stale":
                raise ValueError("An invalidated request was accepted into the queue")
            if kind == "planned" and not injection:
                plan = engine.queue.plan_snapshot()
                if terminal[0]["outcome"] not in ("staged_early", "staged_on_time", "deadline_miss"):
                    raise ValueError("Unexpected planned publication outcome")
                if plan is not None:
                    for _ in range(plan.takeover_index - engine.queue.next_action_index + 1):
                        index = engine.queue.next_action_index
                        action = engine.get_action(None)
                        consumed.append(
                            {"index": index, "task": engine.dispatched_task, "action": action.clone()}
                        )
                    if not equal(action, arrays[f"worker_{label}"]["post_chunk"][0, 0]):
                        raise ValueError("Takeover did not dispatch the new row zero")
            print(f"WORKER {label} kind={kind} outcome={terminal[0]['outcome']}", flush=True)
        completed = True
    finally:
        for event in engine.release.values():
            event.set()
        if stopper is not None:
            stopper.join(5)
        engine.stop()
        arrays["worker_consumed"] = consumed
        result["worker_controls"] = controls
        result["worker_metrics"] = engine.metrics.events
        result["worker_stats"] = asdict(engine.stats)
        result["worker_lifecycle"] = engine.lifecycle
        result["worker_audit"] = {
            "owner_thread": engine.owner_thread,
            "controller_thread": threading.get_ident(),
            "counts": dict(engine.audit.counts),
            "events": engine.audit.events,
            "worker_joined": engine.worker_joined,
            "graph_released": engine.graph_released,
            "original_sampler_restored": engine.original_sampler_restored,
            "failed": engine.failed,
            "failure_traceback": engine.failure_traceback,
            "metrics_closed": engine.metrics.closed,
        }
        result["worker_captures"] = [] if engine.runtime is None else list(engine.runtime.captures)
        result["graph_worker_lifecycle_passed"] = (
            completed
            and engine.worker_joined
            and engine.graph_released
            and engine.original_sampler_restored
            and not engine.failed
        )
        result["graph_identity_engine_integration_passed"] = result["graph_worker_lifecycle_passed"] and len(
            engine.rows
        ) == len(WORKER_EVENTS)
        if not engine.worker_joined:
            raise RuntimeError("Worker exit could not be confirmed")


def termination_requested(_signal, _frame):
    raise TimeoutError("D supervisor requested shutdown before the 300-second hard limit")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execution-head", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
    status = subprocess.check_output(["git", "status", "--porcelain"], cwd=REPO, text=True).splitlines()
    if head != args.execution_head or any(
        line[:3] != "?? " or line[3:] not in UNTRACKED_DOCS for line in status
    ):
        raise ValueError("Registered execution source/protocol must have no uncommitted changes")
    args.output.mkdir(parents=True, exist_ok=False)
    result = {
        "execution_head": head,
        "status": "running",
        "token_graph_equivalence_passed": False,
        "graph_worker_lifecycle_passed": False,
        "graph_identity_engine_integration_passed": False,
        "new_native_episodes": 0,
        "robot_actions": 0,
        "late_policy": "whole_discard_any_late",
        "source": str(SOURCE),
        "source_observation_indices": [0],
        "maximum_main_calls": 120,
        "maximum_captures": 30,
        "max_wall_seconds": 300,
        "untracked_documents_preserved": sorted(UNTRACKED_DOCS),
        **FLAGS,
    }
    arrays, budget = {}, Budget()
    signal.signal(signal.SIGTERM, termination_requested)
    try:
        torch.set_num_threads(1)
        batches, observations, features, specs = fixed_inputs()
        result["inputs"] = specs
        arrays["source_batches"] = cpu(batches)
        result["environment"] = {"python": sys.executable, "torch": torch.__version__}
        result["device"] = torch.cuda.get_device_name()
        if result["device"] != "NVIDIA GeForce RTX 4070 Ti SUPER":
            raise ValueError("The registered GPU changed")
        start = time.perf_counter()
        policy, pre, post, report = load_runtime(POLICY, VLM)
        result["model_load_seconds"] = time.perf_counter() - start
        result["policy_load"] = report
        token_sequence(policy, pre, post, batches, budget, result, arrays)
        if not result["token_graph_equivalence_passed"]:
            raise ValueError("Token sequence did not close")
        # From start through join, the controller only operates CPU observations,
        # Events and CPU queue data. All remaining model/CUDA work is on the owner.
        worker_contract(policy, pre, post, observations, features, budget, result, arrays)
        result["graph_identity_engine_integration_passed"] = False
        if (budget.main_calls, budget.reference_calls, budget.capture_attempts) != (52, 12, 15):
            raise ValueError("Fixed request/capture accounting differs from the protocol")
        if any(
            row["language_length"] != (13 if row["label"] in ("b_bootstrap", "b_to_a") else 12)
            for row in result["worker_requests"]
        ):
            raise ValueError("The registered A/B/A language lengths were not exercised")
        result["graph_identity_engine_integration_passed"] = result["graph_worker_lifecycle_passed"]
        result["status"] = (
            "passed"
            if all(
                result[k]
                for k in (
                    "token_graph_equivalence_passed",
                    "graph_worker_lifecycle_passed",
                    "graph_identity_engine_integration_passed",
                )
            )
            else "failed"
        )
    except Exception:
        result["status"] = "failed"
        result.setdefault("first_failure", {"exception": traceback.format_exc()})
    finally:
        result["budget"] = vars(budget)
        # CPU-only serialization begins after consumption, join and resource closure.
        if result.get("worker_audit", {}).get("worker_joined", True):
            torch.save(arrays, args.output / "request_arrays.pt")
            result["array_records"] = list(arrays)
        else:
            result["arrays_not_serialized"] = "Worker exit unconfirmed; no concurrent hot-path serialization"
        write_json(args.output / "result.json", result)
        print(
            json.dumps(
                {
                    k: v
                    for k, v in result.items()
                    if k
                    in (
                        "status",
                        "first_failure",
                        "budget",
                        "token_graph_equivalence_passed",
                        "graph_worker_lifecycle_passed",
                        "graph_identity_engine_integration_passed",
                    )
                }
            ),
            flush=True,
        )
    return 0 if result["status"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
