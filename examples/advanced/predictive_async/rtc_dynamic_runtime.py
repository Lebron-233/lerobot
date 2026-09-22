"""Dynamic full-RTC inputs and immutable request envelope; original trim is untouched.

The original RTC denoiser/VJP is used verbatim. Only its exact prefix weights
and zero-padded 7D prefix are refreshed in fixed-address device buffers.
"""

import copy
import threading
import time
from dataclasses import asdict, dataclass

import libero_graph_feedback as g
import torch
from profile_libero_cuda_graph_compute import pack
from rtc_execution_queue import RequestStamp
from rtc_graph_runtime import FixedCallGraph, RecordedRTC, RTCGraphRuntime
from smolvla_graph_runtime import signature

from lerobot.configs import RTCAttentionSchedule
from lerobot.policies.rtc.configuration_rtc import RTCConfig
from lerobot.policies.rtc.modeling_rtc import RTCProcessor


def cpu_optional(value):
    return None if value is None else value.detach().cpu().clone()


def config():
    return RTCConfig(
        enabled=True,
        mode="guided",
        execution_horizon=10,
        max_guidance_weight=10.0,
        prefix_attention_schedule=RTCAttentionSchedule.EXP,
    )


class DynamicPrefixRTC(RecordedRTC):
    """The capture uses a canonical key; the values retain the actual (d,L)."""

    def __init__(self, cfg, device):
        super().__init__(cfg)
        if cfg != config():
            raise ValueError("Unsupported RTC configuration")
        self.table_cpu = torch.stack(
            [
                torch.stack([RTCProcessor.get_prefix_weights(self, d, end, 50) for end in range(1, 11)])
                for d in range(9)
            ]
        )
        self.table = self.table_cpu.to(device)
        self.weights = self.table[3, 9].clone()
        self.active = (3, 10)

    def select(self, delay, length):
        if type(delay) is not int or not 0 <= delay <= 8 or type(length) is not int or not 1 <= length <= 50:
            raise ValueError("Unsupported dynamic delay/prefix length")
        self.active = (delay, min(length, 10))
        self.weights.copy_(self.table[delay, self.active[1] - 1])

    def get_prefix_weights(self, start, end, total):
        if (start, end, total) != (3, 10, 50):
            raise ValueError("Only canonical captured RTC key is allowed")
        return self.weights


class DynamicRTCGraphRuntime(RTCGraphRuntime):
    """Two graphs per owner; no-prefix and full-guidance, independent of d/L.

    Real eager receives the real prefix and delay. Captured guidance receives a
    50x7 padded prefix and exact effective weights W(d,min(L,10),50). The parent
    RTCProcessor still applies its original 7-to-32 coordinate mask and VJP.
    """

    def __init__(self, model):
        super().__init__(model)
        self.cached = DynamicPrefixRTC(model.config.rtc_config, model.state_proj.weight.device)
        self.actual_delay = 0
        self.latest_weights = self.latest_padded_prefix = None

    def invoke(self, values, guided):
        return self.original(
            None,
            None,
            values[4],
            values[5],
            values[6],
            noise=values[7],
            future_image_tokens=tuple(values[:2]),
            future_image_token_masks=tuple(values[2:4]),
            inference_delay=self.actual_delay if self.model.rtc_processor is self.native else 3,
            execution_horizon=10,
            prev_chunk_left_over=values[8] if guided else None,
        )

    def __call__(
        self,
        images,
        masks,
        tokens,
        token_masks,
        state,
        noise=None,
        *,
        inference_delay=None,
        prev_chunk_left_over=None,
        execution_horizon=None,
        **kwargs,
    ):
        self.check_owner()
        if (
            self.mode not in ("native_eager", "graph")
            or noise is None
            or type(inference_delay) is not int
            or not 0 <= inference_delay <= 8
            or execution_horizon != 10
            or any(value is not None for value in kwargs.values())
        ):
            raise ValueError("Unsupported dynamic RTC overrides")
        prefix = prev_chunk_left_over
        if prefix is not None and (
            prefix.ndim != 2
            or prefix.shape[1] != 7
            or not 0 <= len(prefix) <= 50
            or prefix.dtype != torch.float32
            or prefix.device != state.device
        ):
            raise ValueError("Expected normalized float32 Lx7 prefix on model device")
        length = 0 if prefix is None else len(prefix)
        if length == 0 and inference_delay != 0:
            raise ValueError("Delay requires a prefix")
        if length and length < inference_delay:
            raise ValueError("Insufficient prefix for expected delay")
        guided = length > 0
        self.actual_delay = inference_delay
        image_tokens, image_masks = self.model.encode_image_tokens(images, masks)
        self.rgb_encodings += 1
        values = pack(image_tokens, image_masks, tokens, token_masks, state, noise)
        sig = signature(values)
        if self.input_signature is None:
            self.input_signature = sig
        elif self.input_signature != sig:
            raise ValueError("Input signature changed; recapture is forbidden")
        self.latest_inputs = values + ((prefix,) if guided else ())
        self.latest_weights = self.latest_padded_prefix = None
        if guided:
            self.cached.select(inference_delay, length)
            self.latest_weights = self.cached.weights.detach().clone()
            padded = state.new_zeros((50, 7), dtype=torch.float32)
            padded[:length].copy_(prefix)
            self.latest_padded_prefix = padded
        processor = self.native if self.mode == "native_eager" else self.cached
        self.model.rtc_processor = processor
        processor.scope = "public"
        created = False
        with torch.no_grad():
            if self.mode == "graph":
                if not self.graphs:
                    if guided:
                        raise ValueError("Bootstrap without prefix is required before guided replay")
                    self.cached.select(3, 30)
                    for name, yes in (("no_prefix", False), ("guided", True)):
                        template = values + ((state.new_zeros((50, 7), dtype=torch.float32),) if yes else ())
                        self.graphs[name] = FixedCallGraph(
                            lambda v, flag=yes: self.invoke(v, flag),
                            template,
                            processor,
                            self.model.action_out_proj,
                            yes,
                        )
                    created = True
                actual = values + ((padded,) if guided else ())
                self.latest = self.graphs["guided" if guided else "no_prefix"](actual)
            else:
                self.latest = self.invoke(self.latest_inputs, guided)
        if self.latest.shape != (1, 50, 32) or not torch.isfinite(self.latest).all():
            raise ValueError("Nonfinite or malformed full output")
        self.requests += 1
        self.metadata = {
            "mode": self.mode,
            "guided": guided,
            "capture_created": created,
            "replays": int(self.mode == "graph"),
            "captured_vjps_per_replay": 10 if guided and self.mode == "graph" else 0,
            "expected_delay": inference_delay,
            "prefix_length": length,
            "effective_horizon": min(length, 10),
            "canonical_capture_key": [3, 10, 50],
        }
        return self.latest


@dataclass(frozen=True)
class RTCRequest:
    """Owned snapshots: no shared mutable queue/observation object crosses threads."""

    stamp: RequestStamp
    observation: dict
    prefix: torch.Tensor | None
    prefix_source: int | None

    @classmethod
    def snapshot(cls, stamp, observation, prefix, source):
        if stamp.observation_index != observation["index"]:
            raise ValueError("Request/observation identity mismatch")
        if prefix is not None:
            if (
                prefix.ndim != 2
                or prefix.shape[1] != 7
                or prefix.device.type != "cpu"
                or prefix.dtype != torch.float32
                or not torch.isfinite(prefix).all()
            ):
                raise ValueError("Request prefix must be normalized CPU float32 Lx7")
            if len(prefix) < stamp.expected_delay:
                raise ValueError("Insufficient owned prefix")
        elif stamp.expected_delay:
            raise ValueError("Missing prefix with nonzero delay")
        return cls(
            stamp, copy.deepcopy(observation), None if prefix is None else prefix.detach().clone(), source
        )


class RTCPrefixOwner(g.old.InferenceOwner):
    """Only submit payload changes. Original serve, finish/trim and close are inherited."""

    def submit(self, observation, delay):
        self.budget.take("model")
        stamp, prefix = self.queue.begin(observation["index"], delay)
        row = {
            "stamp": asdict(stamp),
            "requested_at": time.perf_counter(),
            "observation_returned_at": observation["returned_at"],
            "prefix_source": self.incumbent,
            "prefix_rows": 0 if prefix is None else len(prefix),
        }
        context = RTCRequest.snapshot(stamp, observation, prefix, self.incumbent)
        self.rows.append(row)
        self.pending = row
        self.outputs[stamp.request_id] = {"prefix": prefix}
        self.calls.emit("request_submitted", ordinal=self.spec["ordinal"], **row)
        self.tasks.put_nowait((stamp, context, row))
        return stamp


class DynamicRTCPredictor:
    def __init__(self, policy, pre, post, spec):
        self.policy, self.pre, self.post, self.spec = policy, pre, post, spec
        self.runtime = self.generator = None
        self.requests, self.receipt = 0, None

    def activate(self):
        if self.runtime is None:
            self.policy.config.rtc_config = config()
            self.policy.init_rtc_processor()
            self.runtime = DynamicRTCGraphRuntime(self.policy.model)
            self.runtime.__enter__()
            self.runtime.mode = "graph"
            self.generator = torch.Generator(device="cuda").manual_seed(self.spec["policy_seed"])
        self.runtime.check_owner()

    def __call__(self, context):
        if not isinstance(context, RTCRequest):
            raise TypeError("Explicit RTCRequest required, not a bare observation")
        self.activate()
        rt, observation = self.runtime, context.observation
        if self.requests and len(rt.graphs) != 2:
            raise ValueError("Control-time graph preparation is forbidden")
        guided_arm = self.spec["arm"] == "rtc_async"
        supplied = context.prefix
        prefix = supplied.cuda().clone() if guided_arm and supplied is not None and len(supplied) else None
        delay = context.stamp.expected_delay if prefix is not None else 0
        with torch.no_grad():
            self.policy.reset()
            self.pre.reset()
            self.post.reset()
            batch = self.pre(
                g.pilot.worker_batch(observation, self.spec["task_name"].replace("_", " "), "cuda")
            )
            noise = torch.randn((1, 50, 32), generator=self.generator, device="cuda")
            fingerprint = g.observation_fingerprint(observation)
            torch.cuda.synchronize()
            began = time.perf_counter()
            normalized = self.policy.predict_action_chunk(
                batch, noise=noise, prev_chunk_left_over=prefix, inference_delay=delay, execution_horizon=10
            )
            torch.cuda.synchronize()
            ended = time.perf_counter()
            original = normalized.detach().cpu().clone()[0]
            processed = self.post(normalized).detach().cpu().clone()[0]
            out = {
                "original": original,
                "processed": processed,
                "full": rt.latest.detach().cpu().clone(),
                "inputs": g.q.cpu(rt.latest_inputs),
                "noise": noise.detach().cpu().clone(),
                "rtc_metadata": dict(rt.metadata),
                "rtc_weights": cpu_optional(rt.latest_weights),
                "rtc_padded_prefix": cpu_optional(rt.latest_padded_prefix),
                "context_stamp": asdict(context.stamp),
                "context_prefix_source": context.prefix_source,
                "context_prefix": None if supplied is None else supplied.clone(),
                "owner_thread": threading.get_ident(),
                "model_started_at": began,
                "model_returned_at": ended,
                "input_observation_index": observation["index"],
                "input_fingerprint": fingerprint,
                "vision_encodes": 1,
                "capture_created": rt.metadata["capture_created"],
            }
            torch.cuda.synchronize()
        self.requests += 1
        return out

    def close(self):
        if self.runtime is not None:
            self.runtime.__exit__(None, None, None)
            self.receipt = self.runtime.receipt
            self.receipt.update(
                live_requests=self.requests, explicit_noise_draws=self.requests, closed_at=time.perf_counter()
            )
            self.policy.config.rtc_config = None
            self.policy.init_rtc_processor()
