"""Opt-in prefix-only RTC, not an equivalent implementation of EXP guidance.

Only explicit old-action target weights change. The complete RTC VJP, original
Graph capture, feedback loop and actual-consumption trim are reused unchanged.
"""

import threading

import torch
from rtc_commitment_owner import CommittedPrefixOwner
from rtc_dynamic_runtime import DynamicRTCGraphRuntime, DynamicRTCPredictor, RTCPrefixOwner, config
from rtc_graph_runtime import RecordedRTC

from lerobot.configs import RTCAttentionSchedule
from lerobot.policies.rtc.configuration_rtc import RTCConfig
from lerobot.policies.rtc.modeling_rtc import RTCProcessor

COMMITTED = frozenset(("committed_base", "committed_exp", "committed_prefix"))
VARIANTS = ("serialized", "aligned_async", "committed_base", "committed_exp", "committed_prefix")


def prefix_config():
    return RTCConfig(enabled=True, mode="guided", execution_horizon=10,
                     max_guidance_weight=10.0, prefix_attention_schedule=RTCAttentionSchedule.ZEROS)


def schedule_for(variant):
    if variant not in VARIANTS:
        raise ValueError("Unknown frozen prefix-scope variant")
    return "ZEROS" if variant == "committed_prefix" else "EXP"


class PrefixOnlyWeights(RecordedRTC):
    """Cache exact native ZEROS values; canonical capture key is not the real C."""

    def __init__(self, cfg, device):
        super().__init__(cfg)
        if cfg != prefix_config():
            raise ValueError("Only the explicit prefix-only configuration is supported")
        self.table_cpu = torch.stack([
            torch.stack([RTCProcessor.get_prefix_weights(self, d, end, 50) for end in range(1, 11)])
            for d in range(9)])
        self.table = self.table_cpu.to(device)
        self.weights = self.table[3, 9].clone()
        self.active = (3, 10)

    def select(self, delay, length):
        if type(delay) is not int or not 0 <= delay <= 8 or type(length) is not int or not 1 <= length <= 50:
            raise ValueError("Unsupported prefix-only delay/length")
        self.active = (delay, min(length, 10))
        self.weights.copy_(self.table[delay, self.active[1]-1])

    def get_prefix_weights(self, start, end, total):
        if (start, end, total) != (3, 10, 50):
            raise ValueError("Only canonical captured RTC key is allowed")
        return self.weights


class PrefixOnlyGraphRuntime(DynamicRTCGraphRuntime):
    """Independent explicit ZEROS constructor, reusing unchanged dynamic execution.

    The EXP constructor is not weakened, bypassed by a fake config, or patched.
    The original RTCProcessor.denoise_step remains the native and captured formula.
    """

    def __init__(self, model):
        if model.config.num_steps != 10 or model.config.chunk_size != 50 or model.config.rtc_config != prefix_config():
            raise ValueError("Prefix-only runtime requires the frozen ZEROS configuration")
        self.model, self.original, self.original_processor = model, model.sample_actions, model.rtc_processor
        self.owner, self.closed = threading.get_ident(), False
        self.native = RecordedRTC(model.config.rtc_config)
        self.cached = PrefixOnlyWeights(model.config.rtc_config, model.state_proj.weight.device)
        self.graphs, self.captures = {}, []
        self.mode = "native_eager"
        self.latest = self.latest_inputs = self.metadata = None
        self.requests = self.rgb_encodings = 0
        self.input_signature = self.receipt = None
        self.actual_delay = 0
        self.latest_weights = self.latest_padded_prefix = None


class PrefixScopeOwner(CommittedPrefixOwner):
    """Same declared C and receive schedule for all three commitment controls."""

    def submit(self, observation, delay):
        schedule_for(self.spec["variant"])
        stamp = RTCPrefixOwner.submit(self, observation, delay)
        enabled = self.spec["variant"] in COMMITTED and stamp.request_id > 1
        self.pending["commitment"] = {
            "enabled": enabled,
            "compute_cover_estimate": delay,
            "declared_steps": delay if enabled else None,
            "conditioned_steps": delay if self.spec["arm"] == "rtc_async" and self.pending["prefix_rows"] else 0,
            "constraint_schedule": schedule_for(self.spec["variant"]),
            "deferred_polls": 0,
            "boundary_at": None,
        }
        return stamp


class PrefixScopePredictor(DynamicRTCPredictor):
    """Same dynamic payload/preprocess/noise/output path; only formula selection."""

    def activate(self):
        schedule = schedule_for(self.spec["variant"])
        if self.runtime is None:
            self.policy.config.rtc_config = prefix_config() if schedule == "ZEROS" else config()
            self.policy.init_rtc_processor()
            runtime = PrefixOnlyGraphRuntime if schedule == "ZEROS" else DynamicRTCGraphRuntime
            self.runtime = runtime(self.policy.model)
            self.runtime.__enter__()
            self.runtime.mode = "graph"
            self.generator = torch.Generator(device="cuda").manual_seed(self.spec["policy_seed"])
        self.runtime.check_owner()

    def __call__(self, context):
        out = super().__call__(context)
        out["constraint_schedule"] = schedule_for(self.spec["variant"])
        return out
