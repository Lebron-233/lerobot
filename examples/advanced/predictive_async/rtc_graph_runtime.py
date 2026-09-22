"""Fixed-contract full-Jacobian RTC capture; no production policy changes.

Only the fixed EXP / horizon10 / delay3 / 30x7-prefix path is supported here.
Prefix weights are computed by the existing CPU implementation, then cached on
its target device before capture. The original RTC denoise_step is unchanged.
"""

import copy
import threading
import time
from collections import Counter, defaultdict

import torch
from profile_libero_cuda_graph_compute import copy_inputs, pack
from smolvla_graph_runtime import signature

from lerobot.configs import RTCAttentionSchedule
from lerobot.policies.rtc.modeling_rtc import RTCProcessor


class RecordedRTC(RTCProcessor):
    def __init__(self, config):
        super().__init__(config)
        self.scope = 'public'
        self.counts = defaultdict(Counter)

    def denoise_step(self, *args, **kwargs):
        value = super().denoise_step(*args, **kwargs)
        self.counts[self.scope]['steps'] += 1
        prefix = kwargs.get('prev_chunk_left_over', args[1] if len(args) > 1 else None)
        if prefix is not None:
            self.counts[self.scope]['guided_vjps_returned'] += 1
        return value


class DevicePrefixRTC(RecordedRTC):
    """Cache exactly the original weights, not a GPU approximation of EXP."""

    def __init__(self, config, device):
        super().__init__(config)
        if config.prefix_attention_schedule != RTCAttentionSchedule.EXP or config.debug:
            raise ValueError('Only fixed EXP with debugging disabled is supported')
        self.fixed_key = (3, 10, 50)
        cpu = super().get_prefix_weights(*self.fixed_key)
        self.weights = cpu.to(device=device).clone()
        self.cpu_weights = cpu.clone()

    def get_prefix_weights(self, start, end, total):
        if (start, end, total) != self.fixed_key:
            raise ValueError('Unsupported RTC prefix signature; no silent recapture')
        return self.weights


class FixedCallGraph:
    """Capture one complete sampler, including all ten guided VJPs if present."""

    def __init__(self, invoke, inputs, processor, projection, guided):
        self.inputs = tuple(v.detach().clone() for v in inputs)
        self.sig = signature(self.inputs)
        self.replays = 0
        self.graph = torch.cuda.CUDAGraph()
        self.stream = torch.cuda.Stream()
        self.record = {'guided': guided, 'owner': threading.get_ident(), 'setup': 0,
                       'warmup': 0, 'capture': 0, 'projection_shapes': [], 'status': 'preparing'}
        began = time.perf_counter()
        old_scope = processor.scope
        try:
            with torch.no_grad():
                processor.scope = 'setup'
                invoke(self.inputs)
                self.record['setup'] += 1
                self.stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(self.stream):
                    processor.scope = 'warmup'
                    for _ in range(3):
                        invoke(self.inputs)
                        self.record['warmup'] += 1
                torch.cuda.current_stream().wait_stream(self.stream)
                torch.cuda.synchronize()
                before = dict(processor.counts['capture'])
                hook = projection.register_forward_hook(
                    lambda _m, _a, out: self.record['projection_shapes'].append(list(out.shape)))
                processor.scope = 'capture'
                try:
                    with torch.cuda.graph(self.graph, stream=self.stream):
                        self.latest = invoke(self.inputs)
                        self.record['capture'] += 1
                finally:
                    hook.remove()
                torch.cuda.synchronize()
                self.record['captured_steps'] = processor.counts['capture']['steps']-before.get('steps', 0)
                self.record['captured_vjps'] = (processor.counts['capture']['guided_vjps_returned']
                                              - before.get('guided_vjps_returned', 0))
                if (self.record['projection_shapes'] != [[1, 50, 32]]*10 or
                    self.record['captured_steps'] != 10 or self.record['captured_vjps'] != (10 if guided else 0)):
                    raise ValueError('Incomplete captured sampler or RTC Jacobian path')
                self.record['status'] = 'captured'
        except BaseException as exc:
            self.record.update(status='failed', error=repr(exc))
            raise
        finally:
            processor.scope = old_scope
            self.record['seconds'] = time.perf_counter()-began

    def __call__(self, inputs):
        if signature(inputs) != self.sig:
            raise ValueError('Graph input shape/dtype/device changed')
        copy_inputs(self.inputs, inputs)
        self.graph.replay()
        self.replays += 1
        return self.latest.clone()


class RTCGraphRuntime:
    """Sequential single-owner benchmark runtime, not yet a live controller.

    At the first graph request prepare both no-prefix and guided graphs; the
    latter uses a synthetic zero prefix only for preparation. Actual requests
    always copy their real prefix to the guided graph input before replay.
    """

    MODES = ('native_eager', 'cached_eager', 'graph')

    def __init__(self, model):
        c = model.config.rtc_config
        if (model.config.num_steps != 10 or model.config.chunk_size != 50 or
            c is None or not c.enabled or c.mode != 'guided' or c.execution_horizon != 10 or
            c.max_guidance_weight != 10.0 or c.prefix_attention_schedule != RTCAttentionSchedule.EXP or c.debug):
            raise ValueError('RTC configuration differs from frozen fullpath contract')
        self.model, self.original, self.original_processor = model, model.sample_actions, model.rtc_processor
        self.owner, self.closed = threading.get_ident(), False
        self.native = RecordedRTC(c)
        self.cached = DevicePrefixRTC(c, model.state_proj.weight.device)
        self.graphs, self.captures = {}, []
        self.mode = 'native_eager'
        self.latest = self.latest_inputs = self.metadata = None
        self.requests = self.rgb_encodings = 0
        self.input_signature = None
        self.receipt = None

    def check_owner(self):
        if self.closed or threading.get_ident() != self.owner:
            raise RuntimeError('Closed or wrong RTC model owner')

    def __enter__(self):
        self.check_owner()
        self.model.sample_actions = self
        return self

    def __exit__(self, *_args):
        self.check_owner()
        torch.cuda.synchronize()
        self.model.sample_actions = self.original
        self.model.rtc_processor = self.original_processor
        records = [copy.deepcopy(graph.record) for graph in self.graphs.values()]
        replays = {name: graph.replays for name, graph in self.graphs.items()}
        self.graphs.clear()
        self.latest = self.latest_inputs = None
        self.closed = True
        self.receipt = {'requests': self.requests, 'rgb_encodings': self.rgb_encodings,
                        'captures': records, 'replays': replays,
                        'native_steps': {k: dict(v) for k, v in self.native.counts.items()},
                        'cached_steps': {k: dict(v) for k, v in self.cached.counts.items()},
                        'sampler_restored': self.model.sample_actions == self.original,
                        'processor_restored': self.model.rtc_processor is self.original_processor,
                        'graphs_released': not self.graphs, 'owner': self.owner,
                        'close_thread': threading.get_ident()}

    def invoke(self, values, guided):
        return self.original(None, None, values[4], values[5], values[6], noise=values[7],
            future_image_tokens=tuple(values[:2]), future_image_token_masks=tuple(values[2:4]),
            inference_delay=3, execution_horizon=10,
            prev_chunk_left_over=values[8] if guided else None)

    def __call__(self, images, masks, tokens, token_masks, state, noise=None, *,
                 inference_delay=None, prev_chunk_left_over=None, execution_horizon=None,
                 **kwargs):
        self.check_owner()
        if (self.mode not in self.MODES or noise is None or inference_delay != 3 or
            execution_horizon != 10 or any(value is not None for value in kwargs.values())):
            raise ValueError('Unsupported RTC runtime overrides')
        guided = prev_chunk_left_over is not None
        if guided and (prev_chunk_left_over.shape != (30, 7) or
                       prev_chunk_left_over.device != state.device or
                       prev_chunk_left_over.dtype != torch.float32):
            raise ValueError('Expected original normalized 30x7 CUDA float32 prefix')
        images_t, masks_t = self.model.encode_image_tokens(images, masks)
        self.rgb_encodings += 1
        values = pack(images_t, masks_t, tokens, token_masks, state, noise)
        sig = signature(values)
        if self.input_signature is None:
            self.input_signature = sig
        elif self.input_signature != sig:
            raise ValueError('Same-sample input signature changed')
        self.latest_inputs = values + ((prev_chunk_left_over,) if guided else ())
        processor = self.native if self.mode == 'native_eager' else self.cached
        self.model.rtc_processor = processor
        processor.scope = 'public'
        capture_created = False
        with torch.no_grad():
            if self.mode == 'graph':
                if not self.graphs:
                    if guided:
                        raise ValueError('Both graphs must be prepared before the first guided request')
                    for name, yes in (('no_prefix', False), ('guided', True)):
                        template = values + ((state.new_zeros((30, 7), dtype=torch.float32),) if yes else ())
                        graph = FixedCallGraph(lambda v, flag=yes: self.invoke(v, flag), template,
                                               processor, self.model.action_out_proj, yes)
                        self.graphs[name] = graph
                    capture_created = True
                name = 'guided' if guided else 'no_prefix'
                self.latest = self.graphs[name](self.latest_inputs)
            else:
                self.latest = self.invoke(self.latest_inputs, guided)
        if self.latest.shape != (1, 50, 32) or not torch.isfinite(self.latest).all():
            raise ValueError('Nonfinite or wrong full action shape')
        self.requests += 1
        self.metadata = {'mode': self.mode, 'guided': guided, 'capture_created': capture_created,
                         'replays': int(self.mode == 'graph'),
                         'captured_vjps_per_replay': 10 if self.mode == 'graph' and guided else 0}
        return self.latest
