"""Opt-in, single-owner ten-step sampler for the fixed LIBERO engineering experiment."""

import threading
import time

import numpy as np
import torch
from profile_libero_cuda_graph_compute import GraphSampler, invoke, pack


def signature(values):
    return tuple((tuple(v.shape), v.dtype, v.device) for v in values)


class SmolVLAGraphRuntime:
    """Own one task graph; hand consumers a separate output allocation on every replay.

    The experiment has one model thread and no concurrent CUDA work in its process.
    It uses the existing sample_noise implementation at the original sampling point.
    Preparation uses explicit noise and restores CPU/CUDA RNG even on an exception.
    """

    def __init__(self, model):
        if model.config.num_steps != 10:
            raise ValueError("This runtime requires the original ten denoising steps")
        self.model = model
        self.original = model.sample_actions
        self.owner = threading.get_ident()
        self.graph = None
        self.task = None
        self.mode = "eager"
        self.captures = []
        self.latest = self.latest_noise = self.metadata = None
        self.input_signature = None
        self.control_requests = 0
        self.noise_draws = 0

    def _check_owner(self):
        if threading.get_ident() != self.owner:
            raise RuntimeError("Graph preparation, replay and release require the single model owner")

    def __enter__(self):
        self._check_owner()
        self.model.sample_actions = self
        return self

    def __exit__(self, *_exception):
        self.model.sample_actions = self.original
        self.release_graph()
        self.reset_output()

    def reset_output(self):
        self.latest = self.latest_noise = self.metadata = None

    def release_graph(self):
        self._check_owner()
        if self.graph is not None:
            torch.cuda.synchronize()
        self.graph = None
        self.input_signature = None

    def begin_episode(self, mode, task):
        self._check_owner()
        if mode not in ("eager", "graph"):
            raise ValueError("Only eager and graph are registered")
        if task != self.task:
            self.release_graph()
            self.task = task
        self.mode = mode
        self.reset_output()

    def _capture(self, inputs):
        start = time.perf_counter()
        device = inputs[-1].device
        devices = (
            [device.index if device.index is not None else torch.cuda.current_device()]
            if device.type == "cuda"
            else []
        )
        with torch.random.fork_rng(devices=devices):
            # Include the complete eager setup, not just the later graph constructor.
            invoke(self.original, inputs)
            torch.cuda.synchronize()
            eager_setup = time.perf_counter() - start
            self.graph = GraphSampler(self.model, self.original, inputs, self.model.action_out_proj)
            torch.cuda.synchronize()
        self.input_signature = signature(inputs)
        self.captures.append(
            {
                "capture_id": len(self.captures) + 1,
                "task": self.task,
                "projection_shapes": self.graph.capture_projection_shapes.copy(),
                "eager_setup_seconds": eager_setup,
                "preparation_seconds": time.perf_counter() - start,
                "input_shapes": [list(v.shape) for v in self.graph.inputs],
            }
        )

    def __call__(self, images, masks, tokens, token_masks, state, noise=None):
        self._check_owner()
        if noise is None:
            noise = self.model.sample_noise(
                (state.shape[0], self.model.config.chunk_size, self.model.config.max_action_dim), state.device
            )
            self.noise_draws += 1
        inputs = pack(images, masks, tokens, token_masks, state, noise)
        self.latest_noise = noise.detach().clone()
        self.metadata = None
        if self.mode == "graph":
            if self.graph is None:
                self._capture(inputs)
            elif signature(inputs) != self.input_signature:
                raise ValueError("Same-task graph input shape, dtype or device changed")
            before = self.graph.replay_calls
            shared_output = self.graph(images, masks, tokens, token_masks, state, noise=noise)
            # This clone is inside the selector interval and ordered after replay on its stream.
            self.latest = shared_output.clone()
            capture_id = self.captures[-1]["capture_id"]
            replay_count = self.graph.replay_calls - before
        else:
            self.latest = invoke(self.original, inputs)
            capture_id, replay_count = None, 0
        if self.latest.shape != (1, 50, 32) or not torch.isfinite(self.latest).all():
            raise ValueError("Sampler did not return a finite full [1, 50, 32] chunk")
        self.control_requests += 1
        self.metadata = {
            "sampler_mode": self.mode,
            "capture_id": capture_id,
            "replay_count": replay_count,
            "full_chunk_shape": list(self.latest.shape),
            "full_chunk_finite": True,
        }
        return self.latest

    def snapshot(self, normalized, action):
        return {
            "noise": self.latest_noise.detach().cpu().numpy().copy(),
            "full_chunk": self.latest.detach().cpu().numpy().copy(),
            "normalized_action": normalized.detach().cpu().numpy().copy(),
            "postprocessed_action": np.asarray(action).reshape(1, 7).copy(),
        }

    def record_request(self, output, number, normalized, action, journal):
        directory = output / "requests"
        directory.mkdir(exist_ok=True)
        file = directory / f"{number:03d}.npz"
        np.savez_compressed(file, **self.snapshot(normalized, action))
        journal.emit("sampler_request", number=number, arrays=str(file.relative_to(output)), **self.metadata)


def valid_sampler_evidence(mode, shapes, metadata, captures):
    if mode == "eager":
        return shapes == [[1, 50, 32]] * 10
    return (
        shapes == []
        and metadata is not None
        and metadata["sampler_mode"] == "graph"
        and metadata["replay_count"] == 1
        and metadata["full_chunk_shape"] == [1, 50, 32]
        and metadata["full_chunk_finite"] is True
        and any(
            c["capture_id"] == metadata["capture_id"] and c["projection_shapes"] == [[1, 50, 32]] * 10
            for c in captures
        )
    )
