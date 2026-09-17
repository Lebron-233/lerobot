"""Synthetic CPU tests for E-GCR1 accounting, owner and exact-output gates."""

import sys
from pathlib import Path
from threading import Thread
from types import SimpleNamespace

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'examples/advanced/predictive_async'))
import audit_libero_graph_replay as audit  # noqa: E402
import libero_graph_replay as r  # noqa: E402


def test_scope_is_fixed_workload_not_new_rollout():
    s = r.specification()
    assert len(s['fixed_keys']) == len(set(map(tuple, s['fixed_keys']))) == 16
    assert s['fixed_requests'] == 16*2*2
    assert s['total_requests'] == s['fixed_requests']+s['replay_requests'] == 206
    assert s['replay_actions'] == 2*1410 and s['captures'] == 12
    assert not s['new_task_success_claim'] and not s['rtc']
    assert s['qualification_reads'] == s['training'] == 0


@pytest.mark.parametrize('latency,steps', [(.28, 7), (.35, 8), (.351, 9), (.1, 3)])
def test_tail_budget_fixed(latency, steps):
    rows = [{'complete_s': latency} for _ in range(63)]
    assert r.latency(rows)['required_delay_steps'] == steps
    audit.compare_times(r.latency(rows), audit.times(rows))


def test_tail_is_not_hidden_in_average():
    rows = [{'complete_s': .1} for _ in range(62)]+[{'complete_s': .4}]
    assert r.latency(rows)['mean_s'] < .11
    assert r.latency(rows)['required_delay_steps'] == 9


@pytest.mark.parametrize('value', [float('nan'), float('inf'), -1, 0])
def test_invalid_timing_rejected(value):
    with pytest.raises(ValueError):
        r.latency([{'complete_s': value}])


class FakeRuntime:
    def __init__(self, model):
        assert not isinstance(model.sample_actions, FakeRuntime), 'Nested sampler wrappers'
        self.model, self.original = model, model.sample_actions
        self.control_requests = self.rgb_encodings = self.noise_draws = 0
        self.captures, self.graph = [], object()

    def __enter__(self):
        self.model.sample_actions = self

    def begin_episode(self, mode, task):
        self.mode, self.task = mode, task

    def __exit__(self, *args):
        self.model.sample_actions = self.original
        self.graph = None


def test_single_runtime_mode_switch_preserves_original(monkeypatch):
    monkeypatch.setattr(r, 'SmolVLAGraphRuntime', FakeRuntime)
    original = object()
    policy = SimpleNamespace(model=SimpleNamespace(sample_actions=original))
    p = r.FullPath(policy, None, None, 'eager', 'same_task')
    p.activate()
    first = p.runtime
    for mode in ('graph', 'eager', 'graph'):
        p.mode = mode
        p.activate()
        assert p.runtime is first and policy.model.sample_actions is first
    p.close()
    assert policy.model.sample_actions is original
    assert p.receipt['graph_released'] and p.receipt['sampler_restored']
    with pytest.raises(ValueError, match='Closed'):
        p.activate()


def test_owner_cannot_change(monkeypatch):
    monkeypatch.setattr(r, 'SmolVLAGraphRuntime', FakeRuntime)
    p = r.FullPath(SimpleNamespace(model=SimpleNamespace(sample_actions=object())), None, None, 'graph', 'x')
    p.activate()
    errors = []

    def other():
        try:
            p.activate()
        except ValueError as exc:
            errors.append(str(exc))

    thread = Thread(target=other)
    thread.start()
    thread.join(2)
    assert not thread.is_alive() and errors == ['Wrong model owner']
    p.close()


def test_exact_requires_all_50_by_32_values_and_dtype():
    original = torch.zeros(1, 50, 32)
    changed = original.clone()
    changed[0, 49, 31] = 1
    with pytest.raises(ValueError, match='changed'):
        audit.exact(changed, original, 'changed')
    with pytest.raises(ValueError, match='dtype'):
        audit.exact(original.double(), original, 'dtype')
    assert not torch.cuda.is_initialized()
