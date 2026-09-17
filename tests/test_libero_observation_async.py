"""Synthetic CPU controller/owner and independent provenance audit tests.

No scientific policy weights, data, GPU initialization, or robot simulator.
"""

import copy
import sys
import time
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'examples/advanced/predictive_async'))
import audit_libero_observation_async as audit  # noqa: E402
import libero_observation_async as r  # noqa: E402


def test_manifest_has_exact_eight_pairs_and_alternating_order():
    m = r.manifest()
    assert len(m['rows']) == 16 and len(set(r.PAIRS)) == 8
    assert m['rtc'] is False and m['graph_captures'] == 0
    for pair in range(8):
        a, b = m['rows'][2*pair:2*pair+2]
        assert {a['condition'], b['condition']} == {'serialized', 'async'}
        assert a['environment_seed'] == b['environment_seed'] and a['policy_seed'] == b['policy_seed']
        assert a['condition'] == ('serialized' if pair % 2 == 0 else 'async')


def test_history_conflict_stops_instead_of_replacement():
    r.assert_unused([{'task': 0, 'state': 41}])
    with pytest.raises(ValueError, match='already used'):
        r.assert_unused([{'task': 0, 'state': 8}])


@pytest.mark.parametrize('remaining,inflight,delay,expected', [
    (30, False, 7, True), (31, False, 7, False), (8, False, 7, False),
    (9, False, 7, True), (20, True, 7, False), (20, False, 9, False), (0, False, 9, True)])
def test_submission_respects_inflight_cap_and_guard(remaining, inflight, delay, expected):
    assert r.should_submit(remaining, inflight, delay) is expected


def test_delay_uses_nearest_rank_and_does_not_hide_over_cap():
    assert r.estimate_delay([]) == 7
    assert r.estimate_delay([.28]*50) == 7
    assert r.estimate_delay([.401]*50) == 10
    assert r.estimate_delay([100]*20+[.28]*50) == 7


class FakePredictor:
    def __init__(self):
        self.calls = 0

    def __call__(self, observation):
        self.calls += 1
        began = time.perf_counter()
        time.sleep(.065)
        original = (torch.arange(50, dtype=torch.float32)+100*self.calls)[:, None].repeat(1, 7)
        full = torch.zeros(1, 50, 32)
        full[0, :, :7] = original
        return {'original': original, 'processed': original*10, 'full': full, 'noise': torch.zeros(1, 50, 32),
                'model_started_at': began, 'model_returned_at': time.perf_counter(),
                'vision_encodes': 1, 'projection_shapes': [[1, 50, 32]]*10}


class FakeNative:
    def __init__(self):
        self.steps = []

    def step(self, dispatch):
        began = time.perf_counter()
        time.sleep(.002)
        self.steps.append({'segment': 'measurement', 'action': dispatch['command'].tolist(),
            'action_index': dispatch['action_index'], 'observation_index': dispatch['observation_index'],
            'started_at': began, 'returned_at': time.perf_counter()})
        return {}, 0.0, False, False, {'is_success': False}


def synthetic_episode(tmp_path, arm):
    spec = copy.deepcopy(r.manifest()['rows'][0])
    spec['condition'] = arm
    spec['limits']['measurement'] = 24
    budget = r.Budget()
    budget.begin_episode()
    calls = r.e.Calls(tmp_path / (arm+'.jsonl'))
    native, observations = FakeNative(), []

    def observe(raw, index, returned):
        row = {'index': index, 'returned_at': returned, 'state': torch.zeros((1, 8)),
               'eef_quaternion_xyzw': torch.zeros(4), 'raw_eef_position': torch.zeros(3),
               'raw_gripper_qpos': torch.zeros(2), 'raw_pixels': {'image': torch.zeros((2, 2, 3)), 'image2': torch.zeros((2, 2, 3))}}
        observations.append(row)
        return row

    initial = observe({}, 0, time.perf_counter())
    owner = r.InferenceOwner(FakePredictor(), calls, spec, budget)
    try:
        owner.submit(initial, 0)
        owner.receive(block=True)
        control = r.control_loop(owner, native, spec, initial, observe)
    finally:
        owner.close()
        calls.close()
    count = min(1200, max(1, int(np.ceil((control['ended_at']-control['t0'])*20))))
    result = {'spec': spec, 'status': 'completed', 'first_failure': None,
        'worker_joined': not owner.thread.is_alive(), 'environment_closed': True,
        'requests': owner.rows, 'native_steps': [{'segment': 'settling'}]*10+native.steps,
        'budget': {'episodes': 1, 'settling': 10, 'model': len(owner.rows), 'measurement': 24},
        'wall_slots': count, 'no_action_slots': count-24, 'wall_s': control['ended_at']-control['t0'],
        'measured_actions': 24, 'underflows': 0, 'success': False, 'terminal_reason': 'action_limit'}
    arrays = {'observations': observations, 'control': control, 'outputs': owner.outputs}
    return spec, result, arrays, initial


@pytest.mark.parametrize('arm', ['serialized', 'async'])
def test_shared_owner_controller_and_independent_audit(tmp_path, arm):
    spec, result, arrays, initial = synthetic_episode(tmp_path, arm)
    checked = audit.audit_episode(spec, result, arrays, initial)
    assert checked['actions'] == 24 and result['worker_joined']
    assert checked['request_count'] >= 2
    if arm == 'serialized':
        assert checked['actual_delays'] == [0]*checked['request_count'] and not checked['overlaps']
    else:
        assert max(checked['actual_delays']) > 0
    damaged = copy.deepcopy(arrays)
    damaged['control']['dispatches'][0]['source_row'] += 1
    with pytest.raises(ValueError, match='source row'):
        audit.audit_episode(spec, result, damaged, initial)


def test_owner_exception_propagates_and_can_join(tmp_path):
    def fail(_):
        raise ValueError('synthetic model failure')
    budget = r.Budget()
    budget.begin_episode()
    calls = r.e.Calls(tmp_path/'failure.jsonl')
    owner = r.InferenceOwner(fail, calls, r.manifest()['rows'][0], budget)
    try:
        owner.submit({'index': 0, 'returned_at': time.perf_counter()}, 0)
        with pytest.raises(RuntimeError, match='synthetic model failure'):
            owner.receive(block=True)
    finally:
        owner.close()
        calls.close()
    assert not owner.thread.is_alive()


def test_immediate_close_waits_for_queued_request_and_fences_publication(tmp_path):
    budget = r.Budget()
    budget.begin_episode()
    calls = r.e.Calls(tmp_path/'close.jsonl')
    owner = r.InferenceOwner(FakePredictor(), calls, r.manifest()['rows'][0], budget)
    owner.submit({'index': 0, 'returned_at': time.perf_counter()}, 0)
    owner.close()
    calls.close()
    assert not owner.thread.is_alive() and owner.pending is None
    assert owner.rows[0]['decision'] == {'accepted': False, 'reason': 'stale_or_closed'}
    assert owner.queue.pop() is None


def test_cpu_only_and_quantiles_preserve_maximum():
    assert audit.quantiles(list(range(1, 81)))['p99'] == 80
    assert audit.quantiles([]) == {'n': 0}
    assert not torch.cuda.is_initialized()
