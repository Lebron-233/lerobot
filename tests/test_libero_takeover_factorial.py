"""Synthetic diagnostic tests. No policy weights, real Env or confirmation arrays."""

import copy
import sys
import time
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'examples/advanced/predictive_async'))
import audit_libero_takeover_factorial as a  # noqa: E402
import libero_takeover_factorial as r  # noqa: E402

from tests.test_libero_graph_feedback import FakeLivePath, FakeSession  # noqa: E402
from tests.test_libero_observation_async import FakeNative, FakePredictor  # noqa: E402


def test_manifest_exact_diagnostic_scope():
    m = r.manifest()
    assert len(m['rows']) == 8 and len(m['known_diagnostic_identities']) == 2
    assert m['new_qualification'] is False and m['training'] == 0
    for state in (18, 19):
        selected = [v for v in m['rows'] if v['initial_state_id'] == state]
        assert {(v['first_delay'], v['later_delay']) for v in selected} == {(0, 0), (0, 2), (2, 0), (2, 2)}
        assert len({v['policy_seed'] for v in selected}) == 1
        assert len({v['environment_seed'] for v in selected}) == 1
    assert sum(v['reference_ordinal'] is not None for v in m['rows']) == 4


@pytest.mark.parametrize('first,later', [(0, 0), (0, 2), (2, 0), (2, 2)])
def test_imposed_delay_changes_only_installation(tmp_path, first, later):
    spec = copy.deepcopy(r.manifest()['rows'][0])
    spec.update(first_delay=first, later_delay=later)
    spec['limits']['measurement'] = 44
    budget, calls = r.Budget(), r.e.Calls(tmp_path/'calls.jsonl')
    budget.begin_episode()
    owner = r.old.InferenceOwner(FakePredictor(), calls, spec, budget)
    native = FakeNative()
    def observe(raw, index, returned):
        return {'index': index, 'returned_at': returned}
    initial = observe({}, 0, time.perf_counter())
    try:
        owner.submit(initial, 0)
        owner.receive(block=True)
        control = r.controlled_loop(owner, native, spec, initial, observe)
    finally:
        owner.close()
        calls.close()
    assert not owner.thread.is_alive() and owner.pending is None
    assert len(control['dispatches']) == 44
    assert [v['stamp']['observation_index'] for v in owner.rows] == [0, 20, 40]
    assert [v['decision']['actual_delay'] for v in owner.rows] == [0, first, later]
    for d in control['dispatches']:
        original = owner.outputs[d['request_id']]['original'][d['source_row']]
        np.testing.assert_array_equal(d['command'], (original*10).numpy())
    assert not torch.cuda.is_initialized()


def test_first_request_number_boundary():
    spec = {'first_delay': 0, 'later_delay': 2}
    assert [r.delay_for(spec, i) for i in (1, 2, 3, 14)] == [0, 0, 2, 2]
    with pytest.raises(ValueError, match='Invalid'):
        r.delay_for(spec, 0)


def test_factorial_does_not_turn_diagnostic_into_population_claim():
    table = {k: {'success': v} for k, v in [('first0_later0', True), ('first2_later0', True),
                                          ('first0_later2', True), ('first2_later2', False)]}
    got = a.factors(table)
    assert got['first_delay_effect_when_later_zero'] == 0
    assert got['interaction_contrast_success'] == -1
    del table['first2_later0']
    with pytest.raises(ValueError, match='Incomplete'):
        a.factors(table)


def test_exact_rejects_action_or_dtype_change():
    x = torch.zeros(1, 50, 32)
    a.exact(x, x.clone(), 'exact')
    with pytest.raises(ValueError):
        a.exact(x, x.double(), 'dtype')
    other = x.clone()
    other[0, 20, 0] = 1
    with pytest.raises(ValueError):
        a.exact(x, other, 'action')


def test_complete_diagnostic_episode_lifecycle_and_anchor_tamper(tmp_path, monkeypatch):
    monkeypatch.setattr(r.g, 'FullPath', FakeLivePath)
    monkeypatch.setattr(r.g.GraphFeedbackPredictor, 'fresh_noise', lambda _: torch.zeros(1, 50, 32))
    monkeypatch.setattr(r.e, 'NativeSession', FakeSession)
    def observation(raw, language, index, returned):
        image = torch.full((2, 2, 3), index, dtype=torch.uint8)
        row = {'index': index, 'returned_at': returned, 'state': torch.zeros(1, 8),
            'eef_quaternion_xyzw': torch.zeros(4), 'raw_eef_position': torch.zeros(3),
            'raw_gripper_qpos': torch.zeros(2), 'raw_pixels': {'image': image, 'image2': image.clone()},
            'worker_observation': {'image': image, 'state_0': float(index)}}
        return None, None, row
    monkeypatch.setattr(r.e, 'observation', observation)
    monkeypatch.setattr(r, 'load', lambda _: observation({}, '', 0, 0)[2])
    spec = copy.deepcopy(r.manifest()['rows'][1])
    spec['limits']['measurement'] = 24
    calls, budget = r.e.Calls(tmp_path/'calls.jsonl'), r.Budget()
    try:
        record = r.episode(spec, tmp_path, None, None, None, None, budget, calls)
    finally:
        calls.close()
    assert record['status'] == 'completed', record['first_failure']
    arrays = torch.load(tmp_path/'episode_001/arrays.pt', map_location='cpu', weights_only=False)
    checked = a.graph_audit.inspect_episode(spec, record, arrays, arrays['observations'][0])
    assert checked['actions'] == 24 and record['worker_joined']
    assert checked['actual_delays'] == [0, 2]
    original = copy.deepcopy(arrays)
    assert a.check_anchor(record, arrays, record, original)['complete_trajectory_reproduced']
    arrays['control']['dispatches'][20]['command'][0] += 1
    with pytest.raises(ValueError, match='Anchor command'):
        a.check_anchor(record, arrays, record, original)
    assert not torch.cuda.is_initialized()
