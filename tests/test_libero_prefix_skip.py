"""Synthetic-only E-PSI1 tests: independent dose/skip, fixed cadence, evidence tampering."""

import copy
import sys
import time
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'examples/advanced/predictive_async'))
import audit_libero_prefix_skip as a  # noqa: E402
import libero_prefix_skip as r  # noqa: E402

from tests.test_libero_graph_feedback import FakeLivePath, FakeSession  # noqa: E402
from tests.test_libero_observation_async import FakeNative, FakePredictor  # noqa: E402


@pytest.mark.parametrize('dose,skip', [(0, 0), (0, 2), (2, 0), (2, 2)])
def test_queue_separates_real_consumption_from_selected_row(dose, skip):
    q = r.SplitQueue(dose, skip)
    original = torch.arange(350, dtype=torch.float32).reshape(50, 7)
    stamp, prefix = q.begin(0, 0)
    assert prefix is None
    q.finish(stamp, original, original*10)
    for _ in range(20):
        q.pop()
    stamp, prefix = q.begin(20, dose)
    torch.testing.assert_close(prefix, original[20:], rtol=0, atol=0)
    for _ in range(dose):
        q.pop()
    dec = q.finish(stamp, original+1000, original*10+10000)
    assert dec['actual_delay'] == dose and dec['source_row'] == skip
    d = q.pop()
    assert d['action_index'] == 20+dose and d['source_row'] == skip
    assert d['nominal_action_index'] == 20+skip and d['nominal_minus_actual'] == skip-dose
    torch.testing.assert_close(d['original'], original[skip]+1000, rtol=0, atol=0)
    q.close()
    assert q.pop() is None and q.qsize() == 0
    assert q.finish(stamp, original, original)['reason'] == 'stale_or_closed'


def test_scope_and_anchor_selection():
    m = r.manifest()
    assert len(m['rows']) == 8 and not m['new_qualification']
    for state in (18, 19):
        selected = [v for v in m['rows'] if v['initial_state_id'] == state]
        assert {(v['consume'], v['skip']) for v in selected} == {(0, 0), (2, 2), (0, 2), (2, 0)}
        assert len({v['policy_seed'] for v in selected}) == 1
        assert all(v['later_delay'] == v['later_skip'] == 0 for v in selected)
    assert {v['reference_ordinal'] for v in m['rows'] if v['reference_ordinal'] is not None} == {0, 3, 5, 6}


def test_invalid_dose_and_unrealized_consumption_fail():
    with pytest.raises(ValueError):
        r.SplitQueue(True, 0)
    q = r.SplitQueue(2, 0)
    x = torch.zeros(50, 7)
    stamp, _ = q.begin(0, 0)
    q.finish(stamp, x, x)
    for _ in range(20):
        q.pop()
    stamp, _ = q.begin(20, 2)
    with pytest.raises(ValueError, match='consumption'):
        q.finish(stamp, x, x)


@pytest.mark.parametrize('dose,skip', [(0, 0), (0, 2), (2, 0), (2, 2)])
def test_fixed_physical_request_grid_not_affected_by_skipping(tmp_path, dose, skip):
    spec = copy.deepcopy(r.manifest()['rows'][0])
    spec.update(consume=dose, skip=skip)
    spec['limits']['measurement'] = 44
    budget, calls = r.Budget(), r.e.Calls(tmp_path/'calls.jsonl')
    budget.begin_episode()
    owner = r.DiagnosticOwner(FakePredictor(), calls, spec, budget)
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
    assert not owner.thread.is_alive()
    assert [v['stamp']['observation_index'] for v in owner.rows] == [0, 20, 40]
    assert [v['decision']['actual_delay'] for v in owner.rows] == [0, dose, 0]
    assert [v['decision']['source_row'] for v in owner.rows] == [0, skip, 0]
    for d in control['dispatches']:
        out = owner.outputs[d['request_id']]
        np.testing.assert_array_equal(d['command'], out['processed'][d['source_row']].numpy())
    assert not torch.cuda.is_initialized()


@pytest.mark.parametrize('dose,skip', [(0, 0), (0, 2), (2, 0), (2, 2)])
def test_complete_synthetic_episode_independent_audit_and_tamper(tmp_path, monkeypatch, dose, skip):
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
    spec = copy.deepcopy(r.manifest()['rows'][0])
    spec.update(consume=dose, skip=skip, arm=f'consume{dose}_skip{skip}')
    spec['limits']['measurement'] = 44
    calls, budget = r.e.Calls(tmp_path/'calls.jsonl'), r.Budget()
    try:
        rec = r.episode(spec, tmp_path, None, None, None, None, budget, calls)
    finally:
        calls.close()
    assert rec['status'] == 'completed', rec['first_failure']
    arrays = torch.load(tmp_path/'episode_000/arrays.pt', map_location='cpu', weights_only=False)
    row = a.inspect_episode(spec, rec, arrays, arrays['observations'][0])
    assert row['actions'] == 44 and row['requests'] == 3
    assert row['nonaligned_actions_explicit'] == (20-dose if dose != skip else 0)
    damaged = copy.deepcopy(arrays)
    damaged['control']['dispatches'][22]['source_row'] += 1
    with pytest.raises(ValueError, match='source row'):
        a.inspect_episode(spec, rec, damaged, arrays['observations'][0])
    damaged = copy.deepcopy(arrays)
    damaged['control']['dispatches'][22]['nominal_minus_actual'] += 1
    with pytest.raises(ValueError, match='offset'):
        a.inspect_episode(spec, rec, damaged, arrays['observations'][0])
    assert a.ta.check_anchor(rec, arrays, rec, copy.deepcopy(arrays))['complete_trajectory_reproduced']
    assert not torch.cuda.is_initialized()


def test_close_before_install_fences_result(tmp_path):
    spec = r.manifest()['rows'][3]
    budget, calls = r.Budget(), r.e.Calls(tmp_path/'close.jsonl')
    budget.begin_episode()
    owner = r.DiagnosticOwner(FakePredictor(), calls, spec, budget)
    owner.submit({'index': 0, 'returned_at': time.perf_counter()}, 0)
    owner.close()
    calls.close()
    assert not owner.thread.is_alive() and owner.pending is None
    assert not owner.rows[0]['decision']['accepted']


def test_binary_contrasts_do_not_claim_population_effects():
    table = {f'consume{d}_skip{s}': {'success': d == 0} for d, s in ((0, 0), (2, 2), (0, 2), (2, 0))}
    values = a.contrasts(table)
    assert values['old_execution_effect_with_skip0'] == -1
    assert values['skip_effect_with_old0'] == 0 and not values['population_effect_claimed']
    del table['consume0_skip0']
    with pytest.raises(ValueError):
        a.contrasts(table)
