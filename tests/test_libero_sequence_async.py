"""Synthetic tests only: plan provenance, lifecycle and all three live controllers."""

import copy
import sys
import time
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'examples/advanced/predictive_async'))
import audit_libero_sequence_async as a  # noqa: E402
import libero_sequence_async as r  # noqa: E402
from sequence_execution_queue import SequenceExecutionQueue  # noqa: E402

from tests.test_libero_graph_feedback import FakeLivePath, FakeSession  # noqa: E402
from tests.test_libero_observation_async import FakePredictor  # noqa: E402


def chunks():
    x = torch.arange(350, dtype=torch.float32).reshape(50, 7)
    return x, x*10


@pytest.mark.parametrize('delay', [0, 1, 2, 3, 8])
def test_new_prefix_is_preserved_without_time_identity_mislabel(delay):
    q = SequenceExecutionQueue()
    x, y = chunks()
    stamp, prefix = q.begin(0, 0)
    assert prefix is None
    q.finish(stamp, x, y)
    for _ in range(20):
        q.pop()
    stamp, prefix = q.begin(20, 3)
    torch.testing.assert_close(prefix, x[20:], rtol=0, atol=0)
    prefix.zero_()
    for i in range(delay):
        assert q.pop()['source_row'] == 20+i
    dec = q.finish(stamp, x+1000, y+10000)
    assert dec['source_row'] == 0 and dec['actual_delay'] == delay
    assert dec['superseded_tail']['start_row'] == 20+delay
    assert q.qsize() == 50
    seen = set()
    for i in range(50):
        d = q.pop()
        assert d['source_row'] == i and d['action_index'] == 20+delay+i
        assert d['nominal_action_index'] == 20+i and d['nominal_minus_actual'] == -delay
        assert (d['request_id'], i) not in seen
        seen.add((d['request_id'], i))
    assert q.pop() is None


def test_defensive_copies_and_duplicate_publication():
    q = SequenceExecutionQueue()
    x, y = chunks()
    stamp, _ = q.begin(0, 0)
    q.finish(stamp, x, y)
    x.zero_()
    y.zero_()
    first = q.pop()
    first['command'].zero_()
    assert q.finish(stamp, x, y)['reason'] == 'stale_or_closed'
    assert q.pop()['source_row'] == 1
    assert q.pop()['command'].sum() > 0


@pytest.mark.parametrize('event', ['reset', 'close'])
def test_late_epoch_result_is_fenced(event):
    q = SequenceExecutionQueue()
    x, y = chunks()
    stamp, _ = q.begin(0, 0)
    getattr(q, event)()
    assert q.finish(stamp, x, y)['reason'] == 'stale_or_closed'
    assert q.pop() is None
    if event == 'reset':
        new, _ = q.begin(0, 0)
        assert new != stamp and new.epoch != stamp.epoch
        assert q.finish(new, x, y)['accepted']


def test_over_cap_retains_old_plan_instead_of_reinstalling_stale_prefix():
    q = SequenceExecutionQueue()
    x, y = chunks()
    stamp, _ = q.begin(0, 0)
    q.finish(stamp, x, y)
    stamp, _ = q.begin(0, 8)
    for _ in range(9):
        q.pop()
    assert q.finish(stamp, x+1000, y+1000) == {'accepted': False, 'reason': 'expired', 'actual_delay': 9}
    assert q.pop()['request_id'] == 1


def test_bounds_and_prefix_capacity():
    q = SequenceExecutionQueue()
    x, y = chunks()
    with pytest.raises(ValueError):
        q.begin(True, 0)
    with pytest.raises(ValueError):
        q.begin(0, 9)
    stamp, _ = q.begin(0, 0)
    with pytest.raises(RuntimeError):
        q.begin(0, 0)
    with pytest.raises(ValueError):
        q.finish(stamp, x[:40], y[:40])
    bad = x.clone()
    bad[0, 0] = float('nan')
    with pytest.raises(ValueError):
        q.finish(stamp, bad, y)
    q.finish(stamp, x, y)
    for _ in range(48):
        q.pop()
    with pytest.raises(ValueError, match='Insufficient'):
        q.begin(48, 3)


def test_manifest_is_fixed_three_arm_known_development():
    m = r.manifest()
    assert len(m['rows']) == 30 and m['captures'] == 30
    assert not m['new_qualification_claimed'] and not m['forced_wait_in_async']
    assert len(r.PAIRS) == 10 and r.PAIRS[-2:] == ((7, 18), (7, 19))
    for i in range(10):
        values = m['rows'][3*i:3*i+3]
        assert {v['arm'] for v in values} == set(r.ARMS)
        assert len({v['policy_seed'] for v in values}) == len({v['environment_seed'] for v in values}) == 1
        assert all(v['fps'] == 20 and v['max_delay'] == 8 for v in values)


@pytest.mark.parametrize('arm', r.ARMS)
def test_complete_shared_controller_episode_and_independent_audit(tmp_path, monkeypatch, arm):
    monkeypatch.setattr(r.g, 'FullPath', FakeLivePath)
    monkeypatch.setattr(r.g.GraphFeedbackPredictor, 'fresh_noise', lambda _: torch.zeros(1, 50, 32))
    monkeypatch.setattr(r.e, 'NativeSession', FakeSession)
    def observation(raw, language, index, returned):
        image = torch.full((2, 2, 3), index, dtype=torch.uint8)
        rec = {'index': index, 'returned_at': returned, 'state': torch.zeros(1, 8),
            'eef_quaternion_xyzw': torch.zeros(4), 'raw_eef_position': torch.zeros(3),
            'raw_gripper_qpos': torch.zeros(2), 'raw_pixels': {'image': image, 'image2': image.clone()},
            'worker_observation': {'image': image, 'state_0': float(index)}}
        return None, None, rec
    monkeypatch.setattr(r.e, 'observation', observation)
    spec = copy.deepcopy(next(v for v in r.manifest()['rows'] if v['arm'] == arm))
    spec['limits']['measurement'] = 44
    calls, budget = r.e.Calls(tmp_path/'calls.jsonl'), r.Budget()
    try:
        rec, initial = r.episode(spec, tmp_path, None, None, None, None, budget, calls)
    finally:
        calls.close()
    assert rec['status'] == 'completed', rec['first_failure']
    arrays = r.load(tmp_path/f"episode_{spec['ordinal']:03d}/arrays.pt")
    checked = a.inspect_episode(spec, rec, arrays, initial)
    assert checked['actions'] == 44 and checked['requests'] >= 2
    assert checked['plan_rows']['executed'] == 44
    assert sum(checked['plan_rows'].values()) == checked['requests']*50
    if arm == 'sequence_async':
        assert checked['plan_rows']['skipped_head'] == 0 and checked['max_nominal_lag_steps'] > 0
        assert rec['requests'][2]['stamp']['observation_index'] > 40
    damaged = copy.deepcopy(arrays)
    damaged['control']['dispatches'][22]['source_row'] += 1
    with pytest.raises(ValueError, match='source row'):
        a.inspect_episode(spec, rec, damaged, initial)
    damaged = copy.deepcopy(arrays)
    damaged['outputs'][2]['input_fingerprint'] = '0'*64
    with pytest.raises(ValueError, match='live input'):
        a.inspect_episode(spec, rec, damaged, initial)
    if arm == 'sequence_async':
        damaged = copy.deepcopy(arrays)
        damaged['control']['dispatches'][22]['nominal_minus_actual'] += 1
        with pytest.raises(ValueError, match='offset'):
            a.inspect_episode(spec, rec, damaged, initial)
    assert not torch.cuda.is_initialized()


def test_close_with_inflight_result_still_joins_and_fences(tmp_path):
    spec = next(v for v in r.manifest()['rows'] if v['arm'] == 'sequence_async')
    calls, budget = r.e.Calls(tmp_path/'calls.jsonl'), r.Budget()
    budget.begin_episode()
    owner = r.Owner(FakePredictor(), calls, spec, budget)
    owner.submit({'index': 0, 'returned_at': time.perf_counter()}, 0)
    owner.close()
    calls.close()
    assert not owner.thread.is_alive() and owner.pending is None
    assert owner.rows[0]['decision'] == {'accepted': False, 'reason': 'stale_or_closed'}
    assert owner.queue.pop() is None


def test_explicit_offsets_are_not_silently_relabelled():
    q = SequenceExecutionQueue()
    x, y = chunks()
    boot, _ = q.begin(0, 0)
    q.finish(boot, x, y)
    for _ in range(20):
        q.pop()
    stamp, _ = q.begin(20, 2)
    q.pop()
    q.pop()
    q.finish(stamp, x, y)
    row = q.pop()
    assert row['action_index'] == 22 and row['nominal_action_index'] == 20
    np.testing.assert_array_equal(row['command'], y[0])
