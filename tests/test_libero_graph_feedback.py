"""CPU lifecycle/contract regressions. No scientific data, model or Env is loaded."""

import copy
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'examples/advanced/predictive_async'))
import audit_libero_graph_feedback as audit  # noqa: E402
import libero_graph_feedback as r  # noqa: E402

from tests.test_libero_observation_async import FakePredictor  # noqa: E402


def test_contract_and_new_identities():
    m = r.manifest()
    assert len(m['rows']) == 16 and len(set(r.PAIRS)) == 8
    assert m['captures'] == 16 and not m['replay_commands'] and not m['rtc']
    assert m['chunk'] == 50 and m['denoising_steps'] == 10
    assert {x['initial_state_id'] for x in m['rows']} == {10, 11}
    for i in range(8):
        a, b = m['rows'][2*i:2*i+2]
        assert {a['condition'], b['condition']} == {'serialized', 'async'}
        assert a['environment_seed'] == b['environment_seed']
        assert a['policy_seed'] == b['policy_seed']
    with pytest.raises(ValueError, match='Previously used'):
        r.assert_unused([{'task': 0, 'state': 10}])


def test_capture_only_on_bootstrap():
    rt = SimpleNamespace(graph=None, captures=[])
    r.require_ready(rt, True)
    with pytest.raises(ValueError, match='Control-time'):
        r.require_ready(rt, False)
    rt.graph, rt.captures = object(), [{'status': 'captured'}]
    r.require_ready(rt, False)
    with pytest.raises(ValueError, match='preexisting'):
        r.require_ready(rt, True)
    rt.captures.append({'status': 'captured'})
    with pytest.raises(ValueError, match='Control-time'):
        r.require_ready(rt, False)


@pytest.mark.parametrize('close_early', [False, True])
def test_owner_cleanup_on_model_thread_and_final_result_fencing(tmp_path, close_early):
    predictor = FakePredictor()
    closed = []
    def cleanup():
        closed.append(threading.get_ident())
    calls, budget = r.e.Calls(tmp_path/'calls.jsonl'), r.old.Budget()
    budget.begin_episode()
    owner = r.old.InferenceOwner(predictor, calls, r.manifest()['rows'][0], budget,
        on_owner_close=cleanup, request_kind='graph_request')
    try:
        owner.submit({'index': 0, 'returned_at': time.perf_counter()}, 0)
        if not close_early:
            owner.receive(block=True)
        owner.close()
        assert closed == [owner.thread.ident] and closed[0] != threading.get_ident()
        assert not owner.thread.is_alive() and owner.pending is None
        assert owner.cleanup_error is None
        assert owner.rows[0]['decision']['accepted'] is (not close_early)
    finally:
        calls.close()


def test_cleanup_error_propagates_without_thread_leak(tmp_path):
    def cleanup():
        raise RuntimeError('synthetic release failure')
    calls, budget = r.e.Calls(tmp_path/'calls.jsonl'), r.old.Budget()
    budget.begin_episode()
    owner = r.old.InferenceOwner(FakePredictor(), calls, r.manifest()['rows'][0], budget,
        on_owner_close=cleanup)
    owner.submit({'index': 0, 'returned_at': time.perf_counter()}, 0)
    owner.receive(block=True)
    with pytest.raises(ValueError, match='synthetic release failure'):
        owner.close()
    calls.close()
    assert not owner.thread.is_alive()


def test_failed_request_still_closes_on_owner(tmp_path):
    seen = []
    def fail(_):
        raise ValueError('synthetic request failure')
    calls, budget = r.e.Calls(tmp_path/'calls.jsonl'), r.old.Budget()
    budget.begin_episode()
    owner = r.old.InferenceOwner(fail, calls, r.manifest()['rows'][0], budget,
        on_owner_close=lambda: seen.append(threading.get_ident()))
    owner.submit({'index': 0, 'returned_at': time.perf_counter()}, 0)
    with pytest.raises(RuntimeError, match='request failure'):
        owner.receive(block=True)
    owner.close()
    calls.close()
    assert seen == [owner.thread.ident] and not owner.thread.is_alive()


def test_live_fingerprint_includes_images_and_state():
    observation = {'worker_observation': {'image': torch.zeros(2, 2, 3, dtype=torch.uint8), 'state_0': 1.0}}
    h = r.observation_fingerprint(observation)
    changed = copy.deepcopy(observation)
    changed['worker_observation']['image'][0, 0, 0] = 1
    assert h != r.observation_fingerprint(changed)
    changed = copy.deepcopy(observation)
    changed['worker_observation']['state_0'] = 2.0
    assert h != r.observation_fingerprint(changed)


def test_graph_evidence_rejects_fake_eager_hooks_or_recapture():
    inputs = [torch.zeros(1, 2, 3), torch.zeros(1, 2, 3), torch.ones(1, 2, dtype=torch.bool),
              torch.ones(1, 2, dtype=torch.bool), torch.ones(1, 4, dtype=torch.long),
              torch.ones(1, 4, dtype=torch.bool), torch.zeros(1, 32), torch.zeros(1, 50, 32)]
    record = {'predictor_cleanup': {'owner': 12, 'close_thread': 12,
              'captures': [{'input_shapes': [list(v.shape) for v in inputs]}]}}
    req = {'stamp': {'request_id': 2, 'observation_index': 20}}
    out = {'sampler': {'sampler_mode': 'graph', 'capture_id': 1, 'replay_count': 1,
            'full_chunk_shape': [1, 50, 32], 'full_chunk_finite': True},
        'projection_shapes': [], 'vision_encodes': 1, 'capture_created': False,
        'owner_thread': 12, 'input_observation_index': 20, 'inputs': inputs,
        'noise': inputs[7], 'full': torch.zeros(1, 50, 32), 'original': torch.zeros(50, 7)}
    audit.graph_output(out, req, record)
    out['projection_shapes'] = [[1, 50, 32]]*10
    with pytest.raises(ValueError, match='confused'):
        audit.graph_output(out, req, record)
    out['projection_shapes'], out['capture_created'] = [], True
    with pytest.raises(ValueError, match='Control-time'):
        audit.graph_output(out, req, record)
    out['capture_created'] = False
    out['inputs'][0] = torch.zeros(1, 3, 3)
    with pytest.raises(ValueError, match='signature'):
        audit.graph_output(out, req, record)
    assert not torch.cuda.is_initialized()


class FakeLivePath:
    def __init__(self, policy, pre, post, mode, task):
        self.owner, self.receipt = None, None
        self.task = task
        self.runtime = SimpleNamespace(graph=None, captures=[], rgb_encodings=0)
        self.calls = 0

    def activate(self):
        if self.owner is None:
            self.owner = threading.get_ident()
        assert self.owner == threading.get_ident()

    def __call__(self, item):
        self.activate()
        began = time.perf_counter()
        time.sleep(.065)
        self.calls += 1
        inputs = (torch.zeros(1, 2, 3), torch.zeros(1, 2, 3), torch.ones(1, 2, dtype=torch.bool),
                  torch.ones(1, 2, dtype=torch.bool), torch.ones(1, 4, dtype=torch.long),
                  torch.ones(1, 4, dtype=torch.bool), torch.zeros(1, 32), item['noise'])
        if self.runtime.graph is None:
            self.runtime.graph = object()
            self.runtime.captures = [{'status': 'captured', 'owner_thread': self.owner,
                'eager_setup_calls': 1, 'side_stream_warmup_calls': 3, 'capture_calls': 1,
                'projection_shapes': [[1, 50, 32]]*10, 'input_shapes': [list(v.shape) for v in inputs],
                'preparation_seconds': .01}]
        self.runtime.rgb_encodings += 1
        original = (torch.arange(50, dtype=torch.float32)+100*self.calls)[:, None].repeat(1, 7)
        full = torch.zeros(1, 50, 32)
        full[0, :, :7] = original
        return {'full': full, 'original': original, 'processed': original*10, 'inputs': inputs,
                'owner_thread': self.owner, 'model_started_at': began, 'model_completed_at': time.perf_counter(),
                'sampler': {'sampler_mode': 'graph', 'capture_id': 1, 'replay_count': 1,
                            'full_chunk_shape': [1, 50, 32], 'full_chunk_finite': True}}

    def close(self):
        assert threading.get_ident() == self.owner
        self.receipt = {'mode': 'graph', 'owner': self.owner, 'requests': self.calls,
            'rgb_encodings': self.calls, 'noise_draws': 0, 'graph_released': True,
            'sampler_restored': True, 'captures': self.runtime.captures}
        self.runtime.graph = None


class FakeSession:
    def __init__(self, spec, budget, calls, factory):
        self.budget = budget
        self.native_steps = []
        self.returned = {'settling': 0, 'measurement': 0}

    def create_reset(self):
        for _ in range(10):
            self.budget.take('settling')
            self.native_steps.append({'segment': 'settling'})
        self.returned['settling'] = 10
        return {}

    def step(self, dispatch):
        self.budget.take('measurement')
        began = time.perf_counter()
        time.sleep(.002)
        self.native_steps.append({'segment': 'measurement', 'action': dispatch['command'].tolist(),
            'action_index': dispatch['action_index'], 'observation_index': dispatch['observation_index'],
            'started_at': began, 'returned_at': time.perf_counter()})
        self.returned['measurement'] += 1
        return {}, 0., False, False, {'is_success': False}

    def close(self):
        return True


@pytest.mark.parametrize('arm', ['serialized', 'async'])
def test_injected_predictor_shared_episode_and_graph_audit(tmp_path, monkeypatch, arm):
    monkeypatch.setattr(r, 'FullPath', FakeLivePath)
    monkeypatch.setattr(r.GraphFeedbackPredictor, 'fresh_noise', lambda _: torch.zeros(1, 50, 32))
    monkeypatch.setattr(r.e, 'NativeSession', FakeSession)
    def observation(raw, language, index, returned):
        image = torch.full((2, 2, 3), index, dtype=torch.uint8)
        record = {'index': index, 'returned_at': returned, 'state': torch.zeros(1, 8),
            'eef_quaternion_xyzw': torch.zeros(4), 'raw_eef_position': torch.zeros(3),
            'raw_gripper_qpos': torch.zeros(2), 'raw_pixels': {'image': image, 'image2': image.clone()},
            'worker_observation': {'image': image, 'state_0': float(index)}}
        return None, None, record
    monkeypatch.setattr(r.e, 'observation', observation)
    spec = copy.deepcopy(r.manifest()['rows'][0])
    spec['condition'], spec['limits']['measurement'] = arm, 24
    budget, calls = r.old.Budget(), r.e.Calls(tmp_path/'calls.jsonl')
    result, initial = r.old.episode(spec, tmp_path, None, None, None, None, budget, calls, None,
        predictor_factory=r.GraphFeedbackPredictor, request_kind='graph_request')
    calls.close()
    assert result['status'] == 'completed', result['first_failure']
    arrays = torch.load(tmp_path/'episode_000/arrays.pt', map_location='cpu', weights_only=False)
    checked = audit.inspect_episode(spec, result, arrays, initial)
    assert checked['actions'] == 24 and checked['request_count'] >= 2
    assert result['predictor_cleanup']['requests'] == checked['request_count']
    assert len(result['predictor_cleanup']['captures']) == 1
    tampered = copy.deepcopy(arrays)
    tampered['outputs'][2]['input_fingerprint'] = '0'*64
    with pytest.raises(ValueError, match='fingerprint'):
        audit.inspect_episode(spec, result, tampered, initial)
    assert not torch.cuda.is_initialized()
