"""E-GFB1: Graph sampler with live Env feedback, shared observation-origin control.

Both arms use Graph. No replay commands, archived observations, RTC or learning.
"""

import argparse
import faulthandler
import hashlib
import json
import signal
import subprocess
import sys
import threading
import time
import traceback
from pathlib import Path

import libero_observation_async as old
import numpy as np
import torch
from libero_graph_replay import FullPath

q, e, pilot, REPO = old.q, old.e, old.pilot, old.REPO
check, write, digest = old.check, old.write, old.digest
BASELINE = REPO / 'outputs/smolvla_graph_feedback_development_20260917/baseline.json'
PREREQUISITE = REPO / 'outputs/smolvla_graph_replay_b293f661'
PAIRS = ((0, 10), (2, 10), (6, 10), (7, 10), (7, 11), (6, 11), (2, 11), (0, 11))
LIMITS = old.LIMITS


def paths(head):
    return (REPO / f'outputs/smolvla_graph_feedback_preparation_{head[:8]}',
            REPO / f'outputs/smolvla_graph_feedback_{head[:8]}')


def manifest():
    rows = []
    for pair, (task, state) in enumerate(PAIRS):
        for arm in (('serialized', 'async') if pair % 2 == 0 else ('async', 'serialized')):
            rows.append({'ordinal': len(rows), 'pair_index': pair, 'task_id': task,
                'task_name': e.reference.TASK_NAMES[task], 'initial_state_id': state, 'condition': arm,
                'environment_seed': 1160000 + 100*task + state,
                'policy_seed': 1170000 + 100*task + state, 'ready_wall_slots': 1200,
                'fps': 20, 'threshold': 30, 'max_delay': 8, 'guard': 2, 'margin': 1,
                'latency_window': 50, 'latency_quantile': .9,
                'limits': {k: v[0] for k, v in LIMITS.items()}})
    return {'experiment': 'E-GFB1', 'rows': rows, 'chunk': 50, 'denoising_steps': 10,
        'sampler': 'same_graph_both_arms', 'rtc': False, 'context': 'live_current_visual',
        'origin': 'observation_index', 'trim': 'actual_consumed_once',
        'initial_expected_delay': 7, 'captures': 16, 'captures_per_episode': 1,
        'capture_scope': 'bootstrap_only_before_control',
        'internal_per_capture': {'eager_setup': 1, 'warmup': 3, 'capture': 1},
        'latency_limit_s': .35, 'global_limits': {k: v[1] for k, v in LIMITS.items()},
        'soft': old.SOFT, 'hard': old.HARD, 'qualification_reads': 0, 'training': 0,
        'replay_commands': False, 'statistical_noninferiority_claimed': False}


def source_hashes():
    values = old.sources()
    for p in [BASELINE, PREREQUISITE / 'result.json', PREREQUISITE / 'independent_audit.json',
              REPO / 'tests/test_libero_graph_feedback.py',
              REPO / 'docs/experiments/SMOLVLA_GRAPH_FEEDBACK_PLAN.md']:
        values[str(p)] = digest(p)
    return values


def tree_gate(head):
    check(subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True, cwd=REPO).strip() == head,
          'Execution HEAD changed')
    check(old.pending_state() == json.loads(BASELINE.read_text())['pending'], 'Original pending changed')


def assert_unused(history):
    conflicts = set(PAIRS) & {(v['task'], v['state']) for v in history}
    check(not conflicts, f'Previously used identities; no replacements: {sorted(conflicts)}')


def prepare(head):
    tree_gate(head)
    env = q.runtime_environment()
    prior = json.loads((PREREQUISITE / 'independent_audit.json').read_text())
    check(prior['independent_contract_accepted'] and prior['fixed_output_exact']
          and prior['graph_concurrent_budget_passed'], 'Graph prerequisite not accepted')
    history = q.history_inventory(REPO / 'outputs')
    assert_unused(history)
    prep, out = paths(head)
    check(not prep.exists() and not out.exists(), 'Unique preparation/output required')
    check(not torch.cuda.is_initialized(), 'CPU preparation initialized CUDA')
    prep.mkdir()
    write(prep / 'preparation.json', {'head': head, 'manifest': manifest(), 'sources': source_hashes(),
        'environment': env, 'history': history, 'cuda_initialized': False})
    print(json.dumps({'prepared': True, 'prep': str(prep), 'out': str(out),
        'sha256': digest(prep / 'preparation.json'), 'historical_identities': len(history)}), flush=True)


def validate(head, after=False):
    tree_gate(head)
    prep, out = paths(head)
    data = json.loads((prep / 'preparation.json').read_text())
    check(data['head'] == head and data['manifest'] == manifest(), 'Manifest differs')
    check(data['sources'] == source_hashes() and data['environment'] == q.runtime_environment(), 'Source/environment changed')
    history = q.history_inventory(REPO / 'outputs')
    if after:
        history = [v for v in history if not v['path'].startswith(out.name + '/')]
    assert_unused(history)
    check(history == data['history'], 'History changed or concurrent collection')
    body = (prep / 'registration.md').read_text()
    got = json.loads((prep / 'registration_readback.json').read_text())
    check(type(got.get('id')) is int and got.get('body') == body and got.get('issue_url') ==
        'https://api.github.com/repos/Lebron-233/lerobot/issues/1' and all(x in body for x in
        (f'E-GFB1-REGISTER:{head}', str(out), digest(prep / 'preparation.json'))), 'Registration differs')
    return data


def observation_fingerprint(observation):
    """Bind the actual worker input pixels and state, not an archive file identifier."""
    h = hashlib.sha256()
    for k, value in sorted(observation['worker_observation'].items()):
        h.update(k.encode())
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu().numpy()
        if isinstance(value, np.ndarray):
            h.update(str((value.dtype.str, value.shape)).encode())
            h.update(value.tobytes(order='C'))
        else:
            h.update(json.dumps(value, allow_nan=False).encode())
    return h.hexdigest()


def require_ready(runtime, first):
    if first:
        check(runtime.graph is None and not runtime.captures, 'Unexpected preexisting Graph')
    else:
        check(runtime.graph is not None and len(runtime.captures) == 1
              and runtime.captures[0]['status'] == 'captured', 'Control-time capture is forbidden')


class GraphFeedbackPredictor:
    """Reuse the accepted full path; only the inputs now come from live feedback."""

    def __init__(self, policy, pre, post, spec):
        self.spec = spec
        self.path = FullPath(policy, pre, post, 'graph', f"episode_{spec['ordinal']}")
        self.generator = None
        self.requests = 0
        self.receipt = None

    def fresh_noise(self):
        if self.generator is None:
            self.generator = torch.Generator(device='cuda').manual_seed(self.spec['policy_seed'])
        return torch.randn((1, 50, 32), generator=self.generator, device='cuda')

    def __call__(self, observation):
        self.path.activate()
        rt = self.path.runtime
        require_ready(rt, self.requests == 0)
        encoded = rt.rgb_encodings
        noise = self.fresh_noise()
        fingerprint = observation_fingerprint(observation)
        item = {'observation': observation, 'noise': noise,
                'key': [self.spec['task_id'], self.spec['initial_state_id'], observation['index']],
                'language': self.spec['task_name'].replace('_', ' ')}
        out = self.path(item)
        check(len(rt.captures) == 1 and rt.captures[0]['status'] == 'captured', 'Unexpected Graph capture')
        check(rt.rgb_encodings-encoded == 1, 'Each request must encode live RGB exactly once')
        check(out['sampler']['sampler_mode'] == 'graph' and out['sampler']['replay_count'] == 1,
              'Graph replay evidence missing')
        self.requests += 1
        return {**out, 'noise': noise.detach().cpu().clone(),
            'model_returned_at': out['model_completed_at'], 'vision_encodes': 1,
            'projection_shapes': [], 'capture_created': self.requests == 1,
            'input_observation_index': observation['index'], 'input_fingerprint': fingerprint}

    def close(self):
        self.path.close()
        self.receipt = self.path.receipt
        if self.receipt is not None:
            self.receipt.update(live_requests=self.requests, explicit_noise_draws=self.requests,
                                close_thread=threading.get_ident(), closed_at=time.perf_counter())


def summarize(records):
    pairs = []
    for pair, identity in enumerate(PAIRS):
        arms = {v['spec']['condition']: v for v in records if v['spec']['pair_index'] == pair}
        if set(arms) != {'serialized', 'async'} or any(v['status'] != 'completed' for v in arms.values()):
            continue
        pairs.append({'pair_index': pair, 'task_state': list(identity),
            'arms': {a: {k: v[k] for k in ('success', 'measured_actions', 'wall_s', 'no_action_slots',
                'underflows', 'terminal_reason', 'startup_s')} for a, v in arms.items()}})
    return {'pairs': pairs, 'complete_pairs': len(pairs), 'statistical_noninferiority_claimed': False}


def worker(args):
    torch.set_num_threads(1)
    faulthandler.register(signal.SIGUSR1, file=sys.stderr, all_threads=True)
    calls, budget, records = e.Calls(args.output / 'calls.jsonl'), old.Budget(), []
    result = {'experiment': 'E-GFB1', 'execution_head': args.execution_head,
              'status': 'technical_failure', 'first_failure': None}
    try:
        prepared = validate(args.execution_head)
        with pilot.phase(args.output, 'environment_preflight', 30):
            factory = e.reference.make_native_env_factory()
            check(not torch.cuda.is_initialized(), 'Factory preflight initialized CUDA')
        with pilot.phase(args.output, 'load_model', 90):
            check(torch.cuda.get_device_name() == 'NVIDIA GeForce RTX 4070 Ti SUPER', 'GPU changed')
            policy, pre, post, report = e.load_runtime(e.POLICY, e.VLM)
            check(not policy.config.compile_model and policy.config.rtc_config is None, 'Use frozen no-RTC')
            result['vla_loads'] = 1
            write(args.output / 'policy_load.json', report)
        initials, boots = {}, {}
        for spec in manifest()['rows']:
            with pilot.phase(args.output, f"episode_{spec['ordinal']}", 100):
                print(f"E-GFB1 START {spec['ordinal']} {spec['condition']} task={spec['task_id']} state={spec['initial_state_id']}", flush=True)
                record, initial = old.episode(spec, args.output, policy, pre, post, factory, budget, calls,
                    initials.get(spec['pair_index']), predictor_factory=GraphFeedbackPredictor,
                    request_kind='graph_request')
                records.append(record)
                check(record['status'] == 'completed', record['first_failure'] or 'Episode incomplete')
                # Independent audit also rechecks bootstrap arrays; no extra model call here.
                folder = args.output / f"episode_{spec['ordinal']:03d}"
                arrays = torch.load(folder / 'arrays.pt', map_location='cpu', weights_only=False)
                boot = arrays['outputs'][1]
                pair = spec['pair_index']
                if pair in boots:
                    pilot.require_equal(boots[pair]['full'], boot['full'], 'Bootstrap full output differs')
                    pilot.require_equal(boots[pair]['noise'], boot['noise'], 'Bootstrap noise differs')
                else:
                    initials[pair], boots[pair] = initial, {'full': boot['full'], 'noise': boot['noise']}
                del arrays
                print(f"E-GFB1 END {spec['ordinal']} success={record['success']} actions={record['measured_actions']}", flush=True)
        check(all(not p.requires_grad and p.grad is None for p in policy.parameters()), 'VLA not frozen')
        tree_gate(args.execution_head)
        check(source_hashes() == prepared['sources'], 'Sources changed')
        result.update(status='completed', vla_frozen=True, **summarize(records))
    except BaseException:
        result['first_failure'] = traceback.format_exc()
    finally:
        calls.close()
        captures = [c for v in records for c in (v.get('predictor_cleanup') or {}).get('captures', [])]
        result.update(episodes_completed=sum(v['status'] == 'completed' for v in records),
            native_budget=dict(budget.total), graph_captures=len(captures),
            capture_internal={'setup': sum(c['eager_setup_calls'] for c in captures),
                'warmup': sum(c['side_stream_warmup_calls'] for c in captures),
                'capture': sum(c['capture_calls'] for c in captures)},
            attempts=1, retries=0, training_updates=0, qualification_reads=0,
            rtc_steps=0, predictor_forwards=0, real_robot=0, replay_commands=False,
            realtime_qualified=False, risk_thresholds=None)
        write(args.output / 'worker_result.json', result)
    return 0 if result['status'] == 'completed' else 2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execution-head', required=True)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--prepare', action='store_true')
    parser.add_argument('--worker', action='store_true')
    args = parser.parse_args()
    signal.signal(signal.SIGTERM, e.request_shutdown)
    if args.prepare:
        check(not args.worker and args.output is None, 'Preparation cannot launch worker')
        prepare(args.execution_head)
        return 0
    args.output = (args.output or paths(args.execution_head)[1]).resolve()
    return worker(args) if args.worker else old.supervise(args, validate_run=validate, paths_for=paths,
        manifest_for=manifest, worker_script=str(Path(__file__).resolve()), experiment='E-GFB1')


if __name__ == '__main__':
    raise SystemExit(main())
