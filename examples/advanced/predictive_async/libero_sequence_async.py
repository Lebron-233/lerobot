"""E-SEQ1: all-episode live three-arm development test of plan-order takeover.

The old candidate remains frozen. Both asynchronous arms use its unmodified
control_loop; only the versioned candidate queue starts new plans at row zero.
"""

import argparse
import faulthandler
import json
import math
import signal
import subprocess
import sys
import threading
import time
import traceback
from datetime import UTC, datetime
from pathlib import Path

import libero_graph_feedback as g
import torch
from sequence_execution_queue import SequenceExecutionQueue

old, q, e, pilot, REPO = g.old, g.q, g.e, g.pilot, g.REPO
check, write, digest = g.check, g.write, g.digest
BASE_HEAD = 'ff21ccb777724796fbf3dfdb26ff9658a150a501'
LOCK = REPO/'docs/experiments/SMOLVLA_GRAPH_CANDIDATE_LOCK.json'
PRIOR = REPO/'outputs/smolvla_prefix_skip_757171ea'
PAIRS = ((0, 10), (2, 10), (6, 10), (7, 10), (7, 11), (6, 11), (2, 11), (0, 11), (7, 18), (7, 19))
ARMS = ('serialized', 'aligned_async', 'sequence_async')
LIMITS = {'episodes': (1, 30), 'settling': (10, 300), 'measurement': (280, 8400), 'model': (160, 4800)}
ADDITIONS = ('examples/advanced/predictive_async/sequence_execution_queue.py',
             'examples/advanced/predictive_async/libero_sequence_async.py',
             'examples/advanced/predictive_async/audit_libero_sequence_async.py')


def paths(head):
    return (REPO/f'outputs/smolvla_sequence_async_preparation_{head[:8]}',
            REPO/f'outputs/smolvla_sequence_async_{head[:8]}')


def load(path):
    return torch.load(path, map_location='cpu', weights_only=False)


def manifest():
    template = g.manifest()['rows'][0]
    rows = []
    for pair, (task, state) in enumerate(PAIRS):
        shift = pair % 3
        for arm in ARMS[shift:]+ARMS[:shift]:
            rows.append({**template, 'ordinal': len(rows), 'pair_index': pair,
                'task_id': task, 'task_name': e.reference.TASK_NAMES[task], 'initial_state_id': state,
                'condition': 'serialized' if arm == 'serialized' else 'async', 'arm': arm,
                'cohort': 'prior_pilot_development' if pair < 8 else 'known_diagnostic',
                'environment_seed': 1160000+100*task+state, 'policy_seed': 1170000+100*task+state,
                'limits': {k: v[0] for k, v in LIMITS.items()},
                'queue_semantics': 'plan_order_at_takeover' if arm == 'sequence_async' else 'observation_origin'})
    return {**g.manifest(), 'experiment': 'E-SEQ1', 'rows': rows, 'captures': 30,
        'global_limits': {k: v[1] for k, v in LIMITS.items()}, 'base_head': BASE_HEAD,
        'new_candidate': 'sequence_execution_queue_v1', 'known_development_identities': True,
        'candidate_control_loop': 'unchanged old.control_loop', 'forced_wait_in_async': False,
        'new_qualification_claimed': False, 'all_nonbootstrap_latency_limit_s': .35,
        'gate': 'no lost successes vs both controls; budget all arms; async overlap/wait benefit; no queue faults',
        'cadence': 'unchanged qsize rule; actual request indices may differ between queue semantics'}


def frozen_gate():
    lock = json.loads(LOCK.read_text())
    check(all(digest(REPO/p) == h for p, h in lock['core_sha256'].items()), 'Frozen core changed')
    changed = subprocess.check_output(['git', 'diff', '--name-only', BASE_HEAD, '--',
        'src/lerobot', 'examples/advanced/predictive_async'], cwd=REPO, text=True).splitlines()
    check(set(changed) <= set(ADDITIONS), f'Unexpected old source mutation: {changed}')


def sources():
    values = g.source_hashes()
    files = [LOCK, PRIOR/'result.json', PRIOR/'independent_audit.json',
        REPO/'docs/experiments/SMOLVLA_SEQUENCE_ASYNC_PLAN.md',
        REPO/'tests/test_libero_sequence_async.py', *(REPO/p for p in ADDITIONS)]
    values.update({str(p): digest(p) for p in files})
    return values


def tree_gate(head):
    check(subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip() == head,
          'Execution HEAD changed')
    check(old.pending_state() == json.loads(g.BASELINE.read_text())['pending'], 'Old pending changed')
    frozen_gate()


def prepare(head):
    tree_gate(head)
    environment = q.runtime_environment()
    prior = json.loads((PRIOR/'independent_audit.json').read_text())
    check(prior['independent_contract_accepted'] and prior['complete_anchor_trajectories_exact'] == 4,
          'Calibrated prefix diagnostic prerequisite missing')
    history = q.history_inventory(REPO/'outputs')
    check(set(PAIRS) <= {(v['task'], v['state']) for v in history}, 'Expected known development identities missing')
    prep, out = paths(head)
    check(not prep.exists() and not out.exists(), 'Output/preparation already exists')
    check(not torch.cuda.is_initialized(), 'Prepare initialized CUDA')
    prep.mkdir()
    write(prep/'preparation.json', {'head': head, 'manifest': manifest(), 'sources': sources(),
        'environment': environment, 'history': history})
    print(json.dumps({'prepared': True, 'prep': str(prep), 'out': str(out),
        'sha256': digest(prep/'preparation.json'), 'episodes': 30, 'new_qualification': False}), flush=True)


def validate(head, after=False):
    tree_gate(head)
    prep, out = paths(head)
    saved = json.loads((prep/'preparation.json').read_text())
    check(saved['head'] == head and saved['manifest'] == manifest(), 'Specification changed')
    check(saved['sources'] == sources() and saved['environment'] == q.runtime_environment(), 'Source/environment changed')
    history = q.history_inventory(REPO/'outputs')
    if after:
        history = [v for v in history if not v['path'].startswith(out.name+'/')]
    check(history == saved['history'], 'Concurrent collection/history change')
    body = (prep/'registration.md').read_text()
    got = json.loads((prep/'registration_readback.json').read_text())
    check(type(got.get('id')) is int and got.get('body') == body and got.get('issue_url') ==
        'https://api.github.com/repos/Lebron-233/lerobot/issues/1' and all(x in body for x in
        (f'E-SEQ1-REGISTER:{head}', str(out), digest(prep/'preparation.json'))), 'Actual registration differs')
    return saved


class Owner(old.InferenceOwner):
    def __init__(self, predict, calls, spec, budget, **kwargs):
        super().__init__(predict, calls, spec, budget, **kwargs)
        # No task has been submitted; the sole owner is waiting on an empty queue.
        if spec['arm'] == 'sequence_async':
            self.queue = SequenceExecutionQueue(max_delay=8)


class Budget(e.Budget):
    def check(self, kind):
        local, total = LIMITS[kind]
        check(self.episode[kind] < local and self.total[kind] < total, f'E-SEQ1 {kind} budget exhausted')


def episode(spec, output, policy, pre, post, factory, budget, calls, paired=None):
    folder = output/f"episode_{spec['ordinal']:03d}"
    folder.mkdir()
    write(folder/'started.json', {'spec': spec, 'started_at_utc': datetime.now(UTC).isoformat(),
                                'purpose': 'three-arm live development, not independent confirmation'})
    budget.begin_episode()
    native = e.NativeSession(spec, budget, calls, factory)
    owner, predict, observations, control, initial = None, None, [], None, None
    result = {'spec': spec, 'status': 'technical_failure', 'first_failure': None,
        'worker_joined': False, 'environment_closed': False, 'initial_pair_exact': None,
        'controller_thread': threading.get_ident()}
    try:
        def observe(raw, index, returned):
            _, _, row = e.observation(raw, spec['task_name'].replace('_', ' '), index, returned)
            observations.append(row)
            return row
        initial = observe(native.create_reset(), 0, time.perf_counter())
        torch.save(initial, folder/'initial_checkpoint.pt')
        if paired is not None:
            check(e.initial_difference(initial, paired) is None, 'Paired initial observation differs')
            result['initial_pair_exact'] = True
        predict = g.GraphFeedbackPredictor(policy, pre, post, spec)
        owner = Owner(predict, calls, spec, budget, on_owner_close=predict.close, request_kind='graph_request')
        began = time.perf_counter()
        owner.submit(initial, 0)
        owner.receive(block=True)
        result['startup_s'] = time.perf_counter()-began
        control = old.control_loop(owner, native, spec, initial, observe)
        result.update(status='completed', success=control['success'], terminal_reason=control['terminal_reason'])
    except BaseException:
        result['first_failure'] = traceback.format_exc()
    finally:
        if owner is not None:
            control = getattr(owner, 'control_result', control)
            try:
                owner.close()
                result['worker_joined'] = True
            except BaseException:
                result['first_failure'] = result['first_failure'] or traceback.format_exc()
                result['status'] = 'technical_failure'
        if owner is None or result['worker_joined']:
            try:
                result['environment_closed'] = native.close()
            except BaseException:
                result['first_failure'] = result['first_failure'] or traceback.format_exc()
                result['status'] = 'technical_failure'
            if owner is not None:
                result.update(requests=owner.rows, owner_cleanup=owner.cleanup_evidence,
                              predictor_cleanup=predict.receipt)
                torch.save({'observations': observations, 'control': control, 'outputs': owner.outputs}, folder/'arrays.pt')
        if not result['worker_joined'] or not result['environment_closed']:
            result['status'] = 'technical_failure'
        result.update(budget=dict(budget.episode), native_steps=native.native_steps, native_returned=dict(native.returned))
        if control is not None:
            wall = control['ended_at']-control['t0']
            count = min(spec['ready_wall_slots'], max(1, math.ceil(wall*20)))
            result.update(wall_s=wall, measured_actions=len(control['dispatches']), wall_slots=count,
                no_action_slots=count-len(control['dispatches']),
                underflows=sum(v['outcome'] == 'underflow' for v in control['gets']))
        write(folder/'result.json', result)
    return result, initial


def worker(args):
    torch.set_num_threads(1)
    faulthandler.register(signal.SIGUSR1, file=sys.stderr, all_threads=True)
    calls, budget, records = e.Calls(args.output/'calls.jsonl'), Budget(), []
    result = {'experiment': 'E-SEQ1', 'execution_head': args.execution_head,
              'status': 'technical_failure', 'first_failure': None}
    try:
        prepared = validate(args.execution_head)
        with pilot.phase(args.output, 'environment_preflight', 30):
            factory = e.reference.make_native_env_factory()
            check(not torch.cuda.is_initialized(), 'Factory initialized CUDA')
        with pilot.phase(args.output, 'load_model', 90):
            check(torch.cuda.get_device_name() == 'NVIDIA GeForce RTX 4070 Ti SUPER', 'GPU changed')
            policy, pre, post, report = e.load_runtime(e.POLICY, e.VLM)
            check(not policy.config.compile_model and policy.config.rtc_config is None, 'Model configuration changed')
            write(args.output/'policy_load.json', report)
            result['vla_loads'] = 1
        initials, boots = {}, {}
        for spec in manifest()['rows']:
            with pilot.phase(args.output, f"episode_{spec['ordinal']}", 100):
                print(f"E-SEQ1 START {spec['ordinal']}/30 {spec['arm']} task={spec['task_id']} state={spec['initial_state_id']}", flush=True)
                rec, initial = episode(spec, args.output, policy, pre, post, factory, budget, calls,
                                       initials.get(spec['pair_index']))
                records.append(rec)
                check(rec['status'] == 'completed', rec['first_failure'] or 'Incomplete episode')
                arrays = load(args.output/f"episode_{spec['ordinal']:03d}/arrays.pt")
                boot, pair = arrays['outputs'][1], spec['pair_index']
                if pair in boots:
                    pilot.require_equal(boots[pair]['full'], boot['full'], 'Bootstrap full output differs')
                    pilot.require_equal(boots[pair]['noise'], boot['noise'], 'Bootstrap noise differs')
                else:
                    initials[pair], boots[pair] = initial, {'full': boot['full'], 'noise': boot['noise']}
                del arrays
                print(f"E-SEQ1 END {spec['ordinal']} success={rec['success']} actions={rec['measured_actions']}", flush=True)
        check(all(not p.requires_grad and p.grad is None for p in policy.parameters()), 'VLA not frozen')
        tree_gate(args.execution_head)
        check(prepared['sources'] == sources(), 'Sources changed during run')
        result.update(status='completed', frozen_baseline_unchanged=True, vla_frozen=True)
    except BaseException:
        result['first_failure'] = traceback.format_exc()
    finally:
        calls.close()
        captures = [c for v in records for c in (v.get('predictor_cleanup') or {}).get('captures', [])]
        result.update(episodes_completed=sum(v['status'] == 'completed' for v in records),
            outcomes=[{k: v.get(k) for k in ('spec', 'success', 'measured_actions', 'terminal_reason')} for v in records],
            native_budget=dict(budget.total), graph_captures=len(captures),
            capture_internal={'setup': sum(c['eager_setup_calls'] for c in captures),
                'warmup': sum(c['side_stream_warmup_calls'] for c in captures), 'capture': sum(c['capture_calls'] for c in captures)},
            attempts=1, retries=0, training_updates=0, rtc_steps=0, real_robot=0,
            new_qualification_claimed=False, realtime_qualified=False, confirmation_result_unchanged=True)
        write(args.output/'worker_result.json', result)
    return 0 if result['status'] == 'completed' else 2


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--execution-head', required=True)
    p.add_argument('--output', type=Path)
    p.add_argument('--prepare', action='store_true')
    p.add_argument('--worker', action='store_true')
    args = p.parse_args()
    signal.signal(signal.SIGTERM, e.request_shutdown)
    if args.prepare:
        check(not args.worker and args.output is None, 'Prepare cannot launch worker')
        prepare(args.execution_head)
        return 0
    args.output = (args.output or paths(args.execution_head)[1]).resolve()
    return worker(args) if args.worker else old.supervise(args, validate_run=validate, paths_for=paths,
        manifest_for=manifest, worker_script=str(Path(__file__).resolve()), experiment='E-SEQ1')


if __name__ == '__main__':
    raise SystemExit(main())
