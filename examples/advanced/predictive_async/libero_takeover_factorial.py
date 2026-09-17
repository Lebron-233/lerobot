"""E-TFD1: eight known-identity counterfactuals; frozen model, imposed takeover delays.

Diagnostic only: neither a new candidate nor a continuation of E-GFC1 confirmation.
The simulator may wait after the prescribed old actions; wall speed is not a goal.
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

old, q, e, pilot, REPO = g.old, g.q, g.e, g.pilot, g.REPO
check, write, digest = g.check, g.write, g.digest
BASELINE = REPO / 'outputs/smolvla_takeover_factorial_development_20260917/baseline.json'
SOURCE = REPO / 'outputs/smolvla_graph_confirmation_fc5da9d2'
LOCK = REPO / 'docs/experiments/SMOLVLA_GRAPH_CANDIDATE_LOCK.json'
REFERENCES = {(7, 19): {0: 56, 2: 57}, (7, 18): {0: 55, 2: 54}}
LIMITS = {'episodes': (1, 8), 'settling': (10, 80), 'measurement': (280, 2240), 'model': (160, 1280)}
ADDITIONS = ('examples/advanced/predictive_async/libero_takeover_factorial.py',
             'examples/advanced/predictive_async/audit_libero_takeover_factorial.py')


def paths(head):
    return (REPO / f'outputs/smolvla_takeover_factorial_preparation_{head[:8]}',
            REPO / f'outputs/smolvla_takeover_factorial_{head[:8]}')


def load(path):
    return torch.load(path, map_location='cpu', weights_only=False)


def manifest():
    template = g.manifest()['rows'][0]
    rows = []
    orders = (((0, 0), (2, 2), (0, 2), (2, 0)), ((2, 2), (0, 0), (2, 0), (0, 2)))
    for pair, ((task, state), order) in enumerate(zip(REFERENCES, orders, strict=True)):
        for first, later in order:
            rows.append({**template, 'ordinal': len(rows), 'pair_index': pair,
                'task_id': task, 'task_name': e.reference.TASK_NAMES[task], 'initial_state_id': state,
                'condition': 'counterfactual', 'arm': f'first{first}_later{later}',
                'first_delay': first, 'later_delay': later,
                'environment_seed': 1160000+100*task+state, 'policy_seed': 1170000+100*task+state,
                'limits': {k: v[0] for k, v in LIMITS.items()},
                'reference_ordinal': REFERENCES[(task, state)][first] if first == later else None})
    return {'experiment': 'E-TFD1', 'rows': rows, 'known_diagnostic_identities': [list(k) for k in REFERENCES],
        'candidate_head': 'a5e9fddefc0cfa03ea1509493e8b304ec7c06d92',
        'intervention': 'request2 versus request3plus: exactly zero or two old native actions before installation',
        'timing_scope': 'controlled consumed-action schedule, not an online speed qualification',
        'new_qualification': False, 'training': 0, 'rtc': False, 'captures': 8,
        'global_limits': {k: v[1] for k, v in LIMITS.items()}, 'soft': old.SOFT, 'hard': old.HARD,
        'anchor_requirement': 'four full trajectories, inputs, outputs and commands exact to archived originals',
        'normal_failures_do_not_stop': True, 'attempts': 1, 'retries': 0}


def delay_for(spec, request_id):
    check(type(request_id) is int and request_id >= 1, 'Invalid request ID')
    return 0 if request_id == 1 else spec['first_delay'] if request_id == 2 else spec['later_delay']


def frozen_gate():
    lock = json.loads(LOCK.read_text())
    check(all(digest(REPO/p) == h for p, h in lock['core_sha256'].items()), 'Frozen core changed')
    changed = subprocess.check_output(['git', 'diff', '--name-only', 'd3f7dae74757fe56e8224616a0db4ba8bc9da952',
        '--', 'src/lerobot', 'examples/advanced/predictive_async'], cwd=REPO, text=True).splitlines()
    check(set(changed) <= set(ADDITIONS), f'Only diagnostic additions allowed: {changed}')


def sources():
    values = g.source_hashes()
    files = [BASELINE, LOCK, SOURCE/'result.json', SOURCE/'independent_audit.json',
             REPO/'docs/experiments/SMOLVLA_TAKEOVER_FACTORIAL_PLAN.md',
             REPO/'tests/test_libero_takeover_factorial.py', *(REPO/p for p in ADDITIONS)]
    for ordinal in (54, 55, 56, 57):
        folder = SOURCE/f'episode_{ordinal:03d}'
        files += [folder/'result.json', folder/'arrays.pt', folder/'initial_checkpoint.pt']
    values.update({str(p): digest(p) for p in files})
    return values


def tree_gate(head):
    check(subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip() == head,
          'Execution HEAD changed')
    check(old.pending_state() == json.loads(BASELINE.read_text())['pending'], 'Original pending changed')
    frozen_gate()


def prepare(head):
    tree_gate(head)
    environment = q.runtime_environment()
    prior = json.loads((SOURCE/'independent_audit.json').read_text())
    check(prior['independent_contract_accepted'] and not prior['bounded_confirmation_passed'],
          'Prior confirmation status changed')
    # This is deliberately a known-identity diagnostic, not a fresh identity gate.
    for identity, refs in REFERENCES.items():
        for delay, ordinal in refs.items():
            saved = json.loads((SOURCE/f'episode_{ordinal:03d}/result.json').read_text())
            check((saved['spec']['task_id'], saved['spec']['initial_state_id']) == identity, 'Reference identity')
            for req in saved['requests'][1:]:
                if req['decision']['accepted']:
                    check(req['decision']['actual_delay'] == delay, 'Archived delay is not the fixed diagnostic schedule')
    prep, out = paths(head)
    check(not prep.exists() and not out.exists(), 'Unique preparation/output required')
    check(not torch.cuda.is_initialized(), 'CPU preparation initialized CUDA')
    prep.mkdir()
    write(prep/'preparation.json', {'head': head, 'manifest': manifest(), 'sources': sources(),
        'environment': environment, 'history': q.history_inventory(REPO/'outputs'), 'cuda_initialized': False})
    print(json.dumps({'prepared': True, 'prep': str(prep), 'out': str(out),
        'sha256': digest(prep/'preparation.json'), 'episodes': 8, 'new_qualification': False}), flush=True)


def validate(head, after=False):
    tree_gate(head)
    prep, out = paths(head)
    saved = json.loads((prep/'preparation.json').read_text())
    check(saved['head'] == head and saved['manifest'] == manifest(), 'Diagnostic specification changed')
    check(saved['sources'] == sources() and saved['environment'] == q.runtime_environment(), 'Source/environment changed')
    history = q.history_inventory(REPO/'outputs')
    if after:
        history = [v for v in history if not v['path'].startswith(out.name+'/')]
    check(saved['history'] == history, 'Other experiment started or history changed')
    body = (prep/'registration.md').read_text()
    got = json.loads((prep/'registration_readback.json').read_text())
    check(type(got.get('id')) is int and got.get('body') == body and
        got.get('issue_url') == 'https://api.github.com/repos/Lebron-233/lerobot/issues/1' and
        all(x in body for x in (f'E-TFD1-REGISTER:{head}', str(out), digest(prep/'preparation.json'))),
        'Actual registration readback missing or different')
    return saved


class Budget(e.Budget):
    def check(self, kind):
        local, total = LIMITS[kind]
        check(self.episode[kind] < local and self.total[kind] < total, f'E-TFD1 {kind} budget exceeded')


def controlled_loop(owner, native, spec, first, observe, clock=e.Clock):
    """Only the response installation schedule is experimentally intervened on.

    A request still sees current feedback. Exactly d old actions are consumed,
    then the controller waits as necessary for computation before installing.
    No action is fabricated and no native step is dispatched merely for a wait.
    """
    t0, current, next_slot = clock.now(), first, 0
    dispatches, gets, blocks = [], [], []
    result = {'t0': t0, 'dispatches': dispatches, 'gets': gets, 'blocks': blocks,
              'success': False, 'terminal_reason': None}
    try:
        while next_slot < spec['ready_wall_slots'] and len(dispatches) < spec['limits']['measurement']:
            clock.sleep(max(0, t0+next_slot/20-clock.now()))
            slot = max(next_slot, e.slot_at(clock.now(), t0))
            if slot >= spec['ready_wall_slots']:
                break
            if owner.pending is None:
                remaining = owner.queue.qsize()
                estimated = old.estimate_delay(owner.durations)
                if old.should_submit(remaining, False, estimated):
                    owner.submit(current, 0 if remaining == 0 else estimated)
            if owner.pending is not None:
                stamp = owner.pending['stamp']
                target = delay_for(spec, stamp['request_id'])
                consumed = owner.queue.next_index-stamp['observation_index']
                check(consumed <= target, 'Controlled delay overshot')
                owner.pending['diagnostic_target_delay'] = target
                if consumed == target:
                    began = clock.now()
                    received = owner.receive(block=True)
                    check(received['decision']['accepted'] and received['decision']['actual_delay'] == target,
                          'Counterfactual installation differs')
                    blocks.append({'kind': 'controlled_result_wait', 'start': began, 'end': clock.now()})
            slot = max(slot, e.slot_at(clock.now(), t0))
            if slot >= spec['ready_wall_slots']:
                break
            got = owner.queue.pop()
            gets.append({'slot': slot, 'outcome': 'underflow' if got is None else 'action',
                         'action_index': owner.queue.next_index if got is None else got['action_index']})
            check(got is not None, 'Diagnostic prefix exhausted')
            check(got['action_index'] == current['index'] == len(dispatches), 'Action/observation index mismatch')
            dispatch = {**got, 'command': got['command'].numpy().copy(), 'slot': slot,
                'observation_index': current['index'], 'observation_age': clock.now()-current['returned_at'],
                'dispatched_at': clock.now()}
            dispatches.append(dispatch)
            raw, reward, terminated, truncated, info = native.step(dispatch)
            returned = clock.now()
            dispatch.update(returned_at=returned, reward=float(reward), terminated=bool(terminated),
                truncated=bool(truncated), success=bool(info.get('is_success', False)))
            blocks.append({'kind': 'env_busy', 'start': dispatch['dispatched_at'], 'end': returned})
            current = observe(raw, len(dispatches), returned)
            result['success'] = dispatch['success']
            result['terminal_reason'] = e.terminal_reason(terminated, truncated, info)
            if result['terminal_reason']:
                break
            next_slot = slot+1
        if result['terminal_reason'] is None:
            result['terminal_reason'] = 'action_limit' if len(dispatches) >= spec['limits']['measurement'] else 'wall_slot_limit'
    finally:
        result['ended_at'] = clock.now()
        owner.control_result = result
    return result


def episode(spec, output, policy, pre, post, factory, budget, calls):
    folder = output/f"episode_{spec['ordinal']:03d}"
    folder.mkdir()
    write(folder/'started.json', {'spec': spec, 'started_at_utc': datetime.now(UTC).isoformat(),
                                'purpose': 'known-identity factorial diagnostic'})
    budget.begin_episode()
    native = e.NativeSession(spec, budget, calls, factory)
    owner, predict, observations, control = None, None, [], None
    result = {'spec': spec, 'status': 'technical_failure', 'first_failure': None,
        'worker_joined': False, 'environment_closed': False, 'controller_thread': threading.get_ident()}
    try:
        def observe(raw, index, returned):
            _, _, record = e.observation(raw, spec['task_name'].replace('_', ' '), index, returned)
            observations.append(record)
            return record
        initial = observe(native.create_reset(), 0, time.perf_counter())
        torch.save(initial, folder/'initial_checkpoint.pt')
        ordinal = REFERENCES[(spec['task_id'], spec['initial_state_id'])][0]
        reference = load(SOURCE/f'episode_{ordinal:03d}/initial_checkpoint.pt')
        check(e.initial_difference(reference, initial) is None, 'Diagnostic initial state differs from original')
        result['initial_reference_exact'] = True
        predict = g.GraphFeedbackPredictor(policy, pre, post, spec)
        owner = old.InferenceOwner(predict, calls, spec, budget, on_owner_close=predict.close,
                                   request_kind='graph_request')
        began = time.perf_counter()
        owner.submit(initial, 0)
        owner.receive(block=True)
        result['startup_s'] = time.perf_counter()-began
        control = controlled_loop(owner, native, spec, initial, observe)
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
                no_action_slots=count-len(control['dispatches']), underflows=sum(v['outcome'] == 'underflow' for v in control['gets']))
        write(folder/'result.json', result)
    return result


def worker(args):
    torch.set_num_threads(1)
    faulthandler.register(signal.SIGUSR1, file=sys.stderr, all_threads=True)
    calls, records, budget = e.Calls(args.output/'calls.jsonl'), [], Budget()
    result = {'experiment': 'E-TFD1', 'execution_head': args.execution_head, 'status': 'technical_failure', 'first_failure': None}
    try:
        prepared = validate(args.execution_head)
        with pilot.phase(args.output, 'environment_preflight', 30):
            factory = e.reference.make_native_env_factory()
            check(not torch.cuda.is_initialized(), 'Factory preflight initialized CUDA')
        with pilot.phase(args.output, 'load_model', 90):
            check(torch.cuda.get_device_name() == 'NVIDIA GeForce RTX 4070 Ti SUPER', 'GPU changed')
            policy, pre, post, report = e.load_runtime(e.POLICY, e.VLM)
            check(not policy.config.compile_model and policy.config.rtc_config is None, 'Frozen configuration differs')
            write(args.output/'policy_load.json', report)
            result['vla_loads'] = 1
        for spec in manifest()['rows']:
            with pilot.phase(args.output, f"episode_{spec['ordinal']}", 100):
                print(f"E-TFD1 START {spec['ordinal']}/8 {spec['arm']} state={spec['initial_state_id']}", flush=True)
                record = episode(spec, args.output, policy, pre, post, factory, budget, calls)
                records.append(record)
                check(record['status'] == 'completed', record['first_failure'] or 'Incomplete diagnostic')
                print(f"E-TFD1 END {spec['ordinal']} success={record['success']} actions={record['measured_actions']}", flush=True)
        check(all(not p.requires_grad and p.grad is None for p in policy.parameters()), 'Model not frozen')
        tree_gate(args.execution_head)
        check(prepared['sources'] == sources(), 'Sources changed during diagnostic')
        result.update(status='completed', candidate_unchanged=True, vla_frozen=True)
    except BaseException:
        result['first_failure'] = traceback.format_exc()
    finally:
        calls.close()
        captures = [c for row in records for c in (row.get('predictor_cleanup') or {}).get('captures', [])]
        result.update(episodes_completed=sum(v['status'] == 'completed' for v in records),
            outcomes=[{k: row.get(k) for k in ('spec', 'success', 'measured_actions', 'terminal_reason')} for row in records],
            native_budget=dict(budget.total), graph_captures=len(captures),
            capture_internal={'setup': sum(c['eager_setup_calls'] for c in captures),
                'warmup': sum(c['side_stream_warmup_calls'] for c in captures), 'capture': sum(c['capture_calls'] for c in captures)},
            attempts=1, retries=0, training_updates=0, rtc_steps=0, real_robot=0,
            diagnostic_identity_count=2, new_qualification_claimed=False, realtime_qualified=False,
            confirmation_result_unchanged=True)
        write(args.output/'worker_result.json', result)
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
        check(not args.worker and args.output is None, 'Preparation cannot dispatch model')
        prepare(args.execution_head)
        return 0
    args.output = (args.output or paths(args.execution_head)[1]).resolve()
    return worker(args) if args.worker else old.supervise(args, validate_run=validate, paths_for=paths,
        manifest_for=manifest, worker_script=str(Path(__file__).resolve()), experiment='E-TFD1')


if __name__ == '__main__':
    raise SystemExit(main())
