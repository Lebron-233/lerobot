"""E-PSI1: separate incumbent execution from new-prefix omission at request 2.

Known-state causal diagnostic only. Off-diagonal cells intentionally use a
non-deployable row/time association; every actual and nominal index is recorded.
The frozen model, Graph, owner lifecycle and production queue are not modified.
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

import libero_takeover_factorial as t
import torch
from rtc_execution_queue import RequestStamp

g, old, q, e, pilot, REPO = t.g, t.old, t.q, t.e, t.pilot, t.REPO
check, write, digest, load = t.check, t.write, t.digest, t.load
SOURCE = REPO / 'outputs/smolvla_takeover_factorial_787ad7e6'
REFERENCES = {19: {0: 0, 2: 3}, 18: {0: 5, 2: 6}}
LIMITS = t.LIMITS
ADDITIONS = ('examples/advanced/predictive_async/libero_prefix_skip.py',
             'examples/advanced/predictive_async/audit_libero_prefix_skip.py')
BASE_HEAD = 'ad35950ff95d2fc9c2d84b1e715061ce06c3eb83'


def paths(head):
    return (REPO / f'outputs/smolvla_prefix_skip_preparation_{head[:8]}',
            REPO / f'outputs/smolvla_prefix_skip_{head[:8]}')


def manifest():
    template = g.manifest()['rows'][0]
    rows = []
    for pair, state in enumerate((19, 18)):
        order = ((0, 0), (2, 2), (0, 2), (2, 0)) if state == 19 else ((2, 2), (0, 0), (2, 0), (0, 2))
        for consume, skip in order:
            rows.append({**template, 'ordinal': len(rows), 'pair_index': pair, 'task_id': 7,
                'task_name': e.reference.TASK_NAMES[7], 'initial_state_id': state,
                'condition': 'diagnostic', 'arm': f'consume{consume}_skip{skip}',
                'consume': consume, 'skip': skip, 'intervention_request': 2,
                'request_grid': 20, 'later_delay': 0, 'later_skip': 0,
                'environment_seed': 1160700+state, 'policy_seed': 1170700+state,
                'reference_ordinal': REFERENCES[state][consume] if consume == skip else None,
                'limits': {k: v[0] for k, v in LIMITS.items()}})
    return {'experiment': 'E-PSI1', 'rows': rows, 'candidate_head': g.manifest().get('candidate_head',
        'a5e9fddefc0cfa03ea1509493e8b304ec7c06d92'), 'new_qualification': False,
        'frozen_candidate_changed': False, 'request_observation_grid': '0,20,40,...',
        'intervention': 'only request2: old executed rows D and new omitted rows S independently 0/2',
        'off_diagonal': 'non-deployment diagnostic with explicit nominal versus actual time indices',
        'later_requests': 'wait, skip0, fixed physical-index grid; no qsize-induced cadence change',
        'captures': 8, 'global_limits': {k: v[1] for k, v in LIMITS.items()},
        'soft': old.SOFT, 'hard': old.HARD, 'attempts': 1, 'retries': 0}


def frozen_gate():
    lock = json.loads(t.LOCK.read_text())
    check(all(digest(REPO/p) == h for p, h in lock['core_sha256'].items()), 'Frozen core changed')
    changed = subprocess.check_output(['git', 'diff', '--name-only', BASE_HEAD, '--',
        'src/lerobot', 'examples/advanced/predictive_async'], text=True, cwd=REPO).splitlines()
    check(set(changed) <= set(ADDITIONS), f'Only diagnostic additions allowed: {changed}')


def sources():
    values = t.sources()
    files = [SOURCE/'result.json', SOURCE/'independent_audit.json',
        REPO/'docs/experiments/SMOLVLA_PREFIX_SKIP_PLAN.md',
        REPO/'tests/test_libero_prefix_skip.py', *(REPO/p for p in ADDITIONS)]
    for ordinal in (0, 3, 5, 6):
        folder = SOURCE/f'episode_{ordinal:03d}'
        files += [folder/'result.json', folder/'arrays.pt', folder/'initial_checkpoint.pt']
    values.update({str(p): digest(p) for p in files})
    return values


def tree_gate(head):
    check(subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip() == head,
          'Execution HEAD changed')
    check(old.pending_state() == json.loads(t.BASELINE.read_text())['pending'], 'Original pending changed')
    frozen_gate()


def prepare(head):
    tree_gate(head)
    environment = q.runtime_environment()
    prior = json.loads((SOURCE/'independent_audit.json').read_text())
    check(prior['independent_contract_accepted'] and prior['complete_anchor_trajectories_exact'] == 4,
          'TFD calibration missing')
    prep, out = paths(head)
    check(not prep.exists() and not out.exists(), 'Unique preparation/output required')
    check(not torch.cuda.is_initialized(), 'CPU prepare initialized CUDA')
    prep.mkdir()
    write(prep/'preparation.json', {'head': head, 'manifest': manifest(), 'sources': sources(),
        'environment': environment, 'history': q.history_inventory(REPO/'outputs')})
    print(json.dumps({'prepared': True, 'prep': str(prep), 'out': str(out),
        'sha256': digest(prep/'preparation.json'), 'episodes': 8}), flush=True)


def validate(head, after=False):
    tree_gate(head)
    prep, out = paths(head)
    saved = json.loads((prep/'preparation.json').read_text())
    check(saved['head'] == head and saved['manifest'] == manifest(), 'Contract changed')
    check(saved['sources'] == sources() and saved['environment'] == q.runtime_environment(), 'Source/environment changed')
    history = q.history_inventory(REPO/'outputs')
    if after:
        history = [v for v in history if not v['path'].startswith(out.name+'/')]
    check(saved['history'] == history, 'Concurrent collection/history change')
    body = (prep/'registration.md').read_text()
    got = json.loads((prep/'registration_readback.json').read_text())
    check(type(got.get('id')) is int and got.get('body') == body and got.get('issue_url') ==
        'https://api.github.com/repos/Lebron-233/lerobot/issues/1' and all(x in body for x in
        (f'E-PSI1-REGISTER:{head}', str(out), digest(prep/'preparation.json'))), 'Registration differs')
    return saved


class SplitQueue:
    """Diagnostic row cursor, never presented as the observation-origin RTC queue.

    Only the controller uses this queue. The model thread only returns arrays.
    Begin/finish identities, bounds and close fencing remain explicit.
    """

    def __init__(self, consume, skip):
        check(type(consume) is int and consume in (0, 2) and type(skip) is int and skip in (0, 2), 'Invalid factors')
        self.consume, self.skip = consume, skip
        self.next_index, self.epoch, self.serial = 0, 0, 0
        self.pending = self.original = self.processed = None
        self.cursor, self.source, self.origin, self.closed = 0, None, 0, False

    def begin(self, observation_index, expected_delay):
        check(not self.closed and self.pending is None, 'Closed or pending queue')
        check(type(observation_index) is int and observation_index == self.next_index, 'Observation index mismatch')
        check(type(expected_delay) is int and 0 <= expected_delay <= 8, 'Invalid delay')
        self.serial += 1
        stamp = RequestStamp(self.epoch, self.serial, observation_index, expected_delay)
        self.pending = stamp
        return stamp, None if self.original is None else self.original[self.cursor:].clone()

    def finish(self, stamp, original, processed):
        if self.closed or stamp != self.pending or stamp.epoch != self.epoch:
            return {'accepted': False, 'reason': 'stale_or_closed'}
        for value in (original, processed):
            check(value.shape == (50, 7) and value.device.type == 'cpu' and value.is_floating_point()
                  and torch.isfinite(value).all(), 'Invalid CPU action chunk')
        delay = self.next_index-stamp.observation_index
        expected = self.consume if stamp.request_id == 2 else 0
        start = self.skip if stamp.request_id == 2 else 0
        check(delay == expected, 'Intervention consumption mismatch')
        self.original, self.processed = original.clone(), processed.clone()
        self.cursor, self.source, self.origin = start, stamp.request_id, stamp.observation_index
        self.pending = None
        return {'accepted': True, 'reason': 'installed', 'actual_delay': delay,
            'source_row': start, 'takeover_index': self.next_index, 'request_id': stamp.request_id,
            'epoch': self.epoch, 'nominal_minus_actual': start-delay, 'diagnostic_only': True}

    def pop(self):
        if self.closed or self.original is None or self.cursor >= len(self.original):
            return None
        result = {'action_index': self.next_index, 'request_id': self.source, 'source_row': self.cursor,
            'epoch': self.epoch, 'original': self.original[self.cursor].clone(),
            'command': self.processed[self.cursor].clone(), 'nominal_action_index': self.origin+self.cursor,
            'nominal_minus_actual': self.origin+self.cursor-self.next_index}
        self.cursor += 1
        self.next_index += 1
        return result

    def qsize(self):
        return 0 if self.closed or self.original is None else len(self.original)-self.cursor

    def close(self):
        self.closed = True
        self.epoch += 1
        self.pending = self.original = self.processed = None


class DiagnosticOwner(old.InferenceOwner):
    def __init__(self, predict, calls, spec, budget, **kwargs):
        super().__init__(predict, calls, spec, budget, **kwargs)
        # The owner thread is blocked on its empty task queue. No request has been
        # submitted; replace only this new diagnostic instance, not class globals.
        self.queue = SplitQueue(spec['consume'], spec['skip'])


class Budget(e.Budget):
    def check(self, kind):
        local, total = LIMITS[kind]
        check(self.episode[kind] < local and self.total[kind] < total, f'E-PSI1 {kind} budget exhausted')


def controlled_loop(owner, native, spec, first, observe, clock=e.Clock):
    t0, current, next_slot, last_request_index = clock.now(), first, 0, 0
    dispatches, gets, blocks = [], [], []
    result = {'t0': t0, 'dispatches': dispatches, 'gets': gets, 'blocks': blocks,
        'success': False, 'terminal_reason': None}
    try:
        while next_slot < spec['ready_wall_slots'] and len(dispatches) < spec['limits']['measurement']:
            clock.sleep(max(0, t0+next_slot/20-clock.now()))
            slot = max(next_slot, e.slot_at(clock.now(), t0))
            if slot >= spec['ready_wall_slots']:
                break
            index = len(dispatches)
            if index > last_request_index and index % 20 == 0:
                check(owner.pending is None, 'Request grid reached with unresolved result')
                owner.submit(current, spec['consume'] if index == 20 else 0)
                last_request_index = index
            if owner.pending is not None:
                stamp = owner.pending['stamp']
                dose = spec['consume'] if stamp['request_id'] == 2 else 0
                consumed = owner.queue.next_index-stamp['observation_index']
                check(consumed <= dose, 'Old-action consumption overshot')
                owner.pending['intervention_dose'] = dose
                owner.pending['intervention_skip'] = spec['skip'] if stamp['request_id'] == 2 else 0
                if consumed == dose:
                    began = clock.now()
                    owner.receive(block=True)
                    blocks.append({'kind': 'diagnostic_wait', 'start': began, 'end': clock.now()})
            slot = max(slot, e.slot_at(clock.now(), t0))
            if slot >= spec['ready_wall_slots']:
                break
            got = owner.queue.pop()
            check(got is not None, 'Diagnostic queue empty')
            check(got['action_index'] == current['index'] == index, 'Physical action index mismatch')
            gets.append({'slot': slot, 'outcome': 'action', 'action_index': index})
            dispatch = {**got, 'command': got['command'].numpy().copy(), 'slot': slot,
                'observation_index': current['index'], 'observation_age': clock.now()-current['returned_at'],
                'dispatched_at': clock.now()}
            dispatches.append(dispatch)
            raw, reward, terminated, truncated, info = native.step(dispatch)
            returned = clock.now()
            dispatch.update(returned_at=returned, reward=float(reward), terminated=bool(terminated),
                truncated=bool(truncated), success=bool(info.get('is_success', False)))
            blocks.append({'kind': 'env_busy', 'start': dispatch['dispatched_at'], 'end': returned})
            current = observe(raw, index+1, returned)
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
        'purpose': 'known-state prefix/skip diagnostic, not a new qualification'})
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
        ref = REFERENCES[spec['initial_state_id']][0]
        check(e.initial_difference(initial, load(SOURCE/f'episode_{ref:03d}/initial_checkpoint.pt')) is None,
              'Initial physical state differs')
        predict = g.GraphFeedbackPredictor(policy, pre, post, spec)
        owner = DiagnosticOwner(predict, calls, spec, budget, on_owner_close=predict.close,
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
                no_action_slots=count-len(control['dispatches']), underflows=0)
        write(folder/'result.json', result)
    return result


def worker(args):
    torch.set_num_threads(1)
    faulthandler.register(signal.SIGUSR1, file=sys.stderr, all_threads=True)
    calls, budget, records = e.Calls(args.output/'calls.jsonl'), Budget(), []
    result = {'experiment': 'E-PSI1', 'execution_head': args.execution_head,
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
        for spec in manifest()['rows']:
            with pilot.phase(args.output, f"episode_{spec['ordinal']}", 100):
                print(f"E-PSI1 START {spec['ordinal']}/8 {spec['arm']} state={spec['initial_state_id']}", flush=True)
                record = episode(spec, args.output, policy, pre, post, factory, budget, calls)
                records.append(record)
                check(record['status'] == 'completed', record['first_failure'] or 'Incomplete episode')
                print(f"E-PSI1 END {spec['ordinal']} success={record['success']} actions={record['measured_actions']}", flush=True)
        check(all(not p.requires_grad and p.grad is None for p in policy.parameters()), 'Model not frozen')
        tree_gate(args.execution_head)
        check(prepared['sources'] == sources(), 'Sources changed')
        result.update(status='completed', candidate_unchanged=True, vla_frozen=True)
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
        manifest_for=manifest, worker_script=str(Path(__file__).resolve()), experiment='E-PSI1')


if __name__ == '__main__':
    raise SystemExit(main())
