"""E-OBS1: bounded eager serialized/async native experiment, observation-origin chunks.

No RTC guidance, learned context, training, old qualification reads, or hidden warmup.
One owner thread performs all requests; only the controller publishes at Env boundaries.
"""

import argparse
import faulthandler
import hashlib
import json
import math
import os
import queue
import signal
import subprocess
import sys
import threading
import time
import traceback
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

import libero_action_qualification as q
import torch
from rtc_execution_queue import RTCExecutionQueue

REPO, e, pilot = q.e.REPO, q.e, q.pilot
BASELINE = REPO / 'outputs/smolvla_observation_async_development_20260917/baseline.json'
LATENCY = REPO / 'outputs/smolvla_rtc_fullpath_92c39812'
PAIRS = ((0, 8), (2, 8), (6, 8), (7, 8), (7, 9), (6, 9), (2, 9), (0, 9))
LIMITS = {'episodes': (1, 16), 'settling': (10, 160), 'measurement': (280, 4480), 'model': (160, 2560)}
SOFT, HARD = 1500, 1530
require, cpu = q.pilot.require_equal, q.cpu


def check(value, message):
    if not value:
        raise ValueError(message)


def write(path, value):
    with Path(path).open('x') as f:
        json.dump(value, f, indent=2, ensure_ascii=False, allow_nan=False)


def digest(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def paths(head):
    return (REPO / f'outputs/smolvla_observation_async_preparation_{head[:8]}',
            REPO / f'outputs/smolvla_observation_async_{head[:8]}')


def manifest():
    rows = []
    for pair, (task, state) in enumerate(PAIRS):
        for arm in (('serialized', 'async') if pair % 2 == 0 else ('async', 'serialized')):
            rows.append({'ordinal': len(rows), 'pair_index': pair, 'task_id': task,
                'task_name': e.reference.TASK_NAMES[task], 'initial_state_id': state, 'condition': arm,
                'environment_seed': 1140000 + 100*task + state,
                'policy_seed': 1150000 + 100*task + state,
                'ready_wall_slots': 1200, 'fps': 20, 'threshold': 30, 'max_delay': 8,
                'guard': 2, 'margin': 1, 'latency_window': 50, 'latency_quantile': .9,
                'limits': {k: v[0] for k, v in LIMITS.items()}})
    return {'experiment': 'E-OBS1', 'rows': rows, 'chunk': 50, 'denoising_steps': 10,
            'rtc': False, 'context': 'current_visual', 'sampler': 'same_eager_both_arms',
            'origin': 'observation_index', 'trim': 'actual_consumed_once',
            'initial_expected_delay': 7, 'warmup': 'one_real_bootstrap_per_episode_no_extra',
            'global_limits': {k: v[1] for k, v in LIMITS.items()}, 'soft': SOFT, 'hard': HARD,
            'qualification_reads': 0, 'training': 0, 'graph_captures': 0}


def pending_state():
    status = subprocess.check_output(['git', 'status', '--porcelain'], text=True, cwd=REPO)
    return {s[3:]: {'status': s[:2], 'sha256': digest(REPO / s[3:])} for s in status.splitlines()}


def sources():
    files = [BASELINE, LATENCY / 'result.json', LATENCY / 'independent_audit.json']
    for root in (e.POLICY, e.VLM):
        files += sorted(root.glob('*.safetensors')) + sorted(root.glob('*.json'))
    code = subprocess.check_output(['git', 'ls-files', 'src/lerobot',
        'examples/advanced/predictive_async', 'tests/test_libero_observation_async.py',
        'docs/experiments/SMOLVLA_OBSERVATION_ASYNC_PLAN.md'], text=True, cwd=REPO).splitlines()
    files += [REPO / name for name in code]
    return {str(p.relative_to(REPO)) if p.is_relative_to(REPO) else str(p): digest(p) for p in files}


def assert_unused(history):
    overlap = set(PAIRS) & {(r['task'], r['state']) for r in history}
    check(not overlap, f'Proposed identities already used; no replacement: {sorted(overlap)}')


def source_gate(head):
    check(subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True, cwd=REPO).strip() == head,
          'Execution HEAD changed')
    check(pending_state() == json.loads(BASELINE.read_text())['pending'], 'Original pending changed')


def prepare(head):
    source_gate(head)
    environment = q.runtime_environment()
    previous = json.loads((LATENCY / 'independent_audit.json').read_text())
    check(previous['independent_contract_accepted'] and previous['timing']['base']['required_delay_steps'] <= 8,
          'Base full-path prerequisite absent')
    history = q.history_inventory(REPO / 'outputs')
    assert_unused(history)
    prep, out = paths(head)
    check(not prep.exists() and not out.exists(), 'Preparation/output already exists')
    prep.mkdir()
    write(prep / 'preparation.json', {'head': head, 'manifest': manifest(), 'sources': sources(),
        'environment': environment, 'history': history, 'cuda_initialized': torch.cuda.is_initialized()})
    check(not torch.cuda.is_initialized(), 'Preparation initialized CUDA')
    print(json.dumps({'prepared': True, 'sha256': digest(prep / 'preparation.json'),
          'prep': str(prep), 'out': str(out), 'historical_identities': len(history)}), flush=True)


def validate(head, after=False):
    source_gate(head)
    prep, out = paths(head)
    saved = json.loads((prep / 'preparation.json').read_text())
    check(saved['head'] == head and saved['manifest'] == manifest(), 'Manifest changed')
    check(saved['sources'] == sources(), 'Sources changed')
    check(saved['environment'] == q.runtime_environment(), 'Environment changed')
    history = q.history_inventory(REPO / 'outputs')
    if after:
        history = [r for r in history if not r['path'].startswith(out.name + '/')]
    assert_unused(history)
    check(history == saved['history'], 'Concurrent or changed historical collection')
    body = (prep / 'registration.md').read_text()
    got = json.loads((prep / 'registration_readback.json').read_text())
    check(type(got.get('id')) is int and got.get('body') == body and got.get('issue_url') ==
          'https://api.github.com/repos/Lebron-233/lerobot/issues/1' and all(s in body for s in
          (f'E-OBS1-REGISTER:{head}', str(out), digest(prep / 'preparation.json'))), 'Registration differs')
    return saved


class Budget(e.Budget):
    def check(self, kind):
        local, total = LIMITS[kind]
        if self.episode[kind] >= local or self.total[kind] >= total:
            raise RuntimeError(f'E-OBS1 {kind} exhausted before dispatch')


def estimate_delay(durations):
    if not durations:
        return 7
    v = sorted(durations[-50:])
    return math.ceil(v[math.ceil(.9*len(v))-1]*20) + 1


def should_submit(remaining, inflight, delay):
    return not inflight and (remaining == 0 or (delay <= 8 and delay+2 <= remaining <= 30))


class InferenceOwner:
    """Single model owner. Completion is installed only by the Env-boundary controller."""

    def __init__(self, predict, calls, spec, budget):
        self.predict, self.calls, self.spec, self.budget = predict, calls, spec, budget
        self.queue = RTCExecutionQueue(max_delay=8)
        self.tasks, self.results = queue.Queue(maxsize=1), queue.Queue(maxsize=1)
        self.pending = None
        self.rows, self.outputs, self.durations = [], {}, []
        self.incumbent = None
        self.thread = threading.Thread(target=self._run, name=f"eobs-owner-{spec['ordinal']}", daemon=True)
        self.thread.start()

    def submit(self, observation, delay):
        self.budget.take('model')
        stamp, prefix = self.queue.begin(observation['index'], delay)
        row = {'stamp': asdict(stamp), 'requested_at': time.perf_counter(),
               'observation_returned_at': observation['returned_at'],
               'prefix_source': self.incumbent, 'prefix_rows': 0 if prefix is None else len(prefix)}
        self.rows.append(row)
        self.pending = row
        self.outputs[stamp.request_id] = {'prefix': prefix}
        self.calls.emit('request_submitted', ordinal=self.spec['ordinal'], **row)
        self.tasks.put_nowait((stamp, observation, row))
        return stamp

    def _run(self):
        while True:
            task = self.tasks.get()
            if task is None:
                return
            stamp, observation, row = task
            call_id = self.calls.start('eager_request', self.spec['ordinal'], limit=15,
                                      request_id=stamp.request_id, observation_index=stamp.observation_index)
            started = time.perf_counter()
            try:
                output = self.predict(observation)
                ended = time.perf_counter()
                check(ended-started < 15, 'Request exceeded 15 seconds')
                self.calls.emit('call_return', call_id=call_id, elapsed=ended-started)
                self.results.put((stamp, output, {'started_at': started, 'completed_at': ended,
                    'complete_s': ended-started, 'queue_wait_s': started-row['requested_at']}, None))
            except BaseException:
                failure = traceback.format_exc()
                self.calls.emit('call_error', call_id=call_id, exception=failure)
                self.results.put((stamp, None, None, failure))

    def receive(self, block=False):
        if self.pending is None:
            return None
        try:
            stamp, output, timing, failure = self.results.get(block=block, timeout=15 if block else None)
        except queue.Empty:
            if block:
                raise TimeoutError('Owner completion wait exceeded 15 seconds') from None
            return None
        check(stamp.request_id == self.pending['stamp']['request_id'], 'Completion identity differs')
        row = self.pending
        if failure:
            row['first_failure'] = failure
            self.pending = None
            raise RuntimeError(failure)
        decision = self.queue.finish(stamp, output['original'], output['processed'])
        row.update(timing, decision=decision, published_at=time.perf_counter(), first_failure=None)
        self.outputs[stamp.request_id].update(output)
        if decision['accepted']:
            self.incumbent = stamp.request_id
        if stamp.request_id > 1:
            self.durations.append(timing['complete_s'])
        self.calls.emit('request_finished', ordinal=self.spec['ordinal'], **row)
        self.pending = None
        return row

    def close(self):
        self.queue.close()
        deadline = time.perf_counter() + 16
        self.tasks.put(None, timeout=16)
        self.thread.join(max(0, deadline-time.perf_counter()))
        check(not self.thread.is_alive(), 'Owner thread did not exit')
        self.receive(block=False)
        check(self.pending is None, 'Pending request missing its terminal completion')


class EagerPredictor:
    def __init__(self, policy, pre, post, spec):
        self.policy, self.pre, self.post, self.spec = policy, pre, post, spec
        self.generator = None
        self.owner = None

    def __call__(self, observation):
        ident = threading.get_ident()
        if self.owner is None:
            self.owner = ident
            self.generator = torch.Generator(device='cuda').manual_seed(self.spec['policy_seed'])
        check(self.owner == ident, 'Model called by a different owner')
        policy, pre, post = self.policy, self.pre, self.post
        policy.reset()
        pre.reset()
        post.reset()
        shapes, captured = [], {}
        original_encode = policy.model.encode_image_tokens
        original_sample = policy.model.sample_actions_profiled
        encode_count = 0

        def encode(*args, **kwargs):
            nonlocal encode_count
            value = original_encode(*args, **kwargs)
            encode_count += 1
            return value

        def sample(*args, **kwargs):
            value = original_sample(*args, **kwargs)
            captured['full'] = value.detach().cpu().clone()
            return value

        hook = policy.model.action_out_proj.register_forward_hook(lambda _m, _a, value: shapes.append(list(value.shape)))
        policy.model.encode_image_tokens = encode
        policy.model.sample_actions_profiled = sample
        try:
            with torch.no_grad():
                torch.cuda.synchronize()
                began = time.perf_counter()
                language = self.spec['task_name'].replace('_', ' ')
                batch = pre(pilot.worker_batch(observation, language, 'cuda'))
                noise = torch.randn((1, 50, 32), generator=self.generator, device='cuda')
                torch.cuda.synchronize()
                model_started = time.perf_counter()
                timings = {}
                normalized = policy.predict_action_chunk(batch, noise=noise, timings=timings)
                torch.cuda.synchronize()
                model_ended = time.perf_counter()
                processed = post(normalized).detach().cpu().clone()[0]
                original = normalized.detach().cpu().clone()[0]
                noise_cpu = noise.detach().cpu().clone()
                torch.cuda.synchronize()
                ended = time.perf_counter()
            check(original.shape == processed.shape == (50, 7), 'Public action shape differs')
            check(captured['full'].shape == (1, 50, 32), 'Full action shape differs')
            check(all(torch.isfinite(v).all() for v in (original, processed, captured['full'])), 'Nonfinite action')
            require(original, captured['full'][0, :, :7], 'Unpadding differs')
            check(shapes == [[1, 50, 32]]*10 and encode_count == 1, 'Decoder or image encoding count differs')
            return {'original': original, 'processed': processed, 'full': captured['full'], 'noise': noise_cpu,
                'owner_thread': ident, 'model_started_at': model_started, 'model_returned_at': model_ended,
                'projection_shapes': shapes, 'vision_encodes': encode_count,
                'components': {'preprocess_transfer_s': model_started-began, **timings,
                    'public_predict_total_s': model_ended-model_started, 'postprocess_transfer_s': ended-model_ended}}
        finally:
            hook.remove()
            policy.model.encode_image_tokens = original_encode
            policy.model.sample_actions_profiled = original_sample


def control_loop(owner, native, spec, first, observe, clock=e.Clock):
    """Shared controller; the only scheduling contrast is blocking vs consuming old actions."""
    t0 = clock.now()
    current = first
    dispatches, gets, blocks = [], [], []
    result = {'t0': t0, 'dispatches': dispatches, 'gets': gets, 'blocks': blocks,
              'success': False, 'terminal_reason': None}
    next_slot = 0
    try:
        while next_slot < spec['ready_wall_slots'] and len(dispatches) < spec['limits']['measurement']:
            clock.sleep(max(0, t0+next_slot/20-clock.now()))
            slot = max(next_slot, e.slot_at(clock.now(), t0))
            if slot >= spec['ready_wall_slots']:
                break
            owner.receive()
            remaining = owner.queue.qsize()
            delay = estimate_delay(owner.durations)
            if should_submit(remaining, owner.pending is not None, delay):
                owner.submit(current, 0 if remaining == 0 else delay)
            if spec['condition'] == 'serialized' and owner.pending is not None:
                began = clock.now()
                owner.receive(block=True)
                blocks.append({'kind': 'serialized_wait', 'start': began, 'end': clock.now()})
            slot = max(slot, e.slot_at(clock.now(), t0))
            if slot >= spec['ready_wall_slots']:
                break
            got = owner.queue.pop()
            gets.append({'slot': slot, 'outcome': 'underflow' if got is None else 'action',
                         'action_index': owner.queue.next_index if got is None else got['action_index']})
            if got is None:
                next_slot = slot+1
                continue
            check(got['action_index'] == current['index'] == len(dispatches), 'Action/observation index mismatch')
            command = got['command'].numpy().copy()
            dispatch = {**got, 'command': command, 'slot': slot, 'observation_index': current['index'],
                'observation_age': clock.now()-current['returned_at'], 'dispatched_at': clock.now()}
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


def episode(spec, output, policy, pre, post, factory, budget, calls, paired):
    folder = output / f"episode_{spec['ordinal']:03d}"
    folder.mkdir()
    write(folder / 'started.json', {'spec': spec, 'started_at_utc': datetime.now(UTC).isoformat()})
    budget.begin_episode()
    native, owner, observations, control = e.NativeSession(spec, budget, calls, factory), None, [], None
    result = {'spec': spec, 'status': 'technical_failure', 'first_failure': None,
              'worker_joined': False, 'environment_closed': False, 'initial_pair_exact': None}
    initial = None
    try:
        raw = native.create_reset()
        def observe(raw, index, returned):
            _, _, record = e.observation(raw, spec['task_name'].replace('_', ' '), index, returned)
            observations.append(record)
            return record
        initial = observe(raw, 0, time.perf_counter())
        torch.save(initial, folder / 'initial_checkpoint.pt')
        if paired is not None:
            check(e.initial_difference(paired, initial) is None, 'Paired initial observation differs')
            result['initial_pair_exact'] = True
        owner = InferenceOwner(EagerPredictor(policy, pre, post, spec), calls, spec, budget)
        started = time.perf_counter()
        owner.submit(initial, 0)
        owner.receive(block=True)
        result['startup_s'] = time.perf_counter()-started
        control = control_loop(owner, native, spec, initial, observe)
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
                result['requests'] = owner.rows
                torch.save({'observations': observations, 'control': control, 'outputs': owner.outputs}, folder / 'arrays.pt')
        if not result['worker_joined'] or not result['environment_closed']:
            result['status'] = 'technical_failure'
        result.update(budget=dict(budget.episode), native_steps=native.native_steps, native_returned=dict(native.returned))
        if control is not None:
            count = min(spec['ready_wall_slots'], max(1, math.ceil((control['ended_at']-control['t0'])*20)))
            result.update(wall_s=control['ended_at']-control['t0'], measured_actions=len(control['dispatches']),
                wall_slots=count, no_action_slots=count-len(control['dispatches']),
                underflows=sum(g['outcome'] == 'underflow' for g in control['gets']))
        write(folder / 'result.json', result)
    return result, initial


def summarize(records):
    pairs = []
    for index in range(8):
        arms = {r['spec']['condition']: r for r in records if r['spec']['pair_index'] == index}
        if set(arms) != {'serialized', 'async'} or any(r['status'] != 'completed' for r in arms.values()):
            continue
        pairs.append({'pair_index': index, 'task_state': list(PAIRS[index]),
            'arms': {a: {k: r[k] for k in ('success', 'measured_actions', 'wall_s', 'no_action_slots',
                                           'underflows', 'terminal_reason')} for a, r in arms.items()},
            'async_minus_serialized_wall_s': arms['async']['wall_s']-arms['serialized']['wall_s']})
    return {'pairs': pairs, 'complete_pairs': len(pairs), 'statistical_noninferiority_claimed': False}


def worker(args):
    torch.set_num_threads(1)
    faulthandler.register(signal.SIGUSR1, file=sys.stderr, all_threads=True)
    calls, records, budget = e.Calls(args.output / 'calls.jsonl'), [], Budget()
    result = {'experiment': 'E-OBS1', 'execution_head': args.execution_head,
              'status': 'technical_failure', 'first_failure': None}
    try:
        prepared = validate(args.execution_head)
        with pilot.phase(args.output, 'environment_preflight', 30):
            factory = e.reference.make_native_env_factory()
            check(not torch.cuda.is_initialized(), 'Factory preflight initialized CUDA')
        with pilot.phase(args.output, 'load_model', 90):
            check(torch.cuda.get_device_name() == 'NVIDIA GeForce RTX 4070 Ti SUPER', 'GPU changed')
            policy, pre, post, report = e.load_runtime(e.POLICY, e.VLM)
            check(not policy.config.compile_model and policy.config.rtc_config is None, 'Use eager no-RTC')
            write(args.output / 'policy_load.json', report)
            result['vla_loads'] = 1
        initials = {}
        for spec in manifest()['rows']:
            with pilot.phase(args.output, f"episode_{spec['ordinal']}", 100):
                print(f"E-OBS1 START {spec['ordinal']} {spec['condition']} task={spec['task_id']} state={spec['initial_state_id']}", flush=True)
                record, initial = episode(spec, args.output, policy, pre, post, factory, budget, calls,
                                           initials.get(spec['pair_index']))
                records.append(record)
                if spec['pair_index'] not in initials:
                    initials[spec['pair_index']] = initial
                print(f"E-OBS1 END {spec['ordinal']} {record['status']} success={record.get('success')} actions={record.get('measured_actions')}", flush=True)
                check(record['status'] == 'completed', record['first_failure'] or 'Episode incomplete')
        check(all(not p.requires_grad and p.grad is None for p in policy.parameters()), 'VLA not frozen')
        source_gate(args.execution_head)
        check(sources() == prepared['sources'], 'Sources changed during execution')
        result.update(status='completed', vla_frozen=True, **summarize(records))
    except BaseException:
        result['first_failure'] = traceback.format_exc()
    finally:
        calls.close()
        result.update(episodes_completed=sum(r['status'] == 'completed' for r in records),
            native_budget=dict(budget.total), attempts=1, retries=0, training_updates=0, qualification_reads=0,
            graph_captures=0, rtc_steps=0, predictor_forwards=0, real_robot=0,
            realtime_qualified=False, risk_thresholds=None)
        write(args.output / 'worker_result.json', result)
    return 0 if result['status'] == 'completed' else 2


def supervise(args):
    validate(args.execution_head)
    check(args.output == paths(args.execution_head)[1] and not args.output.exists(), 'Unique output required')
    args.output.mkdir()
    write(args.output / 'manifest.json', manifest())
    command = [sys.executable, '-u', __file__, '--execution-head', args.execution_head,
               '--output', str(args.output), '--worker']
    start, utc = time.perf_counter(), datetime.now(UTC).isoformat()
    cursors, pending, active = {}, {}, {}
    reason, terminated, forced = None, None, False
    with (args.output / 'model.log').open('x') as log:
        child = subprocess.Popen(command, cwd=REPO, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            while child.poll() is None:
                q.cov.parent.previous.update_pending(args.output, cursors, pending, active)
                now = time.perf_counter()
                expired = [v for v in pending.values() if now-v['timestamp'] > v['limit']]
                expired_phases = [v for v in active.values() if now-v['at'] > v['limit']]
                if terminated is None and (expired or expired_phases or now-start >= SOFT):
                    reason = {'calls': expired, 'phases': expired_phases, 'soft_timeout': now-start >= SOFT}
                    os.kill(child.pid, signal.SIGUSR1)
                    os.killpg(child.pid, signal.SIGTERM)
                    terminated = now
                if now-start >= HARD or (terminated is not None and now-terminated >= 5):
                    os.killpg(child.pid, signal.SIGKILL)
                    forced = True
                    break
                time.sleep(.1)
        except BaseException:
            reason = reason or {'supervisor_exception': traceback.format_exc()}
        finally:
            if child.poll() is None:
                os.killpg(child.pid, signal.SIGTERM)
                try:
                    child.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL)
                    forced = True
            code = child.wait()
    q.cov.parent.previous.update_pending(args.output, cursors, pending, active)
    path = args.output / 'worker_result.json'
    result = json.loads(path.read_text()) if path.exists() else {'experiment': 'E-OBS1',
        'execution_head': args.execution_head, 'status': 'technical_failure', 'first_failure': 'Worker receipt absent'}
    result['execution'] = {'child_pid': child.pid, 'exit_code': code, 'exit_confirmed': True,
        'started_utc': utc, 'finished_utc': datetime.now(UTC).isoformat(), 'wall_s': time.perf_counter()-start,
        'stop_reason': reason, 'forced': forced, 'pending': pending, 'active': active}
    result['accounting'] = q.natural.trace.journal_accounting(q.natural.trace.read_events(args.output / 'calls.jsonl'))
    a = result['accounting']
    if code or reason or pending or active or a['call_errors'] or a['unknown_calls'] or not a['journal_consistent']:
        result['status'] = 'technical_failure'
    write(args.output / 'result.json', result)
    print(json.dumps({k: result.get(k) for k in ('status', 'episodes_completed', 'complete_pairs', 'first_failure')}), flush=True)
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
    return worker(args) if args.worker else supervise(args)


if __name__ == '__main__':
    raise SystemExit(main())
