"""E-GCR1: exact full-path Graph comparison, then fixed native workload replay.

Replay uses ALL eight archived E-OBS1 async command sequences, not new policy
rollouts or new task-success evidence. No RTC, training, or qualification reads.
"""

import argparse
import copy
import faulthandler
import json
import math
import os
import signal
import subprocess
import sys
import threading
import time
import traceback
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path

import libero_observation_async as old
import torch
from smolvla_graph_runtime import SmolVLAGraphRuntime

q, e, pilot, REPO = old.q, old.e, old.pilot, old.REPO
check, write, digest = old.check, old.write, old.digest
BASELINE = REPO / 'outputs/smolvla_graph_replay_development_20260917/baseline.json'
SOURCE = REPO / 'outputs/smolvla_observation_async_7887dfe7'
FIXED = REPO / 'outputs/smolvla_rtc_fullpath_preparation_92c39812/data.pt'
SOFT, HARD = 1200, 1230
LIMITS = {'episodes': (1, 16), 'settling': (10, 160), 'measurement': (280, 4480)}


def load(path):
    return torch.load(path, map_location='cpu', weights_only=False)


def paths(head):
    return (REPO / f'outputs/smolvla_graph_replay_preparation_{head[:8]}',
            REPO / f'outputs/smolvla_graph_replay_{head[:8]}')


def specification():
    return {'experiment': 'E-GCR1', 'fixed_keys': [[t, s, n] for t in (6, 7)
        for s in (48, 49) for n in (3, 4, 5, 6)], 'fixed_repeats': 2, 'fixed_requests': 64,
        'replay_source': str(SOURCE), 'replay_async_episodes': 8, 'replay_arms': ['eager', 'graph'],
        'replay_requests': 142, 'replay_actions': 2820, 'total_requests': 206,
        'captures': 12, 'internal_calls_per_capture': {'setup': 1, 'warmup': 3, 'capture': 1},
        'fps': 20, 'chunk_size': 50, 'num_steps': 10, 'max_delay': 8, 'margin': 1,
        'latency_limit_s': .35, 'rtc': False, 'new_task_success_claim': False,
        'qualification_reads': 0, 'training': 0, 'soft': SOFT, 'hard': HARD}


def source_hashes():
    result = old.sources()
    files = [BASELINE, FIXED, SOURCE / 'result.json', SOURCE / 'independent_audit.json',
             REPO / 'tests/test_libero_graph_replay.py',
             REPO / 'docs/experiments/SMOLVLA_GRAPH_REPLAY_PLAN.md']
    for spec in old.manifest()['rows']:
        if spec['condition'] == 'async':
            folder = SOURCE / f"episode_{spec['ordinal']:03d}"
            files += [folder / name for name in ('result.json', 'arrays.pt', 'initial_checkpoint.pt')]
    for f in files:
        result[str(f)] = digest(f)
    return result


def tree_gate(head):
    check(subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True, cwd=REPO).strip() == head,
          'Execution HEAD changed')
    check(old.pending_state() == json.loads(BASELINE.read_text())['pending'], 'Original pending changed')


def data():
    check(json.loads((SOURCE / 'independent_audit.json').read_text())['independent_contract_accepted'],
          'Replay source audit absent')
    fixed = load(FIXED)
    check([v['key'] for v in fixed] == specification()['fixed_keys'], 'Fixed identities changed')
    fixed = [{**v, 'noise': v['inputs'][7], 'expected_full': v['archived_full_chunk']} for v in fixed]
    traces = []
    for spec in old.manifest()['rows']:
        if spec['condition'] != 'async':
            continue
        folder = SOURCE / f"episode_{spec['ordinal']:03d}"
        record, arrays = json.loads((folder / 'result.json').read_text()), load(folder / 'arrays.pt')
        check(record['status'] == 'completed', 'Source episode incomplete')
        requests = []
        for req in record['requests']:
            rid, index = req['stamp']['request_id'], req['stamp']['observation_index']
            value = arrays['outputs'][rid]
            requests.append({'key': [spec['task_id'], spec['initial_state_id'], rid],
                'observation': arrays['observations'][index], 'noise': value['noise'],
                'expected_full': value['full'], 'expected_processed': value['processed'],
                'language': spec['task_name'].replace('_', ' '), 'origin': index})
        traces.append({'spec': spec, 'requests': requests, 'initial': load(folder / 'initial_checkpoint.pt'),
                       'commands': [d['command'] for d in arrays['control']['dispatches']]})
    check(len(traces) == 8 and sum(len(t['requests']) for t in traces) == 71 and
          sum(len(t['commands']) for t in traces) == 1410, 'Source replay coverage differs')
    return {'fixed': fixed, 'traces': traces}


def prepare(head):
    tree_gate(head)
    environment = q.runtime_environment()
    prep, out = paths(head)
    check(not prep.exists() and not out.exists(), 'Preparation/output exists; never overwrite')
    values = data()
    check(not torch.cuda.is_initialized(), 'Preparation initialized CUDA')
    prep.mkdir()
    torch.save(values, prep / 'data.pt')
    write(prep / 'preparation.json', {'head': head, 'specification': specification(),
        'sources': source_hashes(), 'data_sha256': digest(prep / 'data.pt'), 'environment': environment})
    print(json.dumps({'prepared': True, 'prep': str(prep), 'out': str(out),
                      'sha256': digest(prep / 'preparation.json')}), flush=True)


def validate(head):
    tree_gate(head)
    prep, out = paths(head)
    saved = json.loads((prep / 'preparation.json').read_text())
    check(saved['head'] == head and saved['specification'] == specification(), 'Contract changed')
    check(saved['environment'] == q.runtime_environment(), 'Environment changed')
    check(saved['sources'] == source_hashes() and saved['data_sha256'] == digest(prep / 'data.pt'), 'Source changed')
    got = json.loads((prep / 'registration_readback.json').read_text())
    body = (prep / 'registration.md').read_text()
    check(type(got.get('id')) is int and got.get('body') == body and got.get('issue_url') ==
          'https://api.github.com/repos/Lebron-233/lerobot/issues/1' and all(x in body for x in
          (f'E-GCR1-REGISTER:{head}', str(out), digest(prep / 'preparation.json'))), 'Registration differs')
    return saved


def latency(rows):
    if not rows:
        return {'n': 0}
    values = sorted(float(r['complete_s']) for r in rows)
    check(all(math.isfinite(v) and 0 < v < 15 for v in values), 'Invalid request latency')
    result = {'n': len(values), 'mean_s': math.fsum(values)/len(values), 'max_s': values[-1],
        **{f'p{int(p*100)}_s': values[math.ceil(p*len(values))-1] for p in (.5, .95, .99)}}
    result['required_delay_steps'] = math.ceil(result['p99_s']*20)+1
    return result


class FullPath:
    """Same public preprocessing/API/postprocessing for both sampler modes."""

    def __init__(self, policy, pre, post, mode, task):
        check(mode in ('eager', 'graph'), 'Unknown sampler')
        self.policy, self.pre, self.post, self.mode, self.task = policy, pre, post, mode, task
        self.runtime, self.owner = None, None
        self.closed = False
        self.receipt = None

    def activate(self):
        ident = threading.get_ident()
        check(not self.closed and self.mode in ('eager', 'graph'), 'Closed predictor or invalid mode')
        if self.runtime is None:
            self.owner = ident
            self.runtime = SmolVLAGraphRuntime(self.policy.model)
            self.runtime.__enter__()
        check(ident == self.owner, 'Wrong model owner')
        self.runtime.begin_episode(self.mode, self.task)

    def __call__(self, item):
        self.activate()
        ident = self.owner
        rt = self.runtime
        with torch.no_grad():
            torch.cuda.synchronize()
            began = time.perf_counter()
            self.policy.reset()
            self.pre.reset()
            self.post.reset()
            batch = self.pre(pilot.worker_batch(item['observation'], item['language'], 'cuda'))
            noise = item['noise'].cuda().clone()
            torch.cuda.synchronize()
            model_started = time.perf_counter()
            normalized = self.policy.predict_action_chunk(batch, noise=noise)
            torch.cuda.synchronize()
            model_ended = time.perf_counter()
            processed = self.post(normalized).detach().cpu().clone()[0]
            original = normalized.detach().cpu().clone()[0]
            full = rt.latest.detach().cpu().clone()
            inputs = q.cpu(rt.latest_inputs)
            torch.cuda.synchronize()
            ended = time.perf_counter()
        check(full.shape == (1, 50, 32) and original.shape == processed.shape == (50, 7), 'Output shape')
        check(all(torch.isfinite(v).all() for v in (full, original, processed)), 'Nonfinite output')
        return {'mode': self.mode, 'key': item['key'], 'full': full, 'original': original,
            'processed': processed, 'inputs': inputs, 'complete_s': ended-began,
            'started_at': began, 'completed_at': ended,
            'model_started_at': model_started, 'model_completed_at': model_ended,
            'parts': {'pre_s': model_started-began, 'model_s': model_ended-model_started,
                      'post_evidence_s': ended-model_ended},
            'sampler': dict(rt.metadata), 'owner_thread': ident}

    def close(self):
        if self.runtime is None:
            self.closed = True
            return
        check(threading.get_ident() == self.owner, 'Release on wrong owner')
        rt = self.runtime
        self.receipt = {'mode': self.mode, 'task': self.task, 'owner': self.owner,
                        'requests': rt.control_requests, 'rgb_encodings': rt.rgb_encodings,
                        'noise_draws': rt.noise_draws, 'captures': copy.deepcopy(rt.captures)}
        rt.__exit__(None, None, None)
        self.closed = True
        self.receipt.update(graph_released=rt.graph is None, sampler_restored=self.policy.model.sample_actions == rt.original)


def exact_output(row, item):
    pilot.require_equal(row['full'], item['expected_full'], 'Complete output changed')
    pilot.require_equal(row['original'], row['full'][0, :, :7], 'Unpadding changed')
    if 'expected_processed' in item:
        pilot.require_equal(row['processed'], item['expected_processed'], 'Processed command changed')
    if 'inputs' in item:
        for a, b in zip(row['inputs'], item['inputs'], strict=True):
            pilot.require_equal(a, b, 'Dynamic graph input differs')
    pilot.require_equal(row['inputs'][7], item['noise'], 'Noise changed')


class Budget(e.Budget):
    def check(self, kind):
        local, total = LIMITS[kind]
        check(self.episode[kind] < local and self.total[kind] < total, 'Native budget exhausted')


class Run:
    def __init__(self, output, policy, pre, post, calls):
        self.output, self.policy, self.pre, self.post, self.calls = output, policy, pre, post, calls
        self.rows, self.runtimes, self.counts = [], [], Counter()

    def invoke(self, predict, item, label, context):
        cid = self.calls.start('model_request', context.get('ordinal', -1), limit=15, label=label)
        started = time.perf_counter()
        try:
            with pilot.phase(self.output, label, 15):
                row = predict(item)
            self.calls.emit('call_return', call_id=cid, elapsed=time.perf_counter()-started)
            return {**row, **context, 'label': label}
        except BaseException:
            self.calls.emit('call_error', call_id=cid, exception=traceback.format_exc())
            raise

    def retain(self, row, item):
        index = len(self.rows)
        torch.save(row, self.output / f'request_{index:03d}.pt')
        exact_output(row, item)
        self.rows.append(row)
        self.counts['exact_requests'] += 1

    def fixed(self, items):
        for group in range(4):
            # One installed sampler, switching modes without nesting runtime wrappers.
            predictor = FullPath(self.policy, self.pre, self.post, 'eager', f'fixed_{group}')
            try:
                for i in range(4*group, 4*group+4):
                    item = items[i]
                    for repeat in range(2):
                        modes = ('eager', 'graph') if (i+repeat) % 2 == 0 else ('graph', 'eager')
                        for mode in modes:
                            predictor.mode = mode
                            row = self.invoke(predictor, item, f'fixed_{i}_{repeat}_{mode}',
                                              {'scope': 'fixed', 'repeat': repeat, 'sample_index': i})
                            self.retain(row, item)
            finally:
                predictor.close()
                self.runtimes.append(predictor.receipt)
        times = {m: latency([v for v in self.rows if v['mode'] == m and v['repeat'] == 1])
                 for m in ('eager', 'graph')}
        return times, times['graph']['required_delay_steps'] <= 8

    def replay(self, trace, mode, ordinal, budget, factory):
        spec = {**trace['spec'], 'ordinal': ordinal, 'condition': 'replay_'+mode}
        folder = self.output / f'episode_{ordinal:03d}'
        folder.mkdir()
        write(folder / 'started.json', {'spec': spec, 'purpose': 'fixed_command_workload_replay_not_new_policy_rollout'})
        budget.begin_episode()
        native = e.NativeSession(spec, budget, self.calls, factory)
        result = {'spec': spec, 'status': 'technical_failure', 'first_failure': None,
                  'environment_closed': False, 'owner_joined': False}
        executor = None
        predictor = None
        future = None
        active_item = None
        rows = []
        try:
            raw = native.create_reset()
            initial = e.observation(raw, spec['task_name'].replace('_', ' '), 0, time.perf_counter())[2]
            check(e.initial_difference(trace['initial'], initial) is None, 'Replay initial state differs')
            torch.save(initial, folder / 'initial_checkpoint.pt')
            predictor = FullPath(self.policy, self.pre, self.post, mode, f'replay_{ordinal}')
            executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix=f'gcr_{ordinal}')

            def launch(item):
                self.counts['replay_requests_dispatched'] += 1
                check(self.counts['replay_requests_dispatched'] <= 142, 'Replay request budget')
                rid = item['key'][2]
                return executor.submit(self.invoke, predictor, item, f'replay_{ordinal}_{rid}',
                                       {'scope': 'replay', 'ordinal': ordinal, 'rid': rid, 'origin': item['origin']})

            def collect(pending, item):
                row = pending.result(timeout=15)
                self.retain(row, item)
                rows.append(row)

            items = trace['requests']
            check(items[0]['origin'] == 0 and items[0]['key'][2] == 1, 'Bootstrap identity')
            collect(launch(items[0]), items[0])
            # Graph capture occurs before native stepping; no concurrent capture/warmup.
            t0, next_slot = time.perf_counter(), 0
            cursor = 1
            for index, command in enumerate(trace['commands']):
                time.sleep(max(0, t0+next_slot/20-time.perf_counter()))
                slot = max(next_slot, e.slot_at(time.perf_counter(), t0))
                if future is not None and future.done():
                    collect(future, active_item)
                    future = None
                if cursor < len(items) and items[cursor]['origin'] == index:
                    check(future is None, 'Previous replay request still in flight at next fixed origin')
                    active_item = items[cursor]
                    future = launch(active_item)
                    cursor += 1
                dispatch = {'slot': slot, 'action_index': index, 'observation_index': index,
                            'command': command.copy()}
                _, _, terminated, truncated, _ = native.step(dispatch)
                check(not (terminated or truncated) or index+1 == len(trace['commands']),
                      'Replay environment terminated before fixed commands completed')
                next_slot = slot+1
            if future is not None:
                collect(future, active_item)
                future = None
            check(cursor == len(items) and len(rows) == len(items), 'Replay request coverage incomplete')
            result.update(status='completed', control_wall_s=time.perf_counter()-t0,
                          initial_exact=True, request_count=len(rows), command_count=len(trace['commands']))
        except BaseException:
            result['first_failure'] = traceback.format_exc()
        finally:
            if executor is not None:
                try:
                    if future is not None:
                        try:
                            future.result(timeout=15)
                        except BaseException:
                            result['first_failure'] = result['first_failure'] or traceback.format_exc()
                            result['status'] = 'technical_failure'
                    executor.submit(predictor.close).result(timeout=15)
                    self.runtimes.append(predictor.receipt)
                    executor.shutdown(wait=True)
                    result['owner_joined'] = True
                except BaseException:
                    result['first_failure'] = result['first_failure'] or traceback.format_exc()
                    result['status'] = 'technical_failure'
                    executor.shutdown(wait=False, cancel_futures=True)
            if executor is None or result['owner_joined']:
                try:
                    result['environment_closed'] = native.close()
                except BaseException:
                    result['first_failure'] = result['first_failure'] or traceback.format_exc()
                    result['status'] = 'technical_failure'
            result.update(native_steps=native.native_steps, budget=dict(budget.episode),
                          request_labels=[v['label'] for v in rows])
            write(folder / 'result.json', result)
        check(result['status'] == 'completed' and result['environment_closed'] and result['owner_joined'],
              result['first_failure'] or 'Replay cleanup incomplete')
        return result


def worker(args):
    torch.set_num_threads(1)
    faulthandler.register(signal.SIGUSR1, file=sys.stderr, all_threads=True)
    result = {'experiment': 'E-GCR1', 'head': args.head, 'status': 'technical_failure', 'first_failure': None}
    calls, run, budget = e.Calls(args.output / 'calls.jsonl'), None, Budget()
    try:
        prepared = validate(args.head)
        values = load(paths(args.head)[0] / 'data.pt')
        with pilot.phase(args.output, 'load_model', 90):
            check(torch.cuda.get_device_name() == 'NVIDIA GeForce RTX 4070 Ti SUPER', 'GPU changed')
            policy, pre, post, report = e.load_runtime(e.POLICY, e.VLM)
            check(policy.config.rtc_config is None and not policy.config.compile_model, 'No RTC or compile permitted')
            write(args.output / 'policy_load.json', report)
        run = Run(args.output, policy, pre, post, calls)
        fixed_times, ready = run.fixed(values['fixed'])
        result.update(fixed_timing=fixed_times, fixed_gate_passed=ready, replay_started=False)
        records = []
        if ready:
            factory = e.reference.make_native_env_factory()
            result['replay_started'] = True
            for pair, trace in enumerate(values['traces']):
                for mode in (('eager', 'graph') if pair % 2 == 0 else ('graph', 'eager')):
                    ordinal = len(records)
                    print(f'E-GCR1 replay {ordinal}/16 {mode} source={trace["spec"]["task_id"]}/{trace["spec"]["initial_state_id"]}', flush=True)
                    with pilot.phase(args.output, f'episode_{ordinal}', 100):
                        records.append(run.replay(trace, mode, ordinal, budget, factory))
            check(len(run.rows) == 206 and sum(len(v['captures']) for v in run.runtimes) == 12,
                  'Formal request/capture totals differ')
            result['replay_timing'] = {m: latency([v for v in run.rows if v['scope'] == 'replay'
                and v['mode'] == m and v['rid'] > 1]) for m in ('eager', 'graph')}
            result['concurrent_graph_latency_passed'] = result['replay_timing']['graph']['required_delay_steps'] <= 8
        check(all(not p.requires_grad and p.grad is None for p in policy.parameters()), 'Weights not frozen')
        check(all(v['graph_released'] and v['sampler_restored'] for v in run.runtimes), 'Graph not released/restored')
        check(prepared['sources'] == source_hashes(), 'Sources changed')
        result.update(status='completed', vla_frozen=True, replay_episodes=len(records))
    except BaseException:
        result['first_failure'] = traceback.format_exc()
    finally:
        calls.close()
        if run is not None:
            write(args.output / 'runtimes.json', run.runtimes)
            result.update(completed_requests=len(run.rows), counts=dict(run.counts))
        result.update(native_budget=dict(budget.total), attempts=1, retries=0, training_updates=0,
                      rtc_steps=0, qualification_reads=0, real_robot=0, new_task_success_claim=False,
                      realtime_qualified=False)
        write(args.output / 'worker_result.json', result)
    return 0 if result['status'] == 'completed' else 2


def supervise(args):
    validate(args.head)
    check(args.output == paths(args.head)[1] and not args.output.exists(), 'Unique output required')
    args.output.mkdir()
    command = [sys.executable, '-u', __file__, '--head', args.head, '--output', str(args.output), '--worker']
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
    result = json.loads(path.read_text()) if path.exists() else {'status': 'technical_failure',
        'head': args.head, 'first_failure': 'Worker receipt absent'}
    accounting = q.natural.trace.journal_accounting(q.natural.trace.read_events(args.output / 'calls.jsonl'))
    result['execution'] = {'pid': child.pid, 'exit_code': code, 'exit_confirmed': True,
        'started_utc': utc, 'finished_utc': datetime.now(UTC).isoformat(), 'wall_s': time.perf_counter()-start,
        'stop_reason': reason, 'forced': forced, 'pending': pending, 'active': active}
    result['accounting'] = accounting
    if code or reason or pending or active or accounting['unknown_calls'] or accounting['call_errors']:
        result['status'] = 'technical_failure'
    write(args.output / 'result.json', result)
    print(json.dumps({k: result.get(k) for k in ('status', 'completed_requests', 'replay_episodes',
          'concurrent_graph_latency_passed', 'first_failure')}), flush=True)
    return 0 if result['status'] == 'completed' else 2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--head', required=True)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--prepare', action='store_true')
    parser.add_argument('--worker', action='store_true')
    args = parser.parse_args()
    signal.signal(signal.SIGTERM, e.request_shutdown)
    if args.prepare:
        check(not args.worker and args.output is None, 'Preparation cannot dispatch')
        prepare(args.head)
        return 0
    args.output = (args.output or paths(args.head)[1]).resolve()
    return worker(args) if args.worker else supervise(args)


if __name__ == '__main__':
    raise SystemExit(main())
