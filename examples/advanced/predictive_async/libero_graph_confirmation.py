"""E-GFC1: independent fixed cohort; reuse the frozen live Graph candidate unchanged."""

import argparse
import faulthandler
import json
import math
import os
import signal
import subprocess
import sys
import time
import traceback
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

import libero_graph_feedback as g
import torch

old, q, e, pilot, REPO = g.old, g.q, g.e, g.pilot, g.REPO
check, write, digest = g.check, g.write, g.digest
CANDIDATE_HEAD = 'a5e9fddefc0cfa03ea1509493e8b304ec7c06d92'
BASELINE = REPO / 'outputs/smolvla_graph_confirmation_development_20260917/baseline.json'
PRIOR = REPO / 'outputs/smolvla_graph_feedback_8112f8d7'
PAIRS = tuple((t, s) for s in range(12, 20) for t in ((0, 2, 6, 7) if s % 2 == 0 else (7, 6, 2, 0)))
LIMITS = {'episodes': (1, 64), 'settling': (10, 640), 'measurement': (280, 17920), 'model': (160, 10240)}
SOFT, HARD = 2400, 2430
ALPHA, MARGIN = .05, .10
ADDITIONS = ('examples/advanced/predictive_async/libero_graph_confirmation.py',
             'examples/advanced/predictive_async/audit_libero_graph_confirmation.py')


def paths(head):
    return (REPO / f'outputs/smolvla_graph_confirmation_preparation_{head[:8]}',
            REPO / f'outputs/smolvla_graph_confirmation_{head[:8]}')


def manifest():
    template = g.manifest()['rows'][0]
    rows = []
    for pair, (task, state) in enumerate(PAIRS):
        for arm in (('serialized', 'async') if pair % 2 == 0 else ('async', 'serialized')):
            rows.append({**template, 'ordinal': len(rows), 'pair_index': pair, 'task_id': task,
                'task_name': e.reference.TASK_NAMES[task], 'initial_state_id': state, 'condition': arm,
                'environment_seed': 1160000+100*task+state, 'policy_seed': 1170000+100*task+state,
                'limits': {k: v[0] for k, v in LIMITS.items()}})
    return {**g.manifest(), 'experiment': 'E-GFC1', 'candidate_head': CANDIDATE_HEAD, 'rows': rows,
        'captures': 64, 'global_limits': {k: v[1] for k, v in LIMITS.items()}, 'soft': SOFT, 'hard': HARD,
        'confirmation': {'pairs': 32, 'alpha': ALPHA, 'margin': MARGIN,
            'primary': 'U95(P(serialized_success_and_async_failure)) < 0.10',
            'bound': 'independent_heterogeneous_Bernoulli_KL_Chernoff_upper',
            'additional_latency_requirement': 'all_nonbootstrap_requests_both_arms <= 0.35',
            'no_optional_stopping': True, 'production_margin_approved': False}}


def frozen_candidate_gate():
    changed = subprocess.check_output(['git', 'diff', '--name-only', CANDIDATE_HEAD, '--',
        'src/lerobot', 'examples/advanced/predictive_async'], text=True, cwd=REPO).splitlines()
    check(set(changed) <= set(ADDITIONS), f'Frozen candidate changed: {changed}')
    check(g.GraphFeedbackPredictor is not None, 'Frozen predictor missing')


def source_hashes():
    values = g.source_hashes()
    for p in [BASELINE, PRIOR/'result.json', PRIOR/'independent_audit.json',
              REPO/'tests/test_libero_graph_confirmation.py',
              REPO/'docs/experiments/SMOLVLA_GRAPH_CONFIRMATION_PLAN.md',
              *(REPO/name for name in ADDITIONS)]:
        values[str(p)] = digest(p)
    return values


def tree_gate(head):
    check(subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True, cwd=REPO).strip() == head,
          'Execution HEAD changed')
    check(old.pending_state() == json.loads(BASELINE.read_text())['pending'], 'Original pending changed')
    frozen_candidate_gate()


def assert_unused(history):
    conflicts = set(PAIRS) & {(v['task'], v['state']) for v in history}
    check(not conflicts, f'Confirmation identities already used; no replacement: {sorted(conflicts)}')


def prepare(head):
    tree_gate(head)
    environment = q.runtime_environment()
    prior = json.loads((PRIOR/'independent_audit.json').read_text())
    check(prior['independent_contract_accepted'] and prior['graph_feedback_chain_completed']
          and prior['online_latency_budget_passed'], 'Candidate prerequisite missing')
    history = q.history_inventory(REPO/'outputs')
    assert_unused(history)
    prep, out = paths(head)
    check(not prep.exists() and not out.exists(), 'Unique preparation/output required')
    check(not torch.cuda.is_initialized(), 'Preparation initialized CUDA')
    prep.mkdir()
    write(prep/'preparation.json', {'head': head, 'candidate_head': CANDIDATE_HEAD,
        'manifest': manifest(), 'sources': source_hashes(), 'environment': environment, 'history': history})
    print(json.dumps({'prepared': True, 'prep': str(prep), 'out': str(out),
        'sha256': digest(prep/'preparation.json'), 'history_count': len(history), 'pairs': len(PAIRS)}), flush=True)


def validate(head, after=False):
    tree_gate(head)
    prep, out = paths(head)
    data = json.loads((prep/'preparation.json').read_text())
    check(data['head'] == head and data['candidate_head'] == CANDIDATE_HEAD and data['manifest'] == manifest(),
          'Confirmation contract changed')
    check(data['sources'] == source_hashes() and data['environment'] == q.runtime_environment(),
          'Source/environment changed')
    history = q.history_inventory(REPO/'outputs')
    if after:
        history = [v for v in history if not v['path'].startswith(out.name+'/')]
    assert_unused(history)
    check(history == data['history'], 'Concurrent or changed history')
    body = (prep/'registration.md').read_text()
    got = json.loads((prep/'registration_readback.json').read_text())
    check(type(got.get('id')) is int and got.get('body') == body and got.get('issue_url') ==
        'https://api.github.com/repos/Lebron-233/lerobot/issues/1' and all(x in body for x in
        (f'E-GFC1-REGISTER:{head}', CANDIDATE_HEAD, str(out), digest(prep/'preparation.json'))),
        'Actual registration readback differs')
    return data


def loss_upper(k, n, alpha=ALPHA):
    """Conservative upper confidence bound for an independent Bernoulli mean.

    Heterogeneous probabilities are allowed. Dependence is not covered.
    This is a KL/Chernoff inversion, not exact Clopper-Pearson for k>0.
    """
    check(type(k) is int and type(n) is int and n > 0 and 0 <= k <= n and 0 < alpha < 1,
          'Invalid confidence inputs')
    if k == n:
        return 1.0
    if k == 0:
        return -math.expm1(math.log(alpha)/n)
    x, lo, hi = k/n, k/n, 1.0
    target = math.log(1/alpha)/n
    for _ in range(80):
        mid = (lo+hi)/2
        divergence = x*math.log(x/mid)+(1-x)*math.log((1-x)/(1-mid))
        if divergence < target:
            lo = mid
        else:
            hi = mid
    return hi


def task_summary(records):
    pairs, counts = [], Counter()
    for pair, identity in enumerate(PAIRS):
        arms = {v['spec']['condition']: v for v in records if v['spec']['pair_index'] == pair}
        check(set(arms) == {'serialized', 'async'} and all(v['status'] == 'completed' for v in arms.values()),
              'Incomplete pair; do not drop or replace')
        a, b = arms['serialized']['success'], arms['async']['success']
        label = 'both_success' if a and b else 'serialized_only' if a else 'async_only' if b else 'neither_success'
        counts[label] += 1
        pairs.append({'pair_index': pair, 'task_state': list(identity), 'outcome': label,
            'arms': {arm: {k: v[k] for k in ('success', 'measured_actions', 'wall_s', 'no_action_slots',
                'underflows', 'terminal_reason', 'startup_s')} for arm, v in arms.items()}})
    k, gains, n = counts['serialized_only'], counts['async_only'], len(pairs)
    upper = loss_upper(k, n)
    return {'pairs': pairs, 'complete_pairs': n, 'discordance': dict(counts),
        'retention': {'lost_pairs': k, 'gained_pairs': gains, 'n': n,
            'observed_success_difference': (gains-k)/n, 'regression_probability_upper95': upper,
            'success_difference_lower95_conservative': -upper, 'margin': MARGIN, 'alpha': ALPHA,
            'bounded_task_retention_passed': upper < MARGIN,
            'interpretation_requires_independent_pairs': True, 'scope': 'fixed_four_tasks_new_states12_to19'}}


class Budget(e.Budget):
    def check(self, kind):
        local, total = LIMITS[kind]
        check(self.episode[kind] < local and self.total[kind] < total,
              f'E-GFC1 {kind} budget exhausted before dispatch')


def worker(args):
    torch.set_num_threads(1)
    faulthandler.register(signal.SIGUSR1, file=sys.stderr, all_threads=True)
    calls, budget, records = e.Calls(args.output/'calls.jsonl'), Budget(), []
    result = {'experiment': 'E-GFC1', 'execution_head': args.execution_head, 'candidate_head': CANDIDATE_HEAD,
              'status': 'technical_failure', 'first_failure': None}
    try:
        prepared = validate(args.execution_head)
        with pilot.phase(args.output, 'environment_preflight', 30):
            factory = e.reference.make_native_env_factory()
            check(not torch.cuda.is_initialized(), 'Factory preflight initialized CUDA')
        with pilot.phase(args.output, 'load_model', 90):
            check(torch.cuda.get_device_name() == 'NVIDIA GeForce RTX 4070 Ti SUPER', 'GPU changed')
            policy, pre, post, report = e.load_runtime(e.POLICY, e.VLM)
            check(not policy.config.compile_model and policy.config.rtc_config is None, 'Frozen candidate differs')
            write(args.output/'policy_load.json', report)
            result['vla_loads'] = 1
        initials, boots = {}, {}
        for spec in manifest()['rows']:
            with pilot.phase(args.output, f"episode_{spec['ordinal']}", 100):
                print(f"E-GFC1 START {spec['ordinal']}/64 {spec['condition']} task={spec['task_id']} state={spec['initial_state_id']}", flush=True)
                record, initial = old.episode(spec, args.output, policy, pre, post, factory, budget, calls,
                    initials.get(spec['pair_index']), predictor_factory=g.GraphFeedbackPredictor,
                    request_kind='graph_request')
                records.append(record)
                check(record['status'] == 'completed', record['first_failure'] or 'Episode incomplete')
                folder = args.output/f"episode_{spec['ordinal']:03d}"
                arrays = torch.load(folder/'arrays.pt', map_location='cpu', weights_only=False)
                pair, boot = spec['pair_index'], arrays['outputs'][1]
                if pair in boots:
                    pilot.require_equal(boots[pair]['full'], boot['full'], 'Paired bootstrap output differs')
                    pilot.require_equal(boots[pair]['noise'], boot['noise'], 'Paired bootstrap noise differs')
                    del boots[pair], initials[pair]
                else:
                    initials[pair], boots[pair] = initial, {'full': boot['full'], 'noise': boot['noise']}
                del arrays
                print(f"E-GFC1 END {spec['ordinal']} success={record['success']} actions={record['measured_actions']}", flush=True)
        check(all(not p.requires_grad and p.grad is None for p in policy.parameters()), 'VLA not frozen')
        tree_gate(args.execution_head)
        check(source_hashes() == prepared['sources'], 'Sources changed during execution')
        result.update(status='completed', vla_frozen=True, **task_summary(records))
    except BaseException:
        result['first_failure'] = traceback.format_exc()
    finally:
        calls.close()
        captures = [c for v in records for c in (v.get('predictor_cleanup') or {}).get('captures', [])]
        result.update(episodes_completed=sum(v['status'] == 'completed' for v in records),
            native_budget=dict(budget.total), graph_captures=len(captures),
            capture_internal={'setup': sum(c['eager_setup_calls'] for c in captures),
                'warmup': sum(c['side_stream_warmup_calls'] for c in captures),
                'capture': sum(c['capture_calls'] for c in captures)}, attempts=1, retries=0,
            training_updates=0, qualification_reads=0, rtc_steps=0, predictor_forwards=0, real_robot=0,
            replay_commands=False, realtime_qualified=False, broad_generalization_claimed=False, risk_thresholds=None)
        write(args.output/'worker_result.json', result)
    return 0 if result['status'] == 'completed' else 2


def supervise(args):
    validate(args.execution_head)
    check(args.output == paths(args.execution_head)[1] and not args.output.exists(), 'Unique output required')
    args.output.mkdir()
    write(args.output/'manifest.json', manifest())
    command = [sys.executable, '-u', __file__, '--execution-head', args.execution_head,
               '--output', str(args.output), '--worker']
    start, utc = time.perf_counter(), datetime.now(UTC).isoformat()
    cursors, pending, active = {}, {}, {}
    reason, terminated, forced = None, None, False
    with (args.output/'model.log').open('x') as log:
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
    path = args.output/'worker_result.json'
    result = json.loads(path.read_text()) if path.exists() else {'experiment': 'E-GFC1',
        'execution_head': args.execution_head, 'status': 'technical_failure', 'first_failure': 'Worker receipt absent'}
    result['execution'] = {'child_pid': child.pid, 'exit_code': code, 'exit_confirmed': True,
        'started_utc': utc, 'finished_utc': datetime.now(UTC).isoformat(), 'wall_s': time.perf_counter()-start,
        'stop_reason': reason, 'forced': forced, 'pending': pending, 'active': active}
    result['accounting'] = q.natural.trace.journal_accounting(q.natural.trace.read_events(args.output/'calls.jsonl'))
    a = result['accounting']
    if code or reason or pending or active or a['call_errors'] or a['unknown_calls'] or not a['journal_consistent']:
        result['status'] = 'technical_failure'
    write(args.output/'result.json', result)
    print(json.dumps({k: result.get(k) for k in ('status', 'episodes_completed', 'complete_pairs', 'retention', 'first_failure')}), flush=True)
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
