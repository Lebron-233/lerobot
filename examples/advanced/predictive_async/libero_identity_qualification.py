"""F-IQ1: frozen identity-centered candidate, eight unseen native initial states.

No training, no old qualification labels, no candidate controlling the Env.
Shared native/extraction helpers are unmodified; this is a new registered execution.
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
import time
import traceback
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path

import libero_action_qualification as q
import torch

REPO, e, pilot = q.e.REPO, q.e, q.pilot
IAR = REPO / 'outputs/smolvla_identity_anchor_1f74c04d'
CHECKPOINT = q.SOURCE / 'centered.pt'
PRESERVED = REPO / 'outputs/smolvla_transactional_updates_development_20260917/preserved_worktree.json'
PAIRS = ((6, 4), (7, 4), (7, 5), (6, 5), (6, 6), (7, 6), (7, 7), (6, 7))
ARMS = ('identity', 'iar_true', 'iar_zero', 'iar_mismatched')
LIMITS = {'encoding': 40, 'decoder': 192, 'predictor': 224}
SOFT, HARD = 1500, 1530


def require(value, message):
    if not value:
        raise ValueError(message)


def write(path, value):
    with Path(path).open('x') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)


def load(path):
    return torch.load(path, map_location='cpu', weights_only=False)


def paths(head):
    return (REPO / f'outputs/smolvla_identity_qualification_preparation_{head[:8]}',
            REPO / f'outputs/smolvla_identity_qualification_{head[:8]}')


def worktree_gate(head):
    require(subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip() == head,
            'Execution HEAD differs')
    lines = subprocess.check_output(['git', 'status', '--porcelain'], cwd=REPO, text=True).splitlines()
    current = {s[3:]: {'status': s[:2], 'sha256': q.digest(REPO/s[3:])} for s in lines}
    require(current == json.loads(PRESERVED.read_text()), 'Original pending state or bytes changed')


def anchors():
    _, val = q.pfx.load_data()
    require(len(val) == 16, 'Anchor coverage changed')
    return sorted(val, key=q.pfx.key)


def anchor_path(sample):
    return IAR / 'predictions' / ('_'.join(map(str, q.pfx.key(sample)))+'.pt')


def source_hashes():
    files = [PRESERVED, CHECKPOINT, q.SOURCE/'independent_audit.json',
             q.SOURCE/'assessment_selected_centered.pt', IAR/'independent_audit.json',
             q.pfx.scale.OLD, q.pfx.scale.NEW, *[anchor_path(s) for s in anchors()]]
    return {str(f.relative_to(REPO)): q.digest(f) for f in files}


def code_hashes():
    names = subprocess.check_output(['git', 'ls-files', 'src/lerobot',
        'examples/advanced/predictive_async', 'pyproject.toml', 'uv.lock'], cwd=REPO, text=True).splitlines()
    return {n: q.digest(REPO/n) for n in names if n.endswith('.py') or n in ('pyproject.toml', 'uv.lock')}


def manifest():
    original = e.fixed_manifest()
    rows = []
    for task, state in PAIRS:
        spec = copy.deepcopy(next(x for x in original['rows']
            if x['task_id'] == task and x['condition'] == 'graph_identity_async'))
        spec.update(ordinal=len(rows), pair_index=len(rows), initial_state_id=state,
                    environment_seed=1120000+100*task+state, policy_seed=1130000+100*task+state,
                    split='qualification', recovery_policy=q.natural.recovery.RECOVERY_POLICY,
                    max_recovery_probes_per_episode=50)
        rows.append(spec)
    return {'experiment': 'F-IQ1', 'rows': rows, 'candidate': 'I+(h(a)-h(0)); alpha=1; frozen centered72',
            'checkpoint': str(CHECKPOINT), 'arms': list(ARMS), 'max_samples_per_episode': 4,
            'limits': LIMITS, 'native_limits': q.NATIVE_LIMITS, 'soft': SOFT, 'hard': HARD,
            **{k: original[k] for k in ('policy_revision', 'vlm_revision', 'assets_revision')}}


def check_unused(history):
    overlap = set(PAIRS) & {(r['task'], r['state']) for r in history}
    require(not overlap, f'Used qualification identities, no replacement allowed: {sorted(overlap)}')


def prepare(head):
    worktree_gate(head)
    env = q.runtime_environment()
    prep, output = paths(head)
    require(not prep.exists() and not output.exists(), 'Preparation/output exists')
    history = q.history_inventory(REPO/'outputs')
    check_unused(history)
    q.checkpoint_valid(load(CHECKPOINT), 'centered')
    for f in (q.SOURCE/'independent_audit.json', IAR/'independent_audit.json'):
        require(json.loads(f.read_text())['independent_contract_accepted'] is True, 'Source not accepted')
    require(not torch.cuda.is_initialized(), 'CPU preparation initialized CUDA')
    prep.mkdir()
    write(prep/'preparation.json', {'execution_head': head, 'manifest': manifest(), 'history': history,
        'source_hashes': source_hashes(), 'code_hashes': code_hashes(), 'runtime_environment': env,
        'cuda_initialized': False, 'new_env': 0, 'model_forwards': 0, 'old_qualification_reads': 0})
    print(json.dumps({'prepared': True, 'prep': str(prep), 'output': str(output),
                      'sha256': q.digest(prep/'preparation.json'), 'historical_identities': len(history)}), flush=True)


def validate(head, after=False):
    worktree_gate(head)
    prep, out = paths(head)
    saved = json.loads((prep/'preparation.json').read_text())
    require(saved['execution_head'] == head and saved['manifest'] == json.loads(json.dumps(manifest())), 'Contract changed')
    require(saved['source_hashes'] == source_hashes() and saved['code_hashes'] == code_hashes(), 'Source/code bytes changed')
    require(saved['runtime_environment'] == q.runtime_environment(), 'Environment changed')
    history = q.history_inventory(REPO/'outputs')
    prior = [x for x in history if not after or not x['path'].startswith(out.name+'/')]
    check_unused(prior)
    require(prior == saved['history'], 'Historical identities changed or concurrent collection')
    body = (prep/'registration.md').read_text()
    post = json.loads((prep/'registration_post.json').read_text())
    got = json.loads((prep/'registration_readback.json').read_text())
    require(type(got.get('id')) is int and got['id'] == post.get('id') and
            got.get('body') == post.get('body') == body and got.get('issue_url') ==
            'https://api.github.com/repos/Lebron-233/lerobot/issues/1' and all(v in body for v in
            (f'F-IQ1-REGISTER:{head}', str(out), q.digest(prep/'preparation.json'))), 'Registration exact gate failed')
    return saved


def predict(model, sample, actions, counts):
    require(counts['predictor']+2 <= LIMITS['predictor'], 'Predictor budget before dispatch')
    # q.residual performs two actual branch forwards and the original precision boundary.
    return q.residual(model, sample, tuple(v.cuda().float() for v in sample['inputs'][:2]),
                      'centered', actions, counts)


def decode(runtime, sample, visual, output, counts, label):
    require(counts['decoder'] < LIMITS['decoder'], 'Decoder budget before dispatch')
    return q.decode(runtime, sample, visual, output, counts, label)


def replay_anchors(model, runtime, output, counts):
    records = []
    residuals = {tuple(x['key']): x for x in load(q.SOURCE/'assessment_selected_centered.pt')
                 if x['context'] == 'true'}
    for s in anchors():
        with pilot.phase(output, f'anchor_{q.pfx.key(s)}', 90):
            ref = load(anchor_path(s))
            ev = predict(model, s, s['actions'], counts)
            for field in ('action_delta', 'zero_delta'):
                for a, b in zip(ev[field], residuals[q.pfx.key(s)][field], strict=True):
                    pilot.require_equal(a, b, 'Anchor original branch differs')
            for a, b in zip(ev['raw_visual'], ref['raw_new']['identity_true'], strict=True):
                pilot.require_equal(a, b, 'Anchor IAR raw differs')
            for a, b in zip(ev['visual'], ref['visual']['identity_true'], strict=True):
                pilot.require_equal(a, b, 'Anchor IAR BF16 differs')
            identity = decode(runtime, s, s['inputs'][:2], output, counts, 'anchor_identity')
            value = decode(runtime, s, ev['visual'], output, counts, 'anchor_iar')
            pilot.require_equal(identity, ref['outputs']['identity'], 'Anchor identity full output differs')
            pilot.require_equal(value, ref['outputs']['identity_true'], 'Anchor IAR full output differs')
            records.append({'key': list(q.pfx.key(s)), 'evidence': ev, 'identity': identity, 'iar': value})
            counts['anchor_exact'] += 1
    torch.save(records, output/'anchor_replay.pt')


def aggregate(rows):
    def mean(x):
        return math.fsum(x)/len(x)
    arms, contrasts = {}, {}
    for name in ARMS:
        groups = defaultdict(list)
        for row in rows:
            if name in row['metrics']:
                groups['/'.join(map(str, row['key'][:2]))].append(row['metrics'][name])
        eps = {k: {m: mean([r[m] for r in values]) for m in ('row0', 'chunk', 'latent')}
               for k, values in sorted(groups.items())}
        arms[name] = {'samples': sum(map(len, groups.values())), 'episodes': eps,
                      'macro': {m: mean([r[m] for r in eps.values()]) for m in ('row0', 'chunk', 'latent')} if eps else {}}
    for control in ('identity', 'iar_mismatched'):
        subset = [r for r in rows if control in r['metrics']]
        groups, details = defaultdict(list), []
        for row in subset:
            a, b = row['metrics']['iar_true']['row0'], row['metrics'][control]['row0']
            groups['/'.join(map(str, row['key'][:2]))].append((a, b))
            details.append({'key': row['key'], 'benefit': b-a,
                'direction': 'improved' if b-a > q.tolerance(b) else 'worsened' if a-b > q.tolerance(b) else 'tied'})
        eps = {k: {'candidate': mean([a for a, _ in values]), 'control': mean([b for _, b in values])}
               for k, values in sorted(groups.items())}
        for v in eps.values():
            v['benefit'] = v['control']-v['candidate']
        total = math.fsum(v['benefit'] for v in eps.values())
        contrasts[control] = {'paired_samples': len(subset), 'paired_episodes': len(eps), 'episodes': eps,
            'macro_benefit': total/len(eps) if eps else None, 'per_sample': details,
            'sample_improved': sum(x['direction'] == 'improved' for x in details),
            'episode_improved': sum(x['benefit'] > q.tolerance(x['control']) for x in eps.values()),
            'leave_one_out': {k: mean([v['benefit'] for j, v in eps.items() if j != k]) for k in eps} if len(eps)>1 else {},
            'largest_episode_net_share': max(x['benefit'] for x in eps.values())/total if total>0 else None}
    checks = {'eight_episodes': {tuple(r['key'][:2]) for r in rows} == set(PAIRS)}
    robust = {}
    for name, c in contrasts.items():
        checks[name+'_coverage'] = c['paired_episodes'] == 8
        checks['row0_better_'+name] = bool(c['episodes']) and c['macro_benefit'] > q.tolerance(
            mean([v['control'] for v in c['episodes'].values()]))
        checks['six_episodes_'+name] = c['episode_improved'] >= 6
        robust['sample_majority_'+name] = c['paired_samples'] > 0 and c['sample_improved'] > c['paired_samples']/2
    i, a = arms['identity']['macro'], arms['iar_true']['macro']
    checks['chunk_not_worse_identity'] = bool(i) and bool(a) and a['chunk'] <= i['chunk']+q.tolerance(i['chunk'])
    loo = contrasts['identity']['leave_one_out']
    robust['all_leave_one_positive'] = bool(i) and len(loo)==8 and all(x > q.tolerance(i['row0']) for x in loo.values())
    return {'metrics': arms, 'contrasts': contrasts, 'primary_checks': checks, 'robustness_checks': robust,
            'heldout_primary_gate_passed': all(checks.values()),
            'heldout_robustness_gate_passed': all(checks.values()) and all(robust.values()),
            'worst_excess_vs_identity': max([0.0]+[r['metrics']['iar_true']['row0']-r['metrics']['identity']['row0'] for r in rows]),
            'delay_counts': dict(Counter(str(r['delay']) for r in rows)), 'statistical_significance_claimed': False}


def evaluate(samples, model, runtime, output, counts):
    donors = q.pfx.donor_map(samples)
    by_key = {q.pfx.key(s): s for s in samples}
    rows = []
    (output/'predictions').mkdir()
    for s in samples:
        k = q.pfx.key(s)
        with pilot.phase(output, f'sample_{k}', 120):
            visuals = {'identity': q.cpu(s['inputs'][:2]), 'oracle': q.cpu(s['future'])}
            evidence = {}
            for context in ('true', 'zero', 'mismatched'):
                if context == 'mismatched' and donors[k] is None:
                    continue
                used = q.pfx.with_prefix(s, by_key[donors[k]]) if context == 'mismatched' else s
                actions = torch.zeros_like(s['actions']) if context == 'zero' else used['actions']
                ev = predict(model, s, actions, counts)
                evidence['iar_'+context], visuals['iar_'+context] = ev, ev['visual']
            values = {name: decode(runtime, s, z, output, counts, name) for name, z in visuals.items()}
            pilot.require_equal(values['identity'], s['archived_full_chunk'], 'Native identity differs')
            for a, b in zip(evidence['iar_zero']['raw_visual'], s['inputs'][:2], strict=True):
                pilot.require_equal(a, b.float(), 'Zero raw differs from identity')
            pilot.require_equal(values['iar_zero'], values['identity'], 'Zero full output differs')
            counts['identity_exact'] += 1
            counts['zero_exact'] += 1
            scores = {name: q.metrics(z, values[name], s, values['oracle']) for name, z in visuals.items() if name!='oracle'}
            row = {'key': list(k), 'delay': s['delay'], 'donor': list(donors[k]) if donors[k] else None, 'metrics': scores}
            torch.save({**row, 'visual': visuals, 'outputs': values, 'residuals': evidence},
                       output/'predictions'/('_'.join(map(str, k))+'.pt'))
            with (output/'metric_rows.jsonl').open('a') as f:
                f.write(json.dumps(row, allow_nan=False)+'\n')
            rows.append(row)
    return rows


def worker(args):
    faulthandler.register(signal.SIGUSR1, file=sys.stderr, all_threads=True)
    torch.set_num_threads(1)
    counts, budget, records, runtime = Counter(), q.Budget(), [], None
    calls = e.Calls(args.output/'calls.jsonl')
    result = {'experiment': 'F-IQ1', 'status': 'technical_failure', 'first_failure': None,
              'heldout_primary_gate_passed': False, 'heldout_robustness_gate_passed': False}
    try:
        validate(args.execution_head)
        with pilot.phase(args.output, 'environment_preflight', 30):
            result['runtime_environment'] = q.runtime_environment()
            native_factory = e.reference.make_native_env_factory()
            require(not torch.cuda.is_initialized(), 'Factory preflight initialized CUDA')
        with pilot.phase(args.output, 'load', 90):
            require(torch.cuda.get_device_name() == 'NVIDIA GeForce RTX 4070 Ti SUPER', 'GPU differs')
            saved = load(CHECKPOINT)
            q.checkpoint_valid(saved, 'centered')
            model = pilot.LightweightFutureLatentPredictor(pilot.config())
            model.load_state_dict(saved['state_dict'], strict=True)
            require(sum(p.numel() for p in model.parameters()) == 69680, 'Predictor size changed')
            model = model.eval().requires_grad_(False).cuda()
            counts['predictor_loads'] += 1
            torch.save(saved, args.output/'frozen_checkpoint_before.pt')
            policy, pre, post, report = e.load_runtime(e.POLICY, e.VLM)
            counts['vla_loads'] += 1
            write(args.output/'policy_load.json', report)
            runtime = q.Runtime(policy.model)
        with torch.no_grad(), pilot.phase(args.output, 'anchor_replay', 180):
            replay_anchors(model, runtime, args.output, counts)
        factory = q.natural.trace.checkpoint_factory(native_factory, calls)
        with pilot.phase(args.output, 'native_collection', 960):
            for spec in manifest()['rows']:
                print(f"F-IQ1 episode {spec['ordinal']} task={spec['task_id']} state={spec['initial_state_id']}", flush=True)
                record, _ = e.run_episode(spec, args.output/f"episode_{spec['ordinal']:03d}", policy, pre, post,
                    factory, budget, calls, engine_class=q.natural.NaturalEngine)
                records.append(record)
                require(record['status']=='completed', record['first_failure'] or 'Native collection incomplete')
        with torch.no_grad(), pilot.phase(args.output, 'offline_evaluation', 240):
            samples = q.extract(records, policy, pre, args.output, counts)
            rows = evaluate(samples, model, runtime, args.output, counts)
        n, m = len(rows), sum(r['donor'] is not None for r in rows)
        expected = {'encoding': n+8, 'decoder': 32+4*n+m, 'predictor': 32+4*n+2*m,
                    'anchor_exact': 16, 'current_exact': 8, 'identity_exact': n, 'zero_exact': n,
                    'vla_loads': 1, 'predictor_loads': 1}
        require(dict(counts)==expected, f'Formal counts differ: {dict(counts)} vs {expected}')
        after = q.opt.freeze_state(model)
        for name, value in after.items():
            pilot.require_equal(value, saved['state_dict'][name], 'Frozen weight changed')
        require(all(not p.requires_grad and p.grad is None for p in policy.parameters()), 'VLA not frozen')
        torch.save(after, args.output/'frozen_weights_after.pt')
        result.update(status='completed', samples=n, mismatched_samples=m, frozen_weights_unchanged=True,
                      vla_frozen=True, expected_counts=expected, **aggregate(rows))
    except BaseException:
        result['first_failure'] = traceback.format_exc()
    finally:
        calls.close()
        if runtime is not None:
            write(args.output/'offline_captures.json', runtime.captures)
            counts['offline_captures'] = len(runtime.captures)
            runtime.release_graph()
            result['graph_released'] = runtime.graph is None
        result.update(execution_head=args.execution_head, counts=dict(counts), native_budget=dict(budget.total),
            episodes_completed=sum(r['status']=='completed' for r in records), predictor_controls_environment=False,
            training_updates=0, backward=0, real_robot=0, old_qualification_reads=0, baseline_qualified=False,
            realtime_qualified=False, predictor_benefit_tested=False, risk_thresholds=None,
            old_confirmation='not_started_untouched', attempts=1, retries=0)
        write(args.output/'worker_result.json', result)
    return 0 if result['status']=='completed' else 2


def supervise(args):
    validate(args.execution_head)
    require(args.output==paths(args.execution_head)[1] and not args.output.exists(), 'Use unused registered output')
    args.output.mkdir()
    write(args.output/'manifest.json', manifest())
    command = [sys.executable, '-u', '-X', 'faulthandler', str(Path(__file__).resolve()),
               '--execution-head', args.execution_head, '--output', str(args.output), '--worker']
    start, utc = time.perf_counter(), datetime.now(UTC).isoformat()
    cursors, pending, active, reason, terminated, forced = {}, {}, {}, None, None, False
    with (args.output/'model.log').open('x') as log:
        child = subprocess.Popen(command, cwd=REPO, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        print(f'F-IQ1 worker={child.pid}', flush=True)
        try:
            while child.poll() is None:
                q.cov.parent.previous.update_pending(args.output, cursors, pending, active)
                now = time.perf_counter()
                expired = [v for v in pending.values() if now-v['timestamp']>v['limit']]
                phases = [v for v in active.values() if now-v['at']>v['limit']]
                if terminated is None and (expired or phases or now-start>=SOFT):
                    reason = {'calls': expired, 'phases': phases, 'soft_timeout': now-start>=SOFT}
                    os.kill(child.pid, signal.SIGUSR1)
                    os.killpg(child.pid, signal.SIGTERM)
                    terminated = now
                if now-start>=HARD or (terminated is not None and now-terminated>=5):
                    os.killpg(child.pid, signal.SIGKILL)
                    forced = True
                    break
                time.sleep(0.1)
        except BaseException:
            reason = reason or {'supervisor_exception': traceback.format_exc()}
            if child.poll() is None:
                os.killpg(child.pid, signal.SIGTERM)
                try:
                    child.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL)
                    forced = True
        code = child.wait()
    q.cov.parent.previous.update_pending(args.output, cursors, pending, active)
    file = args.output/'worker_result.json'
    result = json.loads(file.read_text()) if file.exists() else {'status':'technical_failure',
              'execution_head': args.execution_head, 'first_failure':'Worker receipt absent'}
    result['execution'] = {'child_pid': child.pid, 'child_exit_code': code, 'exit_confirmed': True,
        'started_at_utc': utc, 'finished_at_utc': datetime.now(UTC).isoformat(), 'wall_seconds': time.perf_counter()-start,
        'stop_reason': reason, 'forced_termination': forced, 'pending': pending, 'active': active}
    accounting = q.natural.trace.journal_accounting(q.natural.trace.read_events(args.output/'calls.jsonl'))
    result['accounting'] = accounting
    if code or reason or pending or active or accounting['call_errors'] or accounting['unknown_calls'] or not accounting['journal_consistent']:
        result.update(status='technical_failure', heldout_primary_gate_passed=False, heldout_robustness_gate_passed=False)
        result['first_failure'] = result.get('first_failure') or str(result['execution'])
    write(args.output/'result.json', result)
    print(json.dumps({k:result.get(k) for k in ('status','samples','heldout_primary_gate_passed','heldout_robustness_gate_passed','first_failure')}), flush=True)
    return 0 if result['status']=='completed' else 2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execution-head', required=True)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--prepare', action='store_true')
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args()
    signal.signal(signal.SIGTERM, e.request_shutdown)
    if args.prepare:
        require(not args.worker and args.output is None, 'Prepare cannot dispatch')
        prepare(args.execution_head)
        return 0
    args.output = (args.output or paths(args.execution_head)[1]).resolve()
    return worker(args) if args.worker else supervise(args)


if __name__=='__main__':
    raise SystemExit(main())
