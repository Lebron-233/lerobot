"""F-BRP1: change only the soft-penalty reference, old development data only.

Warm start, optimizer, schedule and coefficients match audited F-ITC1-R1.
References use training labels, never validation for optimization or selection.
"""
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
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path

import libero_identity_training as r
import torch

REPO, q, e, pilot = r.REPO, r.q, r.e, r.pilot
PREVIOUS = REPO / 'outputs/smolvla_identity_training_64a45518'
PRESERVED = REPO / 'outputs/smolvla_best_reference_development_20260917/preserved_worktree.json'
LIMITS = {'decoder': 597, 'predictor': 842, 'updates': 72, 'backward': 72}
CAPTURES, SOFT, HARD = 16, 600, 630
require, digest, write = r.require, r.digest, r.write


def paths(head):
    return (REPO / f'outputs/smolvla_best_reference_preparation_{head[:8]}',
            REPO / f'outputs/smolvla_best_reference_{head[:8]}')


def specification():
    return {'experiment': 'F-BRP1', 'updates': 72, 'warm_start_step': 72, 'seed': r.SEED,
            'lr': r.LR, 'weight_decay': r.WD, 'betas': [0.9, 0.999], 'eps': 1e-8,
            'foreach': False, 'gradient_clip': 1.0, 'coefficients': [1, 1, 1],
            'reference': 'per_training_sample_min_identity_frozen_per_metric',
            'checkpoint_selection': 'fixed_final_72', 'limits': LIMITS, 'captures': CAPTURES,
            'soft': SOFT, 'hard': HARD, 'qualification_reads': 0, 'new_env': 0}


def reference_bounds(sample, control):
    require(sample['split'] == 'train', 'Penalty references are training-only')
    result = {}
    for key in ('row0', 'chunk'):
        a, b = control['identity'][key], control['frozen'][key]
        require(math.isfinite(a) and math.isfinite(b) and min(a, b) >= 0, 'Invalid reference')
        result[key] = min(a, b)
    return result


def source_hashes():
    result = r.source_hashes()
    for file in (PRESERVED, PREVIOUS / 'independent_audit.json', PREVIOUS / 'result.json',
                 PREVIOUS / 'rows.jsonl', PREVIOUS / 'plain.pt', PREVIOUS / 'guarded.pt'):
        result[str(file.relative_to(REPO))] = digest(file)
    return result


def worktree_gate(head):
    require(subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip() == head,
            'Execution HEAD changed')
    status = subprocess.check_output(['git', 'status', '--porcelain'], cwd=REPO, text=True)
    current = {s[3:]: {'status': s[:2], 'sha256': digest(REPO / s[3:])} for s in status.splitlines()}
    require(current == json.loads(PRESERVED.read_text()), 'Original pending files changed')


def load_data():
    audited = json.loads((PREVIOUS / 'independent_audit.json').read_text())
    previous = json.loads((PREVIOUS / 'result.json').read_text())
    require(audited['independent_contract_accepted'] is True and previous['execution_head'] ==
            '64a455185e911a6adc1996b349a085a68b47bb86', 'Previous control is not audited')
    data = r.load_data()
    rows = audited['per_sample']
    require([tuple(x['key']) for x in rows] == [q.pfx.key(s) for s in data[0]], 'Prior pairing differs')
    return data, {tuple(x['key']): x for x in rows}


def prepare(head):
    worktree_gate(head)
    environment = q.runtime_environment()
    prep, out = paths(head)
    require(not prep.exists() and not out.exists(), 'Preparation/output exists')
    data, previous = load_data()
    samples, _, _, _, manifest, weights, scales, _ = data
    targets = {','.join(map(str, q.pfx.key(s))): reference_bounds(s, previous[q.pfx.key(s)]['metrics'])
               for s in samples if s['split'] == 'train'}
    require(len(targets) == 72 and not torch.cuda.is_initialized(), 'Preparation scope differs')
    prep.mkdir()
    write(prep / 'preparation.json', {'execution_head': head, 'specification': specification(),
          'source_hashes': source_hashes(), 'manifest': manifest, 'weights': weights, 'scales': scales,
          'runtime_environment': environment, 'training_targets': targets,
          'new_env': 0, 'qualification_reads': 0, 'model_forwards': 0, 'cuda_initialized': False})
    print(json.dumps({'prepared': True, 'sha256': digest(prep / 'preparation.json'),
                      'prep': str(prep), 'output': str(out)}), flush=True)


def validate(head):
    worktree_gate(head)
    prep, out = paths(head)
    saved = json.loads((prep / 'preparation.json').read_text())
    require(saved['execution_head'] == head and saved['specification'] == specification(), 'Contract changed')
    require(saved['source_hashes'] == source_hashes(), 'Source bytes changed')
    require(saved['runtime_environment'] == q.runtime_environment(), 'Runtime environment changed')
    body = (prep / 'registration.md').read_text()
    post = json.loads((prep / 'registration_post.json').read_text())
    got = json.loads((prep / 'registration_readback.json').read_text())
    require(type(got.get('id')) is int and got['id'] == post.get('id') and
            got.get('body') == post.get('body') == body and got.get('issue_url') ==
            'https://api.github.com/repos/Lebron-233/lerobot/issues/1' and
            all(x in body for x in (f'F-BRP1-REGISTER:{head}', str(out), digest(prep / 'preparation.json'))),
            'Actual-ID registration exact gate failed')
    return saved


def predict(model, sample, actions, counts):
    require(counts['predictor'] + 2 <= LIMITS['predictor'], 'Predictor budget before dispatch')
    return r.prediction(model, sample, actions, counts)


class Runtime(r.Runtime):
    def _capture(self, inputs):
        require(len(self.captures) < CAPTURES, 'Capture budget before dispatch')
        return super()._capture(inputs)


def decode(policy, runtime, sample, visual, out, counts, label, gradients=False):
    require(counts['decoder'] < LIMITS['decoder'], 'Decoder budget before dispatch')
    return r.decode(policy, runtime, sample, visual, out, counts, label, gradients)


def statistics(rows):
    summary = {}
    for split in ('train', 'validation'):
        selected = [x for x in rows if x['split'] == split]
        arms = {}
        for arm in sorted({a for x in selected for a in x['metrics']}):
            groups = defaultdict(list)
            for row in selected:
                if arm in row['metrics']:
                    groups['/'.join(map(str, row['key'][:2]))].append(row['metrics'][arm])
            eps = {k: {m: math.fsum(x[m] for x in v) / len(v) for m in ('row0', 'chunk', 'latent')}
                   for k, v in groups.items()}
            arms[arm] = {'samples': sum(map(len, groups.values())), 'episodes': eps,
                         'macro': {m: math.fsum(x[m] for x in eps.values()) / len(eps)
                                   for m in ('row0', 'chunk', 'latent')}}
        contrasts = {}
        for control in ('identity', 'frozen', 'plain_true', 'guarded_true', 'bestref_mismatched'):
            subset = [x for x in selected if control in x['metrics']]
            groups, details = defaultdict(list), []
            for row in subset:
                a, b = row['metrics']['bestref_true']['row0'], row['metrics'][control]['row0']
                groups['/'.join(map(str, row['key'][:2]))].append((a, b))
                details.append({'key': row['key'], 'benefit': b-a, 'direction':
                    'improved' if b-a > r.p.tolerance(b) else 'worsened' if a-b > r.p.tolerance(b) else 'tied'})
            eps = {k: {'treatment': math.fsum(a for a, _ in v)/len(v),
                       'control': math.fsum(b for _, b in v)/len(v)} for k, v in groups.items()}
            gains = {k: v['control']-v['treatment'] for k, v in eps.items()}
            contrasts[control] = {'samples': len(subset), 'episodes': eps,
                'macro_benefit': math.fsum(gains.values())/len(gains),
                'sample_directions': dict(Counter(x['direction'] for x in details)),
                'episode_improved': sum(gains[k] > r.p.tolerance(v['control']) for k, v in eps.items()),
                'leave_one_out': {k: math.fsum(v for j, v in gains.items() if j != k)/(len(gains)-1) for k in gains},
                'per_sample': details}
        harms = {a: max([0.0]+[x['metrics'][a]['row0']-x['metrics']['identity']['row0'] for x in selected])
                 for a in ('frozen', 'plain_true', 'guarded_true', 'bestref_true')}
        summary[split] = {'metrics': arms, 'contrasts': contrasts, 'worst_excess': harms}
    v, t = summary['validation'], summary['train']
    target = v['metrics']['bestref_true']['macro']
    checks = {}
    for name in ('identity', 'bestref_mismatched'):
        b = v['metrics'][name]['macro']['row0']
        checks[f'row0_better_{name}'] = target['row0'] < b-r.p.tolerance(b)
    b = v['metrics']['frozen']['macro']['row0']
    checks['row0_not_worse_frozen'] = target['row0'] <= b+r.p.tolerance(b)
    for name in ('frozen', 'guarded_true', 'plain_true'):
        b = v['metrics'][name]['macro']['chunk']
        checks[f'chunk_better_{name}'] = target['chunk'] < b-r.p.tolerance(b)
        b = t['worst_excess'][name]
        checks[f'train_worst_better_{name}'] = t['worst_excess']['bestref_true'] < b-r.p.tolerance(b)
    c = v['contrasts']['identity']
    checks['three_of_four_episodes'] = c['episode_improved'] >= 3
    checks['sample_majority'] = c['sample_directions'].get('improved', 0) > 8
    b = v['worst_excess']['frozen']
    checks['validation_worst_not_worse_frozen'] = v['worst_excess']['bestref_true'] <= b+r.p.tolerance(b)
    return {'splits': summary, 'development_checks': checks,
            'development_followup_supported': all(checks.values()), 'independent_qualification_claimed': False}


def worker(args):
    faulthandler.register(signal.SIGUSR1, file=sys.stderr, all_threads=True)
    torch.set_num_threads(1)
    counts, runtime = Counter(), None
    result = {'experiment': 'F-BRP1', 'execution_head': args.execution_head,
              'status': 'technical_failure', 'first_failure': None}
    try:
        with pilot.phase(args.output, 'load_development', 30):
            prepared = validate(args.execution_head)
            data, previous = load_data()
            samples, _, residuals, donors, manifest, weights, scales, refs = data
            require(manifest == prepared['manifest'] and weights == prepared['weights'] and scales == prepared['scales'],
                    'Prepared metadata differs')
            train = [s for s in samples if s['split'] == 'train']
            order = q.cov.schedule(train, 'multi_conditioned')
            require(len(order) == len(set(order)) == 72, 'Training order differs')
            by_key = {q.pfx.key(s): s for s in samples}
        with pilot.phase(args.output, 'load_models', 90):
            require(torch.cuda.get_device_name() == 'NVIDIA GeForce RTX 4070 Ti SUPER', 'GPU differs')
            policy, _, _, report = e.load_runtime(e.POLICY, e.VLM)
            counts['vla_loads'] += 1
            write(args.output / 'policy_load.json', report)
            saved = torch.load(r.p.ARCHIVE / 'centered.pt', map_location='cpu', weights_only=False)
            q.checkpoint_valid(saved, 'centered')
            torch.manual_seed(r.SEED)
            model = pilot.LightweightFutureLatentPredictor(pilot.config()).cuda()
            model.load_state_dict(saved['state_dict'], strict=True)
            require(sum(v.numel() for v in model.parameters()) == 69680, 'Predictor size differs')
            model.eval()
            counts['predictor_loads'] += 1
            torch.save(r.freeze(model), args.output / 'initial.pt')
            runtime = Runtime(policy.model)
        for name in ('controls', 'training', 'predictions'):
            (args.output / name).mkdir()
        with torch.no_grad(), pilot.phase(args.output, 'initial_replay', 180):
            for sample in samples:
                k, ref = q.pfx.key(sample), refs[q.pfx.key(sample)]
                visual, ev = predict(model, sample, sample['actions'], counts)
                for field in ('action_delta', 'zero_delta'):
                    for a, b in zip(ev[field], residuals[k, 'true'][field], strict=True):
                        pilot.require_equal(a, b, 'Initial residual differs')
                for a, b in zip(ev['raw'], ref['raw_new']['identity_true'], strict=True):
                    pilot.require_equal(a, b, 'Initial raw differs')
                outputs, metrics = {}, {}
                for name, old_name, z in (('identity', 'identity', sample['inputs'][:2]),
                    ('old_centered', 'centered', ref['visual']['centered']), ('frozen', 'identity_true', visual)):
                    value = decode(policy, runtime, sample, z, args.output, counts, 'initial_'+name)
                    pilot.require_equal(value, ref['outputs'][old_name], 'Initial output differs')
                    outputs[name], metrics[name] = value, r.offline_metrics(z, value, sample)
                    counts[name+'_exact'] += 1
                torch.save({'key': list(k), 'outputs': outputs, 'metrics': metrics, 'evidence': ev},
                           args.output / 'controls' / ('_'.join(map(str, k))+'.pt'))
        runtime.release_graph()
        model.train().requires_grad_(True)
        optimizer = torch.optim.AdamW(model.parameters(), lr=r.LR, weight_decay=r.WD,
                                     betas=(0.9, 0.999), eps=1e-8, foreach=False)
        with pilot.phase(args.output, 'train_bestref', 180):
            for step, index in enumerate(order, 1):
                sample = train[index]
                k = q.pfx.key(sample)
                bounds = reference_bounds(sample, previous[k]['metrics'])
                require(bounds == prepared['training_targets'][','.join(map(str, k))], 'Training reference changed')
                with pilot.phase(args.output, f'update_{step}', 30):
                    require(counts['updates'] < 72 and counts['backward'] < 72, 'Update budget before dispatch')
                    optimizer.zero_grad(set_to_none=True)
                    visual, ev = predict(model, sample, sample['actions'], counts)
                    value = decode(policy, runtime, sample, visual, args.output, counts, 'train', True)
                    metrics = q.cov.metric_values(visual, value, sample)
                    weight = weights['rows'][index]['case_weight']
                    loss = r.objective(metrics, bounds, scales, weight, 'guarded')
                    require(torch.isfinite(loss), 'Nonfinite loss')
                    counts['backward'] += 1
                    loss.backward()
                    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    require(torch.isfinite(norm) and all(v.grad is None for v in policy.parameters()), 'Gradient/VLA failure')
                    gradients = {n: v.grad.detach().cpu().clone() for n, v in model.named_parameters() if v.grad is not None}
                    require(gradients and all(torch.isfinite(v).all() for v in gradients.values()), 'Invalid gradients')
                    counts['updates'] += 1
                    optimizer.step()
                    record = {'step': step, 'key': list(k), 'weight': weight, 'scales': scales, 'bounds': bounds,
                        'metrics': dict(zip(('latent', 'row0', 'chunk'), map(float, metrics), strict=True)),
                        'objective': float(loss), 'gradient_norm_before_clip': float(norm), 'evidence': ev,
                        'output': value.detach().cpu().clone(), 'clipped_gradients': gradients,
                        'weights_after': r.freeze(model)}
                    torch.save(record, args.output / 'training' / f'{step:03d}.pt')
                    with (args.output / 'training.jsonl').open('a') as f:
                        f.write(json.dumps({x: record[x] for x in ('step', 'key', 'bounds', 'metrics', 'objective', 'gradient_norm_before_clip')})+'\n')
        model.eval().requires_grad_(False)
        torch.save({'step': 72, 'source_step': 72, 'specification': specification(), 'state_dict': r.freeze(model)},
                   args.output / 'bestref.pt')
        rows = []
        with torch.no_grad(), pilot.phase(args.output, 'evaluate_bestref', 180):
            for sample in samples:
                k = q.pfx.key(sample)
                outputs, metrics, evidence = {}, {}, {}
                for context in ('true', 'zero', 'mismatched'):
                    if context == 'mismatched' and donors[k] is None:
                        continue
                    actions = (torch.zeros_like(sample['actions']) if context == 'zero' else
                               by_key[donors[k]]['actions'] if context == 'mismatched' else sample['actions'])
                    visual, ev = predict(model, sample, actions, counts)
                    value = decode(policy, runtime, sample, visual, args.output, counts, context)
                    if context == 'zero':
                        for a, b in zip(ev['raw'], sample['inputs'][:2], strict=True):
                            pilot.require_equal(a, b.float(), 'Zero raw differs')
                        pilot.require_equal(value, refs[k]['outputs']['identity'], 'Zero output differs')
                        counts['zero_exact'] += 1
                    outputs[context], metrics[context], evidence[context] = value, r.offline_metrics(visual, value, sample), ev
                torch.save({'key': list(k), 'donor': list(donors[k]) if donors[k] else None,
                            'outputs': outputs, 'metrics': metrics, 'evidence': evidence},
                           args.output / 'predictions' / ('_'.join(map(str, k))+'.pt'))
                scores = dict(previous[k]['metrics'])
                scores.update({'bestref_'+c: m for c, m in metrics.items()})
                rows.append({'key': list(k), 'split': sample['split'], 'delay': sample['delay'],
                             'donor': list(donors[k]) if donors[k] else None, 'metrics': scores})
        with (args.output / 'rows.jsonl').open('x') as f:
            for row in rows:
                f.write(json.dumps(row, allow_nan=False)+'\n')
        require(all(counts[k] == v for k, v in LIMITS.items()) and counts['zero_exact'] == 88, 'Incomplete counts')
        require(source_hashes() == prepared['source_hashes'], 'Sources changed')
        require(all(not v.requires_grad and v.grad is None for v in policy.parameters()), 'VLA not frozen')
        result.update(status='completed', vla_frozen=True, samples=88, **statistics(rows))
    except BaseException:
        result['first_failure'] = traceback.format_exc()
    finally:
        if runtime is not None:
            write(args.output / 'captures.json', runtime.captures)
            counts['captures'] = len(runtime.captures)
            runtime.release_graph()
            result['graph_released'] = runtime.graph is None
        result.update(counts=dict(counts), specification=specification(), attempts=1, retries=0,
            new_env=0, image_encodings=0, qualification_reads=0, real_robot=0, baseline_qualified=False,
            realtime_qualified=False, predictor_benefit_tested=False, risk_thresholds=None, old_confirmation='untouched')
        write(args.output / 'worker_result.json', result)
    return 0 if result['status'] == 'completed' else 2


def supervise(args):
    validate(args.execution_head)
    require(args.output == paths(args.execution_head)[1] and not args.output.exists(), 'Use unique output')
    args.output.mkdir()
    command = [sys.executable, '-u', str(Path(__file__).resolve()), '--execution-head', args.execution_head,
               '--output', str(args.output), '--worker']
    start, utc = time.perf_counter(), datetime.now(UTC).isoformat()
    cursors, pending, active, reason, terminated, forced = {}, {}, {}, None, None, False
    with (args.output / 'model.log').open('x') as log:
        child = subprocess.Popen(command, cwd=REPO, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        print(f'F-BRP1 worker={child.pid}', flush=True)
        try:
            while child.poll() is None:
                q.cov.parent.previous.update_pending(args.output, cursors, pending, active)
                now = time.perf_counter()
                expired = [v for v in active.values() if now-v['at'] > v['limit']]
                if terminated is None and (expired or now-start >= SOFT):
                    reason = {'expired': expired, 'soft_timeout': now-start >= SOFT}
                    os.kill(child.pid, signal.SIGUSR1)
                    os.killpg(child.pid, signal.SIGTERM)
                    terminated = now
                if now-start >= HARD or (terminated is not None and now-terminated >= 5):
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
    file = args.output / 'worker_result.json'
    result = json.loads(file.read_text()) if file.exists() else {'status': 'technical_failure',
             'first_failure': 'Worker receipt absent', 'execution_head': args.execution_head}
    result['execution'] = {'child_pid': child.pid, 'child_exit_code': code, 'exit_confirmed': True,
        'started_at_utc': utc, 'finished_at_utc': datetime.now(UTC).isoformat(), 'wall_seconds': time.perf_counter()-start,
        'stop_reason': reason, 'forced_termination': forced, 'pending': pending, 'active': active}
    if code or reason or pending or active:
        result['status'] = 'technical_failure'
    write(args.output / 'result.json', result)
    print(json.dumps({k: result.get(k) for k in ('status', 'first_failure', 'development_followup_supported')}), flush=True)
    return 0 if result['status'] == 'completed' else 2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execution-head', required=True)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--prepare', action='store_true')
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args()
    signal.signal(signal.SIGTERM, e.request_shutdown)
    if args.prepare:
        require(not args.worker and args.output is None, 'Prepare cannot start worker')
        prepare(args.execution_head)
        return 0
    args.output = (args.output or paths(args.execution_head)[1]).resolve()
    return worker(args) if args.worker else supervise(args)


if __name__ == '__main__':
    raise SystemExit(main())
