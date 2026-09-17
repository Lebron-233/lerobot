"""F-UDI1: fixed archived-update attribution, not training or checkpoint selection."""
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

import libero_best_reference as b
import torch

r, q, e, pilot = b.r, b.q, b.e, b.pilot
REPO = b.REPO
SOURCE = REPO / 'outputs/smolvla_best_reference_e35cfc17'
PRESERVED = REPO / 'outputs/smolvla_update_diagnostic_development_20260917/preserved_worktree.json'
SENTINELS = ((4, 46, 5), (5, 48, 6))
SOFT, HARD = 480, 510
EXPECTED = {'vla_loads': 1, 'predictor_loads': 1, 'predictor': 580, 'decoder': 290,
            'backward_calls': 72, 'component_vjps': 216, 'pre_exact': 72,
            'gradient_replay': 72, 'state_loads': 217}
require, digest, write = b.require, b.digest, b.write


def load(path):
    return torch.load(path, map_location='cpu', weights_only=False)


def paths(head):
    return (REPO / f'outputs/smolvla_update_diagnostic_preparation_{head[:8]}',
            REPO / f'outputs/smolvla_update_diagnostic_{head[:8]}')


def specification():
    return {'experiment': 'F-UDI1', 'source': str(SOURCE), 'training_samples': 72,
            'sentinels': [list(k) for k in SENTINELS], 'checkpoints': list(range(73)),
            'expected': EXPECTED, 'new_updates': 0, 'checkpoint_selection': False,
            'soft': SOFT, 'hard': HARD, 'gradient_rtol': 1e-5, 'gradient_atol': 1e-7,
            'qualification_reads': 0, 'new_env': 0, 'validation_forward': 0}


def source_hashes():
    files = [PRESERVED, SOURCE / 'initial.pt', SOURCE / 'bestref.pt', SOURCE / 'result.json',
             SOURCE / 'independent_audit.json', SOURCE / 'training.jsonl', SOURCE / 'rows.jsonl']
    files += sorted((SOURCE / 'training').glob('*.pt'))
    # The original development source contains both splits; only train is evaluated here.
    result = b.source_hashes()
    for f in files:
        result[str(f.relative_to(REPO))] = digest(f)
    for f in sorted((SOURCE / 'controls').glob('*.pt')) + sorted((SOURCE / 'predictions').glob('*.pt')):
        result[str(f.relative_to(REPO))] = digest(f)
    return result


def worktree_gate(head):
    require(subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip() == head,
            'Execution HEAD changed')
    status = subprocess.check_output(['git', 'status', '--porcelain'], cwd=REPO, text=True)
    state = {s[3:]: {'status': s[:2], 'sha256': digest(REPO / s[3:])} for s in status.splitlines()}
    require(state == json.loads(PRESERVED.read_text()), 'Preserved worktree changed')


def data():
    require(json.loads((SOURCE / 'independent_audit.json').read_text())['independent_contract_accepted'] is True,
            'BRP source not accepted')
    samples = r.load_data()[0]
    train = [s for s in samples if s['split'] == 'train']
    order = q.cov.schedule(train, 'multi_conditioned')
    require(len(train) == len(order) == len(set(order)) == 72, 'Training coverage changed')
    return [train[i] for i in order]


def prepare(head):
    worktree_gate(head)
    environment = q.runtime_environment()
    prep, out = paths(head)
    require(not prep.exists() and not out.exists(), 'Preparation/output exists')
    ordered = data()
    manifest = []
    for step, s in enumerate(ordered, 1):
        saved = load(SOURCE / 'training' / f'{step:03d}.pt')
        require(saved['key'] == list(q.pfx.key(s)), 'Archived schedule changed')
        manifest.append({'step': step, 'key': saved['key']})
    prep.mkdir()
    write(prep / 'preparation.json', {'execution_head': head, 'specification': specification(),
          'source_hashes': source_hashes(), 'manifest': manifest, 'environment': environment,
          'cuda_initialized': torch.cuda.is_initialized(), 'new_env': 0, 'model_forwards': 0})
    print(json.dumps({'prepared': True, 'sha256': digest(prep / 'preparation.json'),
                      'prep': str(prep), 'output': str(out)}), flush=True)


def validate(head):
    worktree_gate(head)
    prep, out = paths(head)
    saved = json.loads((prep / 'preparation.json').read_text())
    require(saved['specification'] == specification() and saved['execution_head'] == head, 'Contract changed')
    require(saved['source_hashes'] == source_hashes(), 'Source bytes changed')
    require(saved['environment'] == q.runtime_environment(), 'Environment changed')
    body = (prep / 'registration.md').read_text()
    post = json.loads((prep / 'registration_post.json').read_text())
    got = json.loads((prep / 'registration_readback.json').read_text())
    require(type(got.get('id')) is int and post.get('id') == got['id'] and
            post.get('body') == got.get('body') == body and got.get('issue_url') ==
            'https://api.github.com/repos/Lebron-233/lerobot/issues/1' and all(x in body for x in
            (f'F-UDI1-REGISTER:{head}', str(out), digest(prep / 'preparation.json'))), 'Registration differs')
    return saved


def components(metrics, bounds, scales, weight):
    z, a, c = metrics
    return {'latent': z / scales['latent'],
            'row0': weight*a/scales['row0'] + torch.relu(a-bounds['row0'])/scales['row0'],
            'chunk': c/scales['chunk'] + torch.relu(c-bounds['chunk'])/scales['chunk']}


def flat(values):
    return torch.cat([v.detach().cpu().float().reshape(-1) for v in values])


def vector_stats(grads, displacement):
    g = {k: v.double() for k, v in grads.items()}
    d = displacement.double()
    norms = {k: float(v.norm()) for k, v in g.items()}
    cosines = {}
    for a, c in (('latent', 'row0'), ('chunk', 'row0'), ('total', 'row0')):
        den = norms[a]*norms[c]
        cosines[a+'__'+c] = float(torch.dot(g[a], g[c]))/den if den else None
    residual = g['total']-g['latent']-g['row0']-g['chunk']
    return {'norms': norms, 'cosines': cosines,
            'directional': {k: float(torch.dot(v, d)) for k, v in g.items()},
            'component_sum_relative_residual': float(residual.norm())/max(norms['total'], 1e-30)}


def attribution(initial, before, after, final):
    return {m: {'earlier': before[m]-initial[m], 'own': after[m]-before[m],
                'later': final[m]-after[m], 'net': final[m]-initial[m]}
            for m in ('row0', 'chunk', 'latent', 'objective')}


def scores(visual, value, sample, archived):
    result = r.offline_metrics(visual, value, sample)
    z, a, c = (result[k] for k in ('latent', 'row0', 'chunk'))
    sc, bound, w = archived['scales'], archived['bounds'], archived['weight']
    result['objective'] = z/sc['latent']+w*a/sc['row0']+c/sc['chunk']
    result['objective'] += max(0.0, a-bound['row0'])/sc['row0']+max(0.0, c-bound['chunk'])/sc['chunk']
    return result


def forward(model, policy, sample, output, counts):
    require(counts['predictor']+2 <= EXPECTED['predictor'] and counts['decoder'] < EXPECTED['decoder'],
            'Forward budget exhausted before dispatch')
    visual, evidence = r.prediction(model, sample, sample['actions'], counts)
    counts['decoder'] += 1
    with pilot.phase(output, f"decode_{counts['decoder']}", 30):
        v = tuple(t.cuda() for t in sample['inputs'])
        value = policy.model.sample_actions(None, None, v[4], v[5], v[6], noise=v[7].clone(),
            future_image_tokens=tuple(t.to(dtype=v[i].dtype) for i, t in enumerate(visual)),
            future_image_token_masks=tuple(v[2:4]))
        require(value.shape == (1, 50, 32) and torch.isfinite(value).all(), 'Invalid decoder output')
    return visual, evidence, value


def restore(model, weights, counts):
    require(counts['state_loads'] < EXPECTED['state_loads'], 'State load budget exhausted')
    model.load_state_dict(weights, strict=True)
    counts['state_loads'] += 1


def worker(args):
    faulthandler.register(signal.SIGUSR1, file=sys.stderr, all_threads=True)
    torch.set_num_threads(1)
    result = {'experiment': 'F-UDI1', 'execution_head': args.execution_head,
              'status': 'technical_failure', 'first_failure': None}
    counts = Counter()
    try:
        prep = validate(args.execution_head)
        ordered = data()
        initial = load(SOURCE / 'initial.pt')
        for name in ('samples', 'sentinels'):
            (args.output / name).mkdir()
        with pilot.phase(args.output, 'load_models', 90):
            require(torch.cuda.get_device_name() == 'NVIDIA GeForce RTX 4070 Ti SUPER', 'GPU differs')
            policy, _, _, report = e.load_runtime(e.POLICY, e.VLM)
            counts['vla_loads'] += 1
            write(args.output / 'policy_load.json', report)
            torch.manual_seed(r.SEED)
            model = pilot.LightweightFutureLatentPredictor(pilot.config()).cuda().train()
            require(sum(p.numel() for p in model.parameters()) == 69680, 'Predictor size changed')
            counts['predictor_loads'] += 1
            names = [n for n, _ in model.named_parameters()]
            params = tuple(model.parameters())
            write(args.output / 'parameter_layout.json', [{'name': n, 'shape': list(v.shape)}
                                                         for n, v in model.named_parameters()])
        previous, rows = initial, []
        with pilot.phase(args.output, 'own_update_replay', 270):
            for step, sample in enumerate(ordered, 1):
                with pilot.phase(args.output, f'own_{step}', 60):
                    old = load(SOURCE / 'training' / f'{step:03d}.pt')
                    restore(model, previous, counts)
                    model.zero_grad(set_to_none=True)
                    visual, ev, value = forward(model, policy, sample, args.output, counts)
                    for field in ('raw', 'visual', 'action_delta', 'zero_delta'):
                        for a, c in zip(ev[field], old['evidence'][field], strict=True):
                            pilot.require_equal(a, c, 'Pre-update predictor replay differs')
                    pilot.require_equal(value.detach().cpu(), old['output'], 'Pre-update eager output differs')
                    counts['pre_exact'] += 1
                    metrics = q.cov.metric_values(visual, value, sample)
                    loss = r.objective(metrics, old['bounds'], old['scales'], old['weight'], 'guarded')
                    require(counts['backward_calls'] < 72, 'Backward budget exhausted')
                    loss.backward(retain_graph=True)
                    counts['backward_calls'] += 1
                    total = flat([p.grad if p.grad is not None else torch.zeros_like(p) for p in params])
                    norm = torch.nn.utils.clip_grad_norm_(params, 1.0)
                    for n, p in model.named_parameters():
                        if p.grad is None:
                            require(n not in old['clipped_gradients'], 'Missing replay gradient')
                        else:
                            require(n in old['clipped_gradients'] and torch.allclose(p.grad.detach().cpu(),
                                old['clipped_gradients'][n], rtol=1e-5, atol=1e-7), 'Archived gradient replay differs')
                    require(math.isclose(float(norm), old['gradient_norm_before_clip'], rel_tol=1e-5, abs_tol=1e-7),
                            'Gradient norm replay differs')
                    counts['gradient_replay'] += 1
                    grads = {'total': total}
                    for name, term in components(metrics, old['bounds'], old['scales'], old['weight']).items():
                        require(counts['component_vjps'] < 216, 'VJP budget exhausted')
                        values = torch.autograd.grad(term, params, retain_graph=True, allow_unused=True)
                        grads[name] = flat([g if g is not None else torch.zeros_like(p)
                                           for g, p in zip(values, params, strict=True)])
                        counts['component_vjps'] += 1
                    before = scores(visual, value, sample, old)
                    before_output = value.detach().cpu().clone()
                    del loss, metrics, visual, value, values, term
                    displacement = flat([old['weights_after'][n].double()-previous[n].double() for n in names])
                    model.zero_grad(set_to_none=True)
                    restore(model, old['weights_after'], counts)
                    with torch.no_grad():
                        zv, after_ev, after_out = forward(model, policy, sample, args.output, counts)
                    k = '_'.join(map(str, q.pfx.key(sample)))
                    control = load(SOURCE / 'controls' / f'{k}.pt')
                    final = load(SOURCE / 'predictions' / f'{k}.pt')
                    initial_score = scores(control['evidence']['visual'], control['outputs']['frozen'], sample, old)
                    final_score = scores(final['evidence']['true']['visual'], final['outputs']['true'], sample, old)
                    after = scores(zv, after_out, sample, old)
                    row = {'step': step, 'key': old['key'], 'initial': initial_score, 'before': before,
                           'after': after, 'final': final_score, 'attribution': attribution(initial_score, before, after, final_score),
                           'gradients': vector_stats(grads, displacement)}
                    torch.save({'row': row, 'before_evidence': ev, 'before_output': before_output,
                                'after_evidence': after_ev, 'after_output': after_out.detach().cpu(),
                                'gradients': grads, 'displacement': displacement}, args.output / 'samples' / f'{step:03d}.pt')
                    rows.append(row)
                    previous = old['weights_after']
                    del zv, after_out, grads, old, ev, after_ev, control, final
        model.zero_grad(set_to_none=True)
        by_key = {q.pfx.key(s): s for s in ordered}
        with torch.no_grad(), pilot.phase(args.output, 'sentinel_trajectories', 120):
            for checkpoint in range(73):
                weights = initial if checkpoint == 0 else load(SOURCE / 'training' / f'{checkpoint:03d}.pt')['weights_after']
                restore(model, weights, counts)
                for key in SENTINELS:
                    sample = by_key[key]
                    zv, ev, value = forward(model, policy, sample, args.output, counts)
                    torch.save({'key': list(key), 'checkpoint': checkpoint, 'evidence': ev,
                                'output': value.detach().cpu()}, args.output / 'sentinels' /
                               (f'{key[0]}_{key[1]}_{key[2]}_{checkpoint:03d}.pt'))
        require(dict(counts) == EXPECTED, f'Formal counts differ: {dict(counts)}')
        require(all(not p.requires_grad and p.grad is None for p in policy.parameters()), 'VLA changed')
        require(source_hashes() == prep['source_hashes'], 'Archived bytes changed')
        write(args.output / 'rows.json', rows)
        result.update(status='completed', diagnostic_complete=True, vla_frozen=True)
    except BaseException:
        result['first_failure'] = traceback.format_exc()
    result.update(counts=dict(counts), new_updates=0, new_env=0, image_encodings=0, qualification_reads=0,
                  validation_forward=0, optimizer_steps=0, graph_captures=0, attempts=1, retries=0,
                  model_checkpoint_selection=False, deployment_qualified=False, risk_thresholds=None)
    write(args.output / 'worker_result.json', result)
    return 0 if result['status'] == 'completed' else 2


def supervise(args):
    validate(args.execution_head)
    require(args.output == paths(args.execution_head)[1] and not args.output.exists(), 'Unique output required')
    args.output.mkdir()
    command = [sys.executable, '-u', str(Path(__file__).resolve()), '--execution-head', args.execution_head,
               '--output', str(args.output), '--worker']
    start, utc = time.perf_counter(), datetime.now(UTC).isoformat()
    cursors, pending, active = {}, {}, {}
    reason, terminated, forced = None, None, False
    with (args.output / 'model.log').open('x') as log:
        child = subprocess.Popen(command, cwd=REPO, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        print(f'F-UDI1 worker={child.pid}', flush=True)
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
    f = args.output / 'worker_result.json'
    result = json.loads(f.read_text()) if f.exists() else {'status': 'technical_failure',
             'execution_head': args.execution_head, 'first_failure': 'Worker receipt absent'}
    result['execution'] = {'child_pid': child.pid, 'child_exit_code': code, 'exit_confirmed': True,
        'started_at_utc': utc, 'finished_at_utc': datetime.now(UTC).isoformat(), 'wall_seconds': time.perf_counter()-start,
        'stop_reason': reason, 'forced_termination': forced, 'pending': pending, 'active': active}
    if code or reason or pending or active:
        result['status'] = 'technical_failure'
    write(args.output / 'result.json', result)
    print(json.dumps({'status': result['status'], 'first_failure': result['first_failure']}), flush=True)
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
        require(not args.worker and args.output is None, 'Prepare cannot dispatch')
        prepare(args.execution_head)
        return 0
    args.output = (args.output or paths(args.execution_head)[1]).resolve()
    return worker(args) if args.worker else supervise(args)


if __name__ == '__main__':
    raise SystemExit(main())
