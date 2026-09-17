"""F-TUA1: finite-update acceptance on OLD training data; validation only at the end.

One fixed AdamW proposal per training example. Reject restores BOTH weights and
optimizer state. Full training checks are an expensive offline training mechanism,
not a deployable safety gate or a new qualification experiment.
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
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

import libero_best_reference as b
import torch

r, q, e, pilot = b.r, b.q, b.e, b.pilot
REPO = b.REPO
SOURCE = REPO / 'outputs/smolvla_best_reference_e35cfc17'
UDI = REPO / 'outputs/smolvla_update_diagnostic_bec2151e'
PRESERVED = REPO / 'outputs/smolvla_transactional_updates_development_20260917/preserved_worktree.json'
LIMITS = {'predictor': 11210, 'decoder': 5781, 'backward': 72, 'proposals': 72}
CAPTURES, SOFT, HARD = 448, 1500, 1530
require, write, digest = b.require, b.write, b.digest


def load(path):
    return torch.load(path, map_location='cpu', weights_only=False)


def cpu_tree(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {k: cpu_tree(v) for k, v in value.items()}
    if isinstance(value, list):
        return [cpu_tree(v) for v in value]
    if isinstance(value, tuple):
        return tuple(cpu_tree(v) for v in value)
    return copy.deepcopy(value)


def exact_tree(a, c):
    if isinstance(a, torch.Tensor):
        require(isinstance(c, torch.Tensor) and a.dtype == c.dtype and
                torch.equal(a.cpu(), c.cpu()), 'Tensor state differs')
    elif isinstance(a, dict):
        require(isinstance(c, dict) and a.keys() == c.keys(), 'State keys differ')
        for k in a:
            exact_tree(a[k], c[k])
    elif isinstance(a, (list, tuple)):
        require(type(a) is type(c) and len(a) == len(c), 'State sequence differs')
        for x, y in zip(a, c, strict=True):
            exact_tree(x, y)
    else:
        require(type(a) is type(c) and a == c, 'Scalar state differs')


def restore_transaction(model, optimizer, weights, state):
    model.load_state_dict(weights, strict=True)
    optimizer.load_state_dict(cpu_tree(state))
    optimizer.zero_grad(set_to_none=True)
    exact_tree(weights, r.freeze(model))
    exact_tree(state, cpu_tree(optimizer.state_dict()))


def paths(head):
    return (REPO / f'outputs/smolvla_transactional_updates_preparation_{head[:8]}',
            REPO / f'outputs/smolvla_transactional_updates_{head[:8]}')


def specification():
    return {'experiment': 'F-TUA1', 'proposals': 72, 'warm_start_step': 72,
            'loss': 'unchanged_F_BRP1_best_reference', 'lr': r.LR, 'weight_decay': r.WD,
            'betas': [0.9, 0.999], 'eps': 1e-8, 'foreach': False, 'clip_norm': 1.0,
            'proposal_order': 'unchanged_multi_conditioned', 'backtracking': False,
            'on_reject': 'restore_parameters_and_all_optimizer_state_exact',
            'acceptance': 'train_objective_strictly_down_row0_chunk_max_excess_not_up',
            'gate_samples': 72, 'initial_anchor_check': True,
            'validation_usage': 'initial_exact_replay_and_fixed_final_no_selection',
            'limits': LIMITS, 'captures': CAPTURES, 'soft': SOFT, 'hard': HARD,
            'new_env': 0, 'qualification_reads': 0, 'coefficients': [1, 1, 1]}


def source_hashes():
    result = b.source_hashes()
    for f in (PRESERVED, SOURCE / 'result.json', SOURCE / 'independent_audit.json',
              SOURCE / 'rows.jsonl', SOURCE / 'initial.pt', SOURCE / 'bestref.pt',
              UDI / 'result.json', UDI / 'independent_audit.json'):
        result[str(f.relative_to(REPO))] = digest(f)
    return result


def worktree_gate(head):
    require(subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip() == head,
            'Execution HEAD changed')
    status = subprocess.check_output(['git', 'status', '--porcelain'], cwd=REPO, text=True)
    current = {x[3:]: {'status': x[:2], 'sha256': digest(REPO / x[3:])} for x in status.splitlines()}
    require(current == json.loads(PRESERVED.read_text()), 'Preserved worktree changed')


def data():
    require(json.loads((SOURCE / 'independent_audit.json').read_text())['independent_contract_accepted'] is True,
            'BRP source not audited')
    values, prior = b.load_data()
    brp_rows = [json.loads(x) for x in (SOURCE / 'rows.jsonl').read_text().splitlines()]
    require([tuple(x['key']) for x in brp_rows] == [q.pfx.key(s) for s in values[0]], 'BRP coverage differs')
    return values, prior, {tuple(x['key']): x for x in brp_rows}


def prepare(head):
    worktree_gate(head)
    env = q.runtime_environment()
    prep, out = paths(head)
    require(not prep.exists() and not out.exists(), 'Preparation/output already exists')
    values, prior, _ = data()
    samples, _, _, _, manifest, weights, scales, _ = values
    train = [s for s in samples if s['split'] == 'train']
    order = q.cov.schedule(train, 'multi_conditioned')
    targets = {','.join(map(str, q.pfx.key(s))): b.reference_bounds(s, prior[q.pfx.key(s)]['metrics']) for s in train}
    require(len(train) == len(order) == len(set(order)) == 72 and not torch.cuda.is_initialized(), 'Preparation scope')
    prep.mkdir()
    write(prep / 'preparation.json', {'execution_head': head, 'specification': specification(),
          'source_hashes': source_hashes(), 'manifest': manifest, 'weights': weights, 'scales': scales,
          'targets': targets, 'order': [list(q.pfx.key(train[i])) for i in order], 'environment': env,
          'cuda_initialized': False, 'model_forwards': 0, 'new_env': 0})
    print(json.dumps({'prepared': True, 'sha256': digest(prep / 'preparation.json'),
                      'prep': str(prep), 'output': str(out)}), flush=True)


def validate(head):
    worktree_gate(head)
    prep, out = paths(head)
    saved = json.loads((prep / 'preparation.json').read_text())
    require(saved['execution_head'] == head and saved['specification'] == specification(), 'Contract changed')
    require(saved['source_hashes'] == source_hashes() and saved['environment'] == q.runtime_environment(), 'Sources/environment changed')
    body = (prep / 'registration.md').read_text()
    post = json.loads((prep / 'registration_post.json').read_text())
    got = json.loads((prep / 'registration_readback.json').read_text())
    require(type(got.get('id')) is int and got['id'] == post.get('id') and
            got.get('body') == post.get('body') == body and got.get('issue_url') ==
            'https://api.github.com/repos/Lebron-233/lerobot/issues/1' and all(v in body for v in
            (f'F-TUA1-REGISTER:{head}', str(out), digest(prep / 'preparation.json'))), 'Registration differs')
    return saved


def objective_value(metrics, bounds, scales, weight):
    z, a, c = (metrics[k] for k in ('latent', 'row0', 'chunk'))
    return (z/scales['latent'] + weight*a/scales['row0'] + c/scales['chunk'] +
            max(0.0, a-bounds['row0'])/scales['row0'] + max(0.0, c-bounds['chunk'])/scales['chunk'])


def aggregate_training(records, refs, scales, targets, weights):
    require(len(records) == 72 and len({tuple(x['key']) for x in records}) == 72, 'Gate needs all 72 training cases')
    objectives, harms = [], []
    for item in records:
        k = tuple(item['key'])
        require(k in targets and item['split'] == 'train', 'Gate must be training-only')
        m = item['metrics']
        require(all(math.isfinite(m[n]) and m[n] >= 0 for n in ('latent', 'row0', 'chunk')), 'Invalid gate metric')
        objectives.append(objective_value(m, targets[k], scales, weights[k]))
        harms.append(m['row0'] - refs[k]['metrics']['identity']['row0'])
    return {'objective': math.fsum(objectives)/72,
            'row0': math.fsum(x['metrics']['row0'] for x in records)/72,
            'chunk': math.fsum(x['metrics']['chunk'] for x in records)/72,
            'max_excess': max([0.0, *harms])}


def gate(before, candidate, initial):
    keys = ('objective', 'row0', 'chunk', 'max_excess')
    require(all(math.isfinite(d[k]) and d[k] >= 0 for d in (before, candidate, initial) for k in keys), 'Nonfinite gate')
    tol = r.p.tolerance
    checks = {'objective_decreases': candidate['objective'] < before['objective']-tol(before['objective'])}
    for k in ('row0', 'chunk', 'max_excess'):
        checks[k+'_not_worse_incumbent'] = candidate[k] <= before[k]+tol(before[k])
        checks[k+'_not_worse_initial'] = candidate[k] <= initial[k]+tol(initial[k])
    return all(checks.values()), checks


def predict(model, sample, actions, counts):
    require(counts['predictor']+2 <= LIMITS['predictor'], 'Predictor budget exhausted')
    batch = pilot.batch_for([sample], [0], 'cuda')
    head = (batch['tokens'], batch['masks'])
    tail = (batch['action_mask'], batch['state'], batch['delay'])
    a = model(*head, actions.to('cuda'), *tail).delta_tokens
    z = model(*head, torch.zeros_like(actions, device='cuda'), *tail).delta_tokens
    counts['predictor'] += 2
    raw = r.p.transplant(tuple(v.cuda() for v in sample['inputs'][:2]), a, z)
    visual = tuple(v.to(torch.bfloat16).float() for v in raw)
    ev = {'action_delta': q.cpu(a), 'zero_delta': q.cpu(z), 'raw': q.cpu(raw),
          'visual': q.cpu(visual), 'actions': actions.cpu().clone(), 'mask': sample['mask'].clone()}
    return visual, ev


class Runtime(pilot.SmolVLAGraphRuntime):
    def _capture(self, inputs):
        require(len(self.captures) < CAPTURES, 'Graph capture budget exhausted')
        return super()._capture(inputs)


def decode(policy, runtime, sample, visual, out, counts, gradients=False):
    require(counts['decoder'] < LIMITS['decoder'], 'Decoder budget exhausted')
    counts['decoder'] += 1
    values = tuple(v.cuda() for v in sample['inputs'])
    visual = tuple(v.to(device='cuda', dtype=values[i].dtype) for i, v in enumerate(visual))
    with pilot.phase(out, f"decode_{counts['decoder']}", 30):
        if gradients:
            counts['gradient_decoder'] += 1
            value = policy.model.sample_actions(None, None, values[4], values[5], values[6],
                noise=values[7].clone(), future_image_tokens=visual, future_image_token_masks=tuple(values[2:4]))
        else:
            runtime.begin_episode('graph', sample['task'])
            with torch.no_grad():
                value = runtime(None, None, values[4], values[5], values[6], noise=values[7],
                    future_image_tokens=visual, future_image_token_masks=tuple(values[2:4])).detach().cpu().clone()
        require(value.shape == (1, 50, 32) and torch.isfinite(value).all(), 'Invalid decoder output')
    return value


def statistics(rows, accepted):
    translated = [{**x, 'metrics': {**x['metrics'], 'bestref_true': x['metrics']['transaction_true'],
                    **({'bestref_mismatched': x['metrics']['transaction_mismatched']} if 'transaction_mismatched' in x['metrics'] else {})}}
                  for x in rows]
    summary = b.statistics(translated)
    checks = summary['development_checks']
    checks['nonzero_committed_updates'] = accepted > 0
    # Use the unchanged BRP development conditions, with transaction as the candidate.
    summary['development_followup_supported'] = all(checks.values())
    summary['development_checks'] = {k.replace('bestref', 'transaction'): v for k, v in checks.items()}
    for values in summary['splits'].values():
        values['metrics'] = {k: v for k, v in values['metrics'].items() if not k.startswith('bestref_')}
        values['contrasts'] = {k.replace('bestref', 'transaction'): v for k, v in values['contrasts'].items()}
        values['worst_excess'] = {k.replace('bestref', 'transaction'): v for k, v in values['worst_excess'].items()}
    return summary


def worker(args):
    faulthandler.register(signal.SIGUSR1, file=sys.stderr, all_threads=True)
    torch.set_num_threads(1)
    counts, runtime, logs = Counter(), None, []
    result = {'experiment': 'F-TUA1', 'execution_head': args.execution_head, 'status': 'technical_failure', 'first_failure': None}
    try:
        prep = validate(args.execution_head)
        values, prior, historical = data()
        samples, _, residuals, donors, manifest, weights, scales, refs = values
        require(manifest == prep['manifest'] and weights == prep['weights'] and scales == prep['scales'], 'Prepared metadata differs')
        train = [s for s in samples if s['split'] == 'train']
        order = q.cov.schedule(train, 'multi_conditioned')
        require([list(q.pfx.key(train[i])) for i in order] == prep['order'], 'Training order differs')
        by_key = {q.pfx.key(s): s for s in samples}
        targets = {q.pfx.key(s): b.reference_bounds(s, prior[q.pfx.key(s)]['metrics']) for s in train}
        require({','.join(map(str, k)): v for k, v in targets.items()} == prep['targets'], 'Targets differ')
        weight_map = {tuple(x['key']): x['case_weight'] for x in weights['rows']}
        for name in ('controls', 'gates', 'proposals', 'predictions'):
            (args.output / name).mkdir()
        with pilot.phase(args.output, 'load_models', 90):
            require(torch.cuda.get_device_name() == 'NVIDIA GeForce RTX 4070 Ti SUPER', 'GPU differs')
            policy, _, _, report = e.load_runtime(e.POLICY, e.VLM)
            counts['vla_loads'] += 1
            write(args.output / 'policy_load.json', report)
            checkpoint = load(r.p.ARCHIVE / 'centered.pt')
            q.checkpoint_valid(checkpoint, 'centered')
            torch.manual_seed(r.SEED)
            model = pilot.LightweightFutureLatentPredictor(pilot.config()).cuda().eval()
            model.load_state_dict(checkpoint['state_dict'], strict=True)
            require(sum(v.numel() for v in model.parameters()) == 69680, 'Predictor size differs')
            counts['predictor_loads'] += 1
            torch.save(r.freeze(model), args.output / 'initial.pt')
            runtime = Runtime(policy.model)
        controls, incumbent = {}, []
        with torch.no_grad(), pilot.phase(args.output, 'initial_replay', 180):
            for s in samples:
                k, ref = q.pfx.key(s), refs[q.pfx.key(s)]
                visual, ev = predict(model, s, s['actions'], counts)
                for field in ('action_delta', 'zero_delta'):
                    exact_tree(ev[field], residuals[k, 'true'][field])
                exact_tree(ev['raw'], ref['raw_new']['identity_true'])
                outputs, scores = {}, {}
                for name, old, zv in (('identity', 'identity', s['inputs'][:2]),
                    ('old_centered', 'centered', ref['visual']['centered']), ('frozen', 'identity_true', visual)):
                    value = decode(policy, runtime, s, zv, args.output, counts)
                    pilot.require_equal(value, ref['outputs'][old], 'Initial output replay differs')
                    outputs[name], scores[name] = value, r.offline_metrics(zv, value, s)
                    counts[name+'_exact'] += 1
                controls[k] = {'key': list(k), 'metrics': scores, 'outputs': outputs, 'evidence': ev}
                torch.save(controls[k], args.output / 'controls' / ('_'.join(map(str, k))+'.pt'))
                if s['split'] == 'train':
                    incumbent.append({'key': list(k), 'split': 'train', 'metrics': scores['frozen']})
        initial_summary = aggregate_training(incumbent, controls, scales, targets, weight_map)
        current_summary = initial_summary
        write(args.output / 'initial_training_summary.json', initial_summary)
        runtime.release_graph()
        optimizer = torch.optim.AdamW(model.parameters(), lr=r.LR, weight_decay=r.WD,
                                     betas=(0.9, 0.999), eps=1e-8, foreach=False)
        torch.save(cpu_tree(optimizer.state_dict()), args.output / 'initial_optimizer.pt')
        with pilot.phase(args.output, 'transactional_training', 1200):
            for step, index in enumerate(order, 1):
                with pilot.phase(args.output, f'proposal_{step}', 60):
                    require(counts['proposals'] < 72 and counts['backward'] < 72, 'Proposal budget exhausted')
                    s, k = train[index], q.pfx.key(train[index])
                    before_weights, before_opt = r.freeze(model), cpu_tree(optimizer.state_dict())
                    runtime.release_graph()
                    model.train().requires_grad_(True)
                    optimizer.zero_grad(set_to_none=True)
                    visual, ev = predict(model, s, s['actions'], counts)
                    value = decode(policy, runtime, s, visual, args.output, counts, gradients=True)
                    metrics = q.cov.metric_values(visual, value, s)
                    loss = r.objective(metrics, targets[k], scales, weight_map[k], 'guarded')
                    require(torch.isfinite(loss), 'Nonfinite loss')
                    loss.backward()
                    counts['backward'] += 1
                    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    grads = {n: v.grad.detach().cpu().clone() for n, v in model.named_parameters() if v.grad is not None}
                    require(torch.isfinite(norm) and grads and all(torch.isfinite(g).all() for g in grads.values())
                            and all(v.grad is None for v in policy.parameters()), 'Gradient/VLA failure')
                    loss_record = {'evidence': ev, 'output': value.detach().cpu().clone(),
                        'metrics': dict(zip(('latent', 'row0', 'chunk'), map(float, metrics), strict=True)),
                        'objective': float(loss.detach()), 'gradient_norm_before_clip': float(norm),
                        'clipped_gradients': grads}
                    optimizer.step()
                    counts['proposals'] += 1
                    proposed_weights, proposed_opt = r.freeze(model), cpu_tree(optimizer.state_dict())
                    del loss, metrics, visual, value
                    model.eval()
                    candidate = []
                    with torch.no_grad():
                        for item in train:
                            zv, _ = predict(model, item, item['actions'], counts)
                            val = decode(policy, runtime, item, zv, args.output, counts)
                            candidate.append({'key': list(q.pfx.key(item)), 'split': 'train',
                                'visual': tuple(v.detach().cpu().to(torch.bfloat16) for v in zv),
                                'output': val, 'metrics': r.offline_metrics(zv, val, item)})
                    after_summary = aggregate_training(candidate, controls, scales, targets, weight_map)
                    accepted, checks = gate(current_summary, after_summary, initial_summary)
                    torch.save(candidate, args.output / 'gates' / f'{step:03d}.pt')
                    row = {'step': step, 'key': list(k), 'accepted': accepted, 'checks': checks,
                           'before': current_summary, 'proposed': after_summary}
                    if accepted:
                        incumbent, current_summary = candidate, after_summary
                        counts['accepted'] += 1
                    else:
                        restore_transaction(model, optimizer, before_weights, before_opt)
                        counts['rejected'] += 1
                    row['committed'] = current_summary
                    row['accepted_total'] = counts['accepted']
                    torch.save({**row, **loss_record, 'bounds': targets[k], 'scales': scales, 'weight': weight_map[k],
                        'weights_proposed': proposed_weights, 'optimizer_proposed': proposed_opt,
                        'weights_committed': r.freeze(model), 'optimizer_committed': cpu_tree(optimizer.state_dict())},
                        args.output / 'proposals' / f'{step:03d}.pt')
                    logs.append(row)
                    with (args.output / 'transactions.jsonl').open('a') as f:
                        f.write(json.dumps(row, allow_nan=False)+'\n')
                    print(json.dumps({'proposal': step, 'accepted': accepted, 'accepted_total': counts['accepted'],
                                      'train': current_summary}), flush=True)
                    del candidate, proposed_weights, proposed_opt, before_weights, before_opt, loss_record, grads
        model.eval().requires_grad_(False)
        torch.save({'state_dict': r.freeze(model), 'optimizer_state': cpu_tree(optimizer.state_dict()),
                    'proposals': 72, 'accepted': counts['accepted'], 'specification': specification()}, args.output / 'final.pt')
        runtime.release_graph()
        rows = []
        with torch.no_grad(), pilot.phase(args.output, 'final_evaluation', 180):
            for s in samples:
                k = q.pfx.key(s)
                evs, outputs, scores = {}, {}, {}
                for context in ('true', 'zero', 'mismatched'):
                    if context == 'mismatched' and donors[k] is None:
                        continue
                    actions = (torch.zeros_like(s['actions']) if context == 'zero' else
                               by_key[donors[k]]['actions'] if context == 'mismatched' else s['actions'])
                    zv, ev = predict(model, s, actions, counts)
                    val = decode(policy, runtime, s, zv, args.output, counts)
                    if context == 'zero':
                        exact_tree(ev['raw'], tuple(v.float() for v in s['inputs'][:2]))
                        pilot.require_equal(val, controls[k]['outputs']['identity'], 'Zero output differs')
                        counts['zero_exact'] += 1
                    evs[context], outputs[context], scores[context] = ev, val, r.offline_metrics(zv, val, s)
                torch.save({'key': list(k), 'donor': list(donors[k]) if donors[k] else None,
                            'evidence': evs, 'outputs': outputs, 'metrics': scores},
                           args.output / 'predictions' / ('_'.join(map(str, k))+'.pt'))
                base = {('brp_'+n[len('bestref_'):] if n.startswith('bestref_') else n): v
                        for n, v in historical[k]['metrics'].items()}
                base.update({'transaction_'+c: v for c, v in scores.items()})
                rows.append({'key': list(k), 'split': s['split'], 'delay': s['delay'],
                             'donor': list(donors[k]) if donors[k] else None, 'metrics': base})
        final_training = [{'key': x['key'], 'split': 'train', 'metrics': x['metrics']['transaction_true']}
                          for x in rows if x['split'] == 'train']
        final_summary = aggregate_training(final_training, controls, scales, targets, weight_map)
        require(all(math.isclose(final_summary[k], current_summary[k], rel_tol=1e-6, abs_tol=1e-7)
                    for k in current_summary), 'Final model is not the last committed state')
        require(all(counts[k] == v for k, v in LIMITS.items()) and counts['zero_exact'] == 88 and
                counts['accepted']+counts['rejected'] == 72, 'Formal accounting incomplete')
        require(source_hashes() == prep['source_hashes'] and all(not v.requires_grad and v.grad is None
                    for v in policy.parameters()), 'Sources/VLA changed')
        write(args.output / 'rows.json', rows)
        result.update(status='completed', vla_frozen=True, initial_training=initial_summary,
            final_training=final_summary, accepted=counts['accepted'], rejected=counts['rejected'],
            transaction_mechanism_completed=True, **statistics(rows, counts['accepted']))
    except BaseException:
        result['first_failure'] = traceback.format_exc()
    finally:
        if runtime is not None:
            write(args.output / 'captures.json', runtime.captures)
            counts['captures'] = len(runtime.captures)
            runtime.release_graph()
            result['graph_released'] = runtime.graph is None
        result.update(counts=dict(counts), specification=specification(), attempts=1, retries=0,
            new_env=0, image_encodings=0, qualification_reads=0, real_robot=0,
            deployment_qualified=False, risk_thresholds=None, old_confirmation='untouched')
        write(args.output / 'worker_result.json', result)
    return 0 if result['status'] == 'completed' else 2


def supervise(args):
    validate(args.execution_head)
    require(args.output == paths(args.execution_head)[1] and not args.output.exists(), 'Unique output required')
    args.output.mkdir()
    command = [sys.executable, '-u', str(Path(__file__).resolve()), '--execution-head', args.execution_head,
               '--output', str(args.output), '--worker']
    start, utc = time.perf_counter(), datetime.now(UTC).isoformat()
    cursors, pending, active, reason, terminated, forced = {}, {}, {}, None, None, False
    with (args.output / 'model.log').open('x') as log:
        child = subprocess.Popen(command, cwd=REPO, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        print(f'F-TUA1 worker={child.pid}', flush=True)
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
    path = args.output / 'worker_result.json'
    result = json.loads(path.read_text()) if path.exists() else {'status': 'technical_failure',
        'execution_head': args.execution_head, 'first_failure': 'Worker receipt absent'}
    result['execution'] = {'child_pid': child.pid, 'child_exit_code': code, 'exit_confirmed': True,
        'started_at_utc': utc, 'finished_at_utc': datetime.now(UTC).isoformat(), 'wall_seconds': time.perf_counter()-start,
        'stop_reason': reason, 'forced_termination': forced, 'pending': pending, 'active': active}
    if code or reason or pending or active:
        result['status'] = 'technical_failure'
    write(args.output / 'result.json', result)
    print(json.dumps({k: result.get(k) for k in ('status', 'first_failure', 'accepted', 'rejected',
                      'development_followup_supported')}), flush=True)
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
