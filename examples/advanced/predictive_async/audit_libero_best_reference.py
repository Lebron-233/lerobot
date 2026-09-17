"""CPU-only audit for F-BRP1; independent metrics, references and AdamW arithmetic."""
import argparse
import json
import math
import time
import traceback
from collections import Counter, defaultdict
from pathlib import Path

import audit_libero_identity_training as old
import libero_best_reference as b
import numpy as np
import torch

require, compare, load = b.require, old.compare, old.load


def decision(rows):
    tr = [x for x in rows if x['split'] == 'train']
    va = [x for x in rows if x['split'] == 'validation']
    require(len(tr) == 72 and len(va) == 16, 'Decision coverage differs')

    def episodes(subset, arm, metric):
        groups = defaultdict(list)
        for x in subset:
            groups[tuple(x['key'][:2])].append(x['metrics'][arm][metric])
        return {k: float(np.mean(v)) for k, v in groups.items()}

    def mean(arm, metric):
        return float(np.mean(list(episodes(va, arm, metric).values())))

    def tol(x):
        return 1e-7 + 1e-6 * abs(x)

    def harm(subset, arm):
        return max([0.0] + [x['metrics'][arm]['row0'] - x['metrics']['identity']['row0'] for x in subset])

    checks = []
    for control in ('identity', 'bestref_mismatched'):
        c = mean(control, 'row0')
        checks.append(mean('bestref_true', 'row0') < c-tol(c))
    c = mean('frozen', 'row0')
    checks.append(mean('bestref_true', 'row0') <= c+tol(c))
    for control in ('frozen', 'guarded_true', 'plain_true'):
        c = mean(control, 'chunk')
        checks.append(mean('bestref_true', 'chunk') < c-tol(c))
        c = harm(tr, control)
        checks.append(harm(tr, 'bestref_true') < c-tol(c))
    a, c = episodes(va, 'bestref_true', 'row0'), episodes(va, 'identity', 'row0')
    checks.append(sum(a[k] < c[k]-tol(c[k]) for k in a) >= 3)
    checks.append(sum(x['metrics']['bestref_true']['row0'] < x['metrics']['identity']['row0']
                      - tol(x['metrics']['identity']['row0']) for x in va) > 8)
    c = harm(va, 'frozen')
    checks.append(harm(va, 'bestref_true') <= c+tol(c))
    return bool(all(checks))


def audit(output):
    torch.set_num_threads(1)
    require(not torch.cuda.is_initialized(), 'CPU audit entered with CUDA')
    result = json.loads((output / 'result.json').read_text())
    prep = b.validate(result['execution_head'])
    require(output == b.paths(result['execution_head'])[1], 'Output identity differs')
    require(result['status'] == 'completed' and result['first_failure'] is None, 'Run not completed')
    ex = result['execution']
    require(ex['child_exit_code'] == 0 and ex['exit_confirmed'] and not ex['stop_reason'] and
            not ex['forced_termination'] and not ex['pending'] and not ex['active'], 'Unclosed exit')
    require(result['attempts'] == 1 and result['retries'] == 0 and result['vla_frozen'] and
            result['graph_released'], 'Freeze/attempt mismatch')
    compare(result['specification'], b.specification())
    for name in ('new_env', 'image_encodings', 'qualification_reads', 'real_robot'):
        require(result[name] == 0, 'Forbidden operation')
    for name in ('baseline_qualified', 'realtime_qualified', 'predictor_benefit_tested'):
        require(result[name] is False, 'Deployment flag changed')
    require(result['risk_thresholds'] is None and result['old_confirmation'] == 'untouched', 'Scope changed')
    data, previous = b.load_data()
    samples, _, residuals, donors, manifest, weights, scales, refs = data
    compare(manifest, prep['manifest'])
    compare(weights, prep['weights'])
    compare(scales, prep['scales'])
    by_key = {b.q.pfx.key(s): s for s in samples}
    train = [s for s in samples if s['split'] == 'train']
    order = b.q.cov.schedule(train, 'multi_conditioned')
    names = {'_'.join(map(str, k))+'.pt' for k in by_key}
    for folder in ('controls', 'predictions'):
        require({x.name for x in (output / folder).glob('*.pt')} == names, 'Array coverage differs')
    require({x.name for x in (output / 'training').glob('*.pt')} == {f'{i:03d}.pt' for i in range(1, 73)},
            'Training coverage differs')
    numerical = 0
    for k, s in by_key.items():
        c = load(output / 'controls' / ('_'.join(map(str, k))+'.pt'))
        require(c['key'] == list(k), 'Control key differs')
        old.evidence_check(c['evidence'], s, s['actions'])
        for field in ('action_delta', 'zero_delta'):
            for a, z in zip(c['evidence'][field], residuals[k, 'true'][field], strict=True):
                b.pilot.require_equal(a, z, 'Warm start residual differs')
        for name, old_name in (('identity', 'identity'), ('old_centered', 'centered'), ('frozen', 'identity_true')):
            b.pilot.require_equal(c['outputs'][name], refs[k]['outputs'][old_name], 'Initial replay differs')
            score = old.old.score(refs[k]['visual'][old_name], c['outputs'][name], s)
            compare(score, c['metrics'][name])
            compare(score, previous[k]['metrics'][name])
            numerical += 3
    initial = load(b.r.p.ARCHIVE / 'centered.pt')['state_dict']
    current = load(output / 'initial.pt')
    require(current.keys() == initial.keys(), 'Parameter names differ')
    for name in initial:
        b.pilot.require_equal(current[name], initial[name], 'Initial weights differ')
    state, optimizer_arrays, logs = {}, 0, []
    for step, index in enumerate(order, 1):
        s = train[index]
        k = b.q.pfx.key(s)
        saved = load(output / 'training' / f'{step:03d}.pt')
        require(saved['step'] == step and saved['key'] == list(k), 'Update order differs')
        bounds = {m: min(previous[k]['metrics']['identity'][m], previous[k]['metrics']['frozen'][m])
                  for m in ('row0', 'chunk')}
        compare(bounds, saved['bounds'])
        compare(bounds, prep['training_targets'][','.join(map(str, k))])
        compare(saved['weight'], weights['rows'][index]['case_weight'])
        compare(saved['scales'], scales)
        old.evidence_check(saved['evidence'], s, s['actions'])
        score = old.old.score(saved['evidence']['visual'], saved['output'], s)
        compare(score, saved['metrics'])
        compare(old.loss_value(score, bounds, scales, saved['weight'], 'guarded'), saved['objective'])
        numerical += 4
        gradients = saved['clipped_gradients']
        require(gradients and set(gradients) <= set(current) and all(torch.isfinite(v).all() for v in gradients.values()),
                'Invalid gradients')
        norm = float(np.sqrt(sum(np.square(old.array(v)).sum() for v in gradients.values())))
        before = saved['gradient_norm_before_clip']
        require(math.isfinite(before) and math.isclose(norm, before*min(1.0, 1.0/(before+1e-6)),
                rel_tol=1e-5, abs_tol=1e-7) and norm <= 1.00001, 'Clipping differs')
        expected = old.adamw_step(current, gradients, state)
        after = saved['weights_after']
        require(after.keys() == current.keys(), 'Post-update parameters differ')
        for name in current:
            require(np.isfinite(old.array(after[name])).all() and np.allclose(expected[name], old.array(after[name]),
                    rtol=1e-5, atol=1e-7), f'AdamW arithmetic differs: {step}/{name}')
            optimizer_arrays += 1
        current = after
        logs.append({x: saved[x] for x in ('step', 'key', 'bounds', 'metrics', 'objective', 'gradient_norm_before_clip')})
    compare(logs, list(map(json.loads, (output / 'training.jsonl').read_text().splitlines())))
    checkpoint = load(output / 'bestref.pt')
    require(checkpoint['step'] == checkpoint['source_step'] == 72, 'Wrong final checkpoint')
    compare(checkpoint['specification'], b.specification())
    for name in current:
        b.pilot.require_equal(current[name], checkpoint['state_dict'][name], 'Final weights differ')
    require(any(not torch.equal(current[n], initial[n]) for n in initial), 'No parameter change')
    rows = []
    for s in samples:
        k = b.q.pfx.key(s)
        saved = load(output / 'predictions' / ('_'.join(map(str, k))+'.pt'))
        require(saved['key'] == list(k) and saved['donor'] == (list(donors[k]) if donors[k] else None), 'Donor differs')
        contexts = {'true', 'zero', 'mismatched'} if donors[k] else {'true', 'zero'}
        require(set(saved['metrics']) == set(saved['outputs']) == set(saved['evidence']) == contexts, 'Contexts differ')
        scores = dict(previous[k]['metrics'])
        for context in contexts:
            actions = (torch.zeros_like(s['actions']) if context == 'zero' else
                       by_key[donors[k]]['actions'] if context == 'mismatched' else s['actions'])
            old.evidence_check(saved['evidence'][context], s, actions, context == 'zero')
            value = saved['outputs'][context]
            require(value.shape == (1, 50, 32) and torch.isfinite(value).all(), 'Invalid final output')
            if context == 'zero':
                b.pilot.require_equal(value, refs[k]['outputs']['identity'], 'Zero output differs')
            score = old.old.score(saved['evidence'][context]['visual'], value, s)
            compare(score, saved['metrics'][context])
            scores['bestref_'+context] = score
            numerical += 3
        rows.append({'key': list(k), 'split': s['split'], 'delay': s['delay'],
                     'donor': list(donors[k]) if donors[k] else None, 'metrics': scores})
    compare(rows, list(map(json.loads, (output / 'rows.jsonl').read_text().splitlines())))
    summary = b.statistics(rows)
    for name, value in summary.items():
        compare(value, result[name])
    require(decision(rows) == result['development_followup_supported'], 'Independent decision differs')
    captures = json.loads((output / 'captures.json').read_text())
    require(len(captures) <= 16 and all(x['status'] == 'captured' and x['eager_setup_calls'] == 1 and
            x['side_stream_warmup_calls'] == 3 and x['capture_calls'] == 1 for x in captures), 'Capture accounting differs')
    compare(result['counts'], {'vla_loads': 1, 'predictor_loads': 1, 'decoder': 597, 'predictor': 842,
        'gradient_decoder': 72, 'updates': 72, 'backward': 72, 'identity_exact': 88,
        'old_centered_exact': 88, 'frozen_exact': 88, 'zero_exact': 88, 'captures': len(captures)})
    active, events = {}, Counter()
    for event in map(json.loads, (output / 'events.jsonl').read_text().splitlines()):
        name = event['phase']
        events[event['event']] += 1
        if event['event'] == 'started':
            require(name not in active, 'Duplicate phase')
            active[name] = event['limit']
        else:
            require(event['event'] == 'returned' and name in active, 'Unmatched phase')
            require(event['seconds'] <= active.pop(name), 'Phase timeout')
    require(not active and events['started'] == events['returned'] == 674, 'Phase count differs')
    require(b.source_hashes() == prep['source_hashes'] and not torch.cuda.is_initialized(), 'Source or CPU scope changed')
    return {'independent_contract_accepted': True, 'development_followup_supported': decision(rows),
            'numerical_comparisons': numerical, 'optimizer_parameter_arrays_checked': optimizer_arrays,
            'optimizer_arithmetic_steps': 72, 'gradients_independently_redifferentiated': False,
            'phases': dict(events), 'cuda_initialized': False, 'audit_model_forwards': 0,
            'new_env': 0, 'qualification_reads': 0, 'summary': summary, 'per_sample': rows, 'first_failure': None}


def render(result, audited):
    lines = ['# F-BRP1：训练样本最佳参照软惩罚', '',
        f"execution HEAD：`{result.get('execution_head')}`；运行`{result.get('status')}`；独立接纳`{audited['independent_contract_accepted']}`；开发推进`{audited.get('development_followup_supported')}`。", '',
        '原72训练/16反复使用开发验证；只改两项hinge参照为逐训练样本min(Identity,冻结I+delta)。同一起点及72次更新，不选点。',
        'plain/guarded为已审计历史对照，不是本轮重训；R1资格32样本未读取。oracle不是专家动作。', '']
    if not audited['independent_contract_accepted']:
        return '\n'.join(lines + ['```text', str(result.get('first_failure')), str(audited.get('first_failure')), '```', ''])
    for split, s in audited['summary']['splits'].items():
        lines += [f'## {split}（episode等权）', '', '| 条件 | N/episode | 首动作MSE | chunk MSE | token MSE |',
                  '|---|---:|---:|---:|---:|']
        for name, a in s['metrics'].items():
            m = a['macro']
            lines.append(f"| {name} | {a['samples']}/{len(a['episodes'])} | {m['row0']:.12f} | {m['chunk']:.12f} | {m['latent']:.9f} |")
        lines += ['', f"最大正超额首动作误差（相对Identity）：`{json.dumps(s['worst_excess'])}`。", '']
        for name, c in s['contrasts'].items():
            lines.append(f"相对{name}：配对{c['samples']}/{len(c['episodes'])}episode；均值收益{c['macro_benefit']:.12f}；样本{c['sample_directions']}；episode改善{c['episode_improved']}。")
        for name in ('identity', 'frozen'):
            c = s['contrasts'][name]
            lines += ['', f'相对{name}最不利5例（不删例）：']
            for row in sorted(c['per_sample'], key=lambda x: x['benefit'])[:5]:
                lines.append(f"`{row['key']}`: {row['benefit']:.12f} ({row['direction']})")
            lines += ['', f"留一episode：`{json.dumps(c['leave_one_out'])}`。", '']
    lines += ['## 判据与执行', '', '```json', json.dumps({'checks': audited['summary']['development_checks'],
        'counts': result['counts'], 'phases': audited['phases'], 'execution': result['execution']}, indent=2), '```', '',
        '独立CPU审计重算保存指标/参照/目标/AdamW算术，不独立重新求导。不自动调系数、追加训练或改选其他候选。',
        '软惩罚不是硬保证；开发结果不是独立资格或部署收益。所有部署资格仍false、risk_thresholds=null。', '']
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    output = parser.parse_args().output.resolve()
    require(not (output / 'independent_audit.json').exists(), 'Audit exists; do not overwrite')
    start = time.perf_counter()
    try:
        audited = audit(output)
    except BaseException:
        audited = {'independent_contract_accepted': False, 'first_failure': traceback.format_exc()}
    audited['wall_seconds'] = time.perf_counter()-start
    b.write(output / 'independent_audit.json', audited)
    result = json.loads((output / 'result.json').read_text())
    with (output / 'REPORT.md').open('x') as f:
        f.write(render(result, audited))
    b.write(output / 'REPORT.json', {'result': result, 'audit': audited})
    print(json.dumps({k: v for k, v in audited.items() if k not in ('summary', 'per_sample')}), flush=True)
    return 0 if audited['independent_contract_accepted'] else 2


if __name__ == '__main__':
    raise SystemExit(main())
