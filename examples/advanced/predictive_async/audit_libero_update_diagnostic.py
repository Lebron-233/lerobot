"""CPU/NumPy verification of archived-update attribution; no model execution."""
import argparse
import json
import math
import time
import traceback
from collections import Counter
from pathlib import Path

import audit_libero_identity_anchor_probe as old
import audit_libero_identity_training as train_audit
import libero_update_diagnostic as u
import numpy as np
import torch

require, compare, array = u.require, old.compare, old.array


def metric(visual, output, sample, archived):
    result = old.score(visual, output, sample)
    result['objective'] = train_audit.loss_value(result, archived['bounds'], archived['scales'],
                                               archived['weight'], 'guarded')
    return result


def evidence(ev, sample):
    u.pilot.require_equal(ev['actions'], sample['actions'], 'Action prefix differs')
    u.pilot.require_equal(ev['mask'], sample['mask'], 'Action mask differs')
    for i in range(2):
        raw = sample['inputs'][i].float() + (ev['action_delta'][i].float()-ev['zero_delta'][i].float())
        u.pilot.require_equal(raw, ev['raw'][i], 'Raw algebra differs')
        u.pilot.require_equal(raw.to(torch.bfloat16).float(), ev['visual'][i], 'Precision differs')


def independent_vectors(grads, displacement):
    g = {k: array(v) for k, v in grads.items()}
    d = array(displacement)
    require(set(g) == {'total', 'latent', 'row0', 'chunk'} and
            all(v.shape == (69680,) and np.isfinite(v).all() for v in g.values()), 'Gradient vector invalid')
    norm = {k: float(np.linalg.norm(v)) for k, v in g.items()}
    cosine = {}
    for a, b in (('latent', 'row0'), ('chunk', 'row0'), ('total', 'row0')):
        den = norm[a]*norm[b]
        cosine[a+'__'+b] = float(g[a] @ g[b])/den if den else None
    return {'norms': norm, 'cosines': cosine,
            'directional': {k: float(v @ d) for k, v in g.items()},
            'component_sum_relative_residual': float(np.linalg.norm(g['total']-g['latent']-g['row0']-g['chunk']))
                / max(norm['total'], 1e-30)}


def direction(a, b):
    difference = b-a
    tol = 1e-7+1e-6*abs(a)
    return 'decreased' if difference < -tol else 'increased' if difference > tol else 'tied'


def summarize(rows, trajectories):
    stats = {}
    for m in ('row0', 'chunk', 'latent', 'objective'):
        stats[m] = {'own_update': dict(Counter(direction(x['before'][m], x['after'][m]) for x in rows)),
                    'later_updates': dict(Counter(direction(x['after'][m], x['final'][m]) for x in rows)),
                    'mean_attribution': {k: float(np.mean([x['attribution'][m][k] for x in rows]))
                                         for k in ('earlier', 'own', 'later', 'net')}}
    gradients = {'negative_cosines': {k: sum(x['gradients']['cosines'][k] is not None and
                    x['gradients']['cosines'][k] < -1e-6 for x in rows)
                    for k in ('latent__row0', 'chunk__row0', 'total__row0')},
                 'positive_first_order_change': {k: sum(x['gradients']['directional'][k] > 1e-10 for x in rows)
                    for k in ('total', 'latent', 'row0', 'chunk')},
                 'component_sum_relative_residual_max': max(x['gradients']['component_sum_relative_residual'] for x in rows)}
    sentinel_summary = {}
    for key, values in trajectories.items():
        index = next(i for i, x in enumerate(rows, 1) if '/'.join(map(str, x['key'])) == key)
        jumps = [{'update': i, 'update_sample': rows[i-1]['key'],
                  'row0_change': values[i]['row0']-values[i-1]['row0'],
                  'objective_change': values[i]['objective']-values[i-1]['objective']} for i in range(1, 73)]
        sentinel_summary[key] = {'own_update_index': index, 'initial': values[0], 'pre_own': values[index-1],
            'post_own': values[index], 'final': values[72],
            'largest_adverse_updates': sorted(jumps, key=lambda x: x['row0_change'], reverse=True)[:5],
            'largest_beneficial_updates': sorted(jumps, key=lambda x: x['row0_change'])[:5]}
    return {'sample_count': len(rows), 'episode_count': len({tuple(x['key'][:2]) for x in rows}),
            'attribution': stats, 'gradient_description': gradients, 'sentinels': sentinel_summary,
            'warning': 'Own-update means combine different archived model states, not a new trained policy. '
                       'Gradient directions are local mixed-precision diagnostics, not finite-step guarantees.'}


def audit(output):
    torch.set_num_threads(1)
    require(not torch.cuda.is_initialized(), 'Audit initialized CUDA')
    result = json.loads((output / 'result.json').read_text())
    prep = u.validate(result['execution_head'])
    require(output == u.paths(result['execution_head'])[1] and result['status'] == 'completed'
            and result['first_failure'] is None, 'Run not completed')
    ex = result['execution']
    require(ex['exit_confirmed'] and ex['child_exit_code'] == 0 and not ex['stop_reason'] and
            not ex['forced_termination'] and not ex['pending'] and not ex['active'], 'Exit not closed')
    compare(result['counts'], u.EXPECTED)
    require(result['vla_frozen'] and result['attempts'] == 1 and result['retries'] == 0, 'Freeze/attempt differs')
    for k in ('new_updates', 'optimizer_steps', 'new_env', 'image_encodings', 'qualification_reads',
              'validation_forward', 'graph_captures'):
        require(result[k] == 0, 'Scope changed: '+k)
    require(not result['model_checkpoint_selection'] and not result['deployment_qualified']
            and result['risk_thresholds'] is None, 'Deployment status changed')
    ordered = u.data()
    previous = u.load(u.SOURCE / 'initial.pt')
    layout = json.loads((output / 'parameter_layout.json').read_text())
    require(sum(math.prod(x['shape']) for x in layout) == 69680, 'Parameter layout differs')
    names = [x['name'] for x in layout]
    require(len(names) == len(set(names)) and set(names) == set(previous), 'Parameter names differ')
    require({f.name for f in (output / 'samples').glob('*.pt')} == {f'{i:03d}.pt' for i in range(1, 73)},
            'Own-update coverage differs')
    rows, numerical = [], 0
    for step, sample in enumerate(ordered, 1):
        saved = u.load(output / 'samples' / f'{step:03d}.pt')
        archived = u.load(u.SOURCE / 'training' / f'{step:03d}.pt')
        key = '_'.join(map(str, u.q.pfx.key(sample)))
        require(saved['row']['key'] == archived['key'] == list(u.q.pfx.key(sample)), 'Step identity differs')
        require(saved['row']['step'] == step and prep['manifest'][step-1] == {'step': step, 'key': archived['key']},
                'Step schedule differs')
        for field in ('before_evidence', 'after_evidence'):
            evidence(saved[field], sample)
        for field in ('raw', 'visual', 'action_delta', 'zero_delta'):
            for a, b in zip(saved['before_evidence'][field], archived['evidence'][field], strict=True):
                u.pilot.require_equal(a, b, 'Pre-replay evidence differs')
        u.pilot.require_equal(saved['before_output'], archived['output'], 'Pre-replay output differs')
        control = u.load(u.SOURCE / 'controls' / (key+'.pt'))
        final = u.load(u.SOURCE / 'predictions' / (key+'.pt'))
        values = {'initial': metric(control['evidence']['visual'], control['outputs']['frozen'], sample, archived),
                  'before': metric(saved['before_evidence']['visual'], saved['before_output'], sample, archived),
                  'after': metric(saved['after_evidence']['visual'], saved['after_output'], sample, archived),
                  'final': metric(final['evidence']['true']['visual'], final['outputs']['true'], sample, archived)}
        for k, v in values.items():
            compare(v, saved['row'][k])
            numerical += 4
        attr = {m: dict(zip(('earlier', 'own', 'later', 'net'),
                (values['before'][m]-values['initial'][m], values['after'][m]-values['before'][m],
                 values['final'][m]-values['after'][m], values['final'][m]-values['initial'][m]), strict=True))
                for m in ('row0', 'chunk', 'latent', 'objective')}
        compare(attr, saved['row']['attribution'])
        expected_d = torch.cat([(archived['weights_after'][n].double()-previous[n].double()).float().reshape(-1)
                                for n in names])
        u.pilot.require_equal(expected_d, saved['displacement'], 'Actual update vector differs')
        total = array(saved['gradients']['total'])
        clipped = np.concatenate([array(archived['clipped_gradients'].get(n, torch.zeros_like(previous[n]))).reshape(-1)
                                  for n in names])
        require(np.allclose(total*min(1.0, 1/(archived['gradient_norm_before_clip']+1e-6)), clipped,
                            rtol=1e-5, atol=1e-7), 'Total gradient differs from archived clipped gradient')
        g = independent_vectors(saved['gradients'], saved['displacement'])
        compare(g, saved['row']['gradients'])
        row = {'step': step, 'key': archived['key'], **values, 'attribution': attr, 'gradients': g}
        compare(row, saved['row'])
        rows.append(row)
        previous = archived['weights_after']
    compare(rows, json.loads((output / 'rows.json').read_text()))
    trajectories = {}
    expected = {f'{k[0]}_{k[1]}_{k[2]}_{i:03d}.pt' for k in u.SENTINELS for i in range(73)}
    require({f.name for f in (output / 'sentinels').glob('*.pt')} == expected, 'Sentinel coverage differs')
    for key in u.SENTINELS:
        step = next(i for i, s in enumerate(ordered, 1) if u.q.pfx.key(s) == key)
        sample = ordered[step-1]
        archived = u.load(u.SOURCE / 'training' / f'{step:03d}.pt')
        values = []
        for index in range(73):
            v = u.load(output / 'sentinels' / f'{key[0]}_{key[1]}_{key[2]}_{index:03d}.pt')
            require(v['key'] == list(key) and v['checkpoint'] == index, 'Sentinel identity differs')
            evidence(v['evidence'], sample)
            values.append(metric(v['evidence']['visual'], v['output'], sample, archived))
            numerical += 4
        for k, i in (('initial', 0), ('before', step-1), ('after', step), ('final', 72)):
            compare(values[i], rows[step-1][k])
        trajectories['/'.join(map(str, key))] = values
    active, events = {}, Counter()
    for ev in map(json.loads, (output / 'events.jsonl').read_text().splitlines()):
        events[ev['event']] += 1
        if ev['event'] == 'started':
            require(ev['phase'] not in active, 'Duplicate phase')
            active[ev['phase']] = ev['limit']
        else:
            require(ev['event'] == 'returned' and ev['phase'] in active and
                    ev['seconds'] <= active.pop(ev['phase']), 'Phase not closed or timed out')
    require(not active and events['started'] == events['returned'] == 365, 'Phase accounting differs')
    require(u.source_hashes() == prep['source_hashes'] and not torch.cuda.is_initialized(), 'Audit scope changed')
    return {'independent_contract_accepted': True, 'diagnostic_complete': True,
            'numerical_comparisons': numerical, 'gradient_replay_samples': 72, 'sentinel_points': 146,
            'phases': dict(events), 'audit_model_forwards': 0, 'cuda_initialized': False,
            'component_gradients_independently_redifferentiated_by_audit': False,
            'summary': summarize(rows, trajectories), 'per_sample': rows, 'trajectories': trajectories,
            'first_failure': None}


def render(result, audited):
    lines = ['# F-UDI1：已保存更新的梯度与时序归因', '',
             f"execution HEAD：`{result['execution_head']}`。",
             f"运行：`{result['status']}`；独立接纳：`{audited['independent_contract_accepted']}`。", '',
             '仅重放BRP已保存权重，无新优化器更新、验证前向、资格集读取或Env。',
             '72个原训练样本全部纳入；两个哨兵预先指定，不用于选点、调参或部署。', '']
    if audited['independent_contract_accepted']:
        lines += ['## 完整诊断摘要', '', '```json', json.dumps(audited['summary'], ensure_ascii=False, indent=2), '```', '',
                  '## 逐训练样本首动作误差归因', '',
                  '| step | task/state/request | initial | before | after | final | earlier | own | later |',
                  '|---:|---|---:|---:|---:|---:|---:|---:|---:|']
        for x in audited['per_sample']:
            v = [x[k]['row0'] for k in ('initial', 'before', 'after', 'final')]
            v += [x['attribution']['row0'][k] for k in ('earlier', 'own', 'later')]
            lines.append('| '+str(x['step'])+' | '+'/'.join(map(str, x['key']))+' | '+' | '.join(f'{a:.12f}' for a in v)+' |')
    else:
        lines += ['```text', str(result.get('first_failure')), str(audited.get('first_failure')), '```']
    lines += ['', '## 执行与限制', '', '```json', json.dumps({'counts': result.get('counts'),
              'execution': result.get('execution'), 'audit_phases': audited.get('phases')}, indent=2), '```', '',
              'earlier/own/later是固定样本在已保存参数轨迹上的望远镜差分，不是环境物理因果识别。',
              '组件梯度含原缩放和hinge；BF16反向舍入可使组件梯度之和不同于一次合并反传，差异完整报告。',
              '方向导数只是一阶局部描述，不保证有限步下降。CPU审计不重新求导；无新可部署模型或独立资格。',
              '旧阴性结果不变，生产资格false、risk_thresholds=null。', '']
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    require(not (output / 'independent_audit.json').exists(), 'Audit already exists')
    start = time.perf_counter()
    try:
        audited = audit(output)
    except BaseException:
        audited = {'independent_contract_accepted': False, 'first_failure': traceback.format_exc()}
    audited['wall_seconds'] = time.perf_counter()-start
    u.write(output / 'independent_audit.json', audited)
    result = json.loads((output / 'result.json').read_text())
    with (output / 'REPORT.md').open('x') as f:
        f.write(render(result, audited))
    u.write(output / 'REPORT.json', {'result': result, 'audit': audited})
    print(json.dumps({k: v for k, v in audited.items() if k not in ('summary', 'per_sample', 'trajectories')}), flush=True)
    return 0 if audited['independent_contract_accepted'] else 2


if __name__ == '__main__':
    raise SystemExit(main())
