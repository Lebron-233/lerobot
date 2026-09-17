"""Independent CPU/NumPy audit of F-TUA1 gates and complete optimizer rollback.

No model execution or independent gradient differentiation. Gate outcomes are
training constraints, not evidence of held-out generalization by themselves.
"""
import argparse
import copy
import json
import math
import time
import traceback
from collections import Counter
from pathlib import Path

import audit_libero_best_reference as previous_audit
import audit_libero_identity_training as optaudit
import libero_transactional_updates as t
import numpy as np
import torch

require, load = t.require, t.load
score, compare, array = optaudit.old.score, optaudit.old.compare, optaudit.old.array


def aggregate(records, controls, targets, scales, weights):
    require(len(records) == 72 and len({tuple(x['key']) for x in records}) == 72, 'Training gate coverage')
    terms, harm = [], []
    for x in records:
        key = tuple(x['key'])
        require(x['split'] == 'train' and key in targets, 'Nontraining gate record')
        terms.append(optaudit.loss_value(x['metrics'], targets[key], scales, weights[key], 'guarded'))
        harm.append(x['metrics']['row0']-controls[key]['metrics']['identity']['row0'])
    return {'objective': float(np.mean(terms)), 'row0': float(np.mean([x['metrics']['row0'] for x in records])),
            'chunk': float(np.mean([x['metrics']['chunk'] for x in records])), 'max_excess': max(0.0, *harm)}


def gate(before, candidate, initial):
    def tol(v):
        return 1e-7+1e-6*abs(v)
    result = {'objective_decreases': candidate['objective'] < before['objective']-tol(before['objective'])}
    for k in ('row0', 'chunk', 'max_excess'):
        result[k+'_not_worse_incumbent'] = candidate[k] <= before[k]+tol(before[k])
        result[k+'_not_worse_initial'] = candidate[k] <= initial[k]+tol(initial[k])
    return all(result.values()), result


def verify_optimizer(moment_state, proposed, names, initial_group):
    require(len(proposed['param_groups']) == 1, 'Unexpected optimizer groups')
    t.exact_tree(initial_group, proposed['param_groups'])
    ids = proposed['param_groups'][0]['params']
    require(len(ids) == len(names), 'Optimizer parameter order differs')
    expected_ids = {pid for pid, name in zip(ids, names, strict=True) if name in moment_state}
    require(set(proposed['state']) == expected_ids, 'Optimizer state coverage differs')
    for pid, name in zip(ids, names, strict=True):
        if name not in moment_state:
            continue
        m, v, step = moment_state[name]
        state = proposed['state'][pid]
        require(set(state) == {'step', 'exp_avg', 'exp_avg_sq'}, 'Unexpected optimizer state')
        require(float(state['step']) == step, 'AdamW counter differs')
        require(np.allclose(m, array(state['exp_avg']), rtol=1e-5, atol=1e-7) and
                np.allclose(v, array(state['exp_avg_sq']), rtol=1e-5, atol=1e-7), 'AdamW moments differ')


def audit(out):
    torch.set_num_threads(1)
    require(not torch.cuda.is_initialized(), 'Audit initialized CUDA')
    result = json.loads((out / 'result.json').read_text())
    prepared = t.validate(result['execution_head'])
    require(out == t.paths(result['execution_head'])[1] and result['status'] == 'completed'
            and result['first_failure'] is None, 'Experiment incomplete')
    ex = result['execution']
    require(ex['child_exit_code'] == 0 and ex['exit_confirmed'] is True and not ex['stop_reason']
            and not ex['forced_termination'] and not ex['pending'] and not ex['active'], 'Unclosed exit')
    require(result['vla_frozen'] and result['graph_released'] and result['attempts'] == 1 and result['retries'] == 0,
            'Freeze/attempt changed')
    compare(result['specification'], t.specification())
    for name in ('new_env', 'image_encodings', 'qualification_reads', 'real_robot'):
        require(result[name] == 0, 'Forbidden operation')
    require(result['deployment_qualified'] is False and result['risk_thresholds'] is None
            and result['old_confirmation'] == 'untouched', 'Qualification claim changed')
    values, prior, historical = t.data()
    samples, _, residuals, donors, manifest, weights, scales, refs = values
    compare(manifest, prepared['manifest'])
    compare(weights, prepared['weights'])
    compare(scales, prepared['scales'])
    by_key = {t.q.pfx.key(s): s for s in samples}
    train = [s for s in samples if s['split'] == 'train']
    keys = [t.q.pfx.key(s) for s in train]
    order = t.q.cov.schedule(train, 'multi_conditioned')
    compare(prepared['order'], [list(t.q.pfx.key(train[i])) for i in order])
    targets = {k: {m: min(prior[k]['metrics']['identity'][m], prior[k]['metrics']['frozen'][m])
                   for m in ('row0', 'chunk')} for k in keys}
    compare({','.join(map(str, k)): v for k, v in targets.items()}, prepared['targets'])
    weight_map = {tuple(x['key']): x['case_weight'] for x in weights['rows']}
    filenames = {'_'.join(map(str, k))+'.pt' for k in by_key}
    for directory in ('controls', 'predictions'):
        require({p.name for p in (out / directory).glob('*.pt')} == filenames, 'Final/control coverage differs')
    for directory in ('gates', 'proposals'):
        require({p.name for p in (out / directory).glob('*.pt')} == {f'{i:03d}.pt' for i in range(1, 73)}, 'Proposal coverage differs')
    controls, incumbent = {}, {}
    numerical = 0
    for k, s in by_key.items():
        c = load(out / 'controls' / ('_'.join(map(str, k))+'.pt'))
        require(c['key'] == list(k), 'Control key differs')
        optaudit.evidence_check(c['evidence'], s, s['actions'])
        for f in ('action_delta', 'zero_delta'):
            t.exact_tree(c['evidence'][f], residuals[k, 'true'][f])
        metrics = {}
        for n, old in (('identity', 'identity'), ('old_centered', 'centered'), ('frozen', 'identity_true')):
            t.pilot.require_equal(c['outputs'][n], refs[k]['outputs'][old], 'Archived control output differs')
            metrics[n] = score(refs[k]['visual'][old], c['outputs'][n], s)
            compare(metrics[n], c['metrics'][n])
            numerical += 3
        controls[k] = {**c, 'metrics': metrics}
        if k in targets:
            incumbent[k] = {'key': list(k), 'split': 'train', 'metrics': metrics['frozen'],
                            'output': c['outputs']['frozen'], 'visual': c['evidence']['visual']}
    initial_agg = aggregate(list(incumbent.values()), controls, targets, scales, weight_map)
    compare(initial_agg, result['initial_training'])
    compare(initial_agg, json.loads((out / 'initial_training_summary.json').read_text()))
    current_agg = initial_agg
    current = load(out / 'initial.pt')
    t.exact_tree(current, load(t.r.p.ARCHIVE / 'centered.pt')['state_dict'])
    opt_state = load(out / 'initial_optimizer.pt')
    require(not opt_state['state'], 'Optimizer was not fresh')
    group = opt_state['param_groups']
    require(len(group) == 1 and group[0]['lr'] == t.r.LR and group[0]['weight_decay'] == t.r.WD and
            tuple(group[0]['betas']) == (0.9, 0.999) and group[0]['eps'] == 1e-8 and group[0]['foreach'] is False,
            'Optimizer hyperparameters differ')
    moments, logs, accepted, arrays, rejected_own = {}, [], 0, 0, []
    names = list(current)
    for step, index in enumerate(order, 1):
        k, s = keys[index], train[index]
        saved = load(out / 'proposals' / f'{step:03d}.pt')
        require(saved['step'] == step and saved['key'] == list(k), 'Proposal identity differs')
        compare(saved['bounds'], targets[k])
        compare(saved['scales'], scales)
        compare(saved['weight'], weight_map[k])
        optaudit.evidence_check(saved['evidence'], s, s['actions'])
        m = score(saved['evidence']['visual'], saved['output'], s)
        compare(m, saved['metrics'])
        compare(optaudit.loss_value(m, targets[k], scales, weight_map[k], 'guarded'), saved['objective'])
        numerical += 4
        grads = saved['clipped_gradients']
        require(grads and set(grads) <= set(current) and all(torch.isfinite(g).all() for g in grads.values()), 'Invalid gradients')
        norm = float(np.sqrt(sum(np.square(array(g)).sum() for g in grads.values())))
        pre = saved['gradient_norm_before_clip']
        require(math.isfinite(pre) and math.isclose(norm, pre*min(1., 1./(pre+1e-6)), rel_tol=1e-5, abs_tol=1e-7)
                and norm <= 1.00001, 'Clip differs')
        proposed_moments = copy.deepcopy(moments)
        expected = optaudit.adamw_step(current, grads, proposed_moments)
        require(saved['weights_proposed'].keys() == current.keys(), 'Proposed weights differ')
        for name in current:
            require(np.isfinite(array(saved['weights_proposed'][name])).all() and
                    np.allclose(expected[name], array(saved['weights_proposed'][name]), rtol=1e-5, atol=1e-7), 'AdamW arithmetic differs')
            arrays += 1
        verify_optimizer(proposed_moments, saved['optimizer_proposed'], names, group)
        batch = load(out / 'gates' / f'{step:03d}.pt')
        require([tuple(x['key']) for x in batch] == keys and all(x['split'] == 'train' for x in batch), 'Gate sample order differs')
        for x in batch:
            sk = tuple(x['key'])
            require(x['output'].shape == (1, 50, 32) and torch.isfinite(x['output']).all() and
                    all(z.dtype == torch.bfloat16 and z.shape == by_key[sk]['inputs'][i].shape
                        and torch.isfinite(z).all() for i, z in enumerate(x['visual'])), 'Invalid gate arrays')
            metrics = score(x['visual'], x['output'], by_key[sk])
            compare(metrics, x['metrics'])
            x['metrics'] = metrics
            numerical += 3
        proposed_agg = aggregate(batch, controls, targets, scales, weight_map)
        decision, checks = gate(current_agg, proposed_agg, initial_agg)
        require(saved['accepted'] is decision and saved['checks'] == checks, 'Independent acceptance differs')
        compare(saved['before'], current_agg)
        compare(saved['proposed'], proposed_agg)
        if decision:
            accepted += 1
            current, opt_state, moments = saved['weights_proposed'], saved['optimizer_proposed'], proposed_moments
            current_agg = proposed_agg
            incumbent = {tuple(x['key']): x for x in batch}
        else:
            own = next(x for x in batch if tuple(x['key']) == k)
            rejected_own.append({'step': step, 'key': list(k), 'own_row0_increase': own['metrics']['row0']-m['row0'],
                                 'own_chunk_increase': own['metrics']['chunk']-m['chunk'],
                                 'checks_failed': [n for n, v in checks.items() if not v]})
        t.exact_tree(current, saved['weights_committed'])
        t.exact_tree(opt_state, saved['optimizer_committed'])
        compare(saved['committed'], current_agg)
        require(saved['accepted_total'] == accepted, 'Committed count differs')
        logs.append({n: saved[n] for n in ('step', 'key', 'accepted', 'checks', 'before', 'proposed', 'committed', 'accepted_total')})
    compare(logs, [json.loads(x) for x in (out / 'transactions.jsonl').read_text().splitlines()])
    final = load(out / 'final.pt')
    t.exact_tree(current, final['state_dict'])
    t.exact_tree(opt_state, final['optimizer_state'])
    compare(final['specification'], t.specification())
    require(final['accepted'] == accepted and final['proposals'] == 72, 'Final selection differs')
    require(accepted == result['accepted'] and 72-accepted == result['rejected'], 'Acceptance totals differ')
    rows = []
    for k, s in by_key.items():
        saved = load(out / 'predictions' / ('_'.join(map(str, k))+'.pt'))
        require(saved['key'] == list(k) and saved['donor'] == (list(donors[k]) if donors[k] else None), 'Final donor differs')
        contexts = {'true', 'zero', 'mismatched'} if donors[k] else {'true', 'zero'}
        require(set(saved['outputs']) == set(saved['evidence']) == set(saved['metrics']) == contexts, 'Final contexts differ')
        metrics = {}
        for context in contexts:
            actions = (torch.zeros_like(s['actions']) if context == 'zero' else
                       by_key[donors[k]]['actions'] if context == 'mismatched' else s['actions'])
            optaudit.evidence_check(saved['evidence'][context], s, actions, zero=context == 'zero')
            require(saved['outputs'][context].shape == (1, 50, 32) and torch.isfinite(saved['outputs'][context]).all(), 'Final shape')
            metrics[context] = score(saved['evidence'][context]['visual'], saved['outputs'][context], s)
            compare(metrics[context], saved['metrics'][context])
            numerical += 3
        t.pilot.require_equal(saved['outputs']['zero'], controls[k]['outputs']['identity'], 'Zero replay failed')
        if k in incumbent:
            t.pilot.require_equal(saved['outputs']['true'], incumbent[k]['output'], 'Final output not last committed gate')
            for a, c in zip(saved['evidence']['true']['visual'], incumbent[k]['visual'], strict=True):
                t.pilot.require_equal(a.float(), c.float(), 'Final visual not committed')
        base = {('brp_'+n[len('bestref_'):] if n.startswith('bestref_') else n): v for n, v in historical[k]['metrics'].items()}
        base.update({'transaction_'+c: v for c, v in metrics.items()})
        rows.append({'key': list(k), 'split': s['split'], 'delay': s['delay'],
                     'donor': list(donors[k]) if donors[k] else None, 'metrics': base})
    compare(rows, json.loads((out / 'rows.json').read_text()))
    final_agg = aggregate([{'key': x['key'], 'split': 'train', 'metrics': x['metrics']['transaction_true']}
                           for x in rows if x['split'] == 'train'], controls, targets, scales, weight_map)
    compare(final_agg, current_agg)
    compare(final_agg, result['final_training'])
    summary = t.statistics(rows, accepted)
    for n, v in summary.items():
        compare(v, result[n])
    alias = [{**x, 'metrics': {**x['metrics'], 'bestref_true': x['metrics']['transaction_true'],
              **({'bestref_mismatched': x['metrics']['transaction_mismatched']} if 'transaction_mismatched' in x['metrics'] else {})}}
             for x in rows]
    require((previous_audit.decision(alias) and accepted > 0) == result['development_followup_supported'], 'Independent development decision differs')
    captures = json.loads((out / 'captures.json').read_text())
    require(len(captures) <= t.CAPTURES and all(x['status'] == 'captured' and x['eager_setup_calls'] == 1 and
            x['side_stream_warmup_calls'] == 3 and x['capture_calls'] == 1 for x in captures), 'Graph accounting differs')
    expected = {**t.LIMITS, 'vla_loads': 1, 'predictor_loads': 1, 'identity_exact': 88, 'old_centered_exact': 88,
                'frozen_exact': 88, 'gradient_decoder': 72, 'zero_exact': 88, 'captures': len(captures)}
    for k, v in expected.items():
        require(result['counts'].get(k) == v, 'Budget counter differs: '+k)
    require(result['counts'].get('accepted', 0) == accepted and result['counts'].get('rejected', 0) == 72-accepted, 'Gate accounting differs')
    active, events = {}, Counter()
    for event in map(json.loads, (out / 'events.jsonl').read_text().splitlines()):
        events[event['event']] += 1
        name = event['phase']
        if event['event'] == 'started':
            require(name not in active, 'Duplicate phase')
            active[name] = event['limit']
        else:
            require(event['event'] == 'returned' and name in active and event['seconds'] <= active.pop(name), 'Unclosed/expired phase')
    require(not active and events['started'] == events['returned'] == 5857, 'Phase count differs')
    require(t.source_hashes() == prepared['source_hashes'] and not torch.cuda.is_initialized(), 'Audit source/CUDA changed')
    return {'independent_contract_accepted': True, 'transaction_mechanism_completed': True,
        'development_followup_supported': result['development_followup_supported'],
        'accepted': accepted, 'rejected': 72-accepted, 'optimizer_parameter_arrays_checked': arrays,
        'numerical_comparisons': numerical, 'phases': dict(events), 'training_gate_forward_samples': 5184,
        'rollback_model_and_optimizer_exact': True, 'gradients_independently_redifferentiated': False,
        'cuda_initialized': False, 'audit_model_forwards': 0, 'new_env': 0, 'qualification_reads': 0,
        'initial_training': initial_agg, 'final_training': final_agg, 'summary': summary,
        'per_sample': rows, 'transactions': logs, 'rejected_own_changes': rejected_own, 'first_failure': None}


def render(result, audited):
    lines = ['# F-TUA1：全训练集有限步接受与完整回退', '',
        f"execution HEAD：`{result.get('execution_head')}`；运行`{result.get('status')}`；独立接纳`{audited['independent_contract_accepted']}`。", '',
        '原72训练/18episode用于每次接受检查；16开发验证/4episode仅初始exact重放和最终评估，非独立资格。',
        '72次同序AdamW提案，拒绝同时回退参数、moment和step；不是调小学习率或追加搜索。',
        '历史BRP/guarded/plain是此前已审计对照，不是本轮同期重训。oracle是未来视觉冻结策略，不是专家。', '']
    if not audited['independent_contract_accepted']:
        return '\n'.join(lines+['```text', str(result.get('first_failure')), str(audited.get('first_failure')), '```', ''])
    lines += [f"接受{audited['accepted']}，拒绝{audited['rejected']}；开发推进`{audited['development_followup_supported']}`。",
        '训练检查通过由接受规则直接约束，本身不能用作泛化或安全成功证据。', '', '## 训练接受检查', '',
        '```json', json.dumps({'initial': audited['initial_training'], 'final': audited['final_training']}, indent=2), '```', '']
    for split, values in audited['summary']['splits'].items():
        lines += [f'## {split}（episode等权）', '', '| 条件 | N/episode | 首动作MSE | chunk MSE | token MSE |', '|---|---:|---:|---:|---:|']
        for name, value in values['metrics'].items():
            if name.startswith('bestref_'):
                continue
            m = value['macro']
            lines.append(f"| {name} | {value['samples']}/{len(value['episodes'])} | {m['row0']:.12f} | {m['chunk']:.12f} | {m['latent']:.9f} |")
        lines += ['', '下列对比中的候选为transaction；历史BRP指标单独标作brp。', '']
        for name, c in values['contrasts'].items():
            lines.append(f"相对{name}：N={c['samples']}，均值收益{c['macro_benefit']:.12f}，样本{c['sample_directions']}，episode改善{c['episode_improved']}。")
        c = values['contrasts']['frozen']
        lines += ['', '相对冻结候选最不利样本（不删例）：']
        for x in sorted(c['per_sample'], key=lambda x: x['benefit'])[:5]:
            lines.append(f"{x['key']}: {x['benefit']:.12f} ({x['direction']})")
        lines += ['', '留一episode收益：`'+json.dumps(c['leave_one_out'])+'`。', '']
    lines += ['## 更新接受日志', '', '| 提案 | 样本 | 接受 | 提议row0均值 | 保留row0均值 | 未满足检查 |', '|---:|---|---|---:|---:|---|']
    for x in audited['transactions']:
        lines.append(f"| {x['step']} | {x['key']} | {x['accepted']} | {x['proposed']['row0']:.9f} | {x['committed']['row0']:.9f} | {', '.join(k for k,v in x['checks'].items() if not v)} |")
    lines += ['', '## 固定判据与执行', '', '```json', json.dumps({'checks': result['development_checks'],
        'counts': result['counts'], 'phases': audited['phases'], 'execution': result['execution']}, indent=2), '```', '',
        '全训练集扫描增加离线训练开销；不能用于线上需要未知future/oracle的风险门。',
        '没有新Env、图像编码、资格集或真机。无部署/时延/成功率结论；旧R1阴性保持。',
        'CPU审计不重新求梯度或运行模型；验证保存指标、门判读及优化器更新与完整回退算术。', '']
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    out = args.output.resolve()
    require(not (out / 'independent_audit.json').exists(), 'Audit already exists')
    start = time.perf_counter()
    try:
        audited = audit(out)
    except BaseException:
        audited = {'independent_contract_accepted': False, 'first_failure': traceback.format_exc()}
    audited['wall_seconds'] = time.perf_counter()-start
    t.write(out / 'independent_audit.json', audited)
    result = json.loads((out / 'result.json').read_text())
    (out / 'REPORT.md').open('x').write(render(result, audited))
    t.write(out / 'REPORT.json', {'result': result, 'audit': audited})
    print(json.dumps({k:v for k,v in audited.items() if k not in ('summary','per_sample','transactions','rejected_own_changes')}, ensure_ascii=False), flush=True)
    return 0 if audited['independent_contract_accepted'] else 2


if __name__ == '__main__':
    raise SystemExit(main())
