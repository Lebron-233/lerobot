"""F-IQ1 independent CPU reductions, frozen-input checks and native provenance replay."""
import argparse
import copy
import json
import time
import traceback
from collections import Counter, defaultdict
from pathlib import Path
from types import SimpleNamespace

import audit_libero_action_qualification as old
import libero_identity_qualification as r
import numpy as np
import torch

q, require, load, exact = r.q, r.require, r.load, old.exact
compare = old.compare_json


def array(x):
    return x.detach().cpu().float().numpy().astype(np.float64)


def score(visual, value, sample, oracle):
    delta = array(value)[0, :, :7]-array(oracle)[0, :, :7]
    sums, sizes = [], []
    for i in range(2):
        mask = sample['inputs'][i+2].numpy().astype(bool)
        diff = (array(visual[i])-array(sample['future'][i]))[mask]
        sums.append(np.square(diff).sum())
        sizes.append(diff.size)
    require(sum(sizes)>0, 'No visual support')
    return {'row0': float(np.square(delta[0]).mean()), 'chunk': float(np.square(delta).mean()),
            'latent': float(sum(sums)/sum(sizes))}


def decision(rows):
    """No calls to the runner's aggregate/tolerance implementation."""
    if {tuple(x['key'][:2]) for x in rows} != set(r.PAIRS):
        return False, False
    def groups(subset, arm, metric):
        g = defaultdict(list)
        for row in subset:
            g[tuple(row['key'][:2])].append(row['metrics'][arm][metric])
        return {k: float(np.mean(v)) for k, v in g.items()}
    def tol(x):
        return 1e-7+1e-6*abs(x)
    checks, robustness = [], []
    identity_gains = {}
    for control in ('identity','iar_mismatched'):
        subset = [x for x in rows if control in x['metrics']]
        a, b = groups(subset,'iar_true','row0'), groups(subset,control,'row0')
        if len(a)!=8:
            return False, False
        av, bv = float(np.mean(list(a.values()))), float(np.mean(list(b.values())))
        checks += [av < bv-tol(bv), sum(a[k]<b[k]-tol(b[k]) for k in a)>=6]
        wins = sum(x['metrics']['iar_true']['row0'] < x['metrics'][control]['row0']-tol(x['metrics'][control]['row0']) for x in subset)
        robustness.append(wins>len(subset)/2)
        if control=='identity':
            identity_gains = {k:b[k]-a[k] for k in a}
            threshold = tol(bv)
    a = np.mean(list(groups(rows,'iar_true','chunk').values()))
    b = np.mean(list(groups(rows,'identity','chunk').values()))
    checks.append(a<=b+tol(b))
    robustness.append(all(np.mean([v for j,v in identity_gains.items() if j!=k])>threshold for k in identity_gains))
    primary = bool(all(checks))
    return primary, bool(primary and all(robustness))


def check_residual(ev, sample, actions, zero=False):
    exact(ev['actual_actions'], actions, 'Actual committed actions differ')
    exact(ev['actual_mask'], sample['mask'], 'Actual mask differs')
    for i in range(2):
        a, z = ev['action_delta'][i], ev['zero_delta'][i]
        require(a.shape==z.shape==sample['inputs'][i].shape and torch.isfinite(a).all()
                and torch.isfinite(z).all(), 'Invalid branch outputs')
        raw = sample['inputs'][i].float()+(a-z)
        exact(ev['raw_visual'][i],raw,'Identity residual algebra differs')
        exact(ev['visual'][i],raw.to(torch.bfloat16).float(),'Quantization boundary differs')
        if zero:
            exact(a,z,'Zero branch differs')
            exact(raw,sample['inputs'][i].float(),'Zero raw differs')


def replay_native(record, arrays):
    """Reconstruct read-only views required by the original action-source auditor."""
    requests = copy.deepcopy(record['requests'])
    control = copy.deepcopy(arrays['control'])
    for d in control['dispatches']:
        if isinstance(d['command'],torch.Tensor):
            d['command'] = d['command'].cpu().numpy()
    intervals = [(r['model_started_at'],1) for r in requests]+[(r['model_returned_at'],-1) for r in requests]
    active, peak = 0, 0
    for _,change in sorted(intervals):
        active += change
        peak = max(peak,active)
    require(active==0 and peak==1,'Model concurrency changed')
    engine = SimpleNamespace(runtime=SimpleNamespace(captures=record['captures']), owner_thread=record['owner_thread'],
        budget=SimpleNamespace(episode=record['budget']), rows=requests,
        metrics=SimpleNamespace(events=record['metrics']), stats=SimpleNamespace(**record['stats']),
        accepted={v['request_id']:v for v in record['accepted_requests']},
        arrays=arrays['requests'], spec=record['spec'], peak_active_calls=peak)
    audited = q.e.audit_episode(engine,SimpleNamespace(native_steps=record['native_steps']),control)
    compare(json.loads(json.dumps(audited)),record['audit'],'native_source_audit')
    return len(control['dispatches']), audited['planned_takeovers']


def audit(output):
    torch.set_num_threads(1)
    require(not torch.cuda.is_initialized(),'CPU audit entered with CUDA')
    result = json.loads((output/'result.json').read_text())
    prep = r.validate(result['execution_head'],after=True)
    require(output==r.paths(result['execution_head'])[1] and result['status']=='completed'
            and result['first_failure'] is None,'Run not completed')
    ex = result['execution']
    require(ex['child_exit_code']==0 and ex['exit_confirmed'] and not ex['forced_termination']
            and ex['stop_reason'] is None and not ex['pending'] and not ex['active'],'Exit is not closed')
    require(result['attempts']==1 and result['retries']==0 and result['vla_frozen']
            and result['frozen_weights_unchanged'] and result['graph_released'],'Freeze/attempt evidence differs')
    for name in ('training_updates','backward','real_robot','old_qualification_reads'):
        require(result[name]==0,f'Forbidden action {name}')
    for name in ('baseline_qualified','realtime_qualified','predictor_benefit_tested','predictor_controls_environment'):
        require(result[name] is False,f'Unqualified flag {name}')
    require(result['risk_thresholds'] is None and result['old_confirmation']=='not_started_untouched','Safety scope changed')
    compare(result['runtime_environment'],prep['runtime_environment'])
    compare(json.loads((output/'manifest.json').read_text()),prep['manifest'])
    before, after, original = load(output/'frozen_checkpoint_before.pt'),load(output/'frozen_weights_after.pt'),load(r.CHECKPOINT)
    q.checkpoint_valid(before,'centered')
    require(before['state_dict'].keys()==after.keys()==original['state_dict'].keys(),'Weight names differ')
    for k,v in original['state_dict'].items():
        exact(v,before['state_dict'][k],'Initial weights differ')
        exact(v,after[k],'Final frozen weights differ')
    anchors = load(output/'anchor_replay.pt')
    source_samples = r.anchors()
    require([v['key'] for v in anchors]==[list(q.pfx.key(s)) for s in source_samples],'Anchor identities differ')
    residuals = {tuple(x['key']):x for x in load(q.SOURCE/'assessment_selected_centered.pt') if x['context']=='true'}
    for a,s in zip(anchors,source_samples,strict=True):
        ref = load(r.anchor_path(s))
        check_residual(a['evidence'],s,s['actions'])
        for f in ('action_delta','zero_delta'):
            for v,w in zip(a['evidence'][f],residuals[q.pfx.key(s)][f],strict=True):
                exact(v,w,'Anchor original branch differs')
        for v,w in zip(a['evidence']['visual'],ref['visual']['identity_true'],strict=True):
            exact(v,w,'Anchor BF16 differs')
        for v,w in zip(a['evidence']['raw_visual'],ref['raw_new']['identity_true'],strict=True):
            exact(v,w,'Anchor raw differs')
        exact(a['identity'],ref['outputs']['identity'],'Anchor identity differs')
        exact(a['iar'],ref['outputs']['identity_true'],'Anchor IAR differs')
    samples = load(output/'aligned_cache.pt')
    by_key = {q.pfx.key(s):s for s in samples}
    require(8<=len(samples)<=32 and len(by_key)==len(samples),'Sample count/uniqueness differs')
    native_keys, totals, selections = set(),Counter(),[]
    native_actions,takeovers = 0,0
    for spec in prep['manifest']['rows']:
        directory = output/f"episode_{spec['ordinal']:03d}"
        record,arrays = json.loads((directory/'result.json').read_text()),load(directory/'arrays.pt')
        require(record['spec']==spec and record['status']=='completed' and record['first_failure'] is None,'Native status/identity differs')
        for name in ('environment_closed','worker_joined','graph_released','original_sampler_restored','metrics_closed'):
            require(record[name] is True,f'Native cleanup {name}')
        initial = load(directory/'initial_checkpoint.pt')
        require(q.e.initial_difference(initial,arrays['observations'][0]) is None,'First-model initial observation differs')
        actions,take = replay_native(record,arrays)
        native_actions += actions
        takeovers += take
        pairs,excluded = q.cov.selected_pairs(record,arrays)
        selections.append({'ordinal':spec['ordinal'],'requests':[p['request_id'] for p in pairs],'excluded':excluded})
        for p in pairs:
            k = (spec['task_id'],spec['initial_state_id'],p['request_id'])
            native_keys.add(k)
            s = by_key[k]
            require(s['ordinal']==spec['ordinal'] and s['split']=='qualification','Sample origin differs')
            for name in ('current_index','future_index','delay'):
                require(s[name]==p[name],f'Alignment {name}')
            for name in ('actions','mask'):
                exact(s[name],p[name],f'Committed prefix {name}')
            q.validate_prefix(s)
            for a,b in zip(s['inputs'],p['cached']['inputs'],strict=True):
                exact(a,b,'Native cached input differs')
            exact(s['archived_full_chunk'],p['cached']['full_chunk'],'Native full chunk differs')
        for kind,(limit,_) in q.NATIVE_LIMITS.items():
            value = record['budget'].get(kind,0)
            require(type(value) is int and 0<=value<=limit,f'Native budget {kind}')
            totals[kind] += value
        require(record['budget']['episodes']==1 and len(record['captures'])==record['budget']['capture'],'Native capture budget differs')
    require(native_keys==set(by_key),'Native/cache selection coverage differs')
    compare(selections,json.loads((output/'selections.json').read_text()))
    compare(dict(totals),result['native_budget'])
    for kind,(_,limit) in q.NATIVE_LIMITS.items():
        require(totals[kind]<=limit,f'Total native budget {kind}')
    donors = q.pfx.donor_map(samples)
    require({p.name for p in (output/'predictions').glob('*.pt')}=={'_'.join(map(str,k))+'.pt' for k in by_key},'Prediction file coverage differs')
    rows,numerical = [],0
    for k,s in sorted(by_key.items()):
        item = load(output/'predictions'/('_'.join(map(str,k))+'.pt'))
        require(item['key']==list(k) and item['delay']==s['delay'] and
                item['donor']==(list(donors[k]) if donors[k] else None),'Prediction identity/donor differs')
        names = set(r.ARMS)-({'iar_mismatched'} if donors[k] is None else set())
        require(set(item['outputs'])==set(item['visual'])==names|{'oracle'} and set(item['metrics'])==names,'Arm coverage differs')
        require(set(item['residuals'])==names-{'identity'},'Residual coverage differs')
        for name in ('identity','oracle'):
            expected = s['inputs'][:2] if name=='identity' else s['future']
            for a,b in zip(item['visual'][name],expected,strict=True):
                exact(a,b,f'{name} tokens differ')
        exact(item['outputs']['identity'],s['archived_full_chunk'],'Identity output differs')
        exact(item['outputs']['iar_zero'],item['outputs']['identity'],'Zero output differs')
        for name,ev in item['residuals'].items():
            context = name.removeprefix('iar_')
            actions = torch.zeros_like(s['actions']) if context=='zero' else by_key[donors[k]]['actions'] if context=='mismatched' else s['actions']
            check_residual(ev,s,actions,context=='zero')
            for a,b in zip(ev['visual'],item['visual'][name],strict=True):
                exact(a,b,'Decoded candidate evidence differs')
        metrics = {}
        for name,value in item['outputs'].items():
            require(value.shape==(1,50,32) and torch.isfinite(value).all(),'Invalid output')
            if name=='oracle':
                continue
            metrics[name] = score(item['visual'][name],value,s,item['outputs']['oracle'])
            compare(metrics[name],item['metrics'][name])
            numerical += 3
        rows.append({'key':list(k),'delay':s['delay'],'donor':item['donor'],'metrics':metrics})
    compare(rows,[json.loads(x) for x in (output/'metric_rows.jsonl').read_text().splitlines()])
    summary = r.aggregate(rows)
    for k,v in summary.items():
        compare(v,result[k],k)
    primary,robust = decision(rows)
    require(primary==result['heldout_primary_gate_passed'] and robust==result['heldout_robustness_gate_passed'],'Independent gates differ')
    n,m = len(rows),sum(x['donor'] is not None for x in rows)
    expected = {'encoding':n+8,'decoder':32+4*n+m,'predictor':32+4*n+2*m,'anchor_exact':16,
                'current_exact':8,'identity_exact':n,'zero_exact':n,'vla_loads':1,'predictor_loads':1}
    compare(expected,result['expected_counts'])
    captures = json.loads((output/'offline_captures.json').read_text())
    compare({**expected,'offline_captures':len(captures)},result['counts'])
    require(len(captures)<=8 and all(c['status']=='captured' and c['eager_setup_calls']==1 and
            c['side_stream_warmup_calls']==3 and c['capture_calls']==1 for c in captures),'Offline Graph evidence differs')
    require(result['episodes_completed']==totals['episodes']==8,'Episode total differs')
    active,counts = {},Counter()
    for ev in map(json.loads,(output/'events.jsonl').read_text().splitlines()):
        counts[ev['event']] += 1
        name = ev['phase']
        if ev['event']=='started':
            require(name not in active,'Duplicate phase')
            active[name] = ev['limit']
        else:
            require(ev['event']=='returned' and name in active,'Failed/unmatched phase')
            require(ev['seconds']<=active.pop(name),'Phase exceeded budget')
    require(not active and counts['started']==counts['returned'],'Phase not closed')
    accounting = q.natural.trace.journal_accounting(q.natural.trace.read_events(output/'calls.jsonl'))
    require(accounting==result['accounting'] and accounting['journal_consistent'] and
            not accounting['call_errors'] and not accounting['unknown_calls'],'Native calls not closed')
    require(r.source_hashes()==prep['source_hashes'] and not torch.cuda.is_initialized(),'Source/CPU boundary changed')
    return {'independent_contract_accepted':True,'heldout_primary_gate_passed':primary,
            'heldout_robustness_gate_passed':robust,'samples':n,'mismatched_samples':m,
            'native_actions_reaudited':native_actions,'native_takeovers_reaudited':takeovers,
            'numerical_comparisons':numerical,'phases':dict(counts),'cuda_initialized':False,
            'audit_model_forwards':0,'new_env':0,'summary':summary,'per_sample':rows,'first_failure':None}


def render(result,audited):
    lines = ['# F-IQ1：冻结Identity中心化候选的新初态资格','',f"execution HEAD：`{result['execution_head']}`。",
        f"运行：`{result['status']}`；独立接纳：`{audited['independent_contract_accepted']}`；主门：`{audited.get('heldout_primary_gate_passed')}`；稳健门：`{audited.get('heldout_robustness_gate_passed')}`。",'',
        '新task6/7 state4..7；只由Identity异步采集，IAR没有控制Env。旧ACQ1-R1样本未读取。',
        '真实预测器两次前向重构I+(h(a)-h(0))，无训练/选点/幅度调整。oracle是未来视觉、当前state和同language/noise的冻结策略输出，不是专家动作或成功率上界。','']
    if audited['independent_contract_accepted']:
        s = audited['summary']
        lines += ['## Episode等权结果','','| 条件 | N/episode | 首动作MSE | chunk MSE | token MSE |','|---|---:|---:|---:|---:|']
        for name,v in s['metrics'].items():
            m = v['macro']
            lines.append(f"| {name} | {v['samples']}/{len(v['episodes'])} | {m.get('row0',float('nan')):.12f} | {m.get('chunk',float('nan')):.12f} | {m.get('latent',float('nan')):.9f} |")
        for name,c in s['contrasts'].items():
            lines += ['',f'## 相对{name}',f"配对{c['paired_samples']}样本/{c['paired_episodes']}episode；均值收益（对照−IAR）{c['macro_benefit']}；样本改善{c['sample_improved']}、episode改善{c['episode_improved']}。",'',
                      '| task/state/request | 首动作收益 | 方向 |','|---|---:|---|']
            for row in c['per_sample']:
                lines.append(f"| {'/'.join(map(str,row['key']))} | {row['benefit']:.12f} | {row['direction']} |")
            lines += ['',f"逐episode：`{json.dumps(c['episodes'])}`",f"留一episode：`{json.dumps(c['leave_one_out'])}`",f"最大episode净收益份额：`{c['largest_episode_net_share']}`。"]
        lines += ['',f"最坏超额首动作误差：{s['worst_excess_vs_identity']}；delay分布：{s['delay_counts']}。",'',
            '## 固定判据','```json',json.dumps({'primary':s['primary_checks'],'robustness':s['robustness_checks']},indent=2),'```']
    else:
        lines += ['```text',str(result.get('first_failure')),str(audited.get('first_failure')),'```']
    lines += ['','## 执行','```json',json.dumps({k:result.get(k) for k in ('counts','native_budget','execution')},indent=2),'```','',
              '独立审计不执行模型或Env；复用原native来源/选样工具，数值归约及判门另行实现。无真机、闭环收益、模型算子加速或统计显著性声明。',
              '主门/稳健门均过才进入另立合同的完整路径时延和闭环；未过则封存此候选的本次资格结论，不在新样本上调参直到通过。部署资格仍false，risk_thresholds=null。','']
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    output = parser.parse_args().output.resolve()
    require(not (output/'independent_audit.json').exists(),'Audit exists; no overwrite')
    start = time.perf_counter()
    try:
        audited = audit(output)
    except BaseException:
        audited = {'independent_contract_accepted':False,'first_failure':traceback.format_exc()}
    audited['wall_seconds'] = time.perf_counter()-start
    r.write(output/'independent_audit.json',audited)
    result = json.loads((output/'result.json').read_text())
    with (output/'REPORT.md').open('x') as f:
        f.write(render(result,audited))
    r.write(output/'REPORT.json',{'result':result,'audit':audited})
    print(json.dumps({k:v for k,v in audited.items() if k not in ('summary','per_sample')}),flush=True)
    return 0 if audited['independent_contract_accepted'] else 2


if __name__=='__main__':
    raise SystemExit(main())
