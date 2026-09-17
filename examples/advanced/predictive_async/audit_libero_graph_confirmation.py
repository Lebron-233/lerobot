"""CPU-only confirmation audit; no repeated policy, encoding, gradient or physics."""

import argparse
import json
import math
import traceback
from collections import Counter
from pathlib import Path

import audit_libero_graph_feedback as prior
import libero_graph_confirmation as r
import numpy as np
import torch
from scipy.optimize import brentq

check, load, exact = r.check, prior.load, prior.exact


def independent_upper(k, n, alpha=.05):
    check(n > 0 and 0 <= k <= n and 0 < alpha < 1, 'Invalid statistical counts')
    if k == n:
        return 1.0
    if k == 0:
        return float(1-np.power(alpha, 1/n))
    fraction = np.float64(k)/n
    def equation(p):
        return float(n*(fraction*np.log(fraction/p)+(1-fraction)*np.log((1-fraction)/(1-p)))+np.log(alpha))
    return brentq(equation, float(fraction), float(np.nextafter(1., 0.)), xtol=1e-14)


def aggregate(rows):
    arms = {}
    for arm in ('serialized', 'async'):
        selected = [v for v in rows if v['arm'] == arm]
        check(len(selected) == 32, 'Wrong episode denominator')
        times = prior.previous.quantiles([t for v in selected for t in v['request_seconds']])
        check(times['n'] > 0, 'No nonbootstrap timing evidence')
        times['required_delay_steps'] = int(np.ceil(times['p99']*20))+1
        arms[arm] = {'episodes': 32, 'successes': sum(v['success'] for v in selected),
            **{k: sum(v[k] for v in selected) for k in ('wall_s', 'actions', 'no_action_slots', 'takeovers',
                'underflows', 'request_count', 'expired', 'cancelled_at_stop')},
            'model_native_intersections': sum(len(v['overlaps']) for v in selected), 'request_time_s': times,
            'queue_wait_s': prior.previous.quantiles([t for v in selected for t in v['queue_wait_seconds']]),
            'bootstrap_s': prior.previous.quantiles([v['bootstrap_s'] for v in selected]),
            'max_actual_delay': max(d for v in selected for d in v['actual_delays']),
            'requests_above_350ms': sum(t > .35 for v in selected for t in v['request_seconds']),
            'episode_max_request_s': {str(v['ordinal']): max(v['request_seconds'], default=0.) for v in selected}}
    counts, pairs = Counter(), []
    for index, identity in enumerate(r.PAIRS):
        values = {v['arm']: v for v in rows if v['pair_index'] == index}
        check(set(values) == {'serialized', 'async'}, 'Pair missing')
        a, b = values['serialized']['success'], values['async']['success']
        label = 'both_success' if a and b else 'serialized_only' if a else 'async_only' if b else 'neither_success'
        counts[label] += 1
        pairs.append({'pair_index': index, 'task_state': list(identity), 'outcome': label,
            'arms': {arm: {k: v[k] for k in ('success', 'actions', 'wall_s', 'no_action_slots', 'takeovers',
                'terminal_reason', 'startup_s')} for arm, v in values.items()}})
    k, gains = counts['serialized_only'], counts['async_only']
    u = independent_upper(k, 32)
    stats = {'lost_pairs': k, 'gained_pairs': gains, 'n': 32, 'observed_success_difference': (gains-k)/32,
        'regression_probability_upper95': u, 'success_difference_lower95_conservative': -u,
        'alpha': .05, 'margin': .10, 'bounded_task_retention_passed': u < .10,
        'interpretation_requires_independent_pairs': True, 'scope': 'fixed_four_tasks_new_states12_to19'}
    per_task = {str(task): {arm: {'n': 8, 'successes': sum(v['success'] for v in rows
        if v['arm'] == arm and r.PAIRS[v['pair_index']][0] == task)} for arm in arms} for task in (0, 2, 6, 7)}
    latency = all(v['request_time_s']['required_delay_steps'] <= 8 for v in arms.values())
    all_requests = all(v['requests_above_350ms'] == 0 for v in arms.values())
    overlap = arms['async']['model_native_intersections'] > 0
    waiting = arms['async']['no_action_slots']/arms['async']['actions'] < arms['serialized']['no_action_slots']/arms['serialized']['actions']
    return {'arms': arms, 'pairs': pairs, 'per_task': per_task, 'discordance': dict(counts), 'retention': stats,
        'online_latency_budget_passed': latency, 'all_nonbootstrap_within_budget': all_requests,
        'asynchronous_overlap_observed': overlap, 'waiting_reduction_observed': waiting,
        'bounded_confirmation_passed': stats['bounded_task_retention_passed'] and latency and all_requests and overlap and waiting}


def audit(output):
    check(not torch.cuda.is_initialized(), 'CPU audit only')
    result = json.loads((output/'result.json').read_text())
    check(result['status'] == 'completed' and result['first_failure'] is None, 'Formal confirmation incomplete')
    execution = result['execution']
    check(execution['exit_confirmed'] and execution['exit_code'] == 0 and not execution['forced']
          and not execution['active'] and not execution['pending'] and execution['stop_reason'] is None, 'Execution did not close')
    saved = r.validate(result['execution_head'], after=True)
    check(result['candidate_head'] == r.CANDIDATE_HEAD and output == r.paths(result['execution_head'])[1], 'Frozen identity differs')
    check(json.loads((output/'manifest.json').read_text()) == saved['manifest'], 'Saved manifest changed')
    folders = {p.name for p in output.glob('episode_*') if p.is_dir()}
    check(folders == {f'episode_{i:03d}' for i in range(64)}, 'Extra or missing episodes')
    rows, initials, boots, records, total = [], {}, {}, [], Counter()
    for spec in saved['manifest']['rows']:
        folder = output/f"episode_{spec['ordinal']:03d}"
        check(json.loads((folder/'started.json').read_text())['spec'] == spec, 'Started identity differs')
        record = json.loads((folder/'result.json').read_text())
        arrays, checkpoint = load(folder/'arrays.pt'), load(folder/'initial_checkpoint.pt')
        row = prior.inspect_episode(spec, record, arrays, checkpoint)
        rows.append(row)
        records.append(record)
        pair, boot = spec['pair_index'], arrays['outputs'][1]
        if pair in initials:
            check(r.e.initial_difference(initials[pair], checkpoint) is None and record['initial_pair_exact'], 'Paired initial differs')
            exact(boots[pair]['full'], boot['full'], 'Paired bootstrap output differs')
            exact(boots[pair]['noise'], boot['noise'], 'Paired bootstrap noise differs')
            del initials[pair], boots[pair]
        else:
            initials[pair], boots[pair] = checkpoint, {'full': boot['full'], 'noise': boot['noise']}
        total.update(record['budget'])
        del arrays
    check(not initials and len(rows) == 64 and dict(total) == result['native_budget'], 'Coverage/budget mismatch')
    check(all(total[k] <= v[1] for k, v in r.LIMITS.items()), 'Global budget exceeded')
    check(result['graph_captures'] == 64 and result['capture_internal'] == {'setup': 64, 'warmup': 192, 'capture': 64}, 'Capture counts')
    intents, returned, closed = {}, {}, {}
    for event in map(json.loads, (output/'calls.jsonl').read_text().splitlines()):
        cid = event.get('call_id')
        if event['event'] == 'call_intent':
            check(cid not in intents, 'Duplicate intent')
            intents[cid] = event
        elif event['event'] == 'call_return':
            check(cid in intents and cid not in returned and 0 <= event['elapsed'] <= intents[cid]['limit'], 'Invalid return')
            returned[cid] = event
        elif event['event'] == 'owner_resources_closed':
            check(event['ordinal'] not in closed and event['first_failure'] is None, 'Cleanup failure')
            closed[event['ordinal']] = event
        else:
            check(event['event'] != 'call_error', 'Call error')
    check(intents.keys() == returned.keys() and len(closed) == 64, 'Unknown calls or missing cleanup')
    check(sum(v['kind'] == 'graph_request' for v in intents.values()) == total['model'], 'Model call budget mismatch')
    check(sum(v['kind'] == 'native_step' for v in intents.values()) == total['measurement']+640, 'Native call budget mismatch')
    for v in intents.values():
        if v['kind'] == 'environment_close':
            check(closed[v['ordinal']]['completed_at'] <= v['timestamp'], 'Environment closed before owner')
    active, phases = {}, Counter()
    for v in map(json.loads, (output/'events.jsonl').read_text().splitlines()):
        phases[v['event']] += 1
        if v['event'] == 'started':
            check(v['phase'] not in active, 'Duplicate phase')
            active[v['phase']] = v['limit']
        else:
            check(v['event'] == 'returned' and v['phase'] in active and v['seconds'] <= active.pop(v['phase']), 'Phase error/deadline')
    check(not active and phases == {'started': 66, 'returned': 66}, 'Phase coverage mismatch')
    check(all(result[k] == 0 for k in ('training_updates', 'qualification_reads', 'rtc_steps', 'predictor_forwards', 'real_robot'))
          and result['replay_commands'] is False and result['vla_frozen'], 'Frozen scope changed')
    check(result['attempts'] == 1 and result['retries'] == 0, 'Hidden retry')
    summary = aggregate(rows)
    check(summary['discordance'] == result['discordance'], 'Paired outcome counts differ')
    for key, value in summary['retention'].items():
        original = result['retention'][key]
        if type(value) is float:
            check(math.isclose(value, original, rel_tol=1e-10, abs_tol=1e-12), 'Independent confidence bound differs')
        else:
            check(value == original, 'Independent retention decision differs')
    check(result['complete_pairs'] == 32 and result['episodes_completed'] == 64, 'Final denominator mismatch')
    for p, stored in zip(summary['pairs'], result['pairs'], strict=True):
        check(p['pair_index'] == stored['pair_index'] and p['task_state'] == stored['task_state']
              and p['outcome'] == stored['outcome'], 'Final pair identity differs')
        for arm in ('serialized', 'async'):
            check(p['arms'][arm]['actions'] == stored['arms'][arm]['measured_actions'], 'Final pair actions differ')
            for key in ('success', 'wall_s', 'no_action_slots', 'terminal_reason', 'startup_s'):
                check(p['arms'][arm][key] == stored['arms'][arm][key], 'Final pair result differs')
    check(not torch.cuda.is_initialized(), 'Audit initialized CUDA')
    return {'independent_contract_accepted': True, 'candidate_unchanged': True,
        'episodes': 64, 'initial_pairs_exact': 32, 'native_actions_checked': total['measurement'],
        'request_outputs_checked': total['model'], 'calls_closed': len(returned), 'phases': dict(phases),
        'graph_captures': 64, 'runtime_closed_on_owner': 64, **summary, 'per_episode': rows,
        'realtime_qualified': False, 'broad_generalization_claimed': False,
        'audit_model_forwards': 0, 'cuda_initialized': False, 'first_failure': None,
        'limitations': ['Statistical risk bound assumes independent pairs; fixed four tasks only.',
            'Audit does not rerun RGB encoding, model, postprocessing or physics.',
            'Request timings are correlated; no independent-request hard-real-time guarantee.']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    output = args.output.resolve()
    check(not (output/'independent_audit.json').exists(), 'Audit already exists; do not overwrite')
    try:
        checked = audit(output)
    except BaseException:
        checked = {'independent_contract_accepted': False, 'first_failure': traceback.format_exc()}
    r.write(output/'independent_audit.json', checked)
    print(json.dumps({k: v for k, v in checked.items() if k not in ('per_episode', 'pairs')}), flush=True)
    return 0 if checked['independent_contract_accepted'] else 2


if __name__ == '__main__':
    raise SystemExit(main())
