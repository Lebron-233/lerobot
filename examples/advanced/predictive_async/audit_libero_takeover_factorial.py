"""CPU-only E-TFD1 audit; exact anchor reproduction precedes causal interpretation."""

import argparse
import json
import math
import traceback
from collections import Counter
from pathlib import Path

import audit_libero_graph_feedback as graph_audit
import libero_takeover_factorial as r
import numpy as np
import torch

check, load = r.check, r.load


def exact(a, b, label):
    if isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor):
        check(a.shape == b.shape and a.dtype == b.dtype and torch.equal(a, b), label)
    else:
        a, b = np.asarray(a), np.asarray(b)
        check(a.shape == b.shape and a.dtype == b.dtype and np.array_equal(a, b), label)


def compare_request(actual, archived, label):
    for name in ('noise', 'full', 'original', 'processed'):
        exact(actual[name], archived[name], label+':'+name)
    check(len(actual['inputs']) == len(archived['inputs']) == 8, label+':input_count')
    for i, (a, b) in enumerate(zip(actual['inputs'], archived['inputs'], strict=True)):
        exact(a, b, label+f':input_{i}')


def check_anchor(record, arrays, source_record, source_arrays):
    for name in ('success', 'terminal_reason', 'measured_actions'):
        check(record[name] == source_record[name], 'Anchor outcome differs:'+name)
    current, archived = arrays['control']['dispatches'], source_arrays['control']['dispatches']
    check(len(current) == len(archived), 'Anchor action coverage differs')
    for a, b in zip(current, archived, strict=True):
        check((a['action_index'], a['request_id'], a['source_row']) ==
              (b['action_index'], b['request_id'], b['source_row']), 'Anchor action origin differs')
        exact(a['command'], b['command'], 'Anchor command differs')
    check(len(arrays['observations']) == len(source_arrays['observations']), 'Anchor observations coverage')
    for a, b in zip(arrays['observations'], source_arrays['observations'], strict=True):
        check(r.e.initial_difference(a, b) is None, 'Anchor physical observation differs')
    check(arrays['outputs'].keys() == source_arrays['outputs'].keys(), 'Anchor requests coverage')
    for rid, out in arrays['outputs'].items():
        compare_request(out, source_arrays['outputs'][rid], f'anchor_request_{rid}')
    return {'commands_exact': len(current), 'observations_exact': len(arrays['observations']),
            'requests_exact': len(arrays['outputs']), 'complete_trajectory_reproduced': True}


def factors(table):
    check(set(table) == {'first0_later0', 'first2_later2', 'first0_later2', 'first2_later0'}, 'Incomplete factorial')
    s00, s22, s02, s20 = (int(table[k]['success']) for k in
                          ('first0_later0', 'first2_later2', 'first0_later2', 'first2_later0'))
    return {'first_delay_effect_when_later_zero': s20-s00,
            'first_delay_effect_when_later_two': s22-s02,
            'later_delay_effect_when_first_zero': s02-s00,
            'later_delay_effect_when_first_two': s22-s20,
            'interaction_contrast_success': s22-s20-s02+s00,
            'interpretation': 'Deterministic selected-state contrasts only; not population effect estimates.'}


def audit(output):
    check(not torch.cuda.is_initialized(), 'Audit must not initialize CUDA')
    result = json.loads((output/'result.json').read_text())
    check(result['status'] == 'completed' and result['first_failure'] is None, 'Worker incomplete')
    ex = result['execution']
    check(ex['exit_confirmed'] and ex['exit_code'] == 0 and not ex['forced'] and
          ex['stop_reason'] is None and not ex['active'] and not ex['pending'], 'Execution not closed')
    prepared = r.validate(result['execution_head'], after=True)
    check(output == r.paths(result['execution_head'])[1] and
          json.loads((output/'manifest.json').read_text()) == prepared['manifest'], 'Output/manifest differs')
    rows, total, anchors, request_count = [], Counter(), [], 0
    table = {}
    for spec in prepared['manifest']['rows']:
        folder = output/f"episode_{spec['ordinal']:03d}"
        record = json.loads((folder/'result.json').read_text())
        check(json.loads((folder/'started.json').read_text())['spec'] == spec, 'Started identity changed')
        arrays, checkpoint = load(folder/'arrays.pt'), load(folder/'initial_checkpoint.pt')
        measured = graph_audit.inspect_episode(spec, record, arrays, checkpoint)
        for req in record['requests']:
            rid, dec = req['stamp']['request_id'], req['decision']
            d = 0 if rid == 1 else spec['first_delay'] if rid == 2 else spec['later_delay']
            if dec['accepted']:
                check(dec['actual_delay'] == d, 'Intervention dose not realized')
            else:
                check(dec['reason'] == 'stale_or_closed', 'Unexpected result rejection')
            if rid > 1:
                check(req['diagnostic_target_delay'] == d, 'Intervention intent mismatch')
        # All branches start from the same selected initial state and first request inputs.
        zero = r.REFERENCES[(spec['task_id'], spec['initial_state_id'])][0]
        ref_folder = r.SOURCE/f'episode_{zero:03d}'
        ref_arrays = load(ref_folder/'arrays.pt')
        check(r.e.initial_difference(checkpoint, ref_arrays['observations'][0]) is None, 'Initial source mismatch')
        for rid in (1, 2):
            compare_request(arrays['outputs'][rid], ref_arrays['outputs'][rid], f'common_request_{rid}')
        for i in range(21):
            check(r.e.initial_difference(arrays['observations'][i], ref_arrays['observations'][i]) is None,
                  'Pre-intervention physical state differs')
        del ref_arrays
        if spec['reference_ordinal'] is not None:
            ref_folder = r.SOURCE/f"episode_{spec['reference_ordinal']:03d}"
            source_record = json.loads((ref_folder/'result.json').read_text())
            source_arrays = load(ref_folder/'arrays.pt')
            anchors.append({'ordinal': spec['ordinal'], 'reference': spec['reference_ordinal'],
                            **check_anchor(record, arrays, source_record, source_arrays)})
            del source_arrays
        total.update(record['budget'])
        request_count += len(record['requests'])
        row = {'ordinal': spec['ordinal'], 'identity': [spec['task_id'], spec['initial_state_id']],
            'arm': spec['arm'], 'success': record['success'], 'actions': record['measured_actions'],
            'terminal_reason': record['terminal_reason'], 'requests': len(record['requests']),
            'max_actual_delay': max(measured['actual_delays']), 'underflows': measured['underflows'],
            'cancelled_at_stop': measured['cancelled_at_stop'], 'control_wall_s_descriptive': record['wall_s']}
        rows.append(row)
        table.setdefault(str(spec['initial_state_id']), {})[spec['arm']] = row
        del arrays
    check(len(rows) == 8 and len(anchors) == 4 and result['episodes_completed'] == 8, 'Diagnostic coverage changed')
    check(dict(total) == result['native_budget'] and all(total[k] <= v[1] for k, v in r.LIMITS.items()), 'Budget differs')
    check(result['graph_captures'] == 8 and result['capture_internal'] == {'setup': 8, 'warmup': 24, 'capture': 8},
          'Capture accounting changed')
    intents, returns, closes = {}, {}, {}
    for event in map(json.loads, (output/'calls.jsonl').read_text().splitlines()):
        cid = event.get('call_id')
        if event['event'] == 'call_intent':
            check(cid not in intents, 'Duplicate intent')
            intents[cid] = event
        elif event['event'] == 'call_return':
            check(cid in intents and cid not in returns and math.isfinite(event['elapsed']) and
                  0 <= event['elapsed'] <= intents[cid]['limit'], 'Invalid call return')
            returns[cid] = event
        elif event['event'] == 'owner_resources_closed':
            check(event['ordinal'] not in closes and event['first_failure'] is None, 'Owner cleanup error')
            closes[event['ordinal']] = event
        else:
            check(event['event'] != 'call_error', 'Call failed')
    check(intents.keys() == returns.keys() and len(closes) == 8, 'Unclosed calls/resources')
    check(sum(v['kind'] == 'graph_request' for v in intents.values()) == request_count, 'Request count mismatch')
    check(sum(v['kind'] == 'native_step' for v in intents.values()) == total['measurement']+total['settling'], 'Native count mismatch')
    for event in intents.values():
        if event['kind'] == 'environment_close':
            check(closes[event['ordinal']]['completed_at'] <= event['timestamp'], 'Environment closed too early')
    phases, active = Counter(), {}
    for event in map(json.loads, (output/'events.jsonl').read_text().splitlines()):
        phases[event['event']] += 1
        if event['event'] == 'started':
            check(event['phase'] not in active, 'Duplicate phase')
            active[event['phase']] = event['limit']
        else:
            check(event['event'] == 'returned' and event['phase'] in active and
                  event['seconds'] <= active.pop(event['phase']), 'Phase failure/deadline')
    check(not active and phases == {'started': 10, 'returned': 10}, 'Phase coverage mismatch')
    check(result['attempts'] == 1 and result['retries'] == 0 and result['vla_frozen'] and
          all(result[k] == 0 for k in ('training_updates', 'rtc_steps', 'real_robot')), 'Scope changed')
    check(not torch.cuda.is_initialized(), 'CPU audit initialized CUDA')
    return {'independent_contract_accepted': True, 'candidate_unchanged': True,
        'complete_anchor_trajectories_exact': 4, 'anchors': anchors, 'common_first_two_requests_exact': 16,
        'episodes': 8, 'native_actions_checked': total['measurement'], 'request_outputs_checked': request_count,
        'calls_closed': len(returns), 'phases': dict(phases), 'graph_captures': 8, 'rows': rows,
        'factorial_contrasts': {k: factors(v) for k, v in table.items()},
        'qualification_claimed': False, 'deployment_claimed': False, 'audit_model_forwards': 0,
        'cuda_initialized': False, 'confirmation_result_unchanged': True, 'first_failure': None,
        'limitations': ['Known selected cases, one realization per cell; no new independent validation.',
            'Imposed consumed-action delays may wait; not a deployable real-time policy or speed test.',
            'Audit does not rerun model, encoding, postprocessing or physics.']}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    torch.set_num_threads(1)
    output = args.output.resolve()
    check(not (output/'independent_audit.json').exists(), 'Do not overwrite audit')
    try:
        result = audit(output)
    except BaseException:
        result = {'independent_contract_accepted': False, 'first_failure': traceback.format_exc()}
    r.write(output/'independent_audit.json', result)
    print(json.dumps(result), flush=True)
    return 0 if result['independent_contract_accepted'] else 2


if __name__ == '__main__':
    raise SystemExit(main())
