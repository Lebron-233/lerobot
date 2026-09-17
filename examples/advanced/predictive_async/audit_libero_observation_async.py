"""Independent saved-array/time audit for E-OBS1. No model, CUDA or Env execution."""

import argparse
import json
import math
import traceback
from collections import Counter
from pathlib import Path

import libero_observation_async as r
import numpy as np
import torch

check = r.check


def load(path):
    return torch.load(path, map_location='cpu', weights_only=False)


def exact(left, right, label):
    left = left.detach().cpu().numpy() if isinstance(left, torch.Tensor) else np.asarray(left)
    right = right.detach().cpu().numpy() if isinstance(right, torch.Tensor) else np.asarray(right)
    check(left.shape == right.shape and np.array_equal(left, right), label)


def quantiles(values):
    if not values:
        return {'n': 0}
    a = np.sort(np.asarray(values, dtype=np.float64))
    check(np.isfinite(a).all() and np.min(a) >= 0, 'Invalid timing')
    return {'n': len(a), 'mean': float(a.mean()), 'max': float(a[-1]),
            **{f'p{int(p*100)}': float(a[int(np.ceil(p*len(a)))-1]) for p in (.5, .95, .99)}}


def audit_episode(spec, result, arrays, checkpoint, *, output_checker=None):
    check(result['spec'] == spec and result['status'] == 'completed' and result['first_failure'] is None,
          'Episode incomplete or spec differs')
    check(result['worker_joined'] and result['environment_closed'], 'Cleanup incomplete')
    observations, control, outputs = arrays['observations'], arrays['control'], arrays['outputs']
    dispatches, requests = control['dispatches'], result['requests']
    check([o['index'] for o in observations] == list(range(len(dispatches)+1)), 'Observation index holes')
    check(r.e.initial_difference(checkpoint, observations[0]) is None, 'Initial checkpoint differs')
    check([v['stamp']['request_id'] for v in requests] == list(range(1, len(requests)+1)), 'Request identity holes')
    check(set(outputs) == set(range(1, len(requests)+1)), 'Output coverage differs')
    by_id = {v['stamp']['request_id']: v for v in requests}
    installs, previous_end, previous_publish = [], None, None
    for req in requests:
        s, dec = req['stamp'], req['decision']
        rid, index = s['request_id'], s['observation_index']
        check(s['epoch'] == 0 and 0 <= s['expected_delay'] <= 8 and 0 <= index < len(observations), 'Request stamp differs')
        check(observations[index]['returned_at'] == req['observation_returned_at'] <= req['requested_at']
              <= req['started_at'] <= req['completed_at'] <= req['published_at'], 'Request timeline invalid')
        if previous_end is not None:
            check(req['started_at'] >= previous_end and req['requested_at'] >= previous_publish, 'More than one in-flight request')
        previous_end, previous_publish = req['completed_at'], req['published_at']
        check(math.isclose(req['complete_s'], req['completed_at']-req['started_at'], abs_tol=1e-10), 'Complete timing differs')
        check(math.isclose(req['queue_wait_s'], req['started_at']-req['requested_at'], abs_tol=1e-10), 'Queue wait differs')
        check(req['complete_s'] < 15 and req['first_failure'] is None, 'Model deadline or error')
        out = outputs[rid]
        check(out['original'].shape == out['processed'].shape == (50, 7) and out['full'].shape == (1, 50, 32), 'Action shape')
        check(out['noise'].shape == (1, 50, 32), 'Noise shape')
        for field in ('original', 'processed', 'full', 'noise'):
            check(torch.isfinite(out[field]).all(), 'Nonfinite evidence')
        exact(out['original'], out['full'][0, :, :7], 'Unpadding mismatch')
        if output_checker is None:
            check(out['projection_shapes'] == [[1, 50, 32]]*10 and out['vision_encodes'] == 1, 'Model calls mismatch')
        else:
            output_checker(out, req, result)
        check(req['started_at'] <= out['model_started_at'] <= out['model_returned_at'] <= req['completed_at'], 'Model interval not nested')
        prefix_source = req['prefix_source']
        if prefix_source is None:
            check(out['prefix'] is None and index == 0 and rid == 1, 'Unexpected missing prefix')
        else:
            prior = by_id[prefix_source]
            check(prior['decision']['accepted'] and prior['published_at'] <= req['requested_at'], 'Prefix source not installed')
            offset = index-prior['stamp']['observation_index']
            exact(out['prefix'], outputs[prefix_source]['original'][offset:], 'Wrong prefix origin')
            check(len(out['prefix']) == req['prefix_rows'], 'Prefix length')
        consumed = sum(d['dispatched_at'] < req['published_at'] for d in dispatches)
        delay = consumed-index
        check(delay >= 0, 'Negative delay')
        if dec['reason'] != 'stale_or_closed':
            check(dec['actual_delay'] == delay, 'Trim did not use actual consumed actions')
            if dec['accepted']:
                check(delay <= 8 and delay < 50 and dec['takeover_index'] == consumed and dec['source_row'] == delay,
                      'Installed wrong action row')
                installs.append(req)
            else:
                check(dec['reason'] == 'expired' and (delay > 8 or delay >= 50), 'Invalid expiry')
        else:
            check(not dec['accepted'] and req == requests[-1] and req['published_at'] >= control['ended_at'], 'Invalid stop cancellation')
    check(installs and installs[0]['stamp']['request_id'] == 1, 'Bootstrap absent')
    native = [s for s in result['native_steps'] if s['segment'] == 'measurement']
    settling = [s for s in result['native_steps'] if s['segment'] == 'settling']
    check(len(native) == len(dispatches) and len(settling) == 10, 'Native counts differ')
    slots, takeovers, previous_source = [], 0, None
    for index, (d, step) in enumerate(zip(dispatches, native, strict=True)):
        check(d['action_index'] == d['observation_index'] == step['action_index'] == step['observation_index'] == index,
              'Native/controller action identity mismatch')
        candidates = [v for v in installs if v['published_at'] <= d['dispatched_at']]
        active = candidates[-1]
        source = active['stamp']['request_id']
        offset = index-active['stamp']['observation_index']
        check(d['request_id'] == source and d['source_row'] == offset and 0 <= offset < 50, 'Stale or repeated source row')
        exact(d['original'], outputs[source]['original'][offset], 'Normalized action differs')
        exact(d['command'], outputs[source]['processed'][offset], 'Command differs from postprocessed source')
        exact(np.asarray(step['action'], dtype=np.float32), d['command'], 'Native command differs')
        check(observations[index]['returned_at'] <= d['dispatched_at'] <= step['started_at'] <= step['returned_at']
              <= d['returned_at'] == observations[index+1]['returned_at'], 'Native/observation timeline')
        if source != previous_source:
            check(offset == active['decision']['actual_delay'], 'Double trimming or skipped first available row')
            takeovers += int(source != 1)
        previous_source = source
        slots.append(d['slot'])
    check(slots == sorted(set(slots)) and all(0 <= s < 1200 for s in slots), 'Duplicate or invalid wall slot')
    wall = control['ended_at']-control['t0']
    count = min(1200, max(1, int(np.ceil(wall*20))))
    check(result['wall_slots'] == count and result['no_action_slots'] == count-len(dispatches), 'Slot denominator differs')
    check(result['measured_actions'] == len(dispatches) and math.isclose(result['wall_s'], wall, abs_tol=1e-9), 'Wall/action count differs')
    if dispatches:
        check(result['success'] == dispatches[-1]['success'] == control['success'], 'Success differs')
    overlaps = []
    for req in requests[1:]:
        out = outputs[req['stamp']['request_id']]
        for step in native:
            overlap = min(out['model_returned_at'], step['returned_at'])-max(out['model_started_at'], step['started_at'])
            if overlap > 0:
                overlaps.append({'request': req['stamp']['request_id'], 'action_index': step['action_index'], 'seconds': overlap})
    if spec['condition'] == 'serialized':
        check(not overlaps and all(req['decision'].get('actual_delay', 0) == 0 for req in requests), 'Serialized path overlapped or consumed during wait')
    check(result['budget'] == {'episodes': 1, 'settling': 10, 'model': len(requests), 'measurement': len(dispatches)}, 'Episode budget differs')
    for kind, limit in spec['limits'].items():
        check(result['budget'][kind] <= limit, 'Budget exceeded')
    return {'ordinal': spec['ordinal'], 'pair_index': spec['pair_index'], 'arm': spec['condition'],
        'success': result['success'], 'terminal_reason': result['terminal_reason'], 'wall_s': wall,
        'actions': len(dispatches), 'no_action_slots': count-len(dispatches), 'takeovers': takeovers,
        'underflows': result['underflows'], 'request_count': len(requests),
        'expired': sum(v['decision']['reason'] == 'expired' for v in requests),
        'cancelled_at_stop': sum(v['decision']['reason'] == 'stale_or_closed' for v in requests),
        'overlaps': overlaps, 'actual_delays': [v['decision']['actual_delay'] for v in requests if 'actual_delay' in v['decision']],
        'request_seconds': [v['complete_s'] for v in requests[1:]],
        'queue_wait_seconds': [v['queue_wait_s'] for v in requests[1:]],
        'bootstrap_s': requests[0]['complete_s']}


def audit(output):
    check(not torch.cuda.is_initialized(), 'CPU audit only')
    result = json.loads((output / 'result.json').read_text())
    check(result['status'] == 'completed' and result['first_failure'] is None, 'Formal worker incomplete')
    execution = result['execution']
    check(execution['exit_confirmed'] and execution['exit_code'] == 0 and not execution['forced']
          and execution['stop_reason'] is None and not execution['pending'] and not execution['active'], 'Execution not closed')
    prepared = r.validate(result['execution_head'], after=True)
    check(output == r.paths(result['execution_head'])[1], 'Output identity')
    check(json.loads((output / 'manifest.json').read_text()) == prepared['manifest'], 'Saved manifest differs')
    rows, initial, boots, total = [], {}, {}, Counter()
    for spec in prepared['manifest']['rows']:
        folder = output / f"episode_{spec['ordinal']:03d}"
        record = json.loads((folder / 'result.json').read_text())
        check(json.loads((folder / 'started.json').read_text())['spec'] == spec, 'Started identity differs')
        arrays, checkpoint = load(folder / 'arrays.pt'), load(folder / 'initial_checkpoint.pt')
        row = audit_episode(spec, record, arrays, checkpoint)
        rows.append(row)
        pair = spec['pair_index']
        if pair in initial:
            check(r.e.initial_difference(initial[pair], checkpoint) is None and record['initial_pair_exact'], 'Paired initial differs')
            exact(boots[pair]['noise'], arrays['outputs'][1]['noise'], 'Paired bootstrap noise differs')
            exact(boots[pair]['full'], arrays['outputs'][1]['full'], 'Paired initial full action differs')
        else:
            initial[pair], boots[pair] = checkpoint, arrays['outputs'][1]
        total.update(record['budget'])
    check(len(rows) == 16 and len(initial) == 8 and dict(total) == result['native_budget'], 'Global coverage/budget')
    check(all(total[k] <= limit[1] for k, limit in r.LIMITS.items()), 'Global budget exceeded')
    events = list(map(json.loads, (output / 'calls.jsonl').read_text().splitlines()))
    intents, returned = {}, set()
    for event in events:
        cid = event.get('call_id')
        if event['event'] == 'call_intent':
            check(cid not in intents, 'Duplicate intent')
            intents[cid] = event
        elif event['event'] == 'call_return':
            check(cid in intents and cid not in returned, 'Return identity mismatch')
            check(0 <= event['elapsed'] <= intents[cid]['limit'], 'Call deadline')
            returned.add(cid)
        else:
            check(event['event'] != 'call_error', 'Call error')
    check(set(intents) == returned, 'Unknown calls')
    check(sum(v['kind'] == 'eager_request' for v in intents.values()) == total['model'], 'Request ledger differs')
    check(sum(v['kind'] == 'native_step' for v in intents.values()) == total['settling']+total['measurement'], 'Native ledger differs')
    phases, active = Counter(), {}
    for event in map(json.loads, (output / 'events.jsonl').read_text().splitlines()):
        phases[event['event']] += 1
        if event['event'] == 'started':
            check(event['phase'] not in active, 'Duplicate phase')
            active[event['phase']] = event['limit']
        else:
            check(event['event'] == 'returned' and event['phase'] in active, 'Errored phase')
            check(event['seconds'] <= active.pop(event['phase']), 'Phase deadline')
    check(not active and phases == {'started': 18, 'returned': 18}, 'Unclosed phase')
    arms = {}
    for arm in ('serialized', 'async'):
        selected = [v for v in rows if v['arm'] == arm]
        arms[arm] = {'episodes': len(selected), 'successes': sum(v['success'] for v in selected),
            **{k: sum(v[k] for v in selected) for k in ('wall_s', 'actions', 'no_action_slots', 'takeovers', 'underflows',
                                                       'request_count', 'expired', 'cancelled_at_stop')},
            'model_native_intersections': sum(len(v['overlaps']) for v in selected),
            'request_time_s': quantiles([t for v in selected for t in v['request_seconds']]),
            'queue_wait_s': quantiles([t for v in selected for t in v['queue_wait_seconds']])}
    pairs = []
    for pair in range(8):
        a, b = ({v['arm']: v for v in rows if v['pair_index'] == pair}[arm] for arm in ('serialized', 'async'))
        pairs.append({'pair': pair, 'task_state': list(r.PAIRS[pair]),
            'serialized': {k: a[k] for k in ('success', 'actions', 'wall_s', 'no_action_slots', 'takeovers')},
            'async': {k: b[k] for k in ('success', 'actions', 'wall_s', 'no_action_slots', 'takeovers')},
            'async_minus_serialized_wall_s': b['wall_s']-a['wall_s']})
    check(all(result[k] == 0 for k in ('training_updates', 'qualification_reads', 'graph_captures', 'rtc_steps',
                                     'predictor_forwards', 'real_robot')), 'Forbidden activity')
    check(result['attempts'] == 1 and result['retries'] == 0 and result['vla_frozen'], 'Scope changed')
    check(not torch.cuda.is_initialized(), 'Audit initialized CUDA')
    return {'independent_contract_accepted': True, 'episodes': 16, 'initial_pairs_exact': 8,
        'native_actions_checked': total['measurement'], 'request_outputs_checked': total['model'],
        'calls_closed': len(returned), 'phases': dict(phases), 'arms': arms, 'pairs': pairs, 'per_episode': rows,
        'asynchronous_overlap_observed': arms['async']['model_native_intersections'] > 0,
        'waiting_reduction_observed': arms['async']['no_action_slots']/arms['async']['actions'] <
                                      arms['serialized']['no_action_slots']/arms['serialized']['actions'],
        'statistical_noninferiority_claimed': False, 'realtime_qualified': False, 'cuda_initialized': False,
        'audit_model_forwards': 0, 'first_failure': None}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    output = args.output.resolve()
    check(not (output / 'independent_audit.json').exists(), 'Audit exists; do not overwrite')
    try:
        result = audit(output)
    except BaseException:
        result = {'independent_contract_accepted': False, 'first_failure': traceback.format_exc()}
    r.write(output / 'independent_audit.json', result)
    print(json.dumps({k: v for k, v in result.items() if k not in ('per_episode', 'pairs')}), flush=True)
    return 0 if result['independent_contract_accepted'] else 2


if __name__ == '__main__':
    raise SystemExit(main())
