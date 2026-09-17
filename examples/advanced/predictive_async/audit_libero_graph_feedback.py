"""CPU evidence audit for live Graph feedback, not a policy or simulator rerun."""

import argparse
import json
import traceback
from collections import Counter
from pathlib import Path

import audit_libero_observation_async as previous
import libero_graph_feedback as r
import numpy as np
import torch

check, load = r.check, previous.load


def exact(left, right, name):
    check(left.shape == right.shape and left.dtype == right.dtype
          and torch.equal(left, right), name)


def graph_output(out, req, record):
    rid = req['stamp']['request_id']
    cleanup = record['predictor_cleanup']
    check(out['sampler'] == {'sampler_mode': 'graph', 'capture_id': 1, 'replay_count': 1,
          'full_chunk_shape': [1, 50, 32], 'full_chunk_finite': True}, 'Graph replay metadata differs')
    check(out['projection_shapes'] == [] and out['vision_encodes'] == 1, 'Graph/eager evidence confused')
    check(out['capture_created'] == (rid == 1), 'Control-time capture')
    check(out['owner_thread'] == cleanup['owner'] == cleanup['close_thread'], 'Graph owner changed')
    check(out['input_observation_index'] == req['stamp']['observation_index'], 'Wrong live observation')
    check(len(out['inputs']) == 8 and [list(v.shape) for v in out['inputs']] ==
          cleanup['captures'][0]['input_shapes'], 'Graph input signature changed')
    check(all(torch.isfinite(v).all() for v in out['inputs']), 'Nonfinite Graph input')
    check(out['inputs'][2].dtype == out['inputs'][3].dtype == torch.bool, 'Camera mask type')
    exact(out['inputs'][7], out['noise'], 'Noise differs from dynamic input')
    exact(out['original'], out['full'][0, :, :7], 'Full output unpadding differs')


def inspect_episode(spec, record, arrays, checkpoint):
    cleanup, closed = record['predictor_cleanup'], record['owner_cleanup']
    requests, control = record['requests'], arrays['control']
    check(cleanup and cleanup['mode'] == 'graph' and cleanup['graph_released']
          and cleanup['sampler_restored'], 'Graph release/sampler restore missing')
    check(cleanup['requests'] == cleanup['rgb_encodings'] == cleanup['live_requests'] ==
          cleanup['explicit_noise_draws'] == len(requests) and cleanup['noise_draws'] == 0,
          'Live request/encoding/noise accounting differs')
    check(cleanup['owner'] != record['controller_thread'] and closed['owner_thread'] == cleanup['owner'],
          'Owner/controller identity not separated')
    check(closed['first_failure'] is None and control['ended_at'] <= closed['started_at'] <=
          cleanup['closed_at'] <= closed['completed_at'], 'Owner lifecycle timing differs')
    check(len(cleanup['captures']) == 1, 'Expected exactly one bootstrap capture')
    capture = cleanup['captures'][0]
    check(capture['status'] == 'captured' and capture['owner_thread'] == cleanup['owner']
          and capture['eager_setup_calls'] == 1 and capture['side_stream_warmup_calls'] == 3
          and capture['capture_calls'] == 1 and capture['projection_shapes'] == [[1, 50, 32]]*10,
          'Capture accounting differs')
    check(requests[0]['published_at'] <= control['t0'], 'Bootstrap not ready before control')
    row = previous.audit_episode(spec, record, arrays, checkpoint, output_checker=graph_output)
    native = [v for v in record['native_steps'] if v['segment'] == 'measurement']
    for req in requests:
        out = arrays['outputs'][req['stamp']['request_id']]
        obs = arrays['observations'][req['stamp']['observation_index']]
        check(out['input_fingerprint'] == r.observation_fingerprint(obs), 'Live input fingerprint differs')
        check(req['completed_at'] <= closed['started_at'], 'Graph released while request running')
        for when in (req['requested_at'], req['published_at']):
            check(not any(v['started_at'] < when < v['returned_at'] for v in native),
                  'Request publication/submission inside native step')
    jumps = []
    dispatches = control['dispatches']
    for prior, current in zip(dispatches, dispatches[1:], strict=False):
        if prior['request_id'] != current['request_id']:
            delta = np.asarray(current['command'], dtype=np.float64)-np.asarray(prior['command'], dtype=np.float64)
            jumps.append({'action_index': current['action_index'], 'source_row': current['source_row'],
                          'command_l2_7d': float(np.linalg.norm(delta)),
                          'last_coordinate_abs': float(abs(delta[-1]))})
    row.update(startup_s=record['startup_s'], capture_seconds=capture['preparation_seconds'],
               capture_count=1, transition_jumps=jumps, owner_cleanup=closed)
    return row


def aggregate(rows):
    arms = {}
    for arm in ('serialized', 'async'):
        selected = [v for v in rows if v['arm'] == arm]
        times = previous.quantiles([t for v in selected for t in v['request_seconds']])
        check(times['n'] > 0, 'No measured requests')
        times['required_delay_steps'] = int(np.ceil(20*times['p99']))+1
        arms[arm] = {'episodes': len(selected), 'successes': sum(v['success'] for v in selected),
            **{k: sum(v[k] for v in selected) for k in ('wall_s', 'actions', 'no_action_slots', 'takeovers',
                'underflows', 'request_count', 'expired', 'cancelled_at_stop')},
            'model_native_intersections': sum(len(v['overlaps']) for v in selected),
            'request_time_s': times,
            'queue_wait_s': previous.quantiles([t for v in selected for t in v['queue_wait_seconds']]),
            'bootstrap_s': previous.quantiles([v['bootstrap_s'] for v in selected]),
            'startup_s': previous.quantiles([v['startup_s'] for v in selected]),
            'max_actual_delay': max(d for v in selected for d in v['actual_delays']),
            'boundary_command_l2_7d': previous.quantiles([j['command_l2_7d'] for v in selected for j in v['transition_jumps']])}
    pairs, discordance = [], Counter()
    for pair, identity in enumerate(r.PAIRS):
        values = {v['arm']: v for v in rows if v['pair_index'] == pair}
        a, b = values['serialized'], values['async']
        label = ('both_success' if a['success'] and b['success'] else 'serialized_only' if a['success']
                 else 'async_only' if b['success'] else 'neither_success')
        discordance[label] += 1
        pairs.append({'pair': pair, 'task_state': list(identity), 'outcome': label,
            **{arm: {k: values[arm][k] for k in ('success', 'actions', 'wall_s', 'no_action_slots', 'takeovers',
                       'terminal_reason', 'startup_s')} for arm in ('serialized', 'async')}})
    return {'arms': arms, 'pairs': pairs, 'success_discordance': dict(discordance),
        'online_latency_budget_passed': all(v['request_time_s']['required_delay_steps'] <= 8 for v in arms.values()),
        'asynchronous_overlap_observed': arms['async']['model_native_intersections'] > 0,
        'waiting_reduction_observed': arms['async']['no_action_slots']/arms['async']['actions'] <
                                     arms['serialized']['no_action_slots']/arms['serialized']['actions'],
        'no_serialized_success_lost_in_pilot': discordance['serialized_only'] == 0}


def audit(output):
    check(not torch.cuda.is_initialized(), 'CPU audit required')
    result = json.loads((output / 'result.json').read_text())
    check(result['status'] == 'completed' and result['first_failure'] is None, 'Worker incomplete')
    execution = result['execution']
    check(execution['exit_confirmed'] and execution['exit_code'] == 0 and not execution['forced']
          and not execution['active'] and not execution['pending'] and execution['stop_reason'] is None,
          'Execution not closed')
    saved = r.validate(result['execution_head'], after=True)
    check(output == r.paths(result['execution_head'])[1] and json.loads((output / 'manifest.json').read_text())
          == saved['manifest'], 'Manifest/output differs')
    rows, initials, boots, total = [], {}, {}, Counter()
    for spec in saved['manifest']['rows']:
        folder = output / f"episode_{spec['ordinal']:03d}"
        check(json.loads((folder / 'started.json').read_text())['spec'] == spec, 'Started identity differs')
        record = json.loads((folder / 'result.json').read_text())
        arrays, checkpoint = load(folder / 'arrays.pt'), load(folder / 'initial_checkpoint.pt')
        rows.append(inspect_episode(spec, record, arrays, checkpoint))
        pair = spec['pair_index']
        boot = arrays['outputs'][1]
        if pair in initials:
            check(r.e.initial_difference(initials[pair], checkpoint) is None and record['initial_pair_exact'], 'Paired initial differs')
            exact(boots[pair]['full'], boot['full'], 'Bootstrap output differs')
            exact(boots[pair]['noise'], boot['noise'], 'Bootstrap noise differs')
        else:
            initials[pair], boots[pair] = checkpoint, {'full': boot['full'], 'noise': boot['noise']}
        total.update(record['budget'])
    check(len(rows) == 16 and len(initials) == 8 and dict(total) == result['native_budget'], 'Coverage/budget differs')
    check(all(total[k] <= v[1] for k, v in r.LIMITS.items()), 'Global budget exceeded')
    check(result['graph_captures'] == 16 and result['capture_internal'] == {'setup': 16, 'warmup': 48, 'capture': 16},
          'Capture count differs')
    intents, returned, close_events = {}, {}, {}
    for event in map(json.loads, (output / 'calls.jsonl').read_text().splitlines()):
        cid = event.get('call_id')
        if event['event'] == 'call_intent':
            check(cid not in intents, 'Duplicate intent')
            intents[cid] = event
        elif event['event'] == 'call_return':
            check(cid in intents and cid not in returned and 0 <= event['elapsed'] <= intents[cid]['limit'], 'Invalid return')
            returned[cid] = event
        elif event['event'] == 'owner_resources_closed':
            check(event['ordinal'] not in close_events and event['first_failure'] is None, 'Cleanup error')
            close_events[event['ordinal']] = event
        else:
            check(event['event'] != 'call_error', 'Call error')
    check(intents.keys() == returned.keys() and len(close_events) == 16, 'Unknown call or cleanup missing')
    check(sum(v['kind'] == 'graph_request' for v in intents.values()) == total['model'], 'Request count differs')
    check(sum(v['kind'] == 'native_step' for v in intents.values()) == total['measurement']+total['settling'], 'Native count differs')
    for v in intents.values():
        if v['kind'] == 'environment_close':
            check(close_events[v['ordinal']]['completed_at'] <= v['timestamp'], 'Env closed before owner resources')
    active, phases = {}, Counter()
    for v in map(json.loads, (output / 'events.jsonl').read_text().splitlines()):
        phases[v['event']] += 1
        if v['event'] == 'started':
            check(v['phase'] not in active, 'Duplicate phase')
            active[v['phase']] = v['limit']
        else:
            check(v['event'] == 'returned' and v['phase'] in active and
                  v['seconds'] <= active.pop(v['phase']), 'Phase error/deadline')
    check(not active and phases == {'started': 18, 'returned': 18}, 'Unclosed phases')
    check(all(result[k] == 0 for k in ('training_updates', 'qualification_reads', 'rtc_steps', 'predictor_forwards', 'real_robot'))
          and result['replay_commands'] is False, 'Forbidden scope change')
    check(result['attempts'] == 1 and result['retries'] == 0 and result['vla_frozen'], 'Model/scope changed')
    check(not torch.cuda.is_initialized(), 'Audit initialized CUDA')
    return {'independent_contract_accepted': True, 'graph_feedback_chain_completed': True,
        'episodes': 16, 'initial_pairs_exact': 8, 'native_actions_checked': total['measurement'],
        'request_outputs_checked': total['model'], 'calls_closed': len(returned), 'phases': dict(phases),
        'graph_captures': 16, 'runtime_closed_on_owner': 16, **aggregate(rows), 'per_episode': rows,
        'realtime_qualified': False, 'statistical_noninferiority_claimed': False,
        'audit_model_forwards': 0, 'cuda_initialized': False, 'first_failure': None,
        'limitation': 'Audit does not recompute RGB encoding, policy outputs, postprocessing or physics; input fingerprints bind recorded live input.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    output = args.output.resolve()
    check(not (output / 'independent_audit.json').exists(), 'Audit already exists')
    try:
        result = audit(output)
    except BaseException:
        result = {'independent_contract_accepted': False, 'first_failure': traceback.format_exc()}
    r.write(output / 'independent_audit.json', result)
    print(json.dumps({k: v for k, v in result.items() if k not in ('per_episode', 'pairs')}), flush=True)
    return 0 if result['independent_contract_accepted'] else 2


if __name__ == '__main__':
    raise SystemExit(main())
