"""Independent saved-evidence audit for E-SEQ1; no model or simulator rerun."""

import argparse
import json
import math
import traceback
from collections import Counter
from pathlib import Path

import audit_libero_graph_feedback as ga
import audit_libero_takeover_factorial as ta
import libero_sequence_async as r
import numpy as np
import torch

check, load, exact = r.check, r.load, ta.exact
quantiles = ga.previous.quantiles


def inspect_episode(spec, rec, arrays, checkpoint):
    check(rec['spec'] == spec and rec['status'] == 'completed' and rec['first_failure'] is None, 'Episode incomplete')
    check(rec['worker_joined'] and rec['environment_closed'], 'Resources not closed')
    observations, control, outputs = arrays['observations'], arrays['control'], arrays['outputs']
    dispatches, requests = control['dispatches'], rec['requests']
    sequence = spec['arm'] == 'sequence_async'
    check([v['index'] for v in observations] == list(range(len(dispatches)+1)), 'Observation coverage')
    check(r.e.initial_difference(checkpoint, observations[0]) is None, 'Initial differs')
    check([v['stamp']['request_id'] for v in requests] == list(range(1, len(requests)+1)) and
          set(outputs) == set(range(1, len(requests)+1)), 'Request coverage')
    cleanup, closed = rec['predictor_cleanup'], rec['owner_cleanup']
    check(cleanup['graph_released'] and cleanup['sampler_restored'] and closed['first_failure'] is None, 'Close failed')
    check(cleanup['owner'] == cleanup['close_thread'] == closed['owner_thread'] != rec['controller_thread'], 'Owner mismatch')
    check(cleanup['requests'] == cleanup['rgb_encodings'] == cleanup['live_requests'] ==
          cleanup['explicit_noise_draws'] == len(requests) and cleanup['noise_draws'] == 0, 'Model counts')
    check(control['ended_at'] <= closed['started_at'] <= cleanup['closed_at'] <= closed['completed_at'], 'Close chronology')
    check(len(cleanup['captures']) == 1, 'Capture count')
    cap = cleanup['captures'][0]
    check(cap['status'] == 'captured' and cap['owner_thread'] == cleanup['owner'] and
          cap['eager_setup_calls'] == 1 and cap['side_stream_warmup_calls'] == 3 and
          cap['capture_calls'] == 1 and cap['projection_shapes'] == [[1, 50, 32]]*10, 'Capture accounting')
    native = [v for v in rec['native_steps'] if v['segment'] == 'measurement']
    check(len(native) == len(dispatches) and sum(v['segment'] == 'settling' for v in rec['native_steps']) == 10,
          'Native coverage')
    by_id, installs = {v['stamp']['request_id']: v for v in requests}, []
    for pos, req in enumerate(requests):
        stamp, dec = req['stamp'], req['decision']
        rid, index = stamp['request_id'], stamp['observation_index']
        check(stamp['epoch'] == 0 and 0 <= stamp['expected_delay'] <= 8 and 0 <= index < len(observations), 'Stamp')
        check(observations[index]['returned_at'] == req['observation_returned_at'] <= req['requested_at'] <=
              req['started_at'] <= req['completed_at'] <= req['published_at'], 'Request chronology')
        check(req['completed_at'] <= closed['started_at'] and req['first_failure'] is None, 'Release before request end')
        if pos:
            check(requests[pos-1]['published_at'] <= req['requested_at'], 'Multiple inflight')
        check(math.isclose(req['complete_s'], req['completed_at']-req['started_at'], abs_tol=1e-10) and
              0 < req['complete_s'] < 15, 'Request duration')
        check(math.isclose(req['queue_wait_s'], req['started_at']-req['requested_at'], abs_tol=1e-10), 'Queue duration')
        out = outputs[rid]
        check(out['full'].shape == (1, 50, 32) and out['original'].shape == out['processed'].shape == (50, 7)
              and out['noise'].shape == (1, 50, 32), 'Output shape')
        check(all(torch.isfinite(out[k]).all() for k in ('full', 'original', 'processed', 'noise')), 'Nonfinite')
        ga.graph_output(out, req, rec)
        check(out['input_fingerprint'] == r.g.observation_fingerprint(observations[index]), 'Wrong live input')
        check(req['started_at'] <= out['model_started_at'] <= out['model_returned_at'] <= req['completed_at'], 'Model timeline')
        for when in (req['requested_at'], req['published_at']):
            check(not any(v['started_at'] < when < v['returned_at'] for v in native), 'Publication inside native')
        if req['prefix_source'] is None:
            check(rid == 1 and out['prefix'] is None and req['prefix_rows'] == 0, 'Missing prefix')
        else:
            parent = by_id[req['prefix_source']]
            pd = parent['decision']
            check(pd['accepted'] and parent['published_at'] <= req['requested_at'], 'Wrong prefix parent')
            offset = pd['source_row']+index-pd['takeover_index']
            exact(out['prefix'], outputs[req['prefix_source']]['original'][offset:], 'Prefix cursor differs')
            check(req['prefix_rows'] == len(out['prefix']), 'Prefix length')
        consumed = sum(d['dispatched_at'] < req['published_at'] for d in dispatches)
        delay = consumed-index
        check(delay >= 0, 'Negative consumed delay')
        if dec['accepted']:
            start = 0 if sequence else delay
            check(dec['reason'] == 'installed' and dec['actual_delay'] == delay <= 8 and
                  dec['source_row'] == start and dec['takeover_index'] == consumed, 'Wrong installation row')
            if sequence:
                check(dec['origin_semantics'] == 'plan_order_at_takeover' and dec['nominal_minus_actual'] == -delay,
                      'Wrong plan semantics')
                expected_tail = None
                if installs:
                    prev = installs[-1]
                    expected_tail = {'request_id': prev['stamp']['request_id'],
                        'start_row': consumed-prev['decision']['takeover_index'], 'end_row_exclusive': 50,
                        'reason': 'superseded_unexecuted_tail'}
                check(dec['superseded_tail'] == expected_tail, 'Superseded tail missing')
            installs.append(req)
        elif dec['reason'] == 'expired':
            check(dec['actual_delay'] == delay and delay > 8, 'Invalid expiry')
        else:
            check(dec == {'accepted': False, 'reason': 'stale_or_closed'} and req == requests[-1] and
                  req['published_at'] >= control['ended_at'], 'Invalid stop cancellation')
    check(installs and installs[0]['stamp']['request_id'] == 1 and requests[0]['published_at'] <= control['t0'], 'Bootstrap')
    seen, used, ages, offsets, jumps = set(), {}, [], [], []
    for index, (d, step) in enumerate(zip(dispatches, native, strict=True)):
        active = [req for req in installs if req['published_at'] <= d['dispatched_at']][-1]
        rid, dec = active['stamp']['request_id'], active['decision']
        source_row = dec['source_row']+index-dec['takeover_index']
        nominal = active['stamp']['observation_index']+source_row
        check(d['action_index'] == d['observation_index'] == step['action_index'] == step['observation_index'] == index,
              'Physical action identity')
        check(d['request_id'] == rid and d['source_row'] == source_row and 0 <= source_row < 50, 'Wrong source row')
        key = (d['epoch'], rid, source_row)
        check(key not in seen and d['epoch'] == 0, 'Repeated plan row')
        seen.add(key)
        used.setdefault(rid, []).append(source_row)
        if sequence:
            check(d['nominal_action_index'] == nominal and d['nominal_minus_actual'] == nominal-index and
                  d['plan_takeover_index'] == dec['takeover_index'], 'Hidden plan time offset')
        else:
            check(nominal == index, 'Aligned action offset')
        offsets.append(index-nominal)
        exact(d['original'], outputs[rid]['original'][source_row], 'Normalized source')
        exact(d['command'], outputs[rid]['processed'][source_row].numpy(), 'Processed source')
        exact(np.asarray(step['action'], dtype=np.float32), d['command'], 'Native command differs')
        check(observations[index]['returned_at'] <= d['dispatched_at'] <= step['started_at'] <= step['returned_at'] <=
              d['returned_at'] == observations[index+1]['returned_at'], 'Physical chronology')
        ages.append(d['dispatched_at']-active['observation_returned_at'])
        check(ages[-1] >= 0, 'Negative source age')
        if index and dispatches[index-1]['request_id'] != rid:
            prev = np.asarray(dispatches[index-1]['command'], dtype=np.float64)
            curr = np.asarray(d['command'], dtype=np.float64)
            jumps.append({'command_l2_7d': float(np.linalg.norm(curr-prev)),
                          'same_numeric_vector': bool(np.array_equal(prev, curr))})
    installed_ids = [v['stamp']['request_id'] for v in installs]
    plans, partition = [], Counter()
    for req in requests:
        rid, dec = req['stamp']['request_id'], req['decision']
        counts = {'executed': 0, 'skipped_head': 0, 'superseded_tail': 0, 'terminal_tail': 0, 'not_installed': 0}
        if dec['accepted']:
            start, emitted = dec['source_row'], used.get(rid, [])
            check(emitted == list(range(start, start+len(emitted))), 'Noncontiguous plan execution')
            if sequence:
                check(start == 0, 'New prefix omitted')
            counts['executed'], counts['skipped_head'] = len(emitted), start
            counts['terminal_tail' if rid == installed_ids[-1] else 'superseded_tail'] = 50-start-len(emitted)
        else:
            check(rid not in used, 'Uninstalled plan executed')
            counts['not_installed'] = 50
        check(min(counts.values()) >= 0 and sum(counts.values()) == 50, 'Plan row accounting')
        partition.update(counts)
        plans.append({'request_id': rid, **counts})
    check(partition['executed'] == len(dispatches), 'Plan execution count')
    slots = [d['slot'] for d in dispatches]
    check(slots == sorted(set(slots)) and all(0 <= s < spec['ready_wall_slots'] for s in slots), 'Repeated wall slot')
    wall = control['ended_at']-control['t0']
    count = min(spec['ready_wall_slots'], max(1, math.ceil(wall*20)))
    check(rec['wall_slots'] == count and rec['no_action_slots'] == count-len(dispatches) and
          math.isclose(rec['wall_s'], wall, abs_tol=1e-9), 'Wall accounting')
    check(rec['measured_actions'] == len(dispatches) and
          rec['success'] == control['success'] == dispatches[-1]['success'], 'Outcome differs')
    check(rec['terminal_reason'] == control['terminal_reason'], 'Terminal reason differs')
    check(rec['underflows'] == sum(v['outcome'] == 'underflow' for v in control['gets']), 'Underflow accounting')
    check(rec['budget'] == {'episodes': 1, 'settling': 10, 'model': len(requests), 'measurement': len(dispatches)}, 'Budget')
    check(all(rec['budget'][k] <= v for k, v in spec['limits'].items()), 'Episode budget exceeded')
    intersections = 0
    for req in requests[1:]:
        out = outputs[req['stamp']['request_id']]
        intersections += sum(min(out['model_returned_at'], step['returned_at']) >
                             max(out['model_started_at'], step['started_at']) for step in native)
    if spec['arm'] == 'serialized':
        check(intersections == 0 and all(v['decision'].get('actual_delay', 0) == 0 for v in requests), 'Serialized overlapped')
    else:
        check(not any(v['kind'] == 'serialized_wait' for v in control['blocks']), 'Async forced wait')
    return {'ordinal': spec['ordinal'], 'pair_index': spec['pair_index'], 'identity': [spec['task_id'], spec['initial_state_id']],
        'arm': spec['arm'], 'cohort': spec['cohort'], 'success': rec['success'], 'actions': len(dispatches),
        'wall_s': wall, 'terminal_reason': rec['terminal_reason'], 'no_action_slots': rec['no_action_slots'],
        'underflows': rec['underflows'], 'expired': sum(v['decision']['reason'] == 'expired' for v in requests),
        'cancelled_at_stop': sum(v['decision']['reason'] == 'stale_or_closed' for v in requests),
        'requests': len(requests), 'request_seconds': [v['complete_s'] for v in requests[1:]],
        'queue_wait_seconds': [v['queue_wait_s'] for v in requests[1:]], 'bootstrap_s': requests[0]['complete_s'],
        'startup_s': rec['startup_s'], 'capture_s': cap['preparation_seconds'],
        'max_actual_delay': max(v['decision'].get('actual_delay', 0) for v in requests),
        'model_native_intersections': intersections, 'observation_to_action_s': quantiles(ages),
        'max_nominal_lag_steps': max(offsets), 'plan_rows': dict(partition), 'plan_partitions': plans,
        'request_observation_indices': [v['stamp']['observation_index'] for v in requests],
        'boundary_l2': [v['command_l2_7d'] for v in jumps],
        'same_numeric_boundary_commands': sum(v['same_numeric_vector'] for v in jumps),
        'repeated_plan_rows': 0, 'internal_execution_holes': 0}


def aggregate(rows):
    arms = {}
    for arm in r.ARMS:
        chosen = [v for v in rows if v['arm'] == arm]
        times = quantiles([x for v in chosen for x in v['request_seconds']])
        check(times['n'] > 0, 'No nonbootstrap timing')
        times['required_delay_steps'] = math.ceil(times['p99']*20)+1
        partitions = Counter()
        for v in chosen:
            partitions.update(v['plan_rows'])
        arms[arm] = {'episodes': len(chosen), 'successes': sum(v['success'] for v in chosen),
            **{k: sum(v[k] for v in chosen) for k in ('actions', 'wall_s', 'no_action_slots', 'underflows',
                 'expired', 'cancelled_at_stop', 'requests', 'model_native_intersections', 'same_numeric_boundary_commands')},
            'request_time_s': times, 'queue_wait_s': quantiles([t for v in chosen for t in v['queue_wait_seconds']]),
            'bootstrap_s': quantiles([v['bootstrap_s'] for v in chosen]),
            'boundary_l2_7d': quantiles([t for v in chosen for t in v['boundary_l2']]),
            'max_observation_to_action_s': max(v['observation_to_action_s']['max'] for v in chosen),
            'max_nominal_lag_steps': max(v['max_nominal_lag_steps'] for v in chosen),
            'max_actual_delay': max(v['max_actual_delay'] for v in chosen),
            'requests_above_350ms': sum(t > .35 for v in chosen for t in v['request_seconds']),
            'plan_rows': dict(partitions)}
    triples, losses = [], {'vs_serialized': [], 'vs_aligned_async': []}
    cohorts = {}
    for pair, identity in enumerate(r.PAIRS):
        selected = {v['arm']: v for v in rows if v['pair_index'] == pair}
        check(set(selected) == set(r.ARMS), 'Incomplete triple')
        for ref in ('serialized', 'aligned_async'):
            if selected[ref]['success'] and not selected['sequence_async']['success']:
                losses['vs_'+ref].append(list(identity))
        cohort = selected['serialized']['cohort']
        c = cohorts.setdefault(cohort, {arm: {'n': 0, 'successes': 0} for arm in r.ARMS})
        for arm in r.ARMS:
            c[arm]['n'] += 1
            c[arm]['successes'] += int(selected[arm]['success'])
        triples.append({'identity': list(identity), 'cohort': cohort,
            'arms': {arm: {k: v[k] for k in ('success', 'actions', 'wall_s', 'no_action_slots', 'requests')}
                     for arm, v in selected.items()}})
    seq, ser = arms['sequence_async'], arms['serialized']
    gates = {'no_lost_success_vs_serialized': not losses['vs_serialized'],
        'no_lost_success_vs_aligned': not losses['vs_aligned_async'],
        'all_arm_p99_budget': all(v['request_time_s']['required_delay_steps'] <= 8 for v in arms.values()),
        'all_nonbootstrap_budget': all(v['requests_above_350ms'] == 0 for v in arms.values()),
        'sequence_overlap': seq['model_native_intersections'] > 0,
        'sequence_wait_ratio_improved': seq['no_action_slots']/seq['actions'] < ser['no_action_slots']/ser['actions'],
        'sequence_no_queue_fault': seq['underflows'] == seq['expired'] == 0,
        'sequence_contiguous_prefix': seq['plan_rows'].get('skipped_head', 0) == 0}
    return {'arms': arms, 'triples': triples, 'cohorts': cohorts, 'lost_success_identities': losses,
            'development_gates': gates, 'development_followup_supported': all(gates.values())}


def audit(output):
    check(not torch.cuda.is_initialized(), 'CPU only')
    result = json.loads((output/'result.json').read_text())
    check(result['status'] == 'completed' and result['first_failure'] is None, 'Formal run incomplete')
    ex = result['execution']
    check(ex['exit_confirmed'] and ex['exit_code'] == 0 and not ex['forced'] and not ex['active'] and
          not ex['pending'] and ex['stop_reason'] is None, 'Execution unclosed')
    saved = r.validate(result['execution_head'], after=True)
    check(output == r.paths(result['execution_head'])[1] and
          json.loads((output/'manifest.json').read_text()) == saved['manifest'], 'Output/manifest')
    rows, total, initials, boots = [], Counter(), {}, {}
    for spec in saved['manifest']['rows']:
        folder = output/f"episode_{spec['ordinal']:03d}"
        rec = json.loads((folder/'result.json').read_text())
        check(json.loads((folder/'started.json').read_text())['spec'] == spec, 'Started specification')
        arrays, initial = load(folder/'arrays.pt'), load(folder/'initial_checkpoint.pt')
        rows.append(inspect_episode(spec, rec, arrays, initial))
        pair, boot = spec['pair_index'], arrays['outputs'][1]
        if pair in boots:
            check(r.e.initial_difference(initials[pair], initial) is None and rec['initial_pair_exact'], 'Triple initial mismatch')
            ta.compare_request(boot, boots[pair], 'Triple bootstrap')
        else:
            initials[pair], boots[pair] = initial, boot
        total.update(rec['budget'])
        del arrays
    check(len(rows) == 30 and len(boots) == 10 and result['episodes_completed'] == 30, 'Coverage differs')
    check(dict(total) == result['native_budget'] and all(total[k] <= v[1] for k, v in r.LIMITS.items()), 'Global budget')
    check(result['graph_captures'] == 30 and result['capture_internal'] == {'setup': 30, 'warmup': 90, 'capture': 30}, 'Capture count')
    intents, returns, closes = {}, {}, {}
    for ev in map(json.loads, (output/'calls.jsonl').read_text().splitlines()):
        cid = ev.get('call_id')
        if ev['event'] == 'call_intent':
            check(cid not in intents, 'Repeated intent')
            intents[cid] = ev
        elif ev['event'] == 'call_return':
            check(cid in intents and cid not in returns and math.isfinite(ev['elapsed']) and
                  0 <= ev['elapsed'] <= intents[cid]['limit'], 'Call return/deadline')
            returns[cid] = ev
        elif ev['event'] == 'owner_resources_closed':
            check(ev['ordinal'] not in closes and ev['first_failure'] is None, 'Close error')
            closes[ev['ordinal']] = ev
        else:
            check(ev['event'] != 'call_error', 'Call error')
    check(intents.keys() == returns.keys() and len(closes) == 30, 'Unclosed resources or call')
    check(sum(v['kind'] == 'graph_request' for v in intents.values()) == total['model'], 'Model ledger')
    check(sum(v['kind'] == 'native_step' for v in intents.values()) == total['measurement']+300, 'Native ledger')
    for ev in intents.values():
        if ev['kind'] == 'environment_close':
            check(closes[ev['ordinal']]['completed_at'] <= ev['timestamp'], 'Env closed before owner')
    phases, active = Counter(), {}
    for ev in map(json.loads, (output/'events.jsonl').read_text().splitlines()):
        phases[ev['event']] += 1
        if ev['event'] == 'started':
            check(ev['phase'] not in active, 'Repeated phase')
            active[ev['phase']] = ev['limit']
        else:
            check(ev['event'] == 'returned' and ev['phase'] in active and
                  ev['seconds'] <= active.pop(ev['phase']), 'Phase failed')
    check(not active and phases == {'started': 32, 'returned': 32}, 'Phase count')
    check(result['attempts'] == 1 and result['retries'] == 0 and result['vla_frozen'] and
          all(result[k] == 0 for k in ('training_updates', 'rtc_steps', 'real_robot')), 'Scope changed')
    check(result['frozen_baseline_unchanged'] and not result['new_qualification_claimed'] and
          result['confirmation_result_unchanged'], 'Qualification scope changed')
    check(not torch.cuda.is_initialized(), 'Audit initialized CUDA')
    return {'independent_contract_accepted': True, 'frozen_baseline_unchanged': True,
        'episodes': 30, 'initial_triples_exact': 10, 'native_actions_checked': total['measurement'],
        'request_outputs_checked': total['model'], 'calls_closed': len(returns), 'phases': dict(phases),
        'runtime_closed': 30, 'graph_captures': 30, **aggregate(rows), 'per_episode': rows,
        'new_qualification_claimed': False, 'realtime_qualified': False, 'first_failure': None,
        'audit_model_forwards': 0, 'cuda_initialized': False,
        'limitations': ['Known development identities, not statistical confirmation.',
            'Same qsize trigger with different queue lengths can change request cadence.',
            'No repeated plan identity does not imply no semantic command repetition.',
            'CPU audit does not rerun vision, policy, postprocessing, physics or GPU timing.']}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    torch.set_num_threads(1)
    out = args.output.resolve()
    check(not (out/'independent_audit.json').exists(), 'Audit already exists')
    try:
        result = audit(out)
    except BaseException:
        result = {'independent_contract_accepted': False, 'first_failure': traceback.format_exc()}
    r.write(out/'independent_audit.json', result)
    print(json.dumps({k: v for k, v in result.items() if k not in ('per_episode', 'triples')}), flush=True)
    return 0 if result['independent_contract_accepted'] else 2


if __name__ == '__main__':
    raise SystemExit(main())
