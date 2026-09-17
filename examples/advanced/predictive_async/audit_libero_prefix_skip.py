"""CPU evidence audit for E-PSI1. No model/physics rerun or deployment claim."""

import argparse
import json
import math
import traceback
from collections import Counter
from pathlib import Path

import audit_libero_graph_feedback as ga
import audit_libero_takeover_factorial as ta
import libero_prefix_skip as r
import numpy as np
import torch

check, load, exact = r.check, r.load, ta.exact


def inspect_episode(spec, record, arrays, checkpoint):
    check(record['spec'] == spec and record['status'] == 'completed' and record['first_failure'] is None,
          'Incomplete episode or wrong specification')
    check(record['worker_joined'] and record['environment_closed'], 'Resources not closed')
    observations, control, outputs = arrays['observations'], arrays['control'], arrays['outputs']
    dispatches, requests = control['dispatches'], record['requests']
    check(len(observations) == len(dispatches)+1 and
          [v['index'] for v in observations] == list(range(len(observations))), 'Observation coverage')
    check(r.e.initial_difference(checkpoint, observations[0]) is None, 'Initial checkpoint differs')
    check([v['stamp']['request_id'] for v in requests] == list(range(1, len(requests)+1)) and
          set(outputs) == set(range(1, len(requests)+1)), 'Request coverage')
    cleanup, closed = record['predictor_cleanup'], record['owner_cleanup']
    check(cleanup['graph_released'] and cleanup['sampler_restored'] and closed['first_failure'] is None,
          'Predictor cleanup failed')
    check(cleanup['owner'] == cleanup['close_thread'] == closed['owner_thread'] != record['controller_thread'],
          'Owner identity mismatch')
    check(cleanup['requests'] == cleanup['rgb_encodings'] == cleanup['live_requests'] ==
          cleanup['explicit_noise_draws'] == len(requests) and cleanup['noise_draws'] == 0, 'Live model counts')
    check(control['ended_at'] <= closed['started_at'] <= cleanup['closed_at'] <= closed['completed_at'], 'Close chronology')
    check(len(cleanup['captures']) == 1, 'Capture count')
    cap = cleanup['captures'][0]
    check(cap['status'] == 'captured' and cap['owner_thread'] == cleanup['owner'] and
          cap['eager_setup_calls'] == 1 and cap['side_stream_warmup_calls'] == 3 and
          cap['capture_calls'] == 1 and cap['projection_shapes'] == [[1, 50, 32]]*10, 'Capture accounting')
    native = [v for v in record['native_steps'] if v['segment'] == 'measurement']
    settling = [v for v in record['native_steps'] if v['segment'] == 'settling']
    check(len(native) == len(dispatches) and len(settling) == 10, 'Native coverage')
    by_id, installs = {v['stamp']['request_id']: v for v in requests}, []
    for ordinal, req in enumerate(requests):
        stamp, dec = req['stamp'], req['decision']
        rid, obs_index = stamp['request_id'], stamp['observation_index']
        dose, skip = (spec['consume'], spec['skip']) if rid == 2 else (0, 0)
        check(stamp['epoch'] == 0 and obs_index == ordinal*20 and stamp['expected_delay'] == dose,
              'Fixed request grid/dose changed')
        check(observations[obs_index]['returned_at'] == req['observation_returned_at'] <= req['requested_at']
              <= req['started_at'] <= req['completed_at'] <= req['published_at'], 'Request chronology')
        check(req['completed_at'] <= closed['started_at'] and req['first_failure'] is None, 'Request incomplete at release')
        if ordinal:
            check(requests[ordinal-1]['published_at'] <= req['requested_at'], 'Concurrent requests')
            check(req['intervention_dose'] == dose and req['intervention_skip'] == skip, 'Intervention intent')
        check(math.isclose(req['complete_s'], req['completed_at']-req['started_at'], abs_tol=1e-10) and
              0 < req['complete_s'] < 15, 'Request duration')
        check(math.isclose(req['queue_wait_s'], req['started_at']-req['requested_at'], abs_tol=1e-10), 'Queue duration')
        out = outputs[rid]
        check(out['full'].shape == (1, 50, 32) and out['original'].shape == out['processed'].shape == (50, 7)
              and out['noise'].shape == (1, 50, 32), 'Output shape')
        check(all(torch.isfinite(out[k]).all() for k in ('full', 'original', 'processed', 'noise')), 'Nonfinite output')
        ga.graph_output(out, req, record)
        check(out['input_fingerprint'] == r.g.observation_fingerprint(observations[obs_index]), 'Live input mismatch')
        check(req['started_at'] <= out['model_started_at'] <= out['model_returned_at'] <= req['completed_at'], 'Model chronology')
        for when in (req['requested_at'], req['published_at']):
            check(not any(v['started_at'] < when < v['returned_at'] for v in native), 'Publication during native step')
        if req['prefix_source'] is None:
            check(rid == 1 and out['prefix'] is None and req['prefix_rows'] == 0, 'Missing prefix')
        else:
            parent = by_id[req['prefix_source']]
            pd = parent['decision']
            check(pd['accepted'] and parent['published_at'] <= req['requested_at'], 'Prefix source not installed')
            offset = pd['source_row']+obs_index-pd['takeover_index']
            exact(out['prefix'], outputs[req['prefix_source']]['original'][offset:], 'Prefix cursor differs')
            check(req['prefix_rows'] == len(out['prefix']), 'Prefix length')
        consumed = sum(d['dispatched_at'] < req['published_at'] for d in dispatches)
        if dec['accepted']:
            check(dec['reason'] == 'installed' and dec['actual_delay'] == consumed-obs_index == dose and
                  dec['source_row'] == skip and dec['takeover_index'] == consumed and
                  dec['nominal_minus_actual'] == skip-dose and dec['diagnostic_only'], 'Installed intervention differs')
            installs.append(req)
        else:
            check(dec['reason'] == 'stale_or_closed' and req == requests[-1] and
                  req['published_at'] >= control['ended_at'], 'Invalid cancellation')
    check(installs and installs[0]['stamp']['request_id'] == 1 and
          requests[0]['published_at'] <= control['t0'], 'Bootstrap absent')
    mismatches = 0
    for index, (d, step) in enumerate(zip(dispatches, native, strict=True)):
        active = [req for req in installs if req['published_at'] <= d['dispatched_at']][-1]
        rid, dec = active['stamp']['request_id'], active['decision']
        expected_row = dec['source_row']+index-dec['takeover_index']
        nominal = active['stamp']['observation_index']+expected_row
        check(d['action_index'] == d['observation_index'] == step['action_index'] == step['observation_index'] == index,
              'Physical action identity differs')
        check(d['request_id'] == rid and d['source_row'] == expected_row and 0 <= expected_row < 50,
              'Diagnostic source row differs')
        check(d['nominal_action_index'] == nominal and d['nominal_minus_actual'] == nominal-index,
              'Hidden or incorrect time offset')
        check(nominal-index == (spec['skip']-spec['consume'] if rid == 2 else 0), 'Unexpected nominal offset')
        mismatches += int(nominal != index)
        exact(d['original'], outputs[rid]['original'][expected_row], 'Normalized command source differs')
        exact(d['command'], outputs[rid]['processed'][expected_row].numpy(), 'Processed command source differs')
        exact(np.asarray(step['action'], dtype=np.float32), d['command'], 'Native command differs')
        check(observations[index]['returned_at'] <= d['dispatched_at'] <= step['started_at'] <= step['returned_at']
              <= d['returned_at'] == observations[index+1]['returned_at'], 'Physical chronology')
    slots = [d['slot'] for d in dispatches]
    check(slots == sorted(set(slots)) and all(0 <= s < spec['ready_wall_slots'] for s in slots), 'Repeated wall slot')
    wall = control['ended_at']-control['t0']
    count = min(spec['ready_wall_slots'], max(1, math.ceil(wall*20)))
    check(record['wall_slots'] == count and record['no_action_slots'] == count-len(dispatches), 'Wall accounting')
    check(math.isclose(record['wall_s'], wall, abs_tol=1e-9), 'Wall time differs')
    check(record['measured_actions'] == len(dispatches) and record['success'] == control['success'] == dispatches[-1]['success'],
          'Outcome differs')
    check(record['terminal_reason'] == control['terminal_reason'], 'Terminal reason differs')
    check(record['underflows'] == 0 and all(v['outcome'] == 'action' for v in control['gets']), 'Queue underflow')
    check(record['budget'] == {'episodes': 1, 'settling': 10, 'model': len(requests), 'measurement': len(dispatches)}, 'Budget accounting')
    check(all(record['budget'][k] <= v for k, v in spec['limits'].items()), 'Local budget exceeded')
    return {'ordinal': spec['ordinal'], 'identity': [7, spec['initial_state_id']], 'arm': spec['arm'],
        'success': record['success'], 'actions': len(dispatches), 'requests': len(requests),
        'terminal_reason': record['terminal_reason'], 'nonaligned_actions_explicit': mismatches,
        'startup_s': record['startup_s'], 'capture_seconds': cap['preparation_seconds'],
        'cancelled_at_stop': sum(not v['decision']['accepted'] for v in requests),
        'max_nonbootstrap_s': max(v['complete_s'] for v in requests[1:]),
        'control_wall_s_descriptive': wall}


def contrasts(table):
    check(set(table) == {'consume0_skip0', 'consume2_skip2', 'consume0_skip2', 'consume2_skip0'}, 'Missing cell')
    a, b, c, d = (int(table[k]['success']) for k in ('consume0_skip0', 'consume2_skip2', 'consume0_skip2', 'consume2_skip0'))
    return {'old_execution_effect_with_skip0': d-a, 'old_execution_effect_with_skip2': b-c,
        'skip_effect_with_old0': c-a, 'skip_effect_with_old2': b-d,
        'interaction_success': b-c-d+a, 'population_effect_claimed': False}


def audit(output):
    check(not torch.cuda.is_initialized(), 'CPU audit only')
    result = json.loads((output/'result.json').read_text())
    check(result['status'] == 'completed' and result['first_failure'] is None, 'Formal run incomplete')
    ex = result['execution']
    check(ex['exit_confirmed'] and ex['exit_code'] == 0 and not ex['forced'] and not ex['active'] and
          not ex['pending'] and ex['stop_reason'] is None, 'Execution not closed')
    prepared = r.validate(result['execution_head'], after=True)
    check(output == r.paths(result['execution_head'])[1] and
          json.loads((output/'manifest.json').read_text()) == prepared['manifest'], 'Output identity')
    rows, anchors, total, table = [], [], Counter(), {}
    for spec in prepared['manifest']['rows']:
        folder = output/f"episode_{spec['ordinal']:03d}"
        record = json.loads((folder/'result.json').read_text())
        check(json.loads((folder/'started.json').read_text())['spec'] == spec, 'Started identity')
        arrays, initial = load(folder/'arrays.pt'), load(folder/'initial_checkpoint.pt')
        rows.append(inspect_episode(spec, record, arrays, initial))
        zero = r.REFERENCES[spec['initial_state_id']][0]
        original = load(r.SOURCE/f'episode_{zero:03d}/arrays.pt')
        for i in range(21):
            check(r.e.initial_difference(arrays['observations'][i], original['observations'][i]) is None,
                  'Pre-intervention physical state differs')
        for rid in (1, 2):
            ta.compare_request(arrays['outputs'][rid], original['outputs'][rid], f'common_request{rid}')
        del original
        if spec['reference_ordinal'] is not None:
            ref = r.SOURCE/f"episode_{spec['reference_ordinal']:03d}"
            anchors.append({'ordinal': spec['ordinal'], 'reference': spec['reference_ordinal'],
                **ta.check_anchor(record, arrays, json.loads((ref/'result.json').read_text()), load(ref/'arrays.pt'))})
        table.setdefault(str(spec['initial_state_id']), {})[spec['arm']] = rows[-1]
        total.update(record['budget'])
        del arrays
    check(len(rows) == 8 and len(anchors) == 4 and result['episodes_completed'] == 8, 'Coverage differs')
    check(dict(total) == result['native_budget'] and all(total[k] <= v[1] for k, v in r.LIMITS.items()), 'Global budget')
    check(result['graph_captures'] == 8 and result['capture_internal'] == {'setup': 8, 'warmup': 24, 'capture': 8}, 'Capture budget')
    intents, returns, closes = {}, {}, {}
    for ev in map(json.loads, (output/'calls.jsonl').read_text().splitlines()):
        cid = ev.get('call_id')
        if ev['event'] == 'call_intent':
            check(cid not in intents, 'Repeated intent')
            intents[cid] = ev
        elif ev['event'] == 'call_return':
            check(cid in intents and cid not in returns and math.isfinite(ev['elapsed']) and
                  0 <= ev['elapsed'] <= intents[cid]['limit'], 'Call identity or deadline')
            returns[cid] = ev
        elif ev['event'] == 'owner_resources_closed':
            check(ev['ordinal'] not in closes and ev['first_failure'] is None, 'Close failure')
            closes[ev['ordinal']] = ev
        else:
            check(ev['event'] != 'call_error', 'Call failure')
    check(intents.keys() == returns.keys() and len(closes) == 8, 'Unclosed resources or call')
    check(sum(v['kind'] == 'graph_request' for v in intents.values()) == total['model'], 'Model ledger')
    check(sum(v['kind'] == 'native_step' for v in intents.values()) == total['measurement']+80, 'Native ledger')
    for ev in intents.values():
        if ev['kind'] == 'environment_close':
            check(closes[ev['ordinal']]['completed_at'] <= ev['timestamp'], 'Env closed before model owner')
    phases, active = Counter(), {}
    for ev in map(json.loads, (output/'events.jsonl').read_text().splitlines()):
        phases[ev['event']] += 1
        if ev['event'] == 'started':
            check(ev['phase'] not in active, 'Repeated phase')
            active[ev['phase']] = ev['limit']
        else:
            check(ev['event'] == 'returned' and ev['phase'] in active and
                  ev['seconds'] <= active.pop(ev['phase']), 'Phase failure')
    check(not active and phases == {'started': 10, 'returned': 10}, 'Unclosed phase')
    check(result['attempts'] == 1 and result['retries'] == 0 and result['vla_frozen'] and
          all(result[k] == 0 for k in ('training_updates', 'rtc_steps', 'real_robot')), 'Activity scope')
    check(not result['new_qualification_claimed'] and result['confirmation_result_unchanged'], 'Qualification scope')
    check(not torch.cuda.is_initialized(), 'Audit initialized CUDA')
    return {'independent_contract_accepted': True, 'candidate_unchanged': True,
        'complete_anchor_trajectories_exact': 4, 'anchors': anchors, 'common_first_two_requests_exact': 16,
        'episodes': 8, 'native_actions_checked': total['measurement'], 'request_outputs_checked': total['model'],
        'calls_closed': len(returns), 'phases': dict(phases), 'graph_captures': 8, 'runtime_closed': 8,
        'rows': rows, 'factorial_contrasts': {k: contrasts(v) for k, v in table.items()},
        'qualification_claimed': False, 'deployment_claimed': False, 'confirmation_result_unchanged': True,
        'audit_model_forwards': 0, 'cuda_initialized': False, 'first_failure': None,
        'limitations': ['Two selected known states; single realization per cell.',
            'Off-diagonal rows intentionally violate the original timing association; not deployment candidates.',
            'Physical feedback and contact mechanisms are not individually isolated.',
            'CPU audit does not rerun RGB encoding, model, postprocessing or physics.']}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    torch.set_num_threads(1)
    output = args.output.resolve()
    check(not (output/'independent_audit.json').exists(), 'Audit already exists')
    try:
        result = audit(output)
    except BaseException:
        result = {'independent_contract_accepted': False, 'first_failure': traceback.format_exc()}
    r.write(output/'independent_audit.json', result)
    print(json.dumps(result), flush=True)
    return 0 if result['independent_contract_accepted'] else 2


if __name__ == '__main__':
    raise SystemExit(main())
