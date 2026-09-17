"""Independent CPU evidence reduction for E-GCR1; never runs a model or Env."""

import argparse
import json
import math
import traceback
from collections import Counter
from pathlib import Path

import libero_graph_replay as r
import numpy as np
import torch

check = r.check


def exact(a, b, message):
    check(isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor), message+' type')
    check(a.shape == b.shape and a.dtype == b.dtype and torch.equal(a, b), message)


def times(rows):
    x = np.sort(np.asarray([v['complete_s'] for v in rows], dtype=np.float64))
    check(len(x) > 0 and np.isfinite(x).all() and np.min(x) > 0, 'Invalid times')
    result = {'n': len(x), 'mean_s': float(np.mean(x)), 'max_s': float(x[-1]),
              **{f'p{int(p*100)}_s': float(x[int(np.ceil(len(x)*p))-1]) for p in (.5, .95, .99)}}
    result['required_delay_steps'] = int(np.ceil(result['p99_s']*20))+1
    return result


def compare_times(a, b):
    check(a.keys() == b.keys(), 'Timing fields differ')
    for key in a:
        check(math.isclose(a[key], b[key], rel_tol=1e-9, abs_tol=1e-10), 'Timing reduction differs: '+key)


def check_row(row, item, mode):
    check(row['key'] == item['key'] and row['mode'] == mode, 'Wrong input or arm')
    exact(row['full'], item['expected_full'], 'Full output differs')
    exact(row['original'], row['full'][0, :, :7], 'Unpadding differs')
    if 'expected_processed' in item:
        exact(row['processed'], item['expected_processed'], 'Postprocessed output differs')
    if 'inputs' in item:
        check(len(row['inputs']) == len(item['inputs']) == 8, 'Input count')
        for a, b in zip(row['inputs'], item['inputs'], strict=True):
            exact(a, b, 'Input differs')
    exact(row['inputs'][7], item['noise'], 'Noise differs')
    check(row['full'].shape == (1, 50, 32) and row['original'].shape == row['processed'].shape == (50, 7),
          'Invalid output shape')
    check(all(torch.isfinite(row[k]).all() for k in ('full', 'original', 'processed')), 'Nonfinite output')
    check(row['started_at'] <= row['model_started_at'] <= row['model_completed_at'] <= row['completed_at'],
          'Non-nested timing')
    check(math.isclose(row['completed_at']-row['started_at'], row['complete_s'], abs_tol=1e-10), 'Bad full timing')
    check(math.isclose(sum(row['parts'].values()), row['complete_s'], abs_tol=1e-9) and
          all(v >= 0 for v in row['parts'].values()), 'Component accounting')
    metadata = row['sampler']
    check(metadata['sampler_mode'] == mode and metadata['replay_count'] == int(mode == 'graph')
          and metadata['full_chunk_shape'] == [1, 50, 32] and metadata['full_chunk_finite'], 'Sampler evidence')


def audit(output):
    check(not torch.cuda.is_initialized(), 'Audit must stay CPU-only')
    result = json.loads((output / 'result.json').read_text())
    check(result['status'] == 'completed' and result['first_failure'] is None, 'Worker incomplete')
    execution = result['execution']
    check(execution['exit_confirmed'] and execution['exit_code'] == 0 and not execution['forced'] and
          execution['stop_reason'] is None and not execution['pending'] and not execution['active'], 'Worker not closed')
    prep = r.validate(result['head'])
    check(output == r.paths(result['head'])[1], 'Output identity')
    source = r.load(r.paths(result['head'])[0] / 'data.pt')
    rows, index = [], 0
    for i, item in enumerate(source['fixed']):
        for repeat in range(2):
            for mode in (('eager', 'graph') if (i+repeat) % 2 == 0 else ('graph', 'eager')):
                row = r.load(output / f'request_{index:03d}.pt')
                check_row(row, item, mode)
                check(row['scope'] == 'fixed' and row['repeat'] == repeat and row['sample_index'] == i, 'Fixed order')
                rows.append(row)
                index += 1
    fixed_times = {m: times([v for v in rows if v['mode'] == m and v['repeat'] == 1]) for m in ('eager', 'graph')}
    for mode in fixed_times:
        compare_times(fixed_times[mode], result['fixed_timing'][mode])
    ready = fixed_times['graph']['required_delay_steps'] <= 8
    check(ready == result['fixed_gate_passed'] == result['replay_started'], 'Prerequisite gate changed')
    totals, episodes, overlaps = Counter(), [], Counter()
    native_intervals, request_timelines = [], {}
    ordinal = 0
    if ready:
        for trace in source['traces']:
            for mode in (('eager', 'graph') if ordinal//2 % 2 == 0 else ('graph', 'eager')):
                folder = output / f'episode_{ordinal:03d}'
                record = json.loads((folder / 'result.json').read_text())
                expected_spec = {**trace['spec'], 'ordinal': ordinal, 'condition': 'replay_'+mode}
                check(record['spec'] == expected_spec and record['status'] == 'completed' and
                      record['first_failure'] is None and record['environment_closed'] and record['owner_joined'],
                      'Replay incomplete')
                check(json.loads((folder / 'started.json').read_text())['spec'] == expected_spec, 'Started identity')
                check(r.e.initial_difference(r.load(folder / 'initial_checkpoint.pt'), trace['initial']) is None,
                      'Initial replay differs')
                native = [s for s in record['native_steps'] if s['segment'] == 'measurement']
                check(len(native) == len(trace['commands']), 'Replay command coverage differs')
                check(record['budget'] == {'episodes': 1, 'settling': 10, 'measurement': len(native)}, 'Native budget')
                slots = []
                previous_end = -float('inf')
                for n, (step, command) in enumerate(zip(native, trace['commands'], strict=True)):
                    check(step['action_index'] == step['observation_index'] == n, 'Wrong native action index')
                    check(np.array_equal(np.asarray(step['action'], dtype=command.dtype), command), 'Replay command differs')
                    check(previous_end <= step['started_at'] <= step['returned_at'], 'Native overlap or order')
                    previous_end = step['returned_at']
                    slots.append(step['slot'])
                check(slots == sorted(set(slots)), 'Repeated or reordered wall slot')
                current, previous_end = [], -float('inf')
                for item in trace['requests']:
                    row = r.load(output / f'request_{index:03d}.pt')
                    check_row(row, item, mode)
                    check(row['scope'] == 'replay' and row['ordinal'] == ordinal and
                          row['rid'] == item['key'][2] and row['origin'] == item['origin'], 'Replay request order')
                    check(previous_end <= row['started_at'], 'More than one request in flight')
                    previous_end = row['completed_at']
                    if row['rid'] == 1:
                        check(row['completed_at'] <= native[0]['started_at'], 'Capture/bootstrap overlapped native stepping')
                    else:
                        origin = item['origin']
                        check(origin > 0 and native[origin-1]['returned_at'] <= row['started_at'], 'Request before fixed boundary')
                        for step in native:
                            if min(row['model_completed_at'], step['returned_at']) > max(row['model_started_at'], step['started_at']):
                                overlaps[mode] += 1
                    request_timelines[row['label']] = row
                    rows.append(row)
                    current.append(row)
                    index += 1
                check(record['request_labels'] == [v['label'] for v in current], 'Missing request labels')
                totals.update(record['budget'])
                episodes.append({'ordinal': ordinal, 'mode': mode, 'source_task_state':
                    [trace['spec']['task_id'], trace['spec']['initial_state_id']],
                    'commands': len(native), 'requests': len(current), 'control_wall_s': record['control_wall_s']})
                native_intervals.append(native)
                ordinal += 1
        check(index == 206 and ordinal == 16 and totals['measurement'] == 2820, 'Fixed replay denominator')
    check({p.name for p in output.glob('request_*.pt')} == {f'request_{i:03d}.pt' for i in range(index)}, 'Output coverage')
    check(result['completed_requests'] == index and result['native_budget'] == dict(totals), 'Global count')
    runtimes = json.loads((output / 'runtimes.json').read_text())
    check(len(runtimes) == (20 if ready else 4), 'Runtime lifecycle coverage')
    captures = []
    for runtime in runtimes:
        check(runtime['graph_released'] and runtime['sampler_restored'] and runtime['noise_draws'] == 0 and
              runtime['requests'] == runtime['rgb_encodings'], 'Lifecycle or encoding count')
        for capture in runtime['captures']:
            check(capture['status'] == 'captured' and capture['eager_setup_calls'] == 1 and
                  capture['side_stream_warmup_calls'] == 3 and capture['capture_calls'] == 1 and
                  capture['projection_shapes'] == [[1, 50, 32]]*10 and capture['owner_thread'] == runtime['owner'],
                  'Hidden capture cost or owner differs')
            captures.append(capture)
    check(len(captures) == (12 if ready else 4) and sum(v['requests'] for v in runtimes) == index, 'Runtime totals')
    events = list(map(json.loads, (output / 'calls.jsonl').read_text().splitlines()))
    intents, returns = {}, set()
    for event in events:
        cid = event.get('call_id')
        if event['event'] == 'call_intent':
            check(cid not in intents, 'Duplicate intent')
            intents[cid] = event
        elif event['event'] == 'call_return':
            check(cid in intents and cid not in returns and 0 <= event['elapsed'] <= intents[cid]['limit'], 'Invalid return')
            returns.add(cid)
        else:
            check(event['event'] != 'call_error', 'Call failed')
    check(set(intents) == returns, 'Unclosed call')
    check(sum(v['kind'] == 'model_request' for v in intents.values()) == index, 'Model ledger')
    check(sum(v['kind'] == 'native_step' for v in intents.values()) == totals['measurement']+totals['settling'], 'Native ledger')
    active, phases = {}, Counter()
    for event in map(json.loads, (output / 'events.jsonl').read_text().splitlines()):
        phases[event['event']] += 1
        if event['event'] == 'started':
            check(event['phase'] not in active, 'Duplicate phase')
            active[event['phase']] = event['limit']
        else:
            check(event['event'] == 'returned' and event['phase'] in active and
                  event['seconds'] <= active.pop(event['phase']), 'Phase failed or exceeded')
    expected_phases = 1+index+ordinal
    check(not active and phases == {'started': expected_phases, 'returned': expected_phases}, 'Unclosed phase')
    concurrent = {m: times([v for v in rows if v['scope'] == 'replay' and v['mode'] == m and v['rid'] > 1])
                  for m in ('eager', 'graph')} if ready else {}
    if ready:
        for mode in concurrent:
            compare_times(concurrent[mode], result['replay_timing'][mode])
        check(result['concurrent_graph_latency_passed'] == (concurrent['graph']['required_delay_steps'] <= 8), 'Concurrent gate')
    check(all(result[k] == 0 for k in ('training_updates', 'rtc_steps', 'qualification_reads', 'real_robot')) and
          result['attempts'] == 1 and result['retries'] == 0 and result['vla_frozen'], 'Scope changed')
    check(prep['sources'] == r.source_hashes() and not torch.cuda.is_initialized(), 'Source or audit device changed')
    return {'independent_contract_accepted': True, 'fixed_output_exact': True, 'request_outputs_checked': index,
        'fixed_timing': fixed_times, 'concurrent_timing': concurrent, 'replay_episodes': ordinal,
        'native_commands_checked': totals['measurement'], 'initials_exact': ordinal,
        'model_native_intersections': dict(overlaps), 'captures': len(captures), 'phases': dict(phases),
        'calls_closed': len(returns), 'runtime_restored_count': len(runtimes),
        'graph_concurrent_budget_passed': bool(ready and concurrent['graph']['required_delay_steps'] <= 8),
        'new_policy_rollout': False, 'task_success_gain_claimed': False, 'audit_model_forwards': 0,
        'cuda_initialized': False, 'first_failure': None, 'per_episode': episodes}


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
    print(json.dumps({k: v for k, v in result.items() if k != 'per_episode'}), flush=True)
    return 0 if result['independent_contract_accepted'] else 2


if __name__ == '__main__':
    raise SystemExit(main())
