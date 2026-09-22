"""Independent CPU reduction and evidence audit for the fixed RTC Graph benchmark."""

import argparse
import json
import math
import traceback
from collections import Counter
from pathlib import Path

import libero_rtc_graph as r
import numpy as np
import torch

from lerobot.configs import RTCAttentionSchedule
from lerobot.policies.rtc.configuration_rtc import RTCConfig
from lerobot.policies.rtc.modeling_rtc import RTCProcessor

check = r.check


def exact(a, b, name):
    check(isinstance(a,torch.Tensor) and isinstance(b,torch.Tensor) and
          a.dtype==b.dtype and a.shape==b.shape and torch.equal(a,b), name)


def reduce_timing(rows):
    arms={}
    for arm in ('native_eager','cached_eager','graph'):
        x=np.array([v['complete_s'] for v in rows if v['arm']==arm and v['repeat']>0],dtype=np.float64)
        check(len(x)==80 and np.isfinite(x).all() and (x>0).all(),'Timing population differs')
        ordered=np.sort(x)
        def p(value, ordered_values=ordered):
            return float(ordered_values[int(np.ceil(value*len(ordered_values)))-1])
        arms[arm]={'n':len(x),'mean_s':float(x.mean()),'p50_s':p(.5),'p95_s':p(.95),
                   'p99_s':p(.99),'max_s':float(x.max()),'above350ms':int((x>.35).sum()),
                   'required_delay_steps':int(np.ceil(p(.99)*20))+1}
    graph=arms['graph']
    return {'timing':arms,'graph_budget_passed':graph['above350ms']==0 and graph['required_delay_steps']<=8,
            'graph_faster_mean':graph['mean_s']<arms['native_eager']['mean_s']}


def audit(output):
    check(not torch.cuda.is_initialized(),'CPU audit required')
    result=json.loads((output/'result.json').read_text())
    check(result['status']=='completed' and result['first_failure'] is None,'Worker incomplete')
    ex=result['execution']
    check(ex['exit_confirmed'] and ex['exit_code']==0 and not ex['forced'] and
          ex['stop_reason'] is None and not ex['pending'] and not ex['active'],'Execution not closed')
    saved=r.validate(result['execution_head'],after=True)
    check(output==r.paths(result['execution_head'])[1] and
          json.loads((output/'manifest.json').read_text())==saved['specification'],'Output/manifest differs')
    data=r.prior.load(r.DATA)
    timeline=[json.loads(line) for line in (output/'timings.jsonl').read_text().splitlines()]
    rows, by_group, first_guided={}, {}, {}
    cold, input_exact, cross_fields, repeated=0,0,0,0
    previous_end=-math.inf
    number=0
    for sample_index in range(16):
        for repeat in range(6):
            modes=['native_eager','cached_eager','graph']
            shift=(sample_index+repeat)%3
            for arm in modes[shift:]+modes[:shift]:
                row=r.prior.load(output/f'request_{number:03d}.pt')
                sample=data[sample_index]
                check((row['ordinal'],row['sample'],row['repeat'],row['arm'],row['key'],row['guided'])==
                      (number,sample_index,repeat,arm,sample['key'],repeat>0),'Order or identity differs')
                pure={k:v for k,v in row.items() if k not in ('normalized','processed','full','inputs')}
                check(pure==timeline[number],'Timing log and saved arrays disagree')
                check(previous_end<=row['started_at']<row['completed_at'] and
                      math.isclose(row['complete_s'],row['completed_at']-row['started_at'],abs_tol=1e-10),
                      'Invalid request chronology')
                previous_end=row['completed_at']
                check(row['full'].shape==(1,50,32) and row['normalized'].shape==row['processed'].shape==(1,50,7),
                      'Output shape differs')
                for field in ('full','normalized','processed'):
                    check(row[field].device.type=='cpu' and torch.isfinite(row[field]).all(),'Invalid saved output')
                exact(row['normalized'],row['full'][...,:7],'Unpadding differs')
                check(len(row['inputs'])==(9 if repeat else 8),'Input count differs')
                for i in range(8):
                    exact(row['inputs'][i],sample['inputs'][i],f'Dynamic input {i} differs')
                if repeat:
                    exact(row['inputs'][8],sample['prefix'],'Wrong actual prefix')
                else:
                    exact(row['full'],sample['archived_full_chunk'],'No-prefix complete anchor differs')
                    cold+=1
                input_exact+=1
                check(row['metadata']=={'mode':arm,'guided':repeat>0,
                    'capture_created':number==2,'replays':int(arm=='graph'),
                    'captured_vjps_per_replay':10 if arm=='graph' and repeat>0 else 0},'Sampler metadata differs')
                group=(sample_index,repeat)
                if group in by_group:
                    for field in ('full','normalized','processed'):
                        exact(row[field],by_group[group][field],f'Cross-arm {field} differs')
                        cross_fields+=1
                else:
                    by_group[group]=row
                if repeat:
                    key=(sample_index,arm)
                    if key in first_guided:
                        exact(row['full'],first_guided[key],'Repeated prefix output instability')
                        repeated+=1
                    else:
                        first_guided[key]=row['full']
                rows[number]=row
                number+=1
    check(number==len(timeline)==result['completed_requests']==288,'Coverage differs')
    check(cold==48 and input_exact==288 and cross_fields==576 and repeated==192,'Comparison accounting differs')
    reduced=reduce_timing(list(rows.values()))
    for arm,summary in reduced['timing'].items():
        for k,value in summary.items():
            check(math.isclose(value,result['timing'][arm][k],rel_tol=1e-12,abs_tol=1e-12),'Timing reduction differs')
    check(reduced['graph_budget_passed']==result['graph_budget_passed'] and
          reduced['graph_faster_mean']==result['graph_faster_mean'],'Efficiency gate differs')
    rr=result['runtime_receipt']
    check(rr and rr['requests']==rr['rgb_encodings']==288 and rr['owner']==rr['close_thread'] and
          all(rr[k] for k in ('sampler_restored','processor_restored','graphs_released')),'Runtime closure missing')
    check(rr['replays']=={'no_prefix':16,'guided':80} and len(rr['captures'])==2,'Capture/replay counts differ')
    for cap,guided in zip(rr['captures'],(False,True),strict=True):
        check(cap['status']=='captured' and cap['guided']==guided and cap['owner']==rr['owner'] and
              (cap['setup'],cap['warmup'],cap['capture'])==(1,3,1) and cap['captured_steps']==10 and
              cap['captured_vjps']==(10 if guided else 0) and cap['projection_shapes']==[[1,50,32]]*10 and
              math.isfinite(cap['seconds']) and cap['seconds']>0,'Incomplete captured computation')
    check(rr['native_steps']=={'public':{'steps':960,'guided_vjps_returned':800}},'Native RTC accounting')
    check(rr['cached_steps']=={'public':{'steps':960,'guided_vjps_returned':800},
        'setup':{'steps':20,'guided_vjps_returned':10},'warmup':{'steps':60,'guided_vjps_returned':30},
        'capture':{'steps':20,'guided_vjps_returned':10}},'Cached/Graph RTC accounting')
    processor=RTCProcessor(RTCConfig(prefix_attention_schedule=RTCAttentionSchedule.EXP))
    expected=processor.get_prefix_weights(3,10,50).numpy()
    actual=np.asarray(json.loads((output/'prefix_weights.json').read_text()),dtype=np.float32)
    check(np.array_equal(actual,expected),'Cached EXP weights differ from original CPU formula')
    intents,returns={},{}
    for event in map(json.loads,(output/'calls.jsonl').read_text().splitlines()):
        cid=event.get('call_id')
        if event['event']=='call_intent':
            check(cid not in intents and event['kind']=='rtc_full_request','Unexpected call')
            intents[cid]=event
        elif event['event']=='call_return':
            check(cid in intents and cid not in returns and 0<=event['elapsed']<=intents[cid]['limit'],'Invalid return')
            returns[cid]=event
        else:
            check(False,'Unexpected call event')
    check(len(intents)==len(returns)==288 and intents.keys()==returns.keys(),'Unclosed requests')
    phases,active=Counter(),{}
    for event in map(json.loads,(output/'events.jsonl').read_text().splitlines()):
        phases[event['event']]+=1
        if event['event']=='started':
            check(event['phase'] not in active,'Duplicate phase')
            active[event['phase']]=event['limit']
        else:
            check(event['event']=='returned' and event['phase'] in active and
                  event['seconds']<=active.pop(event['phase']),'Phase error')
    check(not active and phases=={'started':17,'returned':17},'Unclosed phases')
    check(result['attempts']==1 and result['retries']==0 and result['vla_frozen'] and
          all(result[k]==0 for k in ('new_env','training_updates','qualification_reads')),'Scope changed')
    check(not torch.cuda.is_initialized(),'CPU audit initialized CUDA')
    return {'independent_contract_accepted':True,'full_rtc_equivalence_passed':True,
        'requests':288,'no_prefix_anchors_exact':cold,'dynamic_inputs_exact':input_exact,
        'cross_arm_output_arrays_exact':cross_fields,'repeated_outputs_exact':repeated,
        'calls_closed':288,'phases':dict(phases),'captures':2,'replayed_guided_vjps':800,
        'capture_extra_sampler_calls':10,**reduced,
        'rtc_graph_cost_candidate_supported':reduced['graph_budget_passed'] and reduced['graph_faster_mean'],
        'first_graph_request_s':rows[2]['complete_s'],'runtime_init_s':result['runtime_init_s'],
        'capture_seconds':[v['seconds'] for v in rr['captures']],
        'live_feedback_tested':False,'task_retention_tested':False,'deployment_qualified':False,
        'audit_model_forwards':0,'cuda_initialized':False,'first_failure':None,
        'limitations':['Fixed delay3, 30x7 prefix, 16 old development inputs; no concurrent native load.',
                       'CPU audit does not rerun model, VJP, visual encoding or GPU timing.']}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    torch.set_num_threads(1)
    output=args.output.resolve()
    check(not (output/'independent_audit.json').exists(),'Audit already exists')
    try:
        result=audit(output)
    except BaseException:
        result={'independent_contract_accepted':False,'first_failure':traceback.format_exc()}
    r.write(output/'independent_audit.json',result)
    print(json.dumps(result,ensure_ascii=False),flush=True)
    return 0 if result['independent_contract_accepted'] else 2


if __name__=='__main__':
    raise SystemExit(main())
