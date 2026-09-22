"""E-RGC1: original RTC, cached RTC and fully captured RTC on 16 fixed raw inputs."""

import argparse
import faulthandler
import json
import math
import signal
import subprocess
import sys
import time
import traceback
from collections import Counter
from pathlib import Path

import libero_observation_async as old
import libero_rtc_fullpath as prior
import torch
from rtc_graph_runtime import RTCGraphRuntime

from lerobot.configs import RTCAttentionSchedule
from lerobot.policies.rtc.configuration_rtc import RTCConfig

q, e, pilot, REPO = old.q, old.e, old.pilot, old.REPO
check, write, digest = old.check, old.write, old.digest
BASE_HEAD = 'e8206030d07ce889d63ce1e282559ecaa9d5aa34'
BASELINE = REPO/'outputs/smolvla_takeover_factorial_development_20260917/baseline.json'
DATA = REPO/'outputs/smolvla_rtc_fullpath_preparation_92c39812/data.pt'
LOCK = REPO/'docs/experiments/SMOLVLA_GRAPH_CANDIDATE_LOCK.json'
ADDITIONS = ('examples/advanced/predictive_async/rtc_graph_runtime.py',
             'examples/advanced/predictive_async/libero_rtc_graph.py',
             'examples/advanced/predictive_async/audit_libero_rtc_graph.py')
MODES = RTCGraphRuntime.MODES


def paths(head):
    return (REPO/f'outputs/smolvla_rtc_graph_preparation_{head[:8]}',
            REPO/f'outputs/smolvla_rtc_graph_{head[:8]}')


def specification():
    return {'experiment': 'E-RGC1', 'keys': [list(k) for k in prior.EXPECTED_KEYS],
            'arms': list(MODES), 'repeats': 6, 'requests': 288, 'calibration_requests': 48,
            'timed_per_arm': 80, 'captures': 2, 'extra_sampler_calls': 10,
            'delay': 3, 'prefix_shape': [30,7], 'horizon': 10, 'guidance': 10.0,
            'schedule': 'EXP', 'chunk': 50, 'steps': 10, 'fps': 20, 'cap': 8, 'margin': 1,
            'all_graph_requests_limit_s': .35, 'native_guidance_formula_unchanged': True,
            'new_env': 0, 'training': 0, 'qualification_reads': 0, 'live_qualified': False}


def source_hashes():
    values = old.sources()
    files = [BASELINE, DATA, LOCK, REPO/'docs/experiments/SMOLVLA_RTC_GRAPH_PLAN.md',
             REPO/'tests/test_rtc_graph_runtime.py', *(REPO/p for p in ADDITIONS)]
    values.update({str(p): digest(p) for p in files})
    return values


def tree_gate(head):
    check(subprocess.check_output(['git','rev-parse','HEAD'], cwd=REPO, text=True).strip() == head, 'HEAD changed')
    check(old.pending_state() == json.loads(BASELINE.read_text())['pending'], 'Original pending changed')
    changed = subprocess.check_output(['git','diff','--name-only',BASE_HEAD,'--',
        'src/lerobot','examples/advanced/predictive_async'], cwd=REPO, text=True).splitlines()
    check(set(changed) <= set(ADDITIONS), 'Frozen source changed')
    lock = json.loads(LOCK.read_text())
    check(all(digest(REPO/p) == h for p,h in lock['core_sha256'].items()), 'Candidate core changed')


def prepare(head):
    tree_gate(head)
    environment = q.runtime_environment()
    data = prior.load(DATA)
    check([s['key'] for s in data] == specification()['keys'], 'Fixed old development identities differ')
    check(all(s['prefix'].shape == (30,7) and s['delay'] == 3 for s in data), 'Prefix contract differs')
    prep, out = paths(head)
    check(not prep.exists() and not out.exists(), 'Unique preparation/output required')
    check(not torch.cuda.is_initialized(), 'CPU prepare initialized CUDA')
    prep.mkdir()
    write(prep/'preparation.json', {'head': head, 'specification': specification(),
        'sources': source_hashes(), 'environment': environment, 'history': q.history_inventory(REPO/'outputs')})
    print(json.dumps({'prepared': True,'prep':str(prep),'out':str(out),'sha256':digest(prep/'preparation.json')}),flush=True)


def validate(head, after=False):
    tree_gate(head)
    prep, out = paths(head)
    saved = json.loads((prep/'preparation.json').read_text())
    check(saved['head'] == head and saved['specification'] == specification(), 'Specification changed')
    check(saved['sources'] == source_hashes() and saved['environment'] == q.runtime_environment(), 'Source/environment changed')
    check(saved['history'] == q.history_inventory(REPO/'outputs'), 'Other experiment changed history')
    body = (prep/'registration.md').read_text()
    got = json.loads((prep/'registration_readback.json').read_text())
    check(type(got.get('id')) is int and got.get('body') == body and got.get('issue_url') ==
          'https://api.github.com/repos/Lebron-233/lerobot/issues/1' and all(v in body for v in
          (f'E-RGC1-REGISTER:{head}',str(out),digest(prep/'preparation.json'))), 'Actual registration differs')
    return saved


def schedule():
    rows = []
    for index in range(16):
        for repeat in range(6):
            shift = (index+repeat)%3
            for arm in MODES[shift:]+MODES[:shift]:
                rows.append({'ordinal':len(rows),'sample':index,'repeat':repeat,'arm':arm,
                             'key':list(prior.EXPECTED_KEYS[index]),'guided':repeat>0})
    return rows


def timings(rows):
    values = {}
    for arm in MODES:
        measured = [r['complete_s'] for r in rows if r['arm'] == arm and r['repeat'] > 0]
        check(len(measured)==80 and all(math.isfinite(v) and v>0 for v in measured), 'Timing coverage')
        ordered = sorted(measured)
        p99 = ordered[math.ceil(.99*len(ordered))-1]
        values[arm] = {'n':len(measured),'mean_s':math.fsum(measured)/len(measured),
            'p50_s':ordered[math.ceil(.5*len(ordered))-1], 'p95_s':ordered[math.ceil(.95*len(ordered))-1],
            'p99_s':p99,'max_s':max(measured),'required_delay_steps':math.ceil(p99*20)+1,
            'above350ms':sum(v>.35 for v in measured)}
    graph = values['graph']
    return {'timing':values,'graph_budget_passed':graph['required_delay_steps']<=8 and graph['above350ms']==0,
            'graph_faster_mean':graph['mean_s']<values['native_eager']['mean_s']}


def request(spec, sample, runtime, policy, pre, post, calls):
    call_id = calls.start('rtc_full_request', spec['sample'], limit=30, request_id=spec['ordinal'])
    entered = time.perf_counter()
    try:
        runtime.mode = spec['arm']
        torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.no_grad():
            policy.reset()
            pre.reset()
            post.reset()
            batch = pre(pilot.worker_batch(sample['observation'],sample['language'],'cuda'))
            noise = sample['inputs'][7].cuda().clone()
            prefix = sample['prefix'].cuda().clone() if spec['guided'] else None
            normalized = policy.predict_action_chunk(batch,noise=noise,prev_chunk_left_over=prefix,
                                                      inference_delay=3,execution_horizon=10)
            processed = post(normalized).detach().cpu().clone()
            normalized = normalized.detach().cpu().clone()
            full = runtime.latest.detach().cpu().clone()
            inputs = q.cpu(runtime.latest_inputs)
            torch.cuda.synchronize()
            end = time.perf_counter()
        row = {**spec,'complete_s':end-start,'started_at':start,'completed_at':end,
               'normalized':normalized,'processed':processed,'full':full,'inputs':inputs,
               'metadata':dict(runtime.metadata)}
        calls.emit('call_return',call_id=call_id,elapsed=time.perf_counter()-entered)
        return row
    except BaseException:
        calls.emit('call_error',call_id=call_id,exception=traceback.format_exc())
        raise


def worker(args):
    torch.set_num_threads(1)
    faulthandler.register(signal.SIGUSR1,file=sys.stderr,all_threads=True)
    calls, rows, counts = e.Calls(args.output/'calls.jsonl'), [], Counter()
    result = {'experiment':'E-RGC1','execution_head':args.execution_head,'status':'technical_failure','first_failure':None}
    policy, runtime = None, None
    try:
        saved = validate(args.execution_head)
        data = prior.load(DATA)
        with pilot.phase(args.output,'load_model',90):
            check(torch.cuda.get_device_name()=='NVIDIA GeForce RTX 4070 Ti SUPER','GPU changed')
            policy,pre,post,report = e.load_runtime(e.POLICY,e.VLM)
            check(not policy.config.compile_model and policy.config.rtc_config is None,'Unexpected model mode')
            write(args.output/'policy_load.json',report)
            counts['vla_loads']+=1
            policy.config.rtc_config=RTCConfig(enabled=True,mode='guided',execution_horizon=10,
                max_guidance_weight=10.0,prefix_attention_schedule=RTCAttentionSchedule.EXP)
            policy.init_rtc_processor()
        start_runtime = time.perf_counter()
        runtime = RTCGraphRuntime(policy.model)
        torch.cuda.synchronize()
        result['runtime_init_s']=time.perf_counter()-start_runtime
        check(torch.equal(runtime.cached.weights.cpu(),runtime.native.get_prefix_weights(3,10,50)),'Cached weights differ')
        write(args.output/'prefix_weights.json',runtime.cached.cpu_weights.tolist())
        with runtime:
            for index,sample in enumerate(data):
                with pilot.phase(args.output,f'sample_{index}',90):
                    references={}
                    for spec in (s for s in schedule() if s['sample']==index):
                        check(len(rows)==spec['ordinal'] and len(rows)<288,'Request budget/order')
                        row=request(spec,sample,runtime,policy,pre,post,calls)
                        rows.append(row)
                        torch.save(row,args.output/f"request_{spec['ordinal']:03d}.pt")
                        with (args.output/'timings.jsonl').open('a') as stream:
                            stream.write(json.dumps({k:v for k,v in row.items() if k not in
                                ('normalized','processed','full','inputs')},allow_nan=False)+'\n')
                        check(row['full'].shape==(1,50,32) and row['normalized'].shape==row['processed'].shape==(1,50,7),'Output shape')
                        for j in range(8):
                            pilot.require_equal(row['inputs'][j],sample['inputs'][j],f'Input{j} differs')
                        if spec['guided']:
                            pilot.require_equal(row['inputs'][8],sample['prefix'],'Prefix input differs')
                        else:
                            pilot.require_equal(row['full'],sample['archived_full_chunk'],'No-prefix anchor differs')
                            counts['no_prefix_exact']+=1
                        pilot.require_equal(row['normalized'],row['full'][...,:7],'Unpadding differs')
                        counts['input_exact']+=1
                        group=spec['repeat']
                        if group not in references:
                            references[group]=row
                        else:
                            for field in ('full','normalized','processed'):
                                pilot.require_equal(row[field],references[group][field],f'Cross-arm {field} differs')
                            counts['cross_arm_exact']+=1
                        if spec['guided'] and spec['repeat']>1:
                            pilot.require_equal(row['full'],references[1]['full'],'Repeat instability')
                    print(f'E-RGC1 completed {index+1}/16 inputs; {len(rows)}/288 requests',flush=True)
        counts.update(requests=len(rows),rgb_encodings=runtime.rgb_encodings)
        check(counts=={'vla_loads':1,'no_prefix_exact':48,'input_exact':288,'cross_arm_exact':192,
                      'requests':288,'rgb_encodings':288},'Formal accounting differs')
        check(all(not p.requires_grad and p.grad is None for p in policy.parameters()),'Policy is not frozen')
        tree_gate(args.execution_head)
        check(saved['sources']==source_hashes(),'Sources changed')
        result.update(status='completed',vla_frozen=True,**timings(rows))
    except BaseException:
        result['first_failure']=traceback.format_exc()
    finally:
        if runtime is not None:
            result['runtime_receipt']=runtime.receipt
        if policy is not None:
            policy.config.rtc_config=None
            policy.init_rtc_processor()
        calls.close()
        result.update(counts=dict(counts),completed_requests=len(rows),attempts=1,retries=0,
                      new_env=0,training_updates=0,qualification_reads=0,realtime_qualified=False)
        write(args.output/'worker_result.json',result)
    return 0 if result['status']=='completed' else 2


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execution-head',required=True)
    parser.add_argument('--prepare',action='store_true')
    parser.add_argument('--worker',action='store_true')
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    signal.signal(signal.SIGTERM,e.request_shutdown)
    if args.prepare:
        check(not args.worker and args.output is None,'Preparation cannot launch model')
        prepare(args.execution_head)
        return 0
    args.output=(args.output or paths(args.execution_head)[1]).resolve()
    return worker(args) if args.worker else old.supervise(args,validate_run=validate,paths_for=paths,
        manifest_for=specification,worker_script=str(Path(__file__).resolve()),experiment='E-RGC1')


if __name__=='__main__':
    raise SystemExit(main())
