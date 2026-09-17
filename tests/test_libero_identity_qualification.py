"""Synthetic CPU contract tests: never create a model, environment or CUDA context."""
import copy
import sys
from collections import Counter
from pathlib import Path

import pytest
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'examples/advanced/predictive_async'))
import audit_libero_identity_qualification as a  # noqa: E402
import libero_identity_qualification as r  # noqa: E402


def rows():
    return [{'key':[t,s,j],'delay':3,'donor':[t,s,(j+1)%4],
             'metrics':{name:{'row0':value,'chunk':value,'latent':2.0}
                        for name,value in [('identity',2.0),('iar_true',1.0),('iar_zero',2.0),('iar_mismatched',3.0)]}}
            for t,s in r.PAIRS for j in range(4)]


def test_positive_gates_and_budget_identity():
    data=rows()
    assert r.aggregate(data)['heldout_primary_gate_passed']
    assert r.aggregate(data)['heldout_robustness_gate_passed']
    assert a.decision(data)==(True,True)
    assert r.LIMITS=={'encoding':40,'decoder':32+4*32+32,'predictor':32+4*32+2*32}
    assert not set(r.PAIRS)&set(r.q.PAIRS)
    assert all(t in (6,7) and 4<=s<=7 for t,s in r.PAIRS)


@pytest.mark.parametrize('arm',['identity','iar_mismatched'])
def test_both_comparators_required(arm):
    data=rows()
    for x in data:
        x['metrics'][arm]['row0']=0.1
    assert not r.aggregate(data)['heldout_primary_gate_passed']
    assert a.decision(data)==(False,False)


def test_chunk_failure_kept():
    data=rows()
    for x in data:
        x['metrics']['iar_true']['chunk']=4.0
    assert a.decision(data)==(False,False)
    assert not r.aggregate(data)['heldout_primary_gate_passed']


def test_mean_improvement_is_not_majority():
    data=rows()
    for x in data:
        x['metrics']['iar_true']['row0']=0.0 if x['key'][2]==0 else 2.1
    assert r.aggregate(data)['heldout_primary_gate_passed']
    assert not r.aggregate(data)['heldout_robustness_gate_passed']
    assert a.decision(data)==(True,False)


def test_mismatch_pairing_uses_only_available_recipients():
    data=rows()
    for x in data[::4]:
        del x['metrics']['iar_mismatched']
        x['donor']=None
    result=r.aggregate(data)
    assert result['contrasts']['identity']['paired_samples']==32
    assert result['contrasts']['iar_mismatched']['paired_samples']==24
    assert a.decision(data)==(True,True)


@pytest.mark.parametrize('empty',[True,False])
def test_missing_episode_or_all_rows_fails(empty):
    data=[] if empty else rows()[4:]
    assert a.decision(data)==(False,False)
    assert not r.aggregate(data)['heldout_primary_gate_passed']


def test_missing_mismatch_episode_fails():
    data=rows()
    for x in data[:4]:
        del x['metrics']['iar_mismatched']
    assert not r.aggregate(data)['heldout_primary_gate_passed']
    assert a.decision(data)==(False,False)


def test_five_episode_wins_not_six():
    data=rows()
    for x in data[20:]:
        x['metrics']['iar_true']['row0']=2.01
    assert a.decision(data)==(False,False)
    assert not r.aggregate(data)['heldout_primary_gate_passed']


def test_large_adverse_example_cannot_be_discarded():
    data=rows()
    data[0]['metrics']['iar_true']['row0']=100.0
    assert r.aggregate(data)['contrasts']['identity']['sample_improved']==31
    assert not r.aggregate(data)['heldout_primary_gate_passed']
    assert a.decision(data)==(False,False)


def test_duplicate_history_conflict_and_no_replacement():
    r.check_unused([{'task':6,'state':0}])
    with pytest.raises(ValueError,match='no replacement'):
        r.check_unused([{'task':6,'state':4}])


def test_forward_caps_apply_before_cuda_or_dispatch():
    with pytest.raises(ValueError,match='Predictor budget'):
        r.predict(None,{},None,Counter(predictor=223))
    with pytest.raises(ValueError,match='Decoder budget'):
        r.decode(None,{},(),None,Counter(decoder=192),'synthetic')
    assert not torch.cuda.is_initialized()


def test_native_manifest_and_seeds():
    m=r.manifest()
    assert [(s['task_id'],s['initial_state_id']) for s in m['rows']]==list(r.PAIRS)
    for x in m['rows']:
        assert x['environment_seed']==1120000+100*x['task_id']+x['initial_state_id']
        assert x['policy_seed']==1130000+100*x['task_id']+x['initial_state_id']
        assert x['condition']=='graph_identity_async'
        assert x['split']=='qualification'
    assert m==copy.deepcopy(m)


def test_independent_numpy_metrics_match_masked_original():
    g=torch.Generator().manual_seed(22)
    v=tuple(torch.randn(1,4,7,generator=g) for _ in range(2))
    future=tuple(torch.randn(1,4,7,generator=g) for _ in range(2))
    masks=(torch.tensor([[True,False,True,True]]),torch.tensor([[False,True,True,False]]))
    sample={'inputs':(*v,*masks),'future':future}
    value=torch.randn(1,50,32,generator=g)
    oracle=torch.randn(1,50,32,generator=g)
    expected=r.q.metrics(v,value,sample,oracle)
    assert a.score(v,value,sample,oracle)==pytest.approx(expected,rel=1e-10,abs=1e-10)


def test_zero_algebra_and_actions_audit():
    identity=(torch.ones(1,4,7,dtype=torch.bfloat16),)*2
    actions=torch.zeros(1,8,7)
    mask=torch.tensor([[True,True,True,False,False,False,False,False]])
    delta=(torch.full((1,4,7),0.25),)*2
    sample={'inputs':identity,'mask':mask}
    ev={'actual_actions':actions,'actual_mask':mask,'action_delta':delta,'zero_delta':delta,
        'raw_visual':tuple(v.float() for v in identity),'visual':tuple(v.float() for v in identity)}
    a.check_residual(ev,sample,actions,zero=True)
    ev['actual_actions']=torch.ones_like(actions)
    with pytest.raises(ValueError):
        a.check_residual(ev,sample,actions,zero=True)


def test_cpu_only():
    assert not torch.cuda.is_initialized()
