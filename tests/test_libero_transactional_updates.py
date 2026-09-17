"""Synthetic CPU tests; no scientific files, VLA, CUDA, Env or network."""
import copy
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'examples/advanced/predictive_async'))
import audit_libero_transactional_updates as a  # noqa: E402
import libero_transactional_updates as t  # noqa: E402


def metrics():
    return {'objective': 3.0, 'row0': 2.0, 'chunk': 1.0, 'max_excess': 0.5}


def test_strict_objective_and_original_anchor():
    initial = metrics()
    good = {**initial, 'objective': 2.9, 'row0': 1.9}
    assert t.gate(initial, good, initial) == a.gate(initial, good, initial)
    assert t.gate(initial, good, initial)[0]
    assert not t.gate(initial, initial, initial)[0]
    drift = {**good, 'row0': 2.00001}
    before = {**initial, 'row0': 2.00001}
    assert not t.gate(before, drift, initial)[0]


@pytest.mark.parametrize('name', ['row0', 'chunk', 'max_excess'])
def test_each_constraint_blocks_descent_in_total_objective(name):
    before = metrics()
    candidate = {**before, 'objective': 2.0, name: before[name]+0.01}
    assert not t.gate(before, candidate, before)[0]
    assert t.gate(before, candidate, before) == a.gate(before, candidate, before)


@pytest.mark.parametrize('value', [float('nan'), float('inf'), -1.0])
def test_bad_gate_numbers_fail_closed(value):
    with pytest.raises(ValueError, match='Nonfinite'):
        t.gate(metrics(), {**metrics(), 'row0': value}, metrics())


def fixture():
    records, controls, bounds, weights = [], {}, {}, {}
    for task in range(6):
        for state in (46, 48, 49):
            for request in range(4):
                key = (task, state, request)
                records.append({'key': list(key), 'split': 'train',
                                'metrics': {'latent': 2., 'row0': 3., 'chunk': 4.}})
                controls[key] = {'metrics': {'identity': {'row0': 2., 'chunk': 5.}}}
                bounds[key] = {'row0': 1., 'chunk': 2.}
                weights[key] = .5
    return records, controls, bounds, {'latent': 2., 'row0': 2., 'chunk': 2.}, weights


def test_full_training_gate_independent_reduction():
    records, controls, targets, scales, weights = fixture()
    x = t.aggregate_training(records, controls, scales, targets, weights)
    assert x == a.aggregate(records, controls, targets, scales, weights)
    assert x == {'objective': 5.75, 'row0': 3., 'chunk': 4., 'max_excess': 1.}


@pytest.mark.parametrize('kind', ['missing', 'duplicate', 'validation'])
def test_gate_rejects_partial_duplicate_or_validation_rows(kind):
    records, controls, targets, scales, weights = fixture()
    if kind == 'missing':
        records.pop()
    elif kind == 'duplicate':
        records[-1] = records[0]
    else:
        records[-1]['split'] = 'validation'
    with pytest.raises(ValueError):
        t.aggregate_training(records, controls, scales, targets, weights)


def model_opt():
    model = torch.nn.Linear(2, 1).double()
    optimizer = torch.optim.AdamW(model.parameters(), lr=t.r.LR, weight_decay=t.r.WD,
                                 betas=(.9, .999), eps=1e-8, foreach=False)
    return model, optimizer


def step(model, optimizer, scale):
    optimizer.zero_grad(set_to_none=True)
    (model(torch.tensor([[1., 2.]], dtype=torch.float64)).square().sum()*scale).backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
    optimizer.step()


@pytest.mark.parametrize('previous_steps', [0, 1, 3])
def test_complete_rollback_including_empty_state_and_next_update(previous_steps):
    torch.manual_seed(19)
    model, optimizer = model_opt()
    for _ in range(previous_steps):
        step(model, optimizer, 1.)
    before, state = t.cpu_tree(model.state_dict()), t.cpu_tree(optimizer.state_dict())
    baseline = copy.deepcopy(model)
    baseline_optimizer = torch.optim.AdamW(baseline.parameters(), lr=t.r.LR, weight_decay=t.r.WD,
                                          betas=(.9, .999), eps=1e-8, foreach=False)
    baseline_optimizer.load_state_dict(copy.deepcopy(state))
    step(model, optimizer, 7.)
    assert any(not torch.equal(v, model.state_dict()[n]) for n,v in before.items())
    t.restore_transaction(model, optimizer, before, state)
    t.exact_tree(before, model.state_dict())
    t.exact_tree(state, optimizer.state_dict())
    step(model, optimizer, .3)
    step(baseline, baseline_optimizer, .3)
    t.exact_tree(model.state_dict(), baseline.state_dict())
    t.exact_tree(optimizer.state_dict(), baseline_optimizer.state_dict())


def test_snapshot_no_aliasing():
    old = {'x': [torch.tensor([1.])], 'group': {'n': 3}}
    saved = t.cpu_tree(old)
    old['x'][0].add_(7.)
    old['group']['n'] = 9
    assert saved['x'][0].item() == 1. and saved['group']['n'] == 3


def test_adamw_audit_after_rejection_preserves_history():
    torch.manual_seed(33)
    model, opt = model_opt()
    state = {}
    for accept in (True, False, True):
        old, os = t.cpu_tree(model.state_dict()), t.cpu_tree(opt.state_dict())
        opt.zero_grad(set_to_none=True)
        model(torch.tensor([[.5, 1.]], dtype=torch.float64)).square().sum().backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        grads = {n: v.grad.clone() for n,v in model.named_parameters()}
        next_state = copy.deepcopy(state)
        expected = a.optaudit.adamw_step(old, grads, next_state)
        opt.step()
        for name in old:
            assert np.allclose(expected[name], model.state_dict()[name].detach().numpy(), rtol=1e-5, atol=1e-7)
        a.verify_optimizer(next_state, t.cpu_tree(opt.state_dict()), list(old), os['param_groups'])
        if accept:
            state = next_state
        else:
            t.restore_transaction(model, opt, old, os)


def test_candidate_names_and_noop_not_claimed_as_success():
    rows = []
    for split, tasks, states in [('train', range(6), [46, 48, 49]), ('validation', [6, 7], [48, 49])]:
        for task in tasks:
            for state in states:
                for request in range(4):
                    v = {'identity': 1. if split == 'train' else 4., 'frozen': 2.,
                         'plain_true': 3.5, 'guarded_true': 3., 'brp_true': 1.5,
                         'transaction_true': .5, 'transaction_mismatched': 2.5}
                    rows.append({'key': [task, state, request], 'split': split,
                                 'metrics': {k: {'row0': x, 'chunk': x, 'latent': 2.} for k,x in v.items()}})
    result = t.statistics(rows, 1)
    assert result['development_followup_supported']
    assert not t.statistics(rows, 0)['development_followup_supported']
    assert 'row0_better_transaction_mismatched' in result['development_checks']
    for split in result['splits'].values():
        assert 'bestref_true' not in split['metrics']
        assert split['metrics']['brp_true']['macro']['row0'] == 1.5
        assert split['metrics']['transaction_true']['macro']['row0'] == .5


def test_budget_before_model_execution():
    assert t.LIMITS['decoder'] == 264+72+72*72+261
    assert t.LIMITS['predictor'] == 176+144+2*72*72+522
    assert t.CAPTURES == 8+72*6+8
    with pytest.raises(ValueError, match='Predictor budget'):
        t.predict(None, {}, None, Counter(predictor=t.LIMITS['predictor']))
    with pytest.raises(ValueError, match='Decoder budget'):
        t.decode(None, None, {}, (), None, Counter(decoder=t.LIMITS['decoder']))
    assert not torch.cuda.is_initialized()
