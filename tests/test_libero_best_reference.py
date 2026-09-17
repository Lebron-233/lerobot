"""Synthetic CPU tests: no scientific data, model, GPU or Env."""
import copy
import sys
from collections import Counter
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'examples/advanced/predictive_async'))
import audit_libero_best_reference as a  # noqa: E402
import libero_best_reference as b  # noqa: E402


def rows():
    result = []
    for split, tasks, states in [('train', range(6), [46, 48, 49]), ('validation', [6, 7], [48, 49])]:
        for t in tasks:
            for s in states:
                for request in range(4):
                    values = {'identity': 1.0 if split == 'train' else 4.0, 'frozen': 2.0,
                              'plain_true': 3.5, 'guarded_true': 3.0, 'bestref_true': 0.5,
                              'bestref_zero': 1.0, 'bestref_mismatched': 2.5}
                    result.append({'key': [t, s, request], 'split': split,
                                   'metrics': {k: {'row0': v, 'chunk': v, 'latent': 2.0} for k, v in values.items()}})
    return result


def test_reference_preserves_better_of_both_not_bad_frozen():
    value = b.reference_bounds({'split': 'train'}, {'identity': {'row0': 8.0, 'chunk': 2.0},
                                                   'frozen': {'row0': 1.0, 'chunk': 5.0}})
    assert value == {'row0': 1.0, 'chunk': 2.0}


@pytest.mark.parametrize('split', ['validation', 'qualification', 'test'])
def test_no_validation_targets(split):
    with pytest.raises(ValueError, match='training-only'):
        b.reference_bounds({'split': split}, {})


@pytest.mark.parametrize('bad', [float('nan'), float('inf'), -1.0])
def test_nonfinite_or_negative_targets_rejected(bad):
    with pytest.raises(ValueError):
        b.reference_bounds({'split': 'train'}, {'identity': {'row0': bad, 'chunk': 1.0},
                                               'frozen': {'row0': 1.0, 'chunk': 1.0}})


def test_only_reference_changes_objective_and_retains_gradient():
    row0 = torch.tensor(2.0, requires_grad=True)
    metrics = (torch.tensor(2.0), row0, torch.tensor(1.0))
    scales = {'latent': 2.0, 'row0': 1.0, 'chunk': 1.0}
    old = b.r.objective(metrics, {'row0': 4.0, 'chunk': 1.0}, scales, 0.5, 'guarded')
    new = b.r.objective(metrics, {'row0': 1.0, 'chunk': 1.0}, scales, 0.5, 'guarded')
    assert new.item()-old.item() == 1.0
    new.backward()
    assert row0.grad.item() == 1.5


def test_reference_objective_matches_independent_arithmetic():
    m = {'latent': 2.0, 'row0': 3.0, 'chunk': 2.0}
    refs = {'row0': 1.0, 'chunk': 1.0}
    scales = {'latent': 4.0, 'row0': 2.0, 'chunk': 1.0}
    actual = b.r.objective(tuple(torch.tensor(m[k]) for k in ['latent', 'row0', 'chunk']), refs, scales, 0.5, 'guarded')
    assert actual.item() == a.old.loss_value(m, refs, scales, 0.5, 'guarded')


def test_budget_before_dispatch():
    with pytest.raises(ValueError, match='Predictor budget'):
        b.predict(None, {}, None, Counter(predictor=841))
    with pytest.raises(ValueError, match='Decoder budget'):
        b.decode(None, None, {}, (), None, Counter(decoder=597), 'test')
    assert b.LIMITS['decoder'] == 3*88+72+2*88+85
    assert b.LIMITS['predictor'] == 2*88+2*72+2*(2*88+85)


def test_complete_positive_gates_independent():
    data = rows()
    assert b.statistics(data)['development_followup_supported']
    assert a.decision(data)


@pytest.mark.parametrize('control', ['frozen', 'guarded_true', 'plain_true'])
def test_all_chunk_comparisons_are_required(control):
    data = rows()
    for x in data:
        if x['split'] == 'validation':
            x['metrics'][control]['chunk'] = 0.1
    assert not b.statistics(data)['development_followup_supported']
    assert not a.decision(data)


@pytest.mark.parametrize('control', ['frozen', 'guarded_true', 'plain_true'])
def test_all_worst_training_comparisons_required(control):
    data = rows()
    for x in data:
        if x['split'] == 'train':
            x['metrics'][control]['row0'] = 0.5
    assert not b.statistics(data)['development_followup_supported']
    assert not a.decision(data)


def test_mismatched_subset_keeps_paired_denominator():
    data = rows()
    for x in data[:3]:
        del x['metrics']['bestref_mismatched']
    assert b.statistics(data)['splits']['train']['contrasts']['bestref_mismatched']['samples'] == 69


def test_no_checkpoint_selection_change():
    spec = b.specification()
    assert spec['checkpoint_selection'] == 'fixed_final_72' and spec['coefficients'] == [1, 1, 1]
    assert spec['lr'] == 0.0001 and spec['seed'] == 20260912
    assert b.specification() == copy.deepcopy(spec)
    assert not torch.cuda.is_initialized()
