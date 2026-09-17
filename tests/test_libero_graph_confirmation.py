"""Frozen-candidate confirmation tests; no scientific data, model or Env execution."""

import copy
import math
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'examples/advanced/predictive_async'))
import audit_libero_graph_confirmation as audit  # noqa: E402
import libero_graph_confirmation as r  # noqa: E402


def test_fixed_balanced_new_cohort_and_unchanged_single_episode_settings():
    m, template = r.manifest(), r.g.manifest()['rows'][0]
    assert len(m['rows']) == 64 and len(set(r.PAIRS)) == 32
    assert {s for _, s in r.PAIRS} == set(range(12, 20))
    for task in (0, 2, 6, 7):
        assert sum(t == task for t, _ in r.PAIRS) == 8
        first = [m['rows'][2*i]['condition'] for i, (t, _) in enumerate(r.PAIRS) if t == task]
        assert first.count('async') == first.count('serialized') == 4
    for pair in range(32):
        a, b = m['rows'][2*pair:2*pair+2]
        assert a['environment_seed'] == b['environment_seed'] == 1160000+100*a['task_id']+a['initial_state_id']
        assert a['policy_seed'] == b['policy_seed'] == 1170000+100*a['task_id']+a['initial_state_id']
        assert {a['condition'], b['condition']} == {'serialized', 'async'}
        for key in ('ready_wall_slots', 'fps', 'threshold', 'max_delay', 'guard', 'margin', 'latency_window', 'latency_quantile', 'limits'):
            assert a[key] == b[key] == template[key]
    assert m['captures'] == 64 and m['latency_limit_s'] == .35
    assert m['candidate_head'] == r.CANDIDATE_HEAD and not m['rtc']


def test_uses_actual_frozen_predictor_and_controller_without_global_contract_rewrites():
    from libero_graph_feedback import GraphFeedbackPredictor
    assert r.g.GraphFeedbackPredictor is GraphFeedbackPredictor
    assert r.old.control_loop is r.g.old.control_loop
    assert r.g.PAIRS == ((0, 10), (2, 10), (6, 10), (7, 10), (7, 11), (6, 11), (2, 11), (0, 11))
    assert r.old.LIMITS['episodes'] == (1, 16)
    assert r.LIMITS['episodes'] == (1, 64)
    assert not torch.cuda.is_initialized()


def test_duplicate_identity_stops_without_replacement():
    r.assert_unused([{'task': 0, 'state': 10}])
    with pytest.raises(ValueError, match='already used'):
        r.assert_unused([{'task': 6, 'state': 17}])


@pytest.mark.parametrize('k', list(range(33)))
def test_independent_confidence_implementation_agrees(k):
    assert r.loss_upper(k, 32) == pytest.approx(audit.independent_upper(k, 32), abs=1e-11)


def test_zero_loss_design_and_one_loss_cannot_pass():
    assert math.ceil(math.log(.05)/math.log(.9)) == 29
    assert r.loss_upper(0, 32) == pytest.approx(1-.05**(1/32))
    assert r.loss_upper(0, 32) < .10 < r.loss_upper(1, 32)
    assert r.loss_upper(32, 32) == 1
    assert all(r.loss_upper(k+1, 32) >= r.loss_upper(k, 32) for k in range(32))


@pytest.mark.parametrize('k,n,alpha', [(-1, 32, .05), (33, 32, .05), (0, 0, .05), (0, 32, 0)])
def test_invalid_statistics_rejected(k, n, alpha):
    with pytest.raises(ValueError):
        r.loss_upper(k, n, alpha)


def test_bound_has_conservative_coverage_on_heterogeneous_bernoulli_examples():
    # Enumerate Poisson-binomial counts; no robot outcome or empirical tuning.
    for probabilities in (np.linspace(.01, .6, 32), np.array([.03]*16+[.3]*16), np.full(32, .1)):
        pmf = np.array([1.])
        for p in probabilities:
            pmf = np.convolve(pmf, np.array([1-p, p]))
        failure = sum(pmf[k] for k in range(33) if r.loss_upper(k, 32) < probabilities.mean())
        assert failure <= .05+1e-12


def test_no_missing_pair_can_be_silently_dropped():
    records = []
    for spec in r.manifest()['rows']:
        records.append({'spec': copy.deepcopy(spec), 'status': 'completed', 'success': True,
            'measured_actions': 100, 'wall_s': 5., 'no_action_slots': 0, 'underflows': 0,
            'terminal_reason': 'native_success', 'startup_s': 1.3})
    summary = r.task_summary(records)
    assert summary['complete_pairs'] == 32 and summary['retention']['bounded_task_retention_passed']
    with pytest.raises(ValueError, match='Incomplete'):
        r.task_summary(records[:-1])
    records[1]['success'] = False
    summary = r.task_summary(records)
    assert summary['retention']['lost_pairs'] == 1 and not summary['retention']['bounded_task_retention_passed']


def test_global_budget_changes_do_not_change_local_model_limit():
    budget = r.Budget()
    budget.begin_episode()
    assert budget.total['episodes'] == 1
    for _ in range(160):
        budget.take('model')
    with pytest.raises(ValueError, match='budget exhausted'):
        budget.take('model')


def synthetic_audit_rows():
    return [{'ordinal': s['ordinal'], 'pair_index': s['pair_index'], 'arm': s['condition'],
        'success': True, 'wall_s': 5., 'actions': 100, 'no_action_slots': int(s['condition'] == 'serialized'),
        'takeovers': 4, 'underflows': 0, 'request_count': 5, 'expired': 0, 'cancelled_at_stop': 0,
        'overlaps': [1] if s['condition'] == 'async' else [], 'request_seconds': [.1]*4,
        'queue_wait_seconds': [.001]*4, 'bootstrap_s': 1.3, 'startup_s': 1.3,
        'actual_delays': [0, 2] if s['condition'] == 'async' else [0, 0],
        'terminal_reason': 'native_success'} for s in r.manifest()['rows']]


def test_confirmation_joint_gate_preserves_all_pair_outcomes():
    rows = synthetic_audit_rows()
    assert audit.aggregate(rows)['bounded_confirmation_passed']
    rows[1]['success'] = False
    got = audit.aggregate(rows)
    assert got['retention']['lost_pairs'] == 1 and not got['bounded_confirmation_passed']
    assert len(got['pairs']) == 32


def test_one_tail_overrun_is_not_hidden_by_more_than_100_requests():
    rows = synthetic_audit_rows()
    rows[1]['request_seconds'][-1] = .351
    got = audit.aggregate(rows)
    assert got['online_latency_budget_passed']
    assert got['arms']['async']['requests_above_350ms'] == 1
    assert not got['all_nonbootstrap_within_budget'] and not got['bounded_confirmation_passed']


def test_no_actual_overlap_does_not_pass_confirmation():
    rows = synthetic_audit_rows()
    for row in rows:
        row['overlaps'] = []
    assert not audit.aggregate(rows)['bounded_confirmation_passed']
