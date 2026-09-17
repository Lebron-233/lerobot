"""Synthetic CPU-only coverage of F-UDI1 algebra, gradients and closed scope."""
import ast
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

EXAMPLES = Path(__file__).resolve().parents[1] / 'examples/advanced/predictive_async'
sys.path.insert(0, str(EXAMPLES))
import audit_libero_update_diagnostic as audit  # noqa: E402
import libero_update_diagnostic as u  # noqa: E402


def test_fixed_budget_and_sentinels():
    assert u.EXPECTED['decoder'] == 2*72+2*73 == 290
    assert u.EXPECTED['predictor'] == 2*u.EXPECTED['decoder']
    assert u.EXPECTED['state_loads'] == 2*72+73
    assert u.EXPECTED['component_vjps'] == 3*72
    assert u.SENTINELS == ((4, 46, 5), (5, 48, 6))
    assert not u.specification()['new_updates']
    assert not u.specification()['checkpoint_selection']


@pytest.mark.parametrize('values', [(1., 2., 3., 4.), (1., 2., .5, 3.), (4., 3., 2., 1.)])
def test_telescoping_attribution_keeps_all_effects(values):
    points = [dict.fromkeys(('row0', 'chunk', 'latent', 'objective'), x) for x in values]
    result = u.attribution(*points)
    for v in result.values():
        assert v['earlier']+v['own']+v['later'] == pytest.approx(v['net'])
    assert result['row0']['own'] == values[2]-values[1]


@pytest.mark.parametrize('a,c', [(0.2, 0.3), (0.8, 0.9), (0.5, 0.5)])
def test_component_loss_and_gradients_match_original_in_fp64(a, c):
    metrics = tuple(torch.tensor(x, dtype=torch.float64, requires_grad=True) for x in (2., a, c))
    bounds, scales = {'row0': .5, 'chunk': .5}, {'latent': 2., 'row0': .1, 'chunk': .2}
    loss = u.r.objective(metrics, bounds, scales, .3, 'guarded')
    total = sum(u.components(metrics, bounds, scales, .3).values())
    assert float(total.detach()) == pytest.approx(float(loss.detach()))
    for v, w in zip(torch.autograd.grad(loss, metrics), torch.autograd.grad(total, metrics), strict=True):
        assert torch.equal(v, w)


def test_independent_vectors_and_zero_norm():
    ones = torch.ones(69680)
    g = {'latent': ones, 'row0': -ones, 'chunk': ones*0, 'total': ones*0}
    displacement = ones*1e-4
    left, right = u.vector_stats(g, displacement), audit.independent_vectors(g, displacement)
    audit.compare(left, right)
    assert left['cosines']['latent__row0'] == pytest.approx(-1)
    assert left['cosines']['chunk__row0'] is None
    assert left['directional']['row0'] < 0


def test_direction_tolerance_and_forgetting():
    assert audit.direction(1., 1.+1e-8) == 'tied'
    assert audit.direction(1., .5) == 'decreased'
    p = [dict.fromkeys(('row0', 'chunk', 'latent', 'objective'), x) for x in (1., 2., .5, 3.)]
    a = u.attribution(*p)['row0']
    assert a['own'] < 0 < a['later']
    assert a['net'] > 0


def test_displacement_preserves_parameter_order():
    values = [torch.tensor([[1., 2.]]), torch.tensor([3.])]
    assert torch.equal(u.flat(values), torch.tensor([1., 2., 3.]))
    assert u.flat(values).device.type == 'cpu'


def test_scope_no_optimizer_step_or_native_factory():
    tree = ast.parse((EXAMPLES / 'libero_update_diagnostic.py').read_text())
    attrs = [n.func.attr for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)]
    assert 'step' not in attrs and 'make_native_env_factory' not in attrs
    assert 'load_state_dict' in attrs and 'backward' in attrs and 'grad' in attrs
    assert not u.specification()['validation_forward']


def test_audit_rejects_bad_vector():
    grads = {k: torch.zeros(4) for k in ('total', 'row0', 'latent', 'chunk')}
    with pytest.raises(ValueError, match='Gradient vector invalid'):
        audit.independent_vectors(grads, torch.zeros(4))


def test_no_cpu_cuda_initialization():
    assert not torch.cuda.is_initialized()
    assert np.isfinite(1.)
