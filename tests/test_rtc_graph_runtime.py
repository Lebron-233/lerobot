"""Synthetic RTC tests only: no checkpoint, saved images or simulator."""

import os
import sys
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'examples/advanced/predictive_async'))
from rtc_graph_runtime import DevicePrefixRTC, FixedCallGraph, RecordedRTC, RTCGraphRuntime  # noqa: E402

from lerobot.configs import RTCAttentionSchedule  # noqa: E402
from lerobot.policies.common.flow_matching import euler_integrate  # noqa: E402
from lerobot.policies.rtc.configuration_rtc import RTCConfig  # noqa: E402
from lerobot.policies.rtc.modeling_rtc import RTCProcessor  # noqa: E402


def config():
    return RTCConfig(enabled=True, mode='guided', execution_horizon=10,
                     max_guidance_weight=10.0, prefix_attention_schedule=RTCAttentionSchedule.EXP)


def integrate(processor, values, projection):
    x, prefix = values
    with torch.no_grad():
        return euler_integrate(lambda z, t: projection(.2*z+.05*torch.sin(z)), x, 10,
            rtc_processor=processor, rtc_enabled=True, inference_delay=3,
            prev_chunk_left_over=prefix, execution_horizon=10)


def test_cached_weights_exact_and_wrong_signature_rejected():
    native, cached = RTCProcessor(config()), DevicePrefixRTC(config(), 'cpu')
    torch.testing.assert_close(native.get_prefix_weights(3, 10, 50), cached.weights, rtol=0, atol=0)
    assert cached.weights.data_ptr() == cached.get_prefix_weights(3, 10, 50).data_ptr()
    with pytest.raises(ValueError, match='signature'):
        cached.get_prefix_weights(2, 10, 50)


@pytest.mark.parametrize('prefix_dim', [7, 32])
@pytest.mark.parametrize('time_value', [1.0, .5, .1])
def test_full_denoiser_derivative_and_absent_coordinates(prefix_dim, time_value):
    native, cached = RTCProcessor(config()), DevicePrefixRTC(config(), 'cpu')
    x = torch.linspace(-1, 1, 1600).reshape(1, 50, 32)
    prefix = torch.full((30, prefix_dim), .1)
    arguments = {'x_t':x, 'prev_chunk_left_over':prefix, 'inference_delay':3, 'time':time_value,
                 'original_denoise_step_partial':lambda z: .25*z, 'execution_horizon':10}
    a, b = native.denoise_step(**arguments), cached.denoise_step(**arguments)
    torch.testing.assert_close(a, b, rtol=0, atol=0)
    tau = 1-time_value
    gamma = 10.0 if tau == 0 else min(((1-tau)/tau)*(((1-tau)**2+tau**2)/(1-tau)**2), 10.0)
    mask = cached.weights[None, :, None]
    padded = torch.zeros_like(x)
    padded[:, :30, :prefix_dim] = prefix
    residual = (padded-(1-.25*time_value)*x)*mask
    residual[..., prefix_dim:] = 0
    expected = .25*x-gamma*(1-.25*time_value)*residual
    torch.testing.assert_close(a, expected, rtol=2e-6, atol=2e-6)
    if prefix_dim == 7:
        torch.testing.assert_close(a[..., 7:], .25*x[..., 7:], rtol=0, atol=0)


def test_original_and_cached_ten_step_outputs_exact():
    values = (torch.linspace(-.5,.5,1600).reshape(1,50,32), torch.full((30,7),.1))
    a = integrate(RecordedRTC(config()), values, torch.nn.Identity())
    b = integrate(DevicePrefixRTC(config(), 'cpu'), values, torch.nn.Identity())
    torch.testing.assert_close(a,b,rtol=0,atol=0)


def test_runtime_rejects_scope_changes_before_cuda():
    cfg = SimpleNamespace(num_steps=5, chunk_size=50, rtc_config=config())
    model = SimpleNamespace(config=cfg)
    with pytest.raises(ValueError, match='configuration'):
        RTCGraphRuntime(model)
    assert not torch.cuda.is_initialized()


@pytest.mark.skipif(os.environ.get('RUN_RTC_GRAPH_CUDA_TESTS') != '1', reason='explicit synthetic CUDA opt-in')
def test_original_rtc_synthetic_capture_feasibility_and_dynamic_prefix():
    p = DevicePrefixRTC(config(), 'cuda')
    native = RecordedRTC(config())
    x = torch.linspace(-.5,.5,1600,device='cuda').reshape(1,50,32)
    prefix = torch.full((30,7),.1,device='cuda')
    projection = torch.nn.Identity()
    graph = FixedCallGraph(lambda v: integrate(p, v, projection), (x,prefix), p, projection, True)
    assert graph.record['captured_steps'] == graph.record['captured_vjps'] == 10
    for value in (.1,-.15,.2):
        prefix.fill_(value)
        expected = integrate(native, (x,prefix), projection)
        actual = graph((x,prefix))
        torch.cuda.synchronize()
        torch.testing.assert_close(actual,expected,rtol=0,atol=0)
    assert graph.replays == 3
    saved = actual.clone()
    prefix.fill_(.4)
    changed = graph((x,prefix))
    torch.cuda.synchronize()
    assert not torch.equal(changed,saved)
    assert torch.equal(actual,saved)
    with pytest.raises(ValueError, match='shape'):
        graph((x,prefix[:20]))


def test_fixed_schedule_and_independent_latency_reduction():
    import audit_libero_rtc_graph as audit
    import libero_rtc_graph as run
    specs=run.schedule()
    assert len(specs)==288
    assert Counter(s['arm'] for s in specs)==dict.fromkeys(run.MODES,96)
    rows=[{**s,'complete_s':100.0 if not s['repeat'] else (.1 if s['arm']=='graph' else .6)} for s in specs]
    a,b=run.timings(rows),audit.reduce_timing(rows)
    assert a['graph_budget_passed'] and b['graph_budget_passed']
    assert a['graph_faster_mean'] and b['graph_faster_mean']
    assert all(v['n']==80 for v in b['timing'].values())
    late=next(s for s in rows if s['arm']=='graph' and s['repeat']>0)
    late['complete_s']=.36
    assert not audit.reduce_timing(rows)['graph_budget_passed']
    with pytest.raises(ValueError,match='population'):
        audit.reduce_timing(rows[18:])


class SyntheticModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.config=SimpleNamespace(num_steps=10,chunk_size=50,rtc_config=config())
        self.state_proj=torch.nn.Linear(1,1).cuda().requires_grad_(False)
        self.action_out_proj=torch.nn.Identity()
        self.rtc_processor=RecordedRTC(config())

    def encode_image_tokens(self, images, masks):
        return tuple(images),tuple(masks)

    def sample_actions(self, images, masks, tokens, token_masks, state, noise=None, **kw):
        context=(kw['future_image_tokens'][0].mean()+kw['future_image_tokens'][1].mean()
                 +tokens.float().mean()*.001+state.mean()*.01)*.01
        with torch.no_grad():
            return euler_integrate(lambda z,t:self.action_out_proj(.2*z+.05*torch.sin(z)+context),noise,10,
                rtc_processor=self.rtc_processor,rtc_enabled=True,inference_delay=kw['inference_delay'],
                prev_chunk_left_over=kw['prev_chunk_left_over'],execution_horizon=kw['execution_horizon'])


@pytest.mark.skipif(os.environ.get('RUN_RTC_GRAPH_CUDA_TESTS')!='1',reason='explicit synthetic CUDA opt-in')
def test_full_runtime_refreshes_inputs_and_restores_processor():
    model=SyntheticModel()
    original,processor=model.sample_actions,model.rtc_processor
    with RTCGraphRuntime(model) as runtime:
        for sample in range(3):
            images=[torch.full((1,4,8),.1+sample*.1,device='cuda') for _ in range(2)]
            masks=[torch.ones((1,4),dtype=torch.bool,device='cuda') for _ in range(2)]
            tokens=torch.full((1,11),sample,dtype=torch.long,device='cuda')
            token_mask=torch.ones_like(tokens,dtype=torch.bool)
            state=torch.full((1,32),.05*sample,device='cuda')
            noise=torch.linspace(-.5,.5,1600,device='cuda').reshape(1,50,32)
            for guided in (False,True):
                prefix=torch.full((30,7),.15*sample-.1,device='cuda') if guided else None
                outputs=[]
                for mode in runtime.MODES:
                    runtime.mode=mode
                    output=model.sample_actions(images,masks,tokens,token_mask,state,noise=noise,
                        inference_delay=3,execution_horizon=10,prev_chunk_left_over=prefix)
                    outputs.append(output.clone())
                torch.cuda.synchronize()
                for value in outputs[1:]:
                    torch.testing.assert_close(outputs[0],value,rtol=0,atol=0)
    assert model.sample_actions==original and model.rtc_processor is processor
    assert runtime.receipt['replays']=={'no_prefix':3,'guided':3}
    assert len(runtime.receipt['captures'])==2
    assert runtime.receipt['sampler_restored'] and runtime.receipt['processor_restored']
    assert runtime.receipt['graphs_released']
