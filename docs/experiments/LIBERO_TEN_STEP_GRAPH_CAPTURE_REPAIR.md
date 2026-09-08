# Ten-step graph capture: concrete constant-construction repair

Date: 2026-09-08. The first graph profile at
`f3894bc4dc4252460285575c75d78437488d461c` failed during capture, before any
paired timing or replay-equivalence result. Its output is preserved in
`outputs/libero_ten_step_graph_f3894bc4/result.json`; command exit1, tool wall10.326s.
Preregistration5587550213 was published and read back exactly before execution.

## Observed failure

Strict model loading and initial eager evaluation succeeded. Capture reached
`VLAFlowMatching.embed_prefix_from_tokens`, line834, where a Python list was passed
to `torch.tensor(..., device='cuda')`. PyTorch rejected the unpinned CPU-to-CUDA
copy during CUDA graph capture. The exception is retained in the original JSON.
Original sampler restoration and final num_steps10 were recorded. No simulator,
task sample, predictor, alternative numerical candidate or precision change ran.

## Minimal repair and scope

Construct the same constants directly on the destination device:

- Prefix block mask: zeros followed by ones for the state tokens, then the existing
  prefix padding and batch expansion. All special/image/language mask values remain0.
- Suffix block mask: the original all-ones vector, retaining its original dtype/shape.
- Euler timestep: `torch.full((batch_size,), time, dtype=float32, device=device)`
  instead of a scalar CPU construction followed by expand. Same scalar arithmetic,
  time grid, dt, ten calls and integrator update; RTC and hard-prefix handling unchanged.

The latter two instances are the same CPU-copy pattern on the supported path that
the first failure had not reached. This is three constant-construction changes in
the existing model/helper, not a new sampler, compile pass, CUDA kernel or precision
change. Weights, masks' logical values and the ten-step computation are unchanged.
The source changes supersede the original prototype's no-src-edit boundary only
for this separately recorded repair. Closed experimental sources are not rewritten.

Targeted tests compare prefix/suffix mask values against the historical lists with
and without image special tokens, padding and batch1/2; they compare full timestep
grids at1/3/10 steps and batch1/2/7. Existing flow tests also compare outputs and
RTC behavior to the literal historical loop, not another call to the new helper.

The targeted suite passed40 tests. Its first run had two new-test expectation
failures: the global image-start marker contains two tokens, not one. The expected
historical list now derives the special-token count from the existing token arrays;
no model code was changed to accommodate that test correction. Ruff subsequently
identified an unbound test closure variable; it was explicitly bound with a default.

## Separately registered validation

Commit/push this repair and test outcome, then register a fresh output directory
and exact SHA before a single new graph validation. Retain the original four
synthetic fixtures, initial same-seed replay check,5 warmup+24 measured pairs,
per-fixture balanced order, all ten projections, fresh image encoding every call,
all-buffer refresh, full padded/selected/postprocessed exact equality,180s bound,
timing scope and false qualification flags. Do not relax equality or delete failures.

This is a concrete technical repair after a preserved failed run, not a repeated
measurement searching for a favorable speed. It does not authorize new native
episodes or access to the separately blocked matched-control result.
