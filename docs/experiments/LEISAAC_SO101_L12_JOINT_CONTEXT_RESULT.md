# L12: held-out coherent-context evidence, with the coverage failure retained

Date:2026-09-07. State fitting source `bf4e025dbf0cacf4777289b90498a8b70d4974fa`;
independent collection and evaluation source
`fab27a3db3f08a7cb6dbce89b594eab8db9915e7`. All model choices were frozen before
the new scenes were collected. The selected25,478-parameter state MLP is epoch29;
the visual predictor is unchanged L6 epoch3. No policy/VLM/action-expert weights
were trained, and no L6/L8/L11 test trajectories were used for fitting.

## Completed independent experiment

Six fresh scenes20270320–25 (RNG seeds, not scheduled dates), each600 actual
actions and601 observations, produced the complete288 registered cases. These
are six trajectory clusters, not288 independent experiments. Current state and
already committed normalized prefix predict future proprioception; current
visual tokens/prefix/current state feed the original L6 visual predictor.

The **new coherent offline teacher** supplies both actual future visual tokens
and actual future model-ready state. Those future values are not inputs to any
learned predictor. All arms share noise, language, feasible action projection,
and compare the first25 normalized actions. This differs from the old visual-
oracle/current-state audit: the new percentages do not replace the old results.

| Context supplied to the frozen decoder | Held-out mean action L1 |
|---|---:|
|Current visual/current state (identity)|0.134787828631|
|L6 future visual/current state (visual-only)|0.137102237366|
|Current visual/learned future state (state-only)|0.121383302506|
|L6 future visual/learned future state (joint)|0.118989221571|
|True future visual/current state (offline control)|0.098817032377|
|Current visual/true future state (offline control)|0.096088568774|

## What is supported, and what failed

| Joint compared with | Mean relative reduction | Improving cases | Improving episode means | Episode-bootstrap95% reduction interval |
|---|---:|---:|---:|---|
|Identity|11.7211%|179/288|6/6|[6.8537%,15.8048%]|
|Visual-only|13.2113%|185/288|6/6|[6.9170%,18.9189%]|
|State-only|1.9723%|139/288|5/6|[-0.7427%,4.2396%]|

Unlike L11's sign-reversing scene-half means, both primary L12 comparisons have
positive mean effects in every one of the six independent trajectory clusters.
This supports an average coherent-action-reference improvement for the joint
successor under this new metric. It does **not** prove L11's reversal had a
single cause or that the original visual-only M3 now passes its old test.

The prespecified full stable-action gate also required at least192/288 improving
cases in each primary contrast. Counts179 and185 do not reach it. Therefore the
full stable-action gate remains **FAIL**, despite the positive average-effect
intervals. Do not lower the threshold, omit harmed cases, or call this uniform
per-case benefit. Joint versus state-only also does not establish added visual
value: its interval crosses zero and fewer than half the cases improve.

Most of the new average gain is compatible with compensation for stale state;
the six-arm factorial makes that attribution limitation explicit. A "state-only"
arm still uses current images: it forecasts only state, not a vision-free policy.

## Frozen artifacts and next boundary

All artifacts are under `/home/rp/Workspace/SmolVLA_RTC/artifacts/`:
`m54l12_state_development_v1` (fixed30-epoch state fitting and96 development
decoder cases), `m54l12_joint_context_test_v1` (new actual trajectories), and
`m54l12_joint_context_evaluation_v1/test_report.json` (all288 fixed cases, both
25/50-action scores and actual/predicted state errors).

No joint-state online deployment or task comparison has run. This is an explicit
new joint-context hypothesis, not a silent change to visual-only runtime. A
subsequent task study must keep visual-only and state-only attribution controls,
use new seeds, and preserve this failed coverage gate. Positive oracle-reference
means alone do not imply higher task success. No weights or thresholds are
selected again using these opened L12 test cases; risk thresholds remain null.
