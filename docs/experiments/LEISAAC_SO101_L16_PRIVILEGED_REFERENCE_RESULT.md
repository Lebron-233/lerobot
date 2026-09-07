# L15–L16 closed: accurate target-time information did not establish added task value

Date: 2026-09-07. L16 outcome execution source:
`4ee3b992d69d11875e8a17b67884e43db458720c`.
All 24 registered conditions completed once with the source unchanged. The
shared simulator exited normally. The closed-outcome audit was integrated only
after the cohort finished; no model, outcome, seed or failed condition was replaced.

## 1. Decision

The state-only reference meets the registered bounded subtask eligibility
criterion: 8/8 successful first-settled-region endpoints, zero technical failures.
Neither privileged-information reference establishes the registered task benefit.
True target-time visual input, even with true target-time state in the additional
control, did not consistently improve this frozen policy over state-only control.

This challenges the assumption that making a visual forecast more accurate will
necessarily improve this task configuration. It is not proof that vision is
unnecessary, that forecasting is impossible, or that a learned predictor cannot
outperform these references. A frozen imperfect policy is not an optimal controller;
these oracle arms are information references, NOT mathematical performance bounds.
The eight-scene intervals remain wide and include improvements as well as harms.

L15 and L15b both failed their development eligibility requirements. Neither
student was deployed or tested on its reserved scenes. L16 is therefore a
three-arm information-value experiment, not an evaluation of either new student.

## 2. Completed incremental-vision development

L15 collected eight new state-only controlled-delay trajectories, each with
1,200 executed actions and 44 actual request/target pairs: 9,600 measured actions,
240 separately recorded setup actions, and 352 pairs. The split is six training
and two validation episodes; two paired flow-noise draws produce 528 training
and 176 validation cases. Every target was acquired only after the corresponding
seven committed normalized actions actually executed. No future state was supplied
to a learned visual predictor.

The conditional teacher supplies true future visual tokens together with the
SAME causal predicted state used by state-only and student decoding. This
isolates the visual input rather than attributing state compensation to vision.
Only the 273,824-parameter visual student is optimized; policy/VLM/action expert,
the L12 state forecaster, and the L6 parent remain frozen. Zero-residual student
initialization matched the state-only output exactly, with maximum difference 0.

| Development profile | Optimizer updates | Training examples processed | Diagnostic selected update | Validation action L1 | Cases improved /176 |
|---|---:|---:|---:|---:|---:|
|State-only reference|0|0|—|0.076867422|—|
|L15 single-case updates|1,600|1,600|1,600|0.069882075|83|
|L15b four-episode gradient accumulation|400|1,600|400|0.071438625|94|

The respective equal-episode mean reductions are 9.0875% and 7.0625%. Neither
reaches the unchanged 118/176 improving-case requirement; neither ever produced
an eligible checkpoint. Maximum coverage during each run was 97/176. These
are reused-development validation results, not independent evidence of efficacy.
Both first backwards produced finite nonzero student gradients and zero policy
gradients. No EMA, risk threshold, model fine-tuning, or test-derived fallback
was introduced. The original L10 tests and reserved L15 scenes remain unopened.

Detailed histories and decisions remain in
[L15 single-case result](LEISAAC_SO101_L15_SINGLE_DEVELOPMENT_RESULT.md) and
[L15b result](LEISAAC_SO101_L15B_DEVELOPMENT_RESULT.md).

## 3. What L16 actually changes

The [preregistered protocol](LEISAAC_SO101_L16_PRIVILEGED_REFERENCE_PLAN.md)
fixes seven steps of simulated observation age, the original ScheduledActionQueue,
the frozen WSAGI policy/processors and L12 epoch29 state model, feasible actions
before commitment, and the same physical environment and full-resolution cameras.

`state_only` generates at observation t from current visual tokens and state
predicted from the current state and seven committed actions. The original
queue withholds that chunk until t+7.

`oracle_visual` stores the causal predicted state from t but defers its new
chunk computation until those seven old actions have actually executed. It then
uses the real visual observation at t+7 and the stored predicted state, staging
the result ON TIME before the queue get at t+7.

`oracle_joint` uses both real visual tokens and real model-ready state at t+7.
These are explicitly privileged inputs, not predictions. Inference pauses
simulated physics in these reference arms, so no real-time deployment claim is
possible. A task stop before t+7 censors the unobserved oracle request.

For each of eight fresh scenes 20270710–20270717, a single real frozen-policy
current-observation bootstrap was generated before any outcome arm. All three
arms installed that same normalized and physical chunk. Each bootstrap capture
and each arm performed 30 logged setup actions, followed by a same-seed reset.
Policy seeds are 3610–3617; the seed integers are identifiers, not scheduled dates.
The three-arm order was fixed and shuffled before any scene outcome was observed.

All 27 commands preceding the first takeover match across arms. Initial state,
object and camera geometry matches. Across the measured pre-takeover interval,
joint-state differences were zero and maximum object-coordinate difference was
1.013278962e-6 m. At observation 27 immediately before the first new action,
joint-state differences were also zero and maximum object-coordinate difference
was 9.536743164e-7 m. We do not call these object trajectories or rendered RGB
bitwise identical. The common-bootstrap protocol reduces one source of variation;
it does not eliminate subsequent simulator/rendering stochasticity.

## 4. All registered task outcomes

The endpoint remains first ten-tick low-speed plate-region occupancy, NOT
contact-release verification or the original three-oranges-plus-rest task.
All conditions use the 120-s simulated-time bound; failures receive cost 120 s.

| Arm | Subgoal successes | Technical failures | Mean restricted simulated time |
|---|---:|---:|---:|
|State-only, using current images|8/8|0/8|26.504167 s|
|True target-time visual + causal predicted state|6/8|0/8|39.975000 s|
|True target-time visual + true target-time state|6/8|0/8|42.945833 s|

All eight scene outcomes are retained. F means no subgoal by the 120-s bound.

| Scene | State-only | Oracle visual | Oracle joint |
|---|---:|---:|---:|
|20270710|12.467|7.633|12.267|
|20270711|20.267|25.133|120 F|
|20270712|47.667|13.067|24.033|
|20270713|13.867|7.500|10.267|
|20270714|21.067|120 F|120 F|
|20270715|49.233|15.100|6.800|
|20270716|36.033|120 F|31.800|
|20270717|11.433|11.367|18.400|

Negative differences below favor the privileged arm. The preregistered paired
bootstrap uses 50,000 draws and RNG 3601; the eight scenes, not individual
control ticks, are the sampling units.

| Reference minus state-only | Mean difference | Paired 95% interval | First four mean | Last four mean | Stable gate |
|---|---:|---|---:|---:|---|
|Oracle visual|+13.470833 s|[-16.712500, +48.434375] s|-10.233333 s|+37.175000 s|Fail|
|Oracle joint|+16.441667 s|[-15.250000, +53.425000] s|+18.075000 s|+14.808333 s|Fail|

Oracle visual is faster in five individual scenes, but its two failed scenes
and lack of consistent mean improvement prevent a stable-benefit conclusion.
The positive mean differences do NOT establish that privileged information
systematically harms the policy: both intervals cross zero. Reference eligibility
passes, but `privileged_visual_reference_positive=false` and
`learned_task_benefit=false`. No new learned student is present in these outcomes.
The shared-bootstrap/new-scene protocol must not be pooled with L14 to claim
that the unchanged state-only model itself improved between cohorts.

## 5. Independent information-boundary and physical-action audit

The closed audit reconstructs each endpoint from pre-auto-reset witnesses and
checks each request's real information timestamps, exact seven-row normalized
commitment, generation step, and queue takeover. It distinguishes a causal
forecast from a privileged reference, and a completed generation from a takeover.

| Audited item | Count |
|---|---:|
|Once-only outcome conditions|24|
|Subgoal endpoints / timeouts / technical failures|20 / 4 / 0|
|Measured physical actions|26,262|
|Setup actions (8 bootstrap preparations + 24 arm preparations)|960|
|Actual takeovers|960|
|Privileged target-time generations|729|
|Actually executed old-prefix actions|6,735|
|Actual shared-simulator exits|1, return code 0|

The separate smoke completed 60 measurement actions and 60 setup actions, with
two target-time generations/takeovers; it is excluded from the cohort totals.

The first audit stopped because it compared canonical arm order with shuffled
execution order for the physical-divergence rows. The audit was corrected to
compare unique (block, arm) keys while still rejecting any changed physical value
or duplicate condition. No raw result or statistic changed. The full subsequent
audit passed, confirming the keyed values agree, along with seven targeted
information-boundary, queue, and audit tests and lint/format checks.

## 6. Research implication and next prerequisite

L15's conditional action-reference loss can be reduced during development, but
case coverage did not qualify. L16 now separately tests the task behavior of the
true visual reference that such an objective seeks to approximate, and finds
no stable incremental benefit over state-only in this bounded setup. Therefore
another decrease in that offline loss is not by itself a sufficient reason to
scale the same task campaign or claim the learned method is effective.

Preserve state-only as a distinct strong control. Before increasing visual
forecasting capacity or running another comparable efficacy campaign, establish
that target-time visual information provides repeatable additional task value
under a separately registered task/control condition. Do not tune on these
opened L16 outcomes or weaken the L15 failed criteria. This is a change in the
order of research investment, not a proof of impossibility or a claim that vision
does not matter. The original real-time readiness failures remain unchanged.

## 7. Retained evidence

Under `/home/rp/Workspace/SmolVLA_RTC/artifacts/`:

- `m54l15_onpolicy_development_v1`: all eight new causal request/target trajectories;
- `m54l15_visual_increment_training_v1`, `m54l15b_balanced_training_v1`: full fits and selections;
- `m54l15_closed_development_audit_v1.json`, `m54l15b_closed_development_audit_v1.json`: selection/prefix audits;
- `m54l16_privileged_smoke_v1`: the one wiring smoke;
- `m54l16_privileged_task_v1`: shared bootstraps, all 24 outcomes, raw events/actions/witnesses and shared log;
- `m54l16_closed_reference_audit_v1.json`: closed outcome, physical-prefix and information-boundary audit.

No student-test collection or repeated task trial was run after these results.
See the [machine-readable closure summary](LEISAAC_SO101_L16_SUMMARY.json).
