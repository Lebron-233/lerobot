# L16: task-space visual-information reference, with a shared real-policy bootstrap

Date:2026-09-07. No L15 test data are used to choose this experiment. Finish
both scheduled development fits before running another simulator workload.
The question is whether accurate target-time visual information itself gives
the frozen policy useful incremental task performance over state-only control.
This is NOT a deployable forecaster, realtime qualification, or mathematical
upper bound: a frozen imperfect policy need not improve when given extra data.

## Arms and strict information boundary

Use the existing controlled seven-step simulated age, original scheduled queue,
fixed WSAGI policy/processors, and immutable L12 epoch29 state predictor.

* `state_only`: at request t compute predicted state from current state and
  the seven committed actions; generate using current visual/predicted state.
* `oracle_visual`: retain that SAME causal predicted-state construction, but
  defer generation until the seven old actions have actually executed. At t+7
  use the real newly observed visual tokens, together with the stored predicted
  state from t, and stage ON TIME immediately before queue get at t+7.
* `oracle_joint`: the same deferred generation, using both real visual tokens
  and real model-ready state observed at t+7. These values are explicitly
  privileged reference inputs, not learned future predictions.

Both oracle arms pause simulated physics during their target-time inference;
there is no claim that real computation fits a control deadline. State-only
has no access to a successor observation when generating its new chunk. Record
generation and visual/state observation indices, all seven actual old commands,
and actual takeovers. A task ending before t+7 censors that pending request,
not an invented completed oracle call. No simulator object poses control actions.

If (and only if) the selected L15/L15b student has passed its separately frozen
six-scene action test, add `student` with exactly those frozen weights: generate
at t using its forecast visual tokens and the same causal predicted state.
Do not deploy an ineligible student. This optional arm is decided and recorded
before opening any L16 scene; there is no selection using L16 outcomes.

## Reduce initial action noise without hiding reset differences

For each scene, before any outcome arm, perform the fixed30 hold setup steps,
reset that scene, and compute ONE real frozen-policy current-observation
bootstrap chunk with explicit flow noise. Store its tokens/state/geometry and
normalized/physical actions. Each arm then performs its own30 logged hold steps,
resets the same scene, checks exact initial geometry, and installs that identical
policy-generated bootstrap in the original queue. No demonstration or future
trajectory supplies those actions. This is a new common-bootstrap protocol,
not a rerun or replacement of old reset/independent-bootstrap cohorts.

The first27 commands must be equal across arms that reach the first takeover.
Report any measured pre-takeover state/object divergence, not just initial pose.
RGB remains stochastic; identical physical geometry is not pixel identity.
One shared simulator/model process serves the cohort; its cleanup is recorded
once globally, never claimed as a separate process exit for every arm.

## Fixed conditions and reporting

New scene seeds20270710--17, policy3610--17. Three arms give24 conditions, all
six permutations plus the first two repeated, shuffled3600. With an eligible
student, use four cyclic and four reverse cyclic orders (32 conditions), also
shuffled3600. All conditions run once. Same standard cameras640x480@30Hz,
CPU PhysX/GPU rendering, one tensor worker, feasible-before-commitment actions,
assets and original pre-reset task witness. No retraining or other project
model runs concurrently. The seed integers are identifiers, not calendar dates.

Keep the existing first-ten-tick slow plate-region occupancy endpoint, not a
contact-release test or the original three-orange-plus-rest task. Stop at that
endpoint/native terminal/120 simulated seconds. Include technical and task
failures with restricted cost120s. Do not replace failed scenes or retries.

Reference eligibility: state-only succeeds in>=7/8 scenes with zero technical
failures. Primary privileged contrast: oracle_visual minus state_only restricted
time; paired50000-draw bootstrap/RNG3601,95% upper bound below0, both four-scene
halves below0, no lower success count or higher technical failure count, and all
paired initial geometries/shared bootstraps match. Report oracle_joint in full
regardless of direction. If a qualified student participates, separately apply
the SAME contrast criteria to student versus state-only. Privileged-reference
success cannot be called learned-model success; no arm can qualify realtime.

Before the outcome cohort run exactly one60-step oracle_visual wiring smoke,
seed20270701/policy3601, without subgoal stopping. It must show two actual
target-time generations and takeovers at27/54, with no technical failure. A
failed smoke blocks this cohort; do not repeatedly smoke until one passes.
The reserved L15 tests remain sealed if development fails, even if an oracle
reference later performs well. All older studies retain their own decisions.
