# L9: repeated bounded execution, including actual full-task capability

Date:2026-09-07. All L8 campaigns remain immutable. Both previously unresolved
warmed qualifications are now confirmed terminated/cleaned up (identity494,
predicted17 dispatched steps, each lost-slot failure); c4799309 was already
pushed. No stale process was resumed or silently replaced.

## Diagnosis

New GC-only instrumentation at2a9f6dbb completed600 identity steps and recorded
zero cyclic collections during its control window. A separate600-step native
profile atd2332f8f found its largest env.step46.621ms, including17.954ms in tensor
clone and10.021ms in camera buffer update. It did not reproduce the old140–164ms
tail, so there is no unique proven cause and GC was not disabled. These are
diagnostics, not additional outcome trials. The first command's mistyped option
was rejected by argparse before any environment was created.

For the single CPU-physics environment, simulator torch intra-op workers were
reduced6→1; model workers were already1. All sensor fidelity, dt,30Hz, CPU
PhysX/GPU RTX, native task witness, limits and feasible_v1 commitment semantics
remain fixed. Native cold initialization is handled by recorded warmed_v2 setup.

## Six-trial development cohort: five strict passes, not six

Source `8bf7ce0d6fd5c5581bb9bf428db3cb27993f0617`;
artifact `m54l9_single_tensor_thread_qualification_v1`.

| Env seed | Mode | Completed task actions | Outcome | Strict qualification |
|---|---|---:|---|---|
|20261222|identity|1200|bounded finish|pass|
|20261222|predicted|1007|native full-task success|pass|
|20261223|predicted|1200|bounded finish;1 late whole-discard|fail|
|20261223|identity|1200|bounded finish|pass|
|20261224|identity|736|native full-task success|pass|
|20261224|predicted|1104|native full-task success|pass|

Total6447 actions,126 planned predictor calls. All completed without lost control
slots or queue underflow; the predeclared zero-deadline requirement fails once.
For that request, stationary startup estimated132.015ms/d5, actual concurrent
latency was180.051ms, one step late. The existing queue correctly whole-discarded
the late chunk and continued control; this is not a control-slot loss.

The native successes use the original three-oranges-plus-rest predicate, captured
before automatic reset, not the first-placement subgoal. For predicted20261222,
all three objects also had ten-tick stable-region evidence and zero measured
terminal linear velocity. These outcomes establish actual online task capability
in some scenes, not robust task efficacy or causality versus identity.

## Calibrated operating profile: all six strict qualifications pass

Source `f2ebc4407598c821c27882766e8a25313aecc6f9`;
artifact `m54l9_delay_floor7_qualification_v1`.

An explicitly new common minimum planned delay7 uses observed concurrent cost
plus the existing margin. Min1 remains the default, cap8 and raw-cap failure
handling are unchanged. This schedules takeover233ms ahead; it does not slow
control, skip physics or artificially sleep to create latency. Identity uses the
same delay floor, so future comparisons cannot attribute that change to predictor
quality. No old run was overwritten or rerun to rescue its gate.

New env seeds20261225–27/policy2425–27, both modes, six runs of1200 control ticks:

* **6/6** complete the40-second bound;7200 executed actions,132 planned predictor
  calls. Zero lost slots, underflows, inference deadline misses or cap violations.
* Maximum tick-start lateness across all six:20.980748ms. All three predicted
  runs' forward P90 are2.73749/2.73439/2.74440ms. Per-step overruns remain logged;
  this is not a zero-jitter or indefinite-duration guarantee.
* No native full-task success before the40-second bound in this cohort. All six
  outcomes are censored, not task failures at a full episode timeout.
* All worker/metrics/simulator cleanup checks pass. Raw timestamps, per-step
  witness, projection counts, calls and takeover rows are in each artifact.

This qualifies bounded engineering execution for a new independent study on this
machine, not robust full-task competence. There is no guarantee the same hardware
will meet deadlines under arbitrary unrelated desktop load.

## Next scientific test

Prepare L10 action-aware predictor distillation on new train/validation trajectories
with a frozen base policy, then open fresh action-test scenes only after selection.
L6's/L8's already opened test episodes are not used for any fitting. L10's code
was prepared in an isolated worktree during qualification (no concurrent GPU
training), then integrated after all six qualifications terminated. The new
objective,400-update bound, selection criterion and independent test gate are in
`LEISAAC_SO101_L10_ACTION_DISTILLATION_PLAN.md`. It does not inherit a task-benefit
pass from either these native successes or the earlier latent metrics.
