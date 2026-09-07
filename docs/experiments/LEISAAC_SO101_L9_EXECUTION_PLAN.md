# L9: repeatable execution before another efficacy campaign

Date: 2026-09-07. User authorization continues. Source on entry:
`c4799309cae25783e87eec0fa85ebf25fc3b32c7` (local and origin agree).
No project simulator/evaluation process remains. Both L8 warmed qualifications
actually finished: identity failed at494 dispatched steps and predicted at17;
both cleaned up. The predicted warm startup probe passed (143.310ms, predictor
1.719ms), so the remaining failure is not its cold predictor kernel.

## Evidence and decision

L8's two warmed failures contain isolated env.step intervals of164.418ms and
140.470ms, whereas median full ticks were23.912ms and22.140ms. The predicted
failure precedes its first planned inference. Diagnose the simulator's sporadic
pause, rather than change horizon/FPS, rerun the old task comparison or retrain
the predictor to explain an unrelated process pause.

Add observational Python GC callback timestamps, without changing collection
behavior, to a new warmed identity development run (env20261221/policy2421,
600-step bound, feasible_v1, native task witness, standard cameras). A technical
failure stays a failure. An automatic collection overlapping a slow tick would
justify a bounded control-only collection scheduling intervention; absence of
overlap instead requires another measured cause. Callbacks are diagnostic
instrumentation, not a new scientific trial or universal causal proof.

Reference API: Python3.11 `gc.callbacks`, `gc.disable`, `gc.enable`, `gc.collect`
are documented at https://docs.python.org/3.11/library/gc.html. Cyclic collection
is distinct from reference counting. No permanent disabling, unlimited episode,
or claim of a hard real-time Python runtime is proposed.

After a supported fix, freeze new qualifications before dispatch: both modes,
multiple new development seeds, the same existing pacing and failure criteria,
bounded memory/cleanup evidence. Only repeatably qualified execution may advance
to a new outcome comparison. Runtime qualification and scientific benefit remain
separate; L8's36 outcome trials and216 action cases stay closed and immutable.

All existing policy/predictor weights, camera fidelity, dt,30Hz, action limits,
feasible_v1 prefix semantics and late whole-discard rules remain unchanged during
this investigation. No external Pro review, new installation or new license
acceptance is needed. New learning would require a separately frozen train/val/
test protocol using new development trajectories, not either prior opened test.

## First diagnostic outcome and narrower next measurement

`m54l9_gc_trace_identity_v1` at `2a9f6dbb` completed600 steps, but recorded zero
GC callbacks during control. Its longest tick59.260ms contained56.636ms in
env.step with no collection. This does **not** support disabling GC. The command
initially used a wrong option name and exited at argparse without constructing
an environment; the corrected invocation is the sole physical run.

Next: a non-real-time native-call profile with the same development
env20261221/policy2421, warmed_v2, native task witness, feasible_v1 and600 steps.
Retain the five slowest per-step profiles, not only a mean that hides isolated
pauses. This is a changed diagnostic, not a retry of an outcome trial. No
timing qualification can be inferred from profiler-instrumented execution.
