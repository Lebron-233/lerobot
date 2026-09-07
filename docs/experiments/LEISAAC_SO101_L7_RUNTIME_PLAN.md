# L7: frozen SO101 predictor in the production asynchronous loop

Date: 2026-09-07. Starting source: `11e8c5a9258a7b5013ee07c440994c24476d397c`.
User requests continued local feasibility validation through DevSpace, with GitHub
records. Prior broad project authorization and operator NVIDIA acceptance remain.

## Question and frozen inputs

L6 tested forecasting offline, not application in the production queue. L7 tests
whether the selected predictor actually runs on committed prefixes and supplies
chunks to real action-dependent SO101 physics while the old queue keeps moving.
This is an engineering feasibility cohort, not a task-success benefit experiment.
The base policy has not yet qualified for the latter; that negative evidence stays.

Use WSAGI `c8c3318dba152b0ba671ff07b4314418d5aa4b4a`, its native processors,
and L6 `m54l6_predictor_training_v1/best.pt` (epoch 3, training source
`40e00e1654779d5e6bfec8f2f06ab57a6c697ff1`, 273,824 parameters). No fitting,
new threshold, checkpoint selection or reopening L6 test trajectories is included.
The old SO100 frozen loader and public rollout validators remain unchanged.

## Two implementation seams

1. Introduce a narrow candidate-validation method on the existing engine. Its
   default remains the old frozen SO100 check. An explicitly named SO101 subclass
   validates the new policy/processor instance association and L6 predictor
   provenance; it supplies the native front/wrist policy camera keys. The queue,
   worker, startup probe, delay planner, residual insertion and whole-discard are
   the same production implementation, not a duplicate or spoofed old candidate.
2. Align pacing with the same absolute origin already used for deadline checks.
   Readback of the prior 221-step run shows mean work 24.999 ms but start lateness
   accumulated to 29.271 ms. `CycleTimer.wait()` currently re-anchors each period;
   short subsequent steps cannot repay ordinary late work. The final 50.016 ms
   step then crossed the original lost-slot threshold. Add an optional absolute
   deadline to that existing timer and use `origin+(tick+1)/30` in this adapter.
   Default timer users are unchanged. Every tick still performs exactly one
   notify/get/step, and both existing full-slot rejection thresholds stay intact.
   This is a pacing correction, not deletion of jitter, ticks or missed slots.

Targeted tests: old candidate rejection still works; mismatched SO101 weights or
processors are rejected; d>0 has native-camera tokens, current normalized state
and normalized committed actions; d0 does not invoke predictor; late chunks remain
whole-discard; absolute pacing repays bounded lateness without skipping steps and
still rejects a true lost slot. Tests affected by the shared engine seam are run.

## Execution, fixed before live results

After clean-source tests and push, use CPU PhysX, GPU RTX/model, standard camera
with nominal reset anchor, original assets and joint limits, zero initial pose,
two RGB 640x480 streams at 30 Hz, sim dt 1/60 and decimation 2. Task text and success
predicate remain unchanged. Native60 and synchronous shortened chunks are excluded.
Startup plus fresh reset-state bootstrap remain outside control timing and recorded.
q=.9, margin=1, guard=2, max_late_steps=2, horizon=8, risk thresholds=null.

First run one new 600-tick identity engineering trial (environment seed 20260930,
policy seed 2001), followed by the same seed in predicted mode. If both finish
without technical failure, run reversed order on seed 20261001 / policy seed 2002,
also 600 ticks. All four attempts are retained; early failures are not silently
rerun. An interface or pacing defect may be fixed with a documented cause and new
source/output namespace; repeat seeds then denote development, not independent tests.

Primary engineering evidence: completed physical dispatches, actual planned
requests and predictor calls, takeover events, normalized-prefix provenance,
control start/work lateness, inference total/delay, underflow, stale/late/cap events,
terminal accounting and clean shutdown. A success means the configured bounded
runtime worked, not universal real-time performance or a statistical task benefit.
The latest two engineering seeds are not used to train or select the predictor.

If the four short trials pass, freeze a second bounded run of 1,800 ticks per mode
on a new seed, in GitHub before launch. A hardware ceiling is reported honestly;
do not reduce FPS/resolution or suspend physics while calling it real time.

All hot-path telemetry remains in memory. Fresh directories bind each source
commit; files materialize after worker join and environment close. Existing L4–L6
and old M3/B4/M5 results are retained unchanged. No physical robot is involved.

## Implementation checks before first live run

209 targeted tests passed across the production engine, cycle timer, existing
SO101 contract/matched adapters and new SO101-predicted binding. This includes
the default SO100 rejection and prior d0/late handling cases, plus a real-worker
fixture demonstrating that a nonzero predictor residual reaches a staged chunk,
uses normalized committed actions, and takes over at the planned index. No old
scientific experiment or held-out cache was rerun. Formatting-only fixes followed.

## Short cohort completion and longer run frozen before execution

All four predeclared 600-tick trials completed under clean source
`09b9fb3f992b9102778683819ea66153ecb150e8`. Each produced 22 planned requests;
the two predicted trials actually invoked the L6 predictor on their planned
requests. All report zero underflows, cap exceedances and inference deadline
misses, with no full-slot technical termination. Individual slow work intervals
remain in the raw telemetry; bounded timing is not zero jitter.

Now freeze an extended engineering pair: environment seed 20261002, policy seed
2003, predicted then identity, 60-second environment horizon / at most 1,800
control ticks per mode, same configuration and unchanged weights. Each mode is
run once. If one has a technical failure, retain it and run the other mode for
the same diagnostic contrast; do not rerun the failed arm to obtain a pass.
The two complete task terminal flags will be reported, but this one seed is not
an efficacy/success-rate experiment. No result-driven tuning is included.

Same-seed initial state, object locations and camera poses matched in the first
short pair, but RGB was not bitwise identical: first-frame uint8 MAE was 0.881
(front) / 0.986 (wrist). Its bootstrap actions already differed before any
predicted takeover. Thus entire trajectory differences cannot be attributed
solely to prediction; these are nominal-seed engineering pairs. We do not replace
real observations with stored images or modify renderer settings to erase this.
