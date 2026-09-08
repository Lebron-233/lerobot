# Matched original-controller diagnostic after the failed single-step screen

## Execution status: not started

Source `697099645e0b87c3228ab1ea625f1b6f32072703` was committed and pushed.
The preregistration POST returned comment 5586127789, created 2026-09-08T13:45:08Z.
The required subsequent body-readback command was blocked by the platform's
safety-state check before execution. No alternate verification route was used.
The control worker was never launched: zero started tuples, zero observed control
outcomes, and the planned control output directory does not exist.
The protocol below remains unexecuted. This status update is documentation only,
not a new execution source or an authorization to bypass the readback prerequisite.
See [the execution status record](LIBERO_SINGLE_STEP_MATCHED_CONTROL_STATUS.json).

## Frozen diagnostic protocol

Date: 2026-09-08. Candidate screen closed at 64/90; no control-condition outcome
on these tuples has yet been opened. Under the user's continued experiment
authorization, the project lead approves this complete matched diagnostic.

## Why this is the next experiment

The fixed 50/1/1 candidate failed its independent native screen. This verdict is
final. Comparing its 64/90 to the old 185/200 cannot isolate the sampler because
the cohorts use different starts and noise seeds. Complete the original 50/1/10
condition on **all the same 90 tuples**, including all 64 candidate successes and
26 failures. Do not choose only failures, weak tasks or a favorable state subset.
This is one existing reference condition, not a search over new candidates.

## Frozen identities and execution

Candidate input: `outputs/libero_single_step_native_ee273bce/`, source
`ee273bcecc89fb2c6fc1b092b2770d134ec32085`. Require its complete closed 90-slot
summary, actual one-step registration and canonical tuple records, not a passed
screen. Read its existing outcomes only; never rerun its policy or overwrite them.

Control: the exact original ten-step sampler with the same frozen policy/VLM,
saved processors/precision, full dual-camera encoding and environment lock.
Keep 50 generated actions / 1 consumed / 10 denoising evaluations, relative Panda
OSC_POSE, 20 Hz simulation, hard reset, ten settling actions, then policy seed,
280 measured actions, native success and unchanged cleanup/raw recording.

All ten Object tasks, task_order_index=0. On each task use state IDs 41-49, in
ascending order. Environment seed 940000+100*task+state; policy seed
950000+100*task+state: **identical to the candidate**. Rename only the condition's
tuple_id prefix to matched_ten_step_reference, not any state, task or seed.
Each control tuple runs once. No old 0-40 outcome is opened and no old confirmation
is repurposed. There is no new checkpoint or parameter candidate.

Commit/push this source and protocol, publish/read back the exact execution SHA,
output path and command before first control rollout. Keep source and dependencies
fixed throughout. Ordinary task failures do not trigger early stopping. A technical
failure stops the worker, preserves the open record and later not_run slots, and
prevents complete-pair inference. No retry or replacement is authorized.

## Prespecified diagnostic analysis

Reconcile native returns, actual ten projection calls [1,50,32] each, empty action
queue, observations and cleanup using the already tested reference auditor.
Compare each pair's saved initial state, quaternion and both actual raw cameras
directly. This tests whether the recorded initial observations really match rather
than merely trusting matching seed labels. Keep all comparisons and mismatches;
do not drop a mismatched pair to produce a nicer statistic.

Only complete, technically valid 90-pair data with all initial observations equal
yield the registered paired statistics. Report all four outcomes: both successful,
single-step-only successful, ten-step-only successful, both unsuccessful. Primary
contrast is single-step success rate minus ten-step success rate. Report per-task
9-pair counts, exact two-sided McNemar descriptive p, and a 20,000-draw paired
bootstrap of differences resampled within each fixed task, seed970001.
Do not independently bootstrap the two arms. A zero-discordance table has p=1.

These are **post-screen matched diagnostic statistics**, not an independent
confirmatory test: the candidate outcomes and its failed screen motivated this
follow-up. Do not claim unconditional prespecified type-I error control or general
noninferiority. All control outcomes remain unopened until this protocol is frozen.
Task-state matching permits a substantially more direct comparison on these cases
than the old unpaired 200-vs-90 comparison, but is not a population theorem.

Report policy timing distributions and per-condition wall separately. Conditions
are run in two temporal blocks, not interleaved, and may have different trajectories
and episode lengths; do not turn their timing ratio into a randomized device-speed
estimate. The earlier alternating synthetic compute comparison remains separate.

## Decision and limits

This diagnostic cannot rescue the 64/90 screen or grant a new qualification from
the original controller's control-arm score. No hypothesis-test result automatically
starts training, asynchronous deployment, new sampling search or an old confirmation.
If the ten-step arm succeeds where one-step failed without corresponding reverse
wins, preserve the original sampler for the research reference and reject direct
one-step replacement. If both arms fail similarly, do not blame denoising alone;
retain the fixed-state difficulty explanation and report the actual paired evidence.

All existing baseline_qualified/realtime_qualified/predictor_benefit_tested flags
remain false and risk_thresholds=null. The eventual M3 question is whether future
context compensates asynchronous observation age without sacrificing the original
action generator's task quality; this experiment is not that benefit test.
