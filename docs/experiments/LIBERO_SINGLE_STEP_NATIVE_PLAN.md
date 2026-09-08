# Fixed single-step candidate: independent native capability screen

Date: 2026-09-08. Registered before any new candidate task outcome is opened.
The user has authorized continued implementation and experiments. As project lead,
I approve the bounded task experiment below, not an unregistered continuation of
the old qualification. Previous 185/200 and task-5 14/20 results remain closed;
their confirmation states 21-40 remain untouched.

## Question and identity

Does the previously fixed 50/1/1 candidate exhibit broadly reliable native task
completion on independently reserved Object initial states? Its measured synthetic
compute gain of 4.12x is not evidence of task preservation. This experiment measures
native success and actual task-path policy timing, not async compensation.

Use the same strict checkpoint, processors, precision, 256x256 dual cameras,
native relative Panda OSC_POSE at 20 Hz and ten settling actions as the frozen
reference. Only the already selected Euler denoising count changes from 10 to 1.
Keep chunk 50, consumption 1, hard resets, 280 measured actions, native BDDL success,
saved normalization and the existing dependency/environment lock.

Policy: `HuggingFaceVLA/smolvla_libero@6721902bc4d61e50a3bfdb11dfb4cb626f05d102`.
VLM: `HuggingFaceTB/SmolVLM2-500M-Video-Instruct@7b375e1b73b11138ff12fe22c8f2822d8fe03467`.
Assets: `lerobot/libero-assets@0b3ea86be5fe169d0fd036ae63d1070ec09e90f6`.

The candidate was selected before these outcomes from the earlier independent
[same-checkpoint archive](https://github.com/zuoxingdong/smolvla-libero-eval/tree/114d19c51bf7655e61fb994f3b344a0257ceb20b).
That archive concerns Spatial, not this Object suite. Its reported success is not
used as local evidence; no other hyperparameter or simulator-version search is run.

## Entire cohort and immutable sampling rule

All ten native LIBERO-Object tasks in the original task_order_index=0 order.
On every task use all remaining unused official initial-state IDs
`[41,42,43,44,45,46,47,48,49]`: 9 distinct starts per task, 90 total tuples.
Task order 0-9, then ascending state ID. One attempt each, one environment at a time.
Environment seed `940000 + 100*task_id + initial_state_id`; policy/noise seed
`950000 + 100*task_id + initial_state_id`, set after hard reset and settling.
The committed runner's registered_tuples() fully enumerates these 90 identities;
registration.json records that full list before worker launch.

IDs 0-40 are excluded, including the old unopened conditional confirmation.
These states are disjoint from local preparation/development; their original
training-demo overlap remains unknown. No favorable task filtering, replacement
state, retry, new noise draw after a failure, or 280-step extension is allowed.

## Decision fixed before execution

This is a capability screen, not the original two-by-200 qualification.
`native_screen_passed=true` requires all 90 completed, zero technical failure,
at least 81/90 native successes overall and at least 8/9 on every task, all records
reconciled and worker exit 0. These are the integer thresholds for the same nominal
90% overall / 80% per-task project requirements at n=9; 8/9 is 88.9%, so the
per-task count threshold is conservative, not a reduction to 7/9.
No source or parameter changes while this cohort runs. Ordinary failures do not
stop execution early; technical failure stops the worker and preserves later slots
as not_run. A blocked tool operation before launch is not an episode failure.

Regardless of screen outcome, baseline_qualified=false, realtime_qualified=false,
predictor_benefit_tested=false and risk_thresholds=null. This screen cannot grant
full qualification or automatically start confirmation, training, or async trials.
If it fails, close this candidate screen without a parameter sweep. If it passes,
it establishes a useful independent native-task milestone, not noninferiority to
the old controller. A matched comparison or separately registered confirmation
would be required for stronger claims; old confirmation is not repurposed here.

## Measurement and execution

Reuse reference.run_episode and reference.audit_tuple with an explicit expected
denoising_steps=1. Their default remains 10 for the old runner. Both paths check
actual action_out_proj count, [1,50,32] projection shapes, empty action queue,
current observation/action alignment, native transitions and cleanup.
The entire frozen model is loaded strictly before the in-memory num_steps override.
Checkpoint/config files are never written; restore in-memory num_steps at exit.

Retain lossless dual cameras at reset/each measured step, raw/normalized states and
actions, first native success, terminal/truncation flags, settling/native call
counts, cleanup, worker command/exit and policy selection timings. Native success
at action 280 still counts. Finite out-of-box commands retain the existing native
clipping/scaling, not an added software clamp. Nonfinite output is a technical fault.

Report per-task success/9 and Wilson 95% intervals (actual observed n for incomplete
tasks); macro success and fixed-task stratified 20,000-draw bootstrap with seed
960001 only if all 90 are observed. Do not convert unrun slots to observations.
Success time is first-success action/20; observed non-success time is 14 seconds.
Report actual policy timing distribution and wall separately; no simulator pause
time is claimed as real-time control and no comparison to old unpaired timings is
presented as a matched causal speed estimate.

Before launch, commit/push this protocol, runner and targeted tests, publish exact
execution HEAD, output directory and command to Issue #1, and read it back.
Use the existing dedicated Python via uv when available; this shell has no uv,
so the documented dedicated interpreter is used directly with no installation.
Keep existing offline EGL variables and process-local system GLVND preload.
GPU experiments run to completion in this conversation, with no automatic restart.
