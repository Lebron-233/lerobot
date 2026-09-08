# Ten-step graph equivalence on recorded native observations

Execution update: source5953905f completed81 exact measured comparisons for tasks0-2,
then stopped technically at the next task's incompatible graph input. The remaining
189 have no complete comparison. The subsequent task-boundary repair is implemented
but unrun because its preregistrationPOST was blocked. See
[the consolidated result](LIBERO_TEN_STEP_GRAPH_RESULT.md); the frozen protocol below
is retained rather than relabeled as270 successful comparisons.

Date: 2026-09-08. The repaired synthetic graph profile at84e38259 completed with
all exact-equality checks, including all ten original projections. Its formal
means are219.914ms eager and63.652ms graph (24pairs); all24 graph samples remain
above50ms. The first failed capture atf3894bc4 is preserved. An extra arithmetic
summary command was blocked after the raw result had already been successfully
read; it was not retried. The measured raw timing fields support the figures above.

## Purpose and fixed source selection

Validate numerical equivalence on actual image/state observations rather than
only synthetic shapes. This is a read-only replay of observations, NOT a native
task retry, new closed-loop controller comparison or training/tuning exercise.

Use only the original closed one-step cohort at
`outputs/libero_single_step_native_ee273bce`, execution source
`ee273bcecc89fb2c6fc1b092b2770d134ec32085`. Its complete90 tuples comprise every
task0-9 and state41-49. For EVERY episode use observation0, floor(measured_actions/2),
and the terminal observation. Three samples per episode,270 total, no success/failure
or task filtering. Preserve source metadata and actual indices in the output.
Never read the separately blocked matched-control result or its trajectories.
No old state0-40 outcome or confirmation sample is read.

Reconstruct the ordinary policy input from lossless saved cameras, recorded EEF
position, original quaternion and gripper qpos. Require the reconstructed8D
model-input state to equal the previously saved state exactly before preprocessing.
No simulator or environment is created. Terminal frames are simply model inputs,
not instructions to resume a terminated episode or new success labels.

## Frozen algorithm and comparisons

Keep the repaired ten-step sampler and experiment-local GraphSampler from84e38259,
same checkpoint/VLM/processors/precision,50/1/10 and fresh image encoding on EVERY
call. Do not change graph regions or numerical settings using these observations.

Capture once from the first source observation. Five paired warmups use that
observation and seeds989990..989994. Then process270 observations in task/state/stage
order, paired equal seed990000+index, alternating eager/graph-first. Reset policy
and processors each call. Copy all graph inputs, including language,state,noise,
masks and fresh visual tokens. Generate fresh noise and include vision, input
copies, selector,postprocessing and CUDA completion in the main timer.

Compare full[1,50,32] chunks, selected normalized[1,7] actions and postprocessed[1,7]
commands with exact equality. Stop at the first mismatch, preserve it, and do not
relax the tolerance. Save full raw output arrays and each pair's timings/indices.
Report all270 pairs and432000 padded scalar comparisons, means, empirical nearest-rank
P95/ranges, count>50ms, and fresh encoding/replay counts. Partial completion is not270.

Commit/push source and this protocol, then publish/read back exact execution SHA
and a fresh output path before running one process under a300s wall bound. No
automatic retry on technical failure and no favorable-sample replacement.

## Interpretation

Passing extends evidence from four synthetic fixtures to270 recorded native
observations across all ten tasks. It does not establish universal functional
equivalence, native success of graph-driven rollouts,20Hz deployment or asynchronous
predictor benefit. This is not a validation set for choosing numerical hyperparameters.
No new controller outcome is generated, and no old failed qualification is changed.

baseline_qualified=false; realtime_qualified=false; predictor_benefit_tested=false;
risk_thresholds=null. The matched-control final result remains blocked and unread.
