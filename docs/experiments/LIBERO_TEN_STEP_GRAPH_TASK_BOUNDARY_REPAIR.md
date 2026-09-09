# Recorded graph validation: task-boundary static-shape repair

## Execution status

2026-09-09 continuation: the existing repair was executed at36597f04 after
preregistration5594244474 and exact readback. All270 real recorded inputs passed
full-chunk/selected/post exact equality; process exited0. Language lengths were
11/12/13 across tasks. See [completed result](LIBERO_TEN_STEP_GRAPH_RECORDED_RESULT.md).
The paragraphs below retain the earlier, unstarted publication attempt as history.

Repair sourcebfaeb0de was committed/pushed. Duplicate lookup for its registration
marker found no comment. The subsequent preregistrationPOST was blocked by the
platform before execution and returned no publicationID. No alternate publication
route was used and no repaired validation was launched. The observed partial
5953905f result below remains the only recorded-input execution.

Date: 2026-09-08. The first recorded-observation validation at5953905f is closed
as a technical failure. Output `outputs/libero_graph_recorded_5953905f/` is retained.
Its command exited1 after63.663s. All81 measured pairs for tasks0,1,2 were exact
on full padded chunks, selected normalized actions and postprocessed commands.
No numerical mismatch was observed. The next task's first graph call was rejected
by the fixed-input shape/dtype/device check before a completed pair was recorded.
The remaining189 pairs are unmeasured, not observed numerical failures.

## Concrete repair

A single fixed-address graph is not a supported representation of arbitrarily
different task input shapes. The original four synthetic fixtures did not expose
this boundary. Keep the shape check and all original preprocessing, rather than
silently padding/truncating inputs or relaxing equality.

For this fixed ten-task observation protocol, explicitly recapture once when the
task ID changes, from that task's first recorded observation. Reuse that graph
for all27 observations within that task. Record all eight input shapes, ten
projection shapes and setup duration for each of the ten task captures. This will
also identify which actual input dimensions differ across the task boundary.
No general graph cache, shape search, modified attention, new precision or sampler
is introduced. This is a known task-boundary preparation step, not a measured
steady-state speedup; capture/setup is reported separately and cannot be omitted
when budgeting task-switch/cold-start latency in a future runtime.

## Separate fixed validation

New source and fresh output require a new preregistration before execution.
Run all270 original observations again as a new engineering validation, retaining
the complete first failed run. This reruns model-input comparisons, not native
episodes. No task success is generated and no original raw record is overwritten.

Same source one-step cohort, all90 initial/midpoint/terminal samples, seeds990000+i,
alternating measurement order, same three paired output comparisons and finite/queue rules.
Five initial paired warmups are unchanged; each task's new capture uses the same
three internal side-stream warmups already in GraphSampler. A separate eager setup
call seeds989900+task supplies that task's capture input. Preserve all10 preparation
costs separately. Keep the300s bound and stop at the first numerical mismatch or
within-task incompatible input. No dynamic fallback or tolerance change.

This concrete scope repair does not permit access to the separately blocked
matched-control summary or its observations. baseline_qualified=false;
realtime_qualified=false; predictor_benefit_tested=false; risk_thresholds=null.
