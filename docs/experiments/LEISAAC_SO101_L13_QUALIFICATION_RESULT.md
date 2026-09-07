# L13 closed: actual causal state calls, but runtime qualification failed

Date:2026-09-07. Execution source `a021d964d61035aea24cdccce7fa94530968fcab`.
Six conditions completed exactly once. New binding uses immutable L6 visual
weights and immutable L12 state epoch29; the default visual-only engine still
passes its original behavior tests. Future state was supplied through the
existing model-ready override, never read from a future environment observation.

| Seed | Variant | Actual task actions | State calls | Visual calls | Qualification |
|---|---|---:|---:|---:|---|
|20270330|visual-only|27|0|1|fail: lost control slot|
|20270330|state-only|113|4|0|fail: lost control slot|
|20270330|joint|71|2|2|fail: lost control slot|
|20270331|joint|170|6|6|fail: lost control slot|
|20270331|state-only|600|22|0|pass at bounded endpoint|
|20270331|visual-only|256|0|9|fail: lost control slot|

Total1237 measurement actions plus180 setup actions;34 state-predictor calls,
18 visual-predictor calls. These are component counts, not52 independent
requests: joint requests contain one call to each component. The two joint runs
actually took over8 chunks; state-only took over26. One visual-only request
finished after its control failure, so10 visual-only calls produced9 takeovers.
No underflow, inference deadline miss or cap violation occurred. Five failures
were control-slot overruns, preserved as technical failures with all raw records.
All six simulator exits, sink closures and dispatch sequences were audited.

Worst environment-step intervals in the five failed runs were201.306,181.704,
113.846,158.898 and156.826ms. Thus the recorded tail is inside the simulator
step boundary, not simply an expensive additional state MLP. This does not
identify the unique physics/render/driver/host-scheduling cause.

After all project runs exited, three20:06:40–42 local snapshots still measured
39–44% GPU utilization and another Python compute process using4637MiB. A
read-only process-path inspection identified it as a different local project.
No other job was suspended, killed or modified. These post-run snapshots show
present resource sharing; they do NOT prove overlap during every failed tick or
establish sharing as the sole cause. Private other-project paths are not copied
into this public report. Resource snapshots and the accounting audit are saved
in `artifacts/m54l13_joint_runtime_qualification_v1/audit_report.json`.

**L13 qualification1/6, FAIL. The conditional40-trial real-time task cohort did
not start.** Its seeds20270410–17 remain unused. L12's offline mean gains and
failed per-case coverage gate also remain unchanged. Engineering failures are
not a task-success estimate and not proof that state forecasting harmed control.

Subsequent method research can use a separately defined simulated-delay control
experiment to isolate observation age from OS/GPU scheduling. Such an experiment
must not be presented as passing this real-time qualification, nor as a replay
or replacement of these six failed/bounded conditions.
