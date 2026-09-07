# L10 development closed: actionable gradients, insufficient case coverage

Date:2026-09-07. No new held-out scene or test report has been opened. The
reserved test seeds20270130–35 remain unused; no further fitting is queued.

## Actual data and training

Eight new development trajectories each completed600 actions/601 observations:
4800 actions,4808 observations. Split6train/2validation,1152 eligible training
anchors/delays and72 validation cases. Projected normalized committed prefixes
stay within a known50-action chunk; targets are actual future visual tokens.
Only the273824-parameter residual predictor was optimized. The WSAGI policy,
VLM, action expert and checkpoint statistics remained frozen. A real backward
through the frozen decoder produced finite nonzero predictor gradients and zero
policy gradients. This is a working action-supervised training path, not a
synthetic loss or surrogate claim of task success.

| Candidate | Source | Fixed updates | Selected update | Validation action L1 | Improving cases |
|---|---|---:|---:|---:|---:|
|Identity reference|same paired validation|0|—|0.097926008127|—|
|Immutable L6 parent|same paired validation|0|—|0.098411329310|29/72|
|L10 absolute action loss|13c27554a747defabc0b7633a277f4248bd59aba|400|300|0.089691894517|37/72|
|L10b case-balanced loss|6cdc937a53347572a4a478ac2e9800c2b8365fb2|800|800|0.086915314068|38/72|

Absolute student mean improves8.4085% versus identity and8.8602% versus parent;
one validation episode improves20.2927% but the other worsens4.3605%.
Balanced student's mean improves11.2439% versus identity and11.6816% versus
parent; episode means improve21.2431% and0.5002%. Still only38/72 cases improve.
Maximum balanced checkpoint coverage was42/72 at update700; none reached48/72.

The absolute candidate satisfied its original minimal validation permission
(>36/72 cases), which remains recorded true. Before collecting any test, its
test was explicitly deferred and a separately frozen balanced-development trial
was run. L10b required48/72 improving cases to open the test; it failed. Do not
retroactively weaken that requirement or use the test to pick between models.
The two candidates also differ in update budget, so their difference is not an
isolated ablation of objective weighting alone. Validation was reused for
development and checkpoint selection; none of these percentages is a blind result.

First real predictor gradient norms were0.2950035(absolute),6.6280589(balanced),
policy gradients0 in both. Training wall times262.468s/592.777s; peak allocated
CUDA memory1651332096/1651334144bytes. These exclude simulator collection and
are not inference benchmarks. Every scheduled update and validation checkpoint
is preserved, including the worse intermediate results.

## Artifacts and decision

Under `/home/rp/Workspace/SmolVLA_RTC/artifacts/`:

* `m54l10_development_v1`: new train/validation data and complete dispatch logs;
* `m54l10_action_training_v1`: absolute objective, all400 updates, selection300;
* `m54l10b_case_balanced_training_v1`: balanced objective, all800 updates,
  diagnostic best800, `validation_qualified=false`;
* no `m54l10_test_v1` directory or either training directory's test_report exists.

The predeclared stable action criterion is not established. Keep both students
offline, do not silently plug them into the L6-specific runtime binding, and do
not claim stable action or task benefit from these development means.

Separately, repeated L9 execution qualifies a new L11 task cohort using the
unchanged L6 parent and newly frozen scenes. That study tests actual task effects
under common calibrated deployment; it does not deploy an unqualified new student
or inherit efficacy from this learning experiment. No concurrent GPU fitting
will run during those task trials.
