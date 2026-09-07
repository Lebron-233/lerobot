# L8 independent validation: first complete task, no stable benefit established

Date: 2026-09-07. This report preserves the L6 and L7 evidence and records the
completed L8 qualification, frozen task comparison, and independent action audit.
The subsequent warmed-readiness investigation is not a replacement for those results.

## Verdict

The frozen WSAGI policy completed the original three-orange-and-return task once,
with a witness captured inside the native termination evaluation before automatic
reset. This establishes actual task capability in one observed case, not reliable
full-task performance. The subsequent independently registered first-placement
comparison and larger action audit do **not** establish stable predictor benefit.

## Native task and qualification

The initial witness implementation read the original configuration object while
Isaac's manager executed a deep copy. Its 3,600-step timeout remains recorded, but
its null witnesses cannot support a pre-reset placement analysis. Commit
`d7a9e724af0459de067e958596b62298571e1a1d` binds the live termination-manager term;
a two-step simulator check exercised the corrected witness.

Three strict, synchronous development runs then produced:

| Environment seed | Completed actions | Result |
| --- | ---: | --- |
| 20261011 | 1,008 | Native full-task success; all three oranges settled before reset |
| 20261012 | 28 | Shoulder target -101.717094 degrees rejected before dispatch |
| 20261013 | 2,058 | Wrist target 95.037186 degrees rejected before dispatch; one orange settled |

The successful native witness retained all three object positions, zero linear
velocities, and the robot joint state. All three objects had satisfied the separate
sustained-placement tracker over the preceding ten ticks. Neither a scripted
controller nor a state teleport was used to create that policy result.

An explicitly different `feasible_v1` action contract subsequently projects native
motor coordinates into their unchanged limits **before normalized queue commitment**.
Both the policy-space prefix and executed postprocessor output therefore describe
the projected actions. The same projection is used in all comparison arms. This
changes execution semantics and is not a retroactive correction of the strict runs.

Under that contract, six qualification seeds 20261014–20261019 each completed
3,600 synchronous actions without a technical termination. None completed the
original full task. Five reached one sustained settled placement; no run reached
two simultaneously settled oranges. Seed 20261017 projected ten components, with
maximum native adjustment 3.1442566; the other five projected none.

The source-bound qualification accounting is retained in
`artifacts/m54l8_baseline_qualification_summary.json`. Its 28,294 executed actions
include the initial witness-defective run and two-step witness check; these are
not additional successful qualification episodes.

## Frozen first-placement comparison

Source: `bec8cc944169e9f70edc29acb61e1e806ab3e62a`.
Protocol: `LEISAAC_SO101_L8_TASK_VALIDATION_PLAN.md` and
`run_so101_task_comparison.py`. Artifacts:
`artifacts/m54l8_first_placement_comparison_v1/`.

Twelve independently specified seed blocks, 20261101–20261112, each ran sync,
identity, and predicted. Mode order was balanced and shuffled before execution.
The separate endpoint was the first sustained settled placement, not original
three-orange task success. Technical failures and timeouts retained the full
120-second restricted-time cost; no failed arm was replaced or dropped.

| Mode | Trials | First-placement successes | Technical failures | Mean restricted time (s) |
| --- | ---: | ---: | ---: | ---: |
| Sync | 12 | 8 | 0 | 55.144444 |
| Identity | 12 | 1 | 11 | 110.452778 |
| Predicted | 12 | 0 | 12 | 120.000000 |

Predicted-minus-identity paired restricted-time difference was +9.547222 seconds;
the registered paired bootstrap 95% interval was [0, 28.641667] seconds. The
registered stable-task-benefit gate failed. Startup failures also mean that some
pairs have no measured initial observation and cannot pass geometry matching.

These data measure this deployment's overall availability and task execution,
not the intrinsic effect of predicted visual context among healthy executions.
Several predicted startup probes spent approximately 87–88 ms on the first
predictor forward and required nine delay steps, exceeding the unchanged cap of
eight. Other technical terminations were control lost-slot failures. The full
per-trial errors, timestamps and outcomes remain in the artifact summary and logs.

## Independent 216-case action audit

Source: `bec8cc944169e9f70edc29acb61e1e806ab3e62a`. Artifacts:
`artifacts/m54l8_independent_action_audit_v1/`.

The fixed L6 epoch-3 model was evaluated without fitting on six new real simulator
trajectories, seeds 20261130–20261135. Twelve predetermined anchors per trajectory
and delays 1, 4, 8 produced all 216 registered cases. Current state and language
were fixed, flow noise paired, and the future visual oracle was an offline
reference only. Prefixes came from already committed feasible actions.

The primary metric is normalized-action L1 over the first 25 generated actions,
with the same feasible projection applied to identity, predicted and oracle.
It is not identical to L6's earlier 18-case, full-chunk reporting condition.

| Metric | Result |
| --- | ---: |
| Identity action L1 | 0.09352347693250825 |
| Predicted action L1 | 0.09164213376223213 |
| Aggregate error reduction | 2.0116266332% |
| Improving cases | 112 / 216 |
| Improving episodes | 5 / 6 |
| Episode-bootstrap 95% reduction interval | [-6.4659605%, 8.8827110%] |

Per-episode reductions were -16.914734%, 3.059953%, 14.829486%, 0.353124%,
5.575872%, and 2.842023%. The registered gate required at least 144 improving
cases and a positive lower confidence bound, as well as episode coverage and
completeness. It failed. Nominal mean improvement must not be reported as stable
action benefit. These newly opened audit episodes are not available for subsequent
model selection, threshold tuning, or repeated confirmatory testing.

## Follow-on warmed-readiness status

The completed negative experiments above remain frozen. A separate protocol in
`LEISAAC_SO101_L8_WARMED_VALIDATION.md` records fixed preparation steps and cold
kernel preparation without relaxing measured control deadlines or prediction caps.
The last confirmed identity readiness result completed 494 measured ticks after
30 separately recorded setup physics steps, then failed the original lost-slot
check. Its statistics were 18 planned requests, zero queue underflows and zero
inference deadline misses. This result is not a readiness pass.

The follow-on predicted readiness result and any subsequent execution must be
confirmed from their own terminal artifacts before being added to this report.
No unobserved completion, successful push, or clean process state is asserted here.

## Scientific decision

Actual full-task capability has been observed, but reliable original-task baseline
performance has not been established. First placement is a separately named,
partial-task endpoint. The larger independent action audit does not confirm stable
benefit, and technical failures prevent a clean live efficacy comparison under
the evaluated deployment. Preserve these negative results while investigating
readiness and action-relevant prediction objectives using fresh development and
validation data. Do not infer task success from latent error reduction or suppress
technical failures to obtain a favorable comparison.
