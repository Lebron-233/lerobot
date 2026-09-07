# L12–L14 closed: coherent action-reference gains do not establish task benefit

Date:2026-09-07. All forty preregistered L14 conditions finished exactly once at
source `5590892185333ce3acd8d8ae79e042147dab0087`. The source stayed unchanged
and clean throughout the cohort. The closed-cohort audit was prepared in an
isolated worktree and integrated only after the last condition terminated.

**L14's synchronous reference meets its prescribed eligibility criterion, but
neither co-primary stable task-effect criterion passes. All forty conditions
completed without a technical failure. This is controlled simulated-delay task
evidence, not real-time deployment qualification.**

## 1. What the new experiments distinguish

The original visual-only context sends predicted future visual tokens together
with current model-ready state to the frozen policy. L12 explicitly introduced
a separate joint-context hypothesis: predict the six non-padding state values
from current state and the already committed action prefix, while retaining the
unchanged L6 visual forecaster. No future environment observation is an input
to either learned predictor. SmolVLA, its visual backbone and action expert,
checkpoint statistics, and the L6 visual weights remained frozen.

The state MLP has25,478 parameters and was selected at epoch29 solely by state
validation MAE after a fixed30-epoch development fit. Joint forecasting has
299,302 predictor parameters including the existing273,824 visual parameters.
This is an explicitly new joint-state/visual candidate, not a retroactive claim
that original visual-only M3 passed its previous validation.

The L12 independent test uses a new coherent action reference with both actual
future visual tokens and actual future state. It is not the old visual-oracle/
current-state metric. Across six held-out trajectory clusters and288 cases:

| Joint compared with | Mean action-reference L1 reduction | Improving cases | Improving episode means | Episode-bootstrap95% reduction interval |
|---|---:|---:|---:|---|
|Identity|11.7211%|179/288|6/6|[6.8537%,15.8048%]|
|Visual-only|13.2113%|185/288|6/6|[6.9170%,18.9189%]|
|State-only|1.9723%|139/288|5/6|[-0.7427%,4.2396%]|

The full prespecified stable-action criteria required192 improving cases in
each primary contrast. They remain failed. The extra contribution of predicted
vision over learned-state-only compensation is also not established. All288
cases and their negative results remain unchanged; no fitting or threshold
selection followed their opening. See [L12 result](LEISAAC_SO101_L12_JOINT_CONTEXT_RESULT.md).

## 2. Real-time failure was not relabeled as a successful deployment

L13 exercised the new causal state override in the actual production worker and
queue. Its six once-only qualification conditions had one bounded pass and five
control-slot failures:1,237 measured actions plus180 logged setup actions,
34 state calls and18 visual calls. The failed environment-step intervals were
113–201ms, with no queue underflow, inference deadline miss or prediction-cap
violation. All six simulators and metrics sinks closed normally.

The conditional L13 real-time task cohort did not start. Shared-GPU utilization
observed afterward is a resource-sharing snapshot, not proof of a unique cause
for each failure; no unrelated job was changed. These limits are preserved in
[L13 qualification](LEISAAC_SO101_L13_QUALIFICATION_RESULT.md).

L14 instead follows its separately registered [controlled-delay protocol](LEISAAC_SO101_L14_CONTROLLED_DELAY_PLAN.md):
compute a new chunk at observation t while simulated physics is paused, stage
it in the original ScheduledActionQueue for t+7, then physically execute the
seven previously committed actions before takeover. No successor observation
is available when the new chunk is computed. The seven steps are simulated
observation age, not a claim that those actions execute concurrently with the
inference computation. Wall-clock real-time qualification remains false.

## 3. Complete five-arm task result

Eight new scene seeds20270510–17 and policy seeds3410–17 were frozen before
execution. These are RNG identifiers, not scheduled dates. Five cyclic orders
and three reverse cyclic orders were shuffled with seed3400. All arms use the
same frozen model components, feasible-before-commitment action projection,
thirty logged setup hold steps followed by a same-seed reset, full standard
dual640x480 cameras,30Hz simulated control and unchanged physics/assets.

The endpoint is `pickorange_first_settled_v1`: the first orange continuously
inside the specified plate region at low speed for ten ticks. This is neither
a contact-release test nor the original three-oranges-plus-rest task. The
120-second restricted cost includes every failure as120 simulated seconds.

| Arm | Successful subgoals | Technical failures | Mean restricted simulated time |
|---|---:|---:|---:|
|Synchronous reference|7/8|0/8|53.141667s|
|Identity: current visual/current state|7/8|0/8|63.133333s|
|Visual-only: predicted visual/current state|5/8|0/8|70.891667s|
|State-only: current visual/predicted state|7/8|0/8|51.704167s|
|Joint: predicted visual/predicted state|6/8|0/8|55.387500s|

State-only still uses current images; it is not a vision-free policy. Sync is a
zero-delay capability reference, whereas the four scheduled arms share exactly
seven simulated delay steps. Their flow noise is explicitly keyed by policy
seed and request step. Reference eligibility requires>=7/8 sync successes with
no technical failures; this cohort meets that bounded criterion, not a universal
or full-task reliability guarantee.

### All paired outcomes

Times below are simulated seconds, rounded only for display. F denotes failure
to reach the subgoal by120s. No successful or unsuccessful condition was replaced.

| Scene seed | Sync | Identity | Visual-only | State-only | Joint |
|---|---:|---:|---:|---:|---:|
|20270510|38.600|79.600|120 F|104.967|39.333|
|20270511|12.700|26.433|120 F|4.867|4.133|
|20270512|99.100|50.833|18.233|13.067|120 F|
|20270513|13.933|94.000|44.133|24.233|94.433|
|20270514|23.133|99.333|18.533|34.400|18.400|
|20270515|120 F|120 F|120 F|120 F|120 F|
|20270516|81.333|25.033|47.067|72.833|34.767|
|20270517|36.333|9.833|79.167|39.267|12.033|

### Registered contrasts: joint minus comparator

Negative time differences favor joint. The same paired50000-draw bootstrap
with RNG3301 and the same first/last-four consistency conditions were used.
All eight initial state/object/camera geometries match for each contrast;
rendered RGB is not assumed bitwise identical.

| Comparator | Mean difference | Paired95% interval | First four | Last four | Stable-effect gate |
|---|---:|---|---:|---:|---|
|Identity|−7.745833s|[−35.595833,+19.983333]s|+1.758333s|−17.250000s|Fail|
|Visual-only|−15.504167s|[−60.600000,+31.550000]s|−11.116667s|−19.891667s|Fail|
|State-only|+3.683333s|[−30.116667,+43.541667]s|+27.691667s|−20.325000s|Fail|

Joint does not meet the identity contrast's success-count or half-consistency
requirements, and both co-primary intervals include zero. State-only has the
lowest mean restricted time here, but this does not establish a stable state-only
advantage or justify selecting a new model using these opened outcomes. In
particular, the data do not establish added task value from forecasting vision
on top of learned state compensation.

## 4. Raw physical commitment and endpoint accounting

A separate read-only audit recomputed every outcome from pre-auto-reset
witnesses, compared the published statistics, and checked the actual normalized
actions against every retained seven-row committed prefix. It verified the
request cadence, explicit noise-seed rule, component-call ablations, and actual
t+7 takeover indices. Incomplete prefixes at task stop are not fabricated as
executed actions or takeovers.

| Accounting item | Verified count |
|---|---:|
|Completed conditions|40|
|Subgoal successes / timeouts / technical failures|32 / 8 / 0|
|Measured physical actions|70,622|
|Separately recorded setup actions|1,200|
|State-predictor calls|946|
|Visual-predictor calls|1,117|
|Actual queue takeovers|2,127|
|Executed old-prefix actions before takeover opportunity|14,917|

Joint requests contain both components, so their call counts must not be added
and labeled distinct requests. All simulator exits and metric closures are
confirmed; all forty schedules passed the raw commitment audit. The separate
100-step smoke and its 30 setup actions are not included in these cohort totals.

The four added audit unit tests cover an actual seven-action prefix, deliberate
prefix mismatch, early task termination without a takeover, and accidental
visual calls in the state-only arm. They pass, with lint and formatting checks.

## 5. Scientific decision and retained boundaries

The joint temporal-context hypothesis now has an independently observed average
coherent-action-reference improvement, but not its required per-case coverage.
The task assessment has a qualified subtask reference and no technical failures,
yet does not establish either co-primary stable task benefit. Therefore missing
real-time readiness is not the only remaining research question: reducing
offline reference-action deviation has not sufficed to produce reliable task
gains in this controlled experiment.

Preserve identity, visual-only and state-only as distinct controls in subsequent
development. A new visual forecaster must justify its incremental contribution
over state-only compensation, rather than relying on a comparison only with
fully stale context. This is a direction for new development data and a new
frozen validation, not permission to tune on L12/L14 outcomes or weaken their
failed criteria. L10's reserved student-test scenes remain unopened. No training
or automatic trial retries are queued at this closure; risk thresholds stay null.

Local evidence under `/home/rp/Workspace/SmolVLA_RTC/artifacts/`:
`m54l12_state_development_v1`, `m54l12_joint_context_test_v1`,
`m54l12_joint_context_evaluation_v1`, `m54l13_joint_runtime_qualification_v1`,
`m54l14_controlled_joint_smoke_v1`, `m54l14_controlled_delay_task_v1`, and
`m54l14_closed_audit_v1.json`. The last two contain every L14 condition, raw
requests/actions/witnesses, setup logs, images and the closed-cohort audit.
See also [machine summary](LEISAAC_SO101_L14_SUMMARY.json).
