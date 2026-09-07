# L18 closed: unaugmented synchronous full-task reference is not qualified

Date: 2026-09-08. All sixteen registered development conditions executed once at
`40634caa63078318d06b680d7b16177daf47206a`, without source changes during the
cohort. No condition was replaced. This follows the independently closed
[L17 study](LEISAAC_SO101_L17_NOISE_REFERENCE_RESULT.md), not a reanalysis that
changes its failed criteria or a new visual-predictor efficacy test.

## 1. Decisive prerequisite result

The existing frozen WSAGI policy was evaluated with NO visual forecaster, NO
future-state module, NO oracle, NO delayed takeover, and NO noise coupling.
Inference paused physics. Even under this synchronous condition, neither of
the two predeclared task strings met the required7/8 native full-task successes
with zero technical failures. Both development qualifications therefore FAIL.

This establishes a missing prerequisite for a claim about robust completion of
the ORIGINAL three-oranges-plus-rest task under the tested candidate, environment,
120-s budget and50-action execution profile. It does not prove that the model
cannot ever complete the task, that vision is useless, or that asynchronous
forecasting is impossible. It also does not retroactively invalidate previously
defined first-placement-subgoal experiments.

"Zero added inference delay" is not zero action age at every control step:
this reference still executes50 generated actions before replanning. These
results do not isolate all possible controller, model, physics or observation
causes; in particular, they must not be described as proof that latency can
never matter. They show that extra asynchronous delay is not required for
the observed native-task failures.

## 2. Exact conditions and task text

The [registered plan](LEISAAC_SO101_L18_SYNC_CAPABILITY_PLAN.md) used eight NEW
development scenes20270910--20270917 and policy seeds3810--3817, two conditions
per scene. These are RNG identifiers, not scheduled dates. Four of each ordering
were shuffled with3800 before any outcome. Both prompts were fixed documentary
literals, not candidates in an adaptive prompt search:

- Existing runtime/training-task literal: `Grab orange and place into plate`.
- Pinned model-card inference-command literal: `Pick up the orange and put it in the plate`.

The actual string was supplied to both the preprocessor and policy batch and
recorded in every candidate manifest. The new argument is synchronous/matched-
candidate only; default behavior and asynchronous candidate bindings stay
unchanged. The policy checkpoint remains
`wsagi/SmolVLA-PickOrange@c8c3318dba152b0ba671ff07b4314418d5aa4b4a`, with its own
processors/statistics, use_amp=false and50 generated/50 executed actions.

Keep CPU PhysX/RTX, one simulator tensor worker, standard dual640x480@30Hz
cameras, feasible_v1 action projection, zero initial joint pose and warmed_v2's
thirty logged hold steps followed by same-seed reset. No task terminator,
camera layout, physical limit, model weight or action scaling was changed.

Each arm continued past any first-occupancy subgoal until original native success
or120 simulated seconds/3600 actions. Inference pausing means that this is NOT
30Hz wall-clock asynchronous qualification. Unlike L17's shared simulator,
this established evaluation entrypoint starts and closes one simulator process
per condition; all sixteen exits were actually verified.

## 3. Complete outcomes

| Task-string profile | Native complete successes | First-occupancy subgoals | Technical failures | Mean restricted native time | Development qualification |
|---|---:|---:|---:|---:|---|
|Existing literal|1/8|6/8|0/8|115.516667 simulated s|Fail|
|Model-card literal|0/8|5/8|0/8|120.000000 simulated s|Fail|

Failures receive the registered120-s cost. No unsupported success-rate claim
is made by pooling the two profiles: these are eight paired scenes, not sixteen
independent trials of a single treatment. The prompt contrast is development
description, not a confirmed statistical effect of wording.

| Scene seed | Existing literal: native outcome / max simultaneous native-box occupancy | Model-card literal: native outcome / max occupancy |
|---|---|---|
|20270910|Timeout /1|Timeout /1|
|20270911|Timeout /0|Timeout /0|
|20270912|Timeout /1|Timeout /0|
|20270913|Timeout /1|Timeout /2|
|20270914|Timeout /3|Timeout /2|
|20270915|Success at84.133333s /3|Timeout /1|
|20270916|Timeout /0|Timeout /2|
|20270917|Timeout /2|Timeout /0|

The single native success completed2,524 physical actions. Its original
pre-auto-reset success witness is retained. All other conditions completed the
full3,600-action bound, for56,524 measured actions in total, plus480 setup
actions. There are no partial/failed technical trials hidden behind the task
failure counts.

Native predicates were independently reconstructed from saved positions and
joint states on every measured step, with zero disagreements. The reconstruction
uses the original THREE simultaneous orange boxes AND original rest ranges,
not the first single-object low-speed circular region.

Of the fifteen native failures, fourteen never had all three objects inside the
native boxes simultaneously. The remaining existing-literal scene20270914 did
reach three-box occupancy at some earlier step, but at timeout the arm satisfied
the rest predicate while only TWO objects remained in the boxes. Thus a sticky
"each object was placed at some time" counter or adding a forced final home
command would not be an equivalent task result. This is descriptive component
evidence, not proof of a unique behavioral cause or permission to change success.

All eight paired initial joint/object/camera geometries match. Prompt-specific
bootstrap actions were generated independently; unlike L17 they are not a
common-action bootstrap. Rendered images and subsequent trajectories are not
claimed to match bitwise.

## 4. What the source-linked data inspection actually adds

A read-only inspection downloaded only metadata and60 parquet files from the
published dataset NAMED by this checkpoint, pinned to
`LightwheelAI/leisaac-pick-orange@fa6e0625d814352b8e6ee1c6d2482194e4da8ed3`.
There are36,293 frames, and all60 first records have frame_index0. No additional
videos, new policy weights, old forecaster tests or reserved student tests were
used. Its task table contains the existing literal exactly.

The checkpoint's train_config has dataset revision=null. Therefore the link
identifies the declared training dataset, but does NOT establish byte-for-byte
identity between this pinned snapshot and the version used by the model author.

The published first-state distribution provides a concrete INITIALIZATION
hypothesis. Its six-component median in native motor coordinates is:

`[-10.9806023, -47.5480938, 55.0251007, 50.0775070, 1.6822472, 7.5724697]`.

All sixty initial wrist_flex motor values are between43.2132721 and50.1270599;
the currently evaluated zero-joint initialization has wrist_flex motor0.
This is a measured discrepancy in the source-linked data, NOT yet a demonstrated
cause of the L18 failures, and not permission to silently replace the initial
pose of completed trials.

A descriptive medoid, minimizing mean six-coordinate range-scaled L1 distance
to all60 first states and breaking ties by lowest episode index, is episode9.
The fixed coordinate divisors are `[200, 200, 200, 200, 200, 100]`; distance is
averaged across all60 records and all six coordinates, not weighted by outcomes.
Its actual observed motor state is
`[-11.0962448120, -47.5487594604, 55.0041503906, 50.1022491455, 0.1027908325, 8.4473562241]`.
Converted physical joint angles are inside all original limits. This is a
training-source-derived candidate for a separately registered prepared-start
test, not a tested initialization or a selected successful control trajectory.
No such prepared-start run was launched in L18, and no demonstration actions
were replayed as policy commands.

An additional attempt to compare full published-dataset statistics with the
checkpoint normalizers was blocked by the tool's safety-status check and did
not execute. It was not retried via a different route. No claim of matched
global means/standard deviations is made, and no normalizer was modified.

## 5. Checks, records, and next decision

Eight targeted tests covering the actual synchronous task-text route, rejection
of asynchronous overrides, native-versus-subgoal qualification, and existing
task evidence passed; lint and formatting checks passed. After closure, all
sixteen rows were regenerated with the same descriptor from raw artifacts.
Unique condition IDs, exact execution source, literal text, zero-pose profile,
50-step execution, absence of predictors/engine, actual30Hz simulation metadata,
all native/subgoal endpoints, setup counts and sixteen clean exits matched.

The next research step is candidate/task/initialization compatibility, not another
same-candidate visual-forecasting efficacy grid or another visual-capacity sweep.
A source-supported prepared-start diagnostic can be separately preregistered,
using new scenes and the unchanged original native task predicate. It must
not be called a fix or qualified baseline before it is actually tested. A new
baseline that passes development still needs independent confirmation before
claiming stable forecasting benefits.

This closure does not require a new Pro review. Existing project authorization
remains in force, but no background training or automatic retries are queued.
L17's failed noise/visual-value criterion, L16's failed privileged-reference
criterion, L15/L15b's failed student qualifications, and real-time readiness
limits retain their original meanings. L10/L15 reserved tests remain unopened.

Local evidence: `artifacts/m54l18_sync_native_capability_v1` (frozen manifest,
sixteen per-condition result/manifest/ticks/events/setup files, images, simulator
logs and final summary). The previously closed L17 records remain in
`artifacts/m54l17_noise_reference_v1` and
`artifacts/m54l17_closed_noise_audit_v1.json`.

See [machine summary](LEISAAC_SO101_L18_SUMMARY.json).
