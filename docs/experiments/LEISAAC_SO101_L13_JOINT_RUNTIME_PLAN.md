# L13: experimental causal joint-context deployment and attribution

Date:2026-09-07. L12's complete288-case test is frozen. Its joint-versus-identity
and joint-versus-visual-only mean effects improve in6/6 trajectory clusters with
positive cluster-bootstrap intervals, but179/288 and185/288 improving cases miss
the192-case requirement. That full stable-action gate is **FAIL** and stays so.
The user authorizes continuing research, not declaring a failed gate passed.

This is a new **experimental task-validation** decision, not promotion as proven
stable benefit. The positive independent mean effects justify a bounded direct
test while retaining state-only attribution. No weights, gains, thresholds or
test cases are selected again. L10 students stay offline and their tests unopened.

## Minimal binding and invariants

Use immutable L6 visual predictor and L12 state MLP epoch29/sourcebf4e025d.
Wrap the former only in the explicit joint/state-only simulation adapter. Add
one protected engine hook returning no future state by default; only the new
validated adapter returns learned model-ready state through SmolVLA's existing
`future_state` API. The default SO100 and visual-only SO101 paths stay unchanged.
No future observation, simulator privileged state or target command is exposed
to the forecaster: inputs are current normalized state and the same committed
policy-normalized prefix already used by the visual predictor. The visual
predictor still receives CURRENT state, not the state model's output.

For state-only, current images remain in use and no visual forecast is computed.
For joint, both forecasts are computed once. Per-request logs distinguish
pipeline, visual and state call counts. Bootstrap is normal RGB/current state;
d0 invokes neither predictor. Existing worker, queue, feasible projection before
commitment, reset invalidation and late-whole-discard semantics are preserved.
New state is already32-dimensional model-ready data and is NOT normalized twice.

## Bounded qualification, before task outcomes

Existing CPU PhysX/RTX, simulator/model tensor threads1, dual640x480@30Hz,
unchanged assets/physics, feasible_v1, warmed_v2, min7/cap8,margin1/guard2,
in-memory telemetry. On new seeds20270330–31/policy3230–31, run visual-only,
state-only,joint in order and reverse order respectively; each600 control ticks
with60s native environment timeout and no subgoal early stop. All six must
complete the bound or native success, preserve source/witness/cleanup, and have
zero lost slots, underflows, deadline misses and cap violations. Verify actual
positive-delay calls and takeovers, including new state calls where required.
Do not replace a failed condition or weaken clocks to pass.

## Task cohort only if all six qualify

Eight new seeds20270410–17/policy3310–17, five arms per scene: sync reference,
identity async, visual-only async, state-only async and joint async. Five cyclic
rotations plus three reverse cyclic rotations, shuffled with3300, balance each
arm's position to within one trial. Forty conditions run once with120s/3600-step
bound and the unchanged `pickorange_first_settled_v1` endpoint. All numbers are
RNG seeds, NOT future scheduled dates. No concurrent model training is allowed.

Primary contrasts: joint minus identity and joint minus visual-only restricted
placement time. Failures including technical receive120s. Preserve all outcomes
and paired initial geometry. Each contrast must have a bootstrap95% upper bound
below0 (50000 scene-cluster draws,RNG3301), both four-scene halves negative,
successes not lower and technical failures not higher. Reliable-reference
prerequisite is >=7/8 synchronous subgoal successes with zero technical failures.
Report all pair differences even if a gate fails. This remains a small pilot.

State-only and joint-versus-state-only are mandatory attribution controls, not a
post-hoc choice of the winning algorithm. Report the original native complete
task flag separately; region occupancy is not verified contact release or three
oranges plus return to rest. Sync is a physics-paused inference capability
reference, not a real-time policy comparison. No old L11 outcomes are pooled.
