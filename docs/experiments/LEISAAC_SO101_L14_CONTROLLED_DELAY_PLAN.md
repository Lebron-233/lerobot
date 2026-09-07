# L14: controlled simulated-delay task experiment, NOT real-time qualification

Date:2026-09-07. L13's real-time qualification failed1/6 and its outcome cohort
remains unopened. This independently named scientific experiment isolates the
effect of stale context from fluctuating host/GPU step latency. It does not
weaken or bypass L13's real-time claims: **no30Hz wall-clock GO is possible here**.

## Method and causality

Reuse the existing `ScheduledActionQueue`, frozen WSAGI policy/processors,
L6 visual epoch3 and L12 state epoch29. Every environment step still advances
actual SO101 physics by1/30 simulated second; no synthetic task outcomes.

At a request observation t, freeze the queue's next7 already committed normalized
actions. Compute the selected context and new chunk while simulated physics is
paused. Stage that chunk for queue takeover index t+7, then execute the seven
old actions through real `env.step` calls. The queue itself withholds the new
actions until t+7. No successor image/state is used to compute this chunk.
This explicitly injects a seven-step **simulated** delay, regardless of actual
inference wall time. It is not concurrent production-worker validation.

The scheduled arms request when active queue size<=30 and no plan is in flight;
the same guard2/cap8, normalized prefix and feasible-before-commitment mapping
apply. The only difference between identity,visual-only,state-only,joint is the
context supplied to the frozen decoder. Bootstrap uses current RGB/current
state in all arms. Sync instead executes50 generated actions before replanning;
it is a zero-delay capability reference, not the primary controlled contrast.

Use the same explicit flow-noise generator seed`policy_seed*100000+t` at request
step t in every arm. Current state always feeds the visual predictor. Only the
learned state module supplies future model-ready state in state/joint arms.
Neither privileged simulator state nor any future teacher enters live action
generation. Load the same frozen model components in every arm, even unused
ones; record real start geometry, wall cost, action requests and takeover rows.

## Fixed experiment

Eight NEW scenes20270510–17, policy3410–17, five arms per scene. Use the same
five cyclic rotations plus three reverse cyclic rotations shuffled3400. Forty
conditions, one run each; no failed condition replacement. These numbers are
RNG seeds, not future scheduled dates. No concurrent fitting of this project.

Preserve the standard dual640x480@30Hz cameras,CPU PhysX/RTX,tensor threads1,
assets,reset randomization and feasible_v1. Thirty logged hold setup steps then
same-seed reset in all arms. Each condition ends at120 simulated seconds/3600
actions or the existing `pickorange_first_settled_v1` endpoint (ten consecutive
slow region-occupancy steps), with native full-task success separately recorded.
No claim of contact release or three-orange completion from that subgoal.

Primary contrast is joint-minus-identity restricted simulated placement time;
joint-minus-visual-only is co-primary. Unsuccessful/technical conditions get120s.
Each contrast: eight matched initial geometries; paired bootstrap50000 draws,
RNG3301,95% upper bound<0; first and last four differences each average<0;
joint successes not lower, technical failures not higher. Reference eligibility
requires>=7/8 sync successes and0 technical failures. Report state-only and
joint-minus-state-only regardless of direction. Reuse the already tested summary
implementation, not the L13 real-time qualification flag.

Before the cohort, test that all seven committed actions execute before takeover
and that state-only never calls the visual forecaster. Complete only a single
100-step joint wiring smoke atseed20270501/policy3401, no outcome inference and
no repeated smoke until success. If the smoke fails technically, stop this
protocol and preserve the failure. No model selection follows these outcomes.

L12's288-case test remains frozen; all L14 results are a new, lower-level
controlled-delay task assessment. Positive results would motivate dedicated
resource-isolated real-time replication, not substitute for it.
