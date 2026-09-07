# L11: new task comparison after repeated runtime qualification

Date:2026-09-07. This experiment tests the immutable L6 epoch3 predictor, not
either new L10 training candidate. Its decision does not depend on L10 validation
scores. No L10/L10b training or other model workload runs concurrently.

## Eligibility and operating profile

L9's common minimum-delay7 profile completed all six preregistered40-second
qualifications:7200 actions,132 measured predictor calls, zero control lost slots,
queue underflows, inference deadline misses or cap violations. The earlier
single-thread/min1 qualification is5/6 and stays that way; its single late result
was whole-discarded. Three original-task successes there demonstrate capability,
not a stable advantage of either mode.

Use the same effective simulator tensor threads1, model threads1, CPU PhysX/GPU
RTX, standard dual640x480@30Hz cameras, unchanged physics/assets, warmed_v2,
feasible_v1 projected-before-commitment actions. Both async modes use minimum7,
maximum8; no artificial delay, future state, alternative queue or risk threshold.

## Registered outcome cohort

Twelve entirely new environment seeds20270210–20270221, policy2610–2621. Each
gets sync,identity,predicted in the same six-permutations-twice balanced order
shuffled with2088. These integers are RNG seeds, not future scheduled dates.
Every condition runs once, at most120 simulated seconds/3600 control steps.
Use the existing `pickorange_first_settled_v1` endpoint: first object continuously
inside the declared plate region at low velocity for ten ticks. Stop there and
retain the original native full-task success flag separately. A single-orange
subgoal is not the original three-orange-plus-rest task or a contact-sensor proof.

This is a new operating-profile cohort; do not pool it with L8 or retry a failed
old seed. Both async arms receive the same timing/feasible preparation. Sync is
the frozen policy's capability reference; its inference does not progress physics,
so its simulated-time cost is not a real-time latency claim.

Primary contrast: predicted minus identity restricted first-placement time,
failure (including technical) assigned120s. Report success counts, all technical
failures, per-pair geometry agreement, projection counts and action-change
descriptives. RGB need not be bitwise identical, so do not treat a lone pair as
a deterministic intervention. No failed run is replaced or removed.

Keep the previously specified stable-effect criterion: all12 paired initial
geometries match; paired-bootstrap50000 draws/RNG2089,95% upper bound for the
time difference below0; both six-pair halves negative; predicted successes not
lower and technical failures not higher than identity. A **reliable-reference
prerequisite** additionally requires at least10/12 synchronous subgoal successes
and zero synchronous technical failures. A benefit-on-reliable-reference claim
requires both criteria; if either fails, report the failed condition explicitly.
Twelve episodes remain a bounded pilot, not proof of universal reliability.

L6 and both L10 candidates stay frozen during outcomes. These new task scenes
must not later be described as held-out if used to tune any next policy,
predictor, risk rule or run configuration.
