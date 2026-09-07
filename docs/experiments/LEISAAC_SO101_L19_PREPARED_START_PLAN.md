# L19: source-defined preparation before reliable native-task qualification

Registered after L18 closure at `d25f8b0ad186f30c9fb5e7f921005953f0dabd4a`.
The user authorizes project implementation/experiments and GitHub publication;
no additional reviewer is required. This is not a new forecasting experiment.

## Question and fixed candidate

L18's zero-additional-delay frozen policy produced 1/8 native successes with
the dataset task text and 0/8 with a separately documented model-card string.
Neither met the previously registered native-task prerequisite. Do not add
prompts or infer that additional asynchronous latency is the sole problem.

The published training dataset at `fa6e0625d814352b8e6ee1c6d2482194e4da8ed3`
has 60 episodes; its first-state wrist motor coordinate ranges 43.2133--50.1271,
unlike the current zero-joint initialization. This is an observed covariate
difference, not a proven root cause. The policy training config does not pin
the dataset revision, so this is not a byte-exact reconstruction claim.

Use the single observed frame-0 state minimizing mean range-scaled L1 distance
to all 60 first states (motor scales 200 for five arm coordinates, 100 for
gripper; lowest episode breaks ties). This selects episode 9, fixed before
any L19 scene: [-11.0962448120, -47.5487594604, 55.0041503906, 50.1022491455,
0.1027908325, 8.4473562241] in native motor coordinates. The source-defined
conversion and provenance are in `so101_prepared_start.py`. All 60 initial
states, not task scores or held-out forecasts, determine this preparation.
The runner rechecks the selection from the pinned local tables before dispatch.

Only the initial robot joint configuration changes. No demonstration actions,
object relocations, success-driven reset, forced return-to-rest, or state writes
during an episode are permitted. All arms retain native object/camera reset
randomization; the wrist camera necessarily follows the changed initial joints.
Every prepared reset checks the actually measured joint positions before task
dispatch. Thirty real hold setup steps are logged, followed by a same-seed reset
as in L18; these steps do not count as task actions.

## Fixed once-only development cohort

Eight NEW scenes 20271010--20271017, policy 3910--3917; zero versus
`training_medoid_v1` initial pose in each scene. Four forward/four reverse orders
shuffled by 3900, 16 total conditions, each run once to native three-oranges-plus-
rest success or 120 simulated seconds / 3600 actions. The numbers are RNG seeds,
not scheduled dates. No failed condition is replaced or retried.

Frozen WSAGI checkpoint/processors and their original statistics; text exactly
`Grab orange and place into plate` (the published training dataset text).
Existing synchronous select_action, generated/executed chunk 50; no future
predictors, oracle, coupled noise, action expert fine-tuning or RTC. CPU PhysX,
GPU RTX, single tensor worker, standard dual 640x480 cameras, simulated 30Hz,
feasible-before-execution action projection, physics/assets unchanged.
Physics pauses for inference: this cannot qualify wall-clock real-time control.

Primary eligibility is native full-task success >=7/8 with zero technical
failures for the prepared condition; retain the paired zero control in full.
Report native counts and restricted completion time (failures including
technical =120s), first-region occupancy only as descriptive, and all native
placement/rest components. Independent pre-reset witness reconstruction is
required. Same-seed initial objects/front camera should match; joint/wrist pose
is deliberately different and must not be described as fully matched geometry.
Per-scene results, paired time differences and an eight-scene bootstrap
(50000 draws, seed 3901) are descriptive development evidence, not confirmation.

Passing is only permission for a NEW frozen independent native-task qualification,
not proof of forecasting benefit. Failing blocks scaling the current forecasting
campaign; do not lower the native threshold, use early occupancy as success, or
reinterpret old tests as development data. L10/L15 reserved tests stay unopened.
No model fitting is part of L19. No unrelated GPU process is interrupted.

## Targeted checks and immutable records

Check source-medoid selection, motor/degree/radian conversion including gripper,
joint-name ordering and an intentionally wrong reset; verify the prepared driver
distinguishes native/early endpoints and expected pose from actual measurements.
Publish source/plan before any environment dispatch. Use a new output root,
preserve raw per-step witnesses and simulator logs, keep execution HEAD fixed
until all sixteen terminal records exist, then audit and publish closure.
