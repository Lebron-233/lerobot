# L17 closed: shared noise did not establish visual value or reliable full-task control

Date: 2026-09-08 (Asia/Shanghai). Execution source:
`96a14379e57a0fac36d91219582b2c598a5f518c`.
All 32 preregistered conditions completed once. The live source was unchanged;
the independently prepared audit was integrated only after the last simulator
condition and shared process terminated. No training, resampling of failed
scenes, model selection, or student-test opening occurred.

## 1. What was tested

The [frozen L17 plan](LEISAAC_SO101_L17_NOISE_REFERENCE_PLAN.md) separates two
factors: fresh versus absolute-action-indexed Gaussian noise, and current versus
actual target-time visual information. All four arms retain the frozen WSAGI
policy, L12 epoch29 causal state predictor, processors, physical limits,
feasible-before-commitment actions, cameras, seven-step simulated age, and the
original ScheduledActionQueue. No RTC guidance or visual student was enabled.

`fresh_state` and `coupled_state` generate at observation t using current visual
tokens and the state predicted at t from current state plus the seven committed
actions. The corresponding oracle arms store that same causal state prediction,
execute the seven old actions, and generate using real visual tokens observed
at t+7. Neither oracle receives actual future state. Oracle input is a privileged
information reference, not a learned forecast or a performance upper bound.

Fresh noise follows the existing request-seed rule. Coupled noise takes a window
from a per-scene Gaussian field indexed by intended absolute action time; repeated
action indices reuse noise exactly. This changes noise correlation, not the
policy weights, and does not mathematically force continuous action outputs.

Each scene uses one common real-policy bootstrap across all four arms. Its first
50 noise rows match the original fresh bootstrap. All arms execute the same
first 27 normalized and physical commands. Initial geometries match. Including
observation 27, measured joint-state differences are zero and maximum object
coordinate difference is 1.430511475e-6 m. RGB and subsequent trajectories are
not claimed to be bitwise identical.

## 2. Full-task follow-through, not only early subgoal stopping

Eight new scene seeds 20270810--20270817 and policy seeds 3710--3717 were used.
The identifiers are RNG seeds, not scheduled dates. Unlike L16, every arm
continued after first stable plate-region occupancy until the original native
three-oranges-plus-rest condition or 120 simulated seconds (3,600 actions).
The first-occupancy endpoint remains the registered primary information-value
metric. Native full-task success is a separately recorded secondary outcome.

The subgoal is ten consecutive low-speed ticks in the declared plate region,
not a contact-release test. All task timeouts receive the registered 120-second
restricted cost. No technical failure occurred, so the difference between a
subgoal and a full-task endpoint cannot be attributed to abandoned technical
runs in this cohort.

| Arm | Subgoal successes | Native full-task successes | Technical failures | Mean restricted subgoal time | Mean restricted native time |
|---|---:|---:|---:|---:|---:|
|Fresh noise / state-only|5/8|0/8|0/8|64.675000 s|120.000000 s|
|Fresh noise / true target visual|5/8|0/8|0/8|62.195833 s|120.000000 s|
|Coupled noise / state-only|5/8|0/8|0/8|71.154167 s|120.000000 s|
|Coupled noise / true target visual|5/8|1/8|0/8|53.850000 s|114.379167 s|

The single native success is coupled_oracle, scene20270812: 2,251 actions,
75.033333 simulated seconds. It used privileged visual observations, not a
learned visual predictor. Its pre-auto-reset native witness has all three
orange positions in the original boxes, all measured object linear velocities
zero, and all six joints within the original rest ranges. It does not establish
a deployable learned-method advantage or reliable full-task competence.

There were 20 subgoal attainments across 32 conditions, but only one reached
the native complete task. These are descriptive counts from four different
conditions on eight paired scenes, not 32 independent replicates of one policy.
In the other 31 runs, all three oranges were NEVER simultaneously inside the
native task boxes. Thus the observed full-task failures are not merely cases
where placement succeeded and only the final return-to-rest was missing.

| Arm | Maximum simultaneous native-box occupancy: 0 / 1 / 2 / 3 oranges | Runs ever reaching all three boxes |
|---|---|---:|
|Fresh/state|3 / 0 / 5 / 0|0|
|Fresh/oracle|2 / 2 / 4 / 0|0|
|Coupled/state|3 / 2 / 3 / 0|0|
|Coupled/oracle|3 / 0 / 4 / 1|1|

This component decomposition is post-closure description using the existing
native predicate, not a new fitted endpoint or a causal explanation of failure.

## 3. All registered comparisons, including unfavorable directions

Negative differences below favor the first named arm. Bootstrap resampling
units are the eight scene pairs, with the preregistered 50,000 draws/RNG3701.

| Contrast in restricted subgoal time | Mean difference | Paired 95% interval | First four scenes | Last four scenes |
|---|---:|---|---:|---:|
|Coupled oracle minus coupled state (primary)|-17.304167 s|[-66.779167, +35.485625] s|-23.475000 s|-11.133333 s|
|Fresh oracle minus fresh state|-2.479167 s|[-34.366667, +33.304167] s|+19.125000 s|-24.083333 s|
|Coupled state minus fresh state|+6.479167 s|[-51.887813, +61.658333] s|+31.133333 s|-18.175000 s|
|Coupled oracle minus fresh oracle|-8.345833 s|[-26.391667, +4.126042] s|-11.466667 s|-5.225000 s|
|Noise-by-information interaction|-14.825000 s|[-63.450000, +37.508333] s|-42.600000 s|+12.950000 s|

All intervals include zero. The primary's two half-means are negative, but
its confidence interval is not, and coupled_state achieves only5/8 subgoals,
below the registered7/8 reference requirement. Therefore both reference
eligibility and the privileged visual-value decision FAIL. Prior L16's8/8
reference result remains valid for its own scenes/protocol; it is not carried
forward as an eligibility pass on these new scenes.

All secondary contrasts and the native restricted-time statistics remain in
the machine summary and raw report. No secondary point estimate or the lone
native oracle success replaces the failed primary decision.

### Did shared Gaussian rows guarantee smaller action jumps?

No such guarantee was assumed. The preregistered descriptive quantity is the
range-normalized L2 jump in commanded physical targets at actual takeovers,
averaged within an episode and then equally across the eight episodes:

| Arm | Mean takeover command-jump descriptor |
|---|---:|
|Fresh/state|0.052826323|
|Coupled/state|0.081414156|
|Fresh/oracle|0.054236826|
|Coupled/oracle|0.111653545|

Coupled point estimates are larger here. This is a command-space description,
not measured mechanical jerk, a fitted objective, or a statistically established
cause of the task outcomes. Noise overlap alone must not be described as RTC
inpainting, action continuity, or a verified fix for the earlier failures.

## 4. Closed accounting and independent native-rule reconstruction

| Item | Verified count |
|---|---:|
|Once-only outcome conditions|32|
|Measured physical actions|113,851|
|Setup actions: 8 common bootstraps plus 32 arms, each30|1,200|
|Actual queue takeovers|4,206|
|Actual old-prefix actions|29,442|
|Privileged target-time generations|2,078|
|Native successes / native timeouts / technical failures|1 / 31 / 0|
|Disagreements with independently reconstructed native predicate|0|
|Actual shared simulator exits|1, return code0|

The single smoke's60 measurement plus60 setup actions are excluded. The audit
checks complete condition identities, true information times, noise-window
first/last rows against the saved Gaussian field, shared normalized/physical
bootstrap, each actual committed prefix, and original-task truth for all
113,851 pre-reset witnesses. It reconstructs the registered scene-level
statistics without another model evaluation. All checks pass. Eleven targeted
noise, queue-information, and native-versus-subgoal tests pass, with lint checks.

## 5. Research decision

L17 does not justify promoting noise coupling, scaling incremental visual
training, or declaring a stable learned visual benefit. It also does not prove
that vision is useless or that generative sampling is the unique cause of past
results. The primary interval is wide, reference eligibility fails on these
scenes, and the only original-task success belongs to a privileged arm.

The next priority is a task-matched, reproducible baseline for the complete
outcome one intends to claim. Do not transfer an early-occupancy qualification
to the compound three-orange task. Before another visual-efficacy campaign,
separately register a zero-delay synchronous full-task capability diagnostic on
fresh development scenes and inspect task/checkpoint alignment. Its result must
decide whether the current candidate/task is an appropriate basis, rather than
continually changing noise or increasing forecaster capacity. No checkpoint,
task predicate or physics contract is silently replaced by this decision.

Original L10/L15 reserved tests remain unopened. L15/L15b failed development,
L16 failed privileged-value validation, and L13 failed real-time readiness
remain unchanged. This study pauses physics during inference and therefore
cannot qualify concurrent30Hz execution, which remains a separate requirement.

## 6. Retained artifacts

Under `/home/rp/Workspace/SmolVLA_RTC/artifacts/`:
`m54l17_noise_smoke_v1`, `m54l17_noise_reference_v1` (32 terminal rows,
shared bootstraps/Gaussian fields, raw actions/requests/pre-reset witnesses,
setup records and shared simulator log), and
`m54l17_closed_noise_audit_v1.json` (native reconstruction, component counts,
noise/prefix accounting and scene-level statistics).

See [machine summary](LEISAAC_SO101_L17_SUMMARY.json). No new training or
automatic retry is queued at closure. The next full-task diagnostic requires
its own frozen development protocol, not a rerun of any opened L17 scene.
