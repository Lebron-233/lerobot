# L8: qualify task ability before claiming predictor benefits

Starting source: `95f8e6f3d817e349950788f1cfa032fdc3678ac6`.
Date: 2026-09-07. Local DevSpace execution; user authorizes continued project work.

## Question and interpretation

L7 establishes bounded online execution, not task efficacy. L8 first asks whether
the frozen WSAGI policy has usable pick/place ability across new development
initializations. The nine L4 configurations used one common development seed;
they do not measure reliability across scenes. We will not tune on L6 test
episodes or reinterpret the L7 identity action-limit failure as a benefit.

The publisher's currently retrieved LeIsaac-Training README reports zero full
three-orange rounds in its strict 20-round SmolVLA evaluation, despite an earlier
model-card 2/5 figure. Its per-orange measure is not our official environment
success. These are publisher reports, not reproduced results. Source:
https://github.com/vitorcen/LeIsaac-Training (retrieved 2026-09-07).

## Phase A: baseline diagnosis (development, no efficacy claim)

Freeze the same WSAGI revision, native processors, language, 50-step chunks,
CPU PhysX/GPU RTX, standard cameras, 30 Hz timebase, zero initial pose and 120 s
environment horizon. Run synchronous capability episodes at environment seeds
20261010, 20261011, 20261012, 20261013; policy seeds 2000,2001,2002,2003.
No clipping, changed joint limits, scene simplification, reward or success edits.
Each episode is run once. All failures, including technical failures, are kept.

Add observational task evidence at the native termination check, before Isaac's
automatic reset. Retain the unchanged native success predicate and capture the
three orange positions/velocities, plate position and measured joint positions.
Record official box membership separately from a stricter diagnostic:
radial distance <0.10 m, |height difference| <0.07 m, speed <=0.05 m/s for ten
consecutive control steps. The latter is called settled placement, not native
success. A historical near-gripper boolean alone is not a successful grasp.
Record lift >=0.03 m above the settled tabletop height only as manipulation
progress, not success. No sticky-OR across different oranges defines success.

Development decision: if >=3/4 episodes achieve a settled single-orange placement,
qualify that explicitly named subtask on independent seeds before testing delay
benefit; three-orange+rest success remains reported separately. Otherwise diagnose
one concrete physical/interface cause or prepare an independent task-capable
candidate. Do not promote a partial metric after seeing results or keep sampling
until a lucky full success appears.

## Phase B: causal evidence, conditional on a qualified base

Freeze a new independent protocol before outcome evaluation: matched policies,
processors, initial scene states, control cadence, delay schedule and paired
noise; compare synchronous/no-delay reference, stale identity and predicted.
Use genuinely future-blind committed prefixes. Future RGB/latent oracle may be
used only in a separately labelled offline diagnostic, never the deployable arm.
Report per-episode outcomes, action feasibility, settled placements, full native
success, control and inference failures, and episode-level uncertainty. Same seed
does not imply identical rendered RGB or perfect trajectory matching.

## Execution and preservation

Keep every actual run tied to a clean source commit and fresh artifact directory.
Only task-evidence instrumentation needed here is enabled; old L7 defaults remain.
Test native-boolean preservation, pre-reset provenance and consecutive placement
counting before actual runs. No new model training is part of Phase A. Broader
project authorization remains in force, but any new candidate/training/benchmark
identity is written down before execution. No external Pro review is required.

## First-run instrumentation correction

The first seed20261010 execution completed 3600 actions and native timeout, but
the new witness was null: Isaac's manager deepcopies configuration callables.
The server held the construction-time object instead of the live manager term.
Bind the actual success callable via `termination_manager.get_term_cfg` after
construction and reject missing evidence. Keep the first outcome as a timeout
with unavailable settled-placement evidence. Do not rerun its policy. Continue
the three remaining predeclared seeds; the missing secondary metric is not a PASS.

## Phase A result and independently named feasible-action successor

The first complete native task success is now observed: seed20261011 finished at
1008 actions, all three oranges simultaneously settled for the preceding ten
steps, with a pre-reset native-success witness. Seed20261010 timed out; 20261012
stopped after 28 actions on shoulder_lift=-101.7171 degrees; 20261013 stopped
after 2058 actions on wrist_flex=95.0372 degrees (one orange already settled).
Thus the original Phase A reliability gate is NOT met. These outcomes are fixed.

The next concrete intervention is **so101_feasible_actions_v1**, not a predictor
benefit claim. The old strict transport and all L7 results stay unchanged. The new
contract projects native motor targets into the existing physical feasible set
BEFORE queue commitment, and inverse-normalizes the projected targets using the
same checkpoint action mean/std. The queue stores both projected normalized and
physical actions, so the predictor sees the prefix actually committed to execute.
Post-dispatch clipping, changing joint limits, and preserving an unprojected
predictor prefix would be incorrect. Both comparison arms must share projection.
Unmodified in-range normalized values are preserved exactly. Projection counts
and magnitudes are recorded and reported separately from task/predictor effects.

Qualify this independent execution contract with six fresh synchronous baseline
episodes: environment seeds20261014–20261019, policy seeds2014–2019, same120s,
same model/environment settings and pre-reset task evidence. No model or predictor
training. The usable single-placement baseline gate is >=4/6 episodes reaching
settled placement without technical failure; full three-orange+rest outcomes are
reported independently. This is development qualification, not held-out efficacy.
Only then freeze new independent comparison seeds and trial order. This change
is covered by the user's explicit project-wide authorization; no further approval
loop is introduced.

## Qualification result and frozen efficacy protocol

The six new feasible-action baselines completed without technical failure. Five
of six reached a settled placement; none completed native three-orange+rest.
The declared single-placement capability gate passes (5/6), NOT a full-task
reliability gate. Only seed20261017 needed projection: ten scalar components,
largest adjustment3.1443 motor units. Different scene sets do not establish a
causal improvement attributable to projection.

The independent efficacy task is **pickorange_first_settled_v1**: in the same
three-orange scene, complete the first grasp/place cycle, defined as any orange
remaining in the declared settled-placement region for ten consecutive ticks.
Stop at this observable subgoal, native termination, technical failure, or120s.
Do not report this as original three-orange task success. Native success flags
and pre-reset witnesses are unchanged; early subgoal stop leaves full-task
outcome unobserved. This saves time without changing the first-placement endpoint.

Run12 independent blocks, environment seeds20261101–20261112 and policy
seeds2101–2112. Each block contains `sync`, `identity`, `predicted` with the same
feasible-action contract. The six permutations of these modes are repeated twice
and shuffled with Python Random(2088) before any test outcome. All use120s bounds,
30Hz, standard cameras, CPU PhysX, GPU policy, zero pose, frozen WSAGI and (only
predicted) the unchanged L6 epoch3 predictor. No fitting or test-driven adaptation.

Primary task endpoint: restricted time to first settled placement, in simulated
seconds; noncompletion or technical failure is scored120s. The primary contrast
is predicted minus identity, paired by block. Report success counts, actual
technical failures, native flags, and physical execution counts alongside it.
The synchronous50-action policy is only a task-capability reference, not an
isolated delay ablation: its replanning schedule differs from the async engine.
Only identity versus predicted shares the same production scheduling algorithm.

Use50000 paired episode bootstrap draws, NumPy seed2089, percentile95% intervals.
Stable task-benefit evidence requires the primary difference's upper interval
bound below zero, negative mean differences in both predeclared six-block halves,
no decrease in success count and no increase in technical failures. Otherwise
report the outcome as negative or inconclusive, not a successful efficacy proof.
Require matching initial physical state/object/camera poses; rendered pixels are
not assumed identical. Keep every attempted condition, including technical stops.

Independent mechanistic audit: six new baseline-generated600-step trajectories,
environment seeds20261130–20261135, policy2230–2235, same feasible contract. No
real-time claim for this token-collection diagnostic. Use anchors0,50,...,550,
d=1/4/8 (216 cases if all eligible). Each prefix is projected normalized actions
from the already generated50-step chunk, not future replanning. Compare identity
and frozen predicted contexts with the actual future-visual oracle, holding
current normalized state, language and per-case noise fixed. Primary action error
is normalized feasible-action L1 on the first25 rows; report all50 secondarily.
No future state enters the compared calls. Aggregate uncertainty resamples the
six episodes, not216 correlated frames. Stable action evidence requires positive
95% reduction interval, >=5/6 improving episode means and >=2/3 improving cases.
Missing predefined cases are reported, not replaced. This audit does not replace
the actual task experiment or provide a new tuning/selection set.
