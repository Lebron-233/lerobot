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
