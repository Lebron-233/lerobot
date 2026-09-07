# L17: distinguish target-time information from generative-noise turnover

Preregistered after L16 closure at402b3b260b232320bcd9e8ab3a295b65d1f5459e.
The user authorizes continued project experiments and GitHub publication without
another reviewer. All old outcomes and unopened L10/L15 student tests stay intact.

## Hypothesis and nonclaims

L16 found no stable incremental task value from exact target-time vision. One
untested control factor is independent flow-noise resampling at each chunk:
overlapping future commands are generated from unrelated Gaussian inputs.
Here we test coupling Gaussian inputs by absolute action index. This is an
experimental noise schedule, NOT RTC inpainting, action continuity enforcement,
a trained predictor, or proof of the cause of L16 failures. All policy weights,
denoising steps, state predictor, actions/limits and camera settings stay fixed.

RTC's original paper motivates distinguishing chunk consistency from latency
(Black et al.,2025, arXiv:2506.07339). Our noise coupling is not that algorithm
and no result from that paper is inherited. No RTC guidance is enabled.

## Frozen 2x2 design and information boundary

Eight NEW scenes20270810--17 and policy3710--17. Four arms:
fresh_state, fresh_oracle, coupled_state, coupled_oracle. Use all four cyclic
orders plus their four reverse cyclic orders, shuffled with3700.
All32 conditions run once. No replacing failures,
changing model or noise rule, or stopping early on favorable outcomes.

Fresh noise uses the unchanged L16 seed policy_seed*100000+request_step.
Coupled noise uses a fixed per-episode Gaussian field indexed by intended
absolute ACTION time: bootstrap noise rows0:50; a chunk taking over at t+7 uses
rows(t+7):(t+57). Construct the field from independent50x32 normal draws keyed
by policy_seed*100000+50*block, so bootstrap first50 is EXACTLY the same as fresh.
All draws are made before any outcome arm. Each chunk retains standard-normal
marginals; overlapping action indices reuse exactly the same noise row. This
does not guarantee identical or continuous generated actions.

State arms compute at t from current visual and causally predicted t+7 state.
Oracle arms freeze the same predicted-state construction at t, execute seven
old committed actions, then use real t+7 VISUAL with that stored predicted state
to generate and stage on time. They do NOT get real t+7 state. Both retain the
original ScheduledActionQueue, actual physical prefix execution, guard2 and
the feasible-before-commitment action contract. All flows use50x32 noise,10
denoising steps. Bootstrap is a shared REAL policy chunk, no scripted trajectory.

Use one shared simulator;30 hold steps for each scene's bootstrap and each arm,
followed by same-seed reset. Check initial state/object/camera geometry and all
first27 commands. Both noise modes load exactly the same policy/state weights.
Capture scene-level reset and pre-first-takeover physical differences; RGB is
not assumed bitwise identical. Setup is separately counted; no unrelated GPU
job is stopped or changed. Inference pauses physics: NO wall-clock30Hz claim.

## Endpoints and full-task follow-through

Unlike L16's early stop, ALL arms continue until native three-oranges-plus-rest
termination or120 simulated seconds/3600 actions. Keep the native success flag
from before auto-reset. Report native full-task success and its restricted time
as distinct secondary outcomes; first occupancy NEVER implies native success.

Primary remains the already defined first ten-tick slow plate-region occupancy.
Record its first attainment independently of later trajectory events. However,
any technical failure invalidates the trial and costs120s in primary analysis,
even if an earlier occupancy existed; report that earlier observation separately.
Timeout without first occupancy also costs120s. All failed trials stay in the
denominator. No contact-release claim is made.

Primary contrast: coupled_oracle minus coupled_state mean restricted first-
occupancy time. Same eight-scene paired bootstrap50000 draws/RNG3701; require
95% upper endpoint<0, both four-scene halves<0, no fewer successes or more
technical failures. Coupled_state eligibility>=7/8 first occupancy successes,
zero technical failures. Secondary: fresh oracle-minus-state, coupling effect
within each context, and difference-of-differences. Publish all regardless of
direction. Native task counts/times and per-takeover physical range-normalized
command jumps are descriptive, not tuned objectives or substitute gate criteria.

If the primary fails, do not scale visual training on that basis. If it passes,
record only privileged-information value under this noise/control condition;
require independent replication before fitting a successor on its held-out data.
L17 cannot turn L15 failures or L13 real-time failure into passes.

## Bounded implementation checks

Tests must reject out-of-bound noise windows, verify exact overlap and shared
bootstrap, unchanged fresh-noise behavior, and distinguish first occupancy from
native full-task success. Then ONE60-step coupled_oracle wiring smoke at
env20270801/policy3701 (no endpoint stopping) must show target-time generations
and takeovers27/54 with the actual seven-row prefix. A technical failure blocks
the cohort; no repeated smoke until success. Record/push the plan and source
before either smoke or outcome execution. Audit final raw commitments and noise
labels without rerunning any model after outcomes close.
