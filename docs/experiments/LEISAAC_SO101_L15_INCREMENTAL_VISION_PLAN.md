# L15: visual contribution conditional on the state-only controller

Date: 2026-09-07. L12/L14 are closed and are not fitting data. This protocol
tests a new visual student, not another selection of the immutable L6 parent.
The user authorizes implementation, local experiments and GitHub records.

## Question and change

L14 did not establish additional visual value over state-only compensation.
Two possible mismatches deserve an independently specified experiment: prior
visual training trajectories followed synchronous 50-action chunks, whereas
deployment replans with committed prefixes; and L10's action supervision kept
current state, whereas the stronger state-only comparator uses predicted state.
Neither mismatch is declared the proven cause of previous negative results.

Collect new trajectories using the existing state-only controlled-delay
controller, with L12 epoch29 and seven simulated delay steps. Preserve the
original scheduled queue, 30Hz simulation, full cameras, physical assets,
feasible-before-commitment actions and separately logged thirty setup steps.
Physics pauses during inference: no wall-clock realtime claim is possible.
At each planned request save the current tokens, state, its frozen predicted
state, language, and seven committed normalized actions. Capture target tokens
at t+7 BEFORE applying the new chunk. Audit that the seven actions between t
and t+7 exactly equal the saved prefix. Never use target tokens to control
these collection trajectories. Native episode terminals prevent post-reset
targets; incomplete terminal prefixes remain explicitly accounted for.

New development seeds 20270610--20270617 / policy3510--3517, six training
episodes and two validation episodes. Maximum1200 measured actions per episode,
no first-placement early stopping; stop on native terminal or technical error.
Keep all complete requests, including poor task trajectories. Each episode must
have at least eight complete pairs; a technical collection failure blocks the
study rather than inviting seed replacement. Different lengths are reported.
The integers are RNG identifiers, not future scheduled dates.

## Frozen teacher and one bounded student

The primary reference is the frozen policy using actual future visual tokens
and the SAME predicted state available to the state-only controller at t.
Thus teacher/student/baseline differ in visual input, not proprioception. This
conditional visual-reference metric differs from the coherent true-future-state
teacher used in L12, and from the current-state teacher used in L10. Report
those distinctions explicitly; a lower reference error is not task success.

The student retains the273824-parameter visual predictor architecture. Initialize
its hidden parameters from immutable L6 but zero its final residual projection:
before fitting it is exactly state-only, not an already harmful visual update.
Only student parameters train. WSAGI policy/VLM/expert, processors, L12 state
forecaster and L6 comparison weights stay frozen. Input state to the VISUAL
student stays current state; predicted state goes only to the action decoder.

Fixed1600 AdamW updates, lr1e-4, weight_decay1e-4, gradient clip1, seed3500.
Uniformly sample training episodes and their complete queries; choose one of
two fixed paired-noise draws per query. Batch1 avoids batch-dependent decoder
rounding. Cache the no-gradient teacher and state-only outputs before fitting.
Loss = e/(b+0.02) + 0.5*relu((e-b)/(b+0.02)) +
0.0005*latent_SmoothL1 +0.005*latent_cosine_distance. Here e and b are first25
feasible normalized-action MAE to the conditional reference, for student and
state-only respectively. No risk score, runtime gate or action correction is
learned; risk thresholds remain null.

Evaluate at update0 and every200 updates on two held development episodes,
both fixed noise draws per complete query. Record all checkpoints/metrics.
Eligibility: strictly lower mean error than both state-only and old L6-joint,
each validation episode mean lower than state-only, and at least2/3 individual
noise/query cases improved. Select the lowest equal-episode mean among eligible
checkpoints. Without any eligible checkpoint publish the diagnostic best but
leave the new held-out scenes unopened. Do not deploy an ineligible student.

## Independent evaluation, conditional on eligibility

Freeze and publish the selected checkpoint before collecting six new state-only
trajectories: env20270630--20270635 / policy3530--3535. All collection rules and
1200-action bounds match development. No old L10 reserved tests are consumed.
Evaluate once with two newly keyed paired noise draws per complete query.
Report state-only, old L6-joint, new student, every episode and every case.

Primary stable incremental-action criterion versus state-only: all six complete
episodes, at least2/3 improved noise/query cases, at least5/6 improved episode
means, positive lower95% episode-cluster bootstrap reduction bound (50000 draws,
RNG3501). Report the contrast to L6 even if unfavorable. Use equal episode
weights for the mean; overlapping requests/noise draws are not independent
experiments. All failures and terminal-censored query counts stay visible.
These are d7 controlled-age claims only, not a d1--8 generalization.

Only a qualified new frozen student can motivate a separately preregistered
controlled task comparison against state-only, with identical noise, timing,
projection, task definition and fresh scenes. Neither positive action error
nor an oracle comparison will be called stable task benefit. No realtime
qualification, base-policy training, old-test refitting or unrelated GPU-job
interference is authorized by this experiment's technical scope.
