# L10: action-aware residual prediction on new development trajectories

Date: 2026-09-07. This is a new learning experiment, not a reanalysis or retuning
of L6/L8 test episodes. Code preparation uses an isolated worktree while the L9
qualification runs; no GPU training runs concurrently with timed simulation.
No task-efficacy campaign starts without separately qualified execution.

## Hypothesis and fixed candidate

L8 did not establish stable action benefit for a latent-reconstruction-only
predictor. Test whether directly supervising the outputs of the frozen action
expert improves action alignment, while retaining a small latent objective.
Keep the WSAGI policy/VLM/statistics/action expert frozen. Initialize a new
273824-parameter predictor from L6 epoch3. The parent checkpoint and all earlier
test conclusions stay unchanged. No new risk threshold or inference gating.

For current state s, current tokens Z, already committed feasible normalized
prefix U, delay d, and paired noise e:

    Z_hat = Z + predictor(Z,U,s,d)
    A_teacher = feasible(policy(Z_future, s, e))   [stop gradient]
    A_student = feasible(policy(Z_hat,    s, e))
    L = mean(abs(A_student[:25] - A_teacher[:25]))
        + 0.001 * SmoothL1(Z_hat, Z_future)
        + 0.01 * cosine_distance(Z_hat, Z_future)

Gradients pass through the frozen policy to the predictor only. Future state is
never an input. The future-visual teacher is an offline reference, not a ground
truth action or task reward. Online computation would remain a single predictor
plus the original policy, not teacher inference.

## Data and fit, frozen before collection

New development env seeds20270110–20270117, policy2510–2517. First6 episodes
train, last2 validation. Each collects at most600 actions from the frozen base
policy in50-action committed chunks, feasible_v1, standard cameras, CPU PhysX,
30Hz simulation, no real-time claim. Use measured states and actual future image
tokens. Record every dispatch and save any interrupted prefix; no seed retries.
Train anchors0,25,...,575 and d1..8 must remain within the same committed chunk.

Fixed400 updates, batch1, AdamW lr1e-4/weight_decay1e-4, grad clip1, RNG2500.
Fixed validation anchors0,50,...,550 and delays1/4/8, paired noise,72 cases.
Validate at updates0/100/200/300/400; choose minimum mean first25-action L1.
The original L6 parent is update0. Require a nonzero selected update, lower
validation error than both identity and the parent, and more than36/72 cases
improving over identity before opening the held-out phase. Otherwise stop as a
development failure; test remains unopened. No learning-rate/epoch sweep.

The first real backward pass must produce finite nonzero predictor gradients,
and all policy parameters must remain requires_grad=false with no gradients.
If the gradient path fails technically, fix that path before interpreting fit
quality; do not substitute a synthetic score or a weaker post-hoc objective.

## Independent evaluation after selection

Only after checkpoint selection is saved/published, collect test env seeds
20270130–20270135, policy2530–2535, with the same600-action protocol. These scenes
are never inputs to training or selection. Fixed12 anchors and d1/4/8 give216
cases, evaluated once with identity, L6 parent, selected student and offline
future-visual reference, same state/noise/feasible projection.

Report first25-action L1 (primary), full50-action L1 (secondary), latent losses,
all per-episode/case results and an episode-cluster bootstrap (50000 draws,
RNG2501). Stable action gate: all216 cases present, at least144 improve over
identity, at least5/6 episode means improve, and95% reduction interval lower
bound>0. Any failure remains negative evidence. This is not task success.

Runtime integration or task-level benefit of a selected student needs a new
binding and a separately frozen evaluation. It does not inherit L7's/L9's
predictor deployment pass merely because its tensor shapes match.
