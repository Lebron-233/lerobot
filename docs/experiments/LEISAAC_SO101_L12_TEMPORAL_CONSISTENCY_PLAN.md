# L12: test temporal consistency before another visual-student search

Date: 2026-09-07. User authorization covers continued project implementation,
training and validation through DevSpace; no external Pro review is requested.
L11 and all opened L6/L8 test evidence stay immutable and are not fitting data.

## Hypothesis and explicit scientific change

The current production branch predicts visual tokens at takeover t+d but supplies
the **current** normalized joint state to the frozen policy. This is observed in
`PredictiveAsyncInferenceEngine._run_request`; it is not proof that this mismatch
caused the L11 sign reversal. Test that hypothesis rather than claim it as fact.

L12 is a separately named **joint future-context successor**, not another result
for visual-only M3. It adds a small supervised proprioceptive state residual
predictor using ONLY current model-ready state, already committed normalized
actions and delay. The immutable L6 visual predictor, policy, VLM, action expert,
normalizers, task and action feasibility contract remain frozen. Neither future
state nor future RGB is observable online. True future state/visual tokens are
used only as supervised targets and offline oracle controls. No risk threshold,
full RTC, new queue, action-expert training or horizon beyond eight is introduced.

This work may establish that state compensation alone explains an improvement.
In that case do not attribute it to future visual prediction. Keep identity,
visual-only, learned-state-only and joint contexts separate.

## Development (no held-out claims)

Reuse only the already designated L10 **development** cache, episodes0–5 for
fitting and6–7 for validation. Its reuse is explicitly development, not blind
confirmation. Never open L10 reserved test, L6/L8 test or L11 trajectories to fit.
Eligible labels require all committed actions t:t+d from one already generated
chunk, d1..8, and t+d before terminal/reset. State inputs/targets are the actual
32-dimensional model-ready vectors, first6 active and remaining26 padding.

Model: MLP [state6, flattened masked8x6 actions, mask8, delay/8] ->128->128->6
state residual, SiLU activations, zero-initialized final layer; append unchanged
padding. Seed3100, AdamW lr0.001/weight_decay0.0001, batch256,30 fixed epochs,
gradient clip1. Choose lowest validation mean absolute state error. No action
metric is used to select this checkpoint, no policy parameters are trained.

On both validation episodes evaluate the same12 anchors0:50:550, delays1/4/7/8,
paired noise310000+episode*10000+anchor*10+d. Freeze selected state weights first.
The **new** coherent teacher is policy(Z[t+d],state[t+d],same language/noise).
Compare identity, frozen L6 visual-only, learned-state-only, joint, oracle-visual-
only and oracle-state-only. Report all cases, first25 and full50 feasible
normalized-action L1; separately state prediction error and per-delay results.
This teacher differs from the old future-visual/current-state action audit. Do
not compare percentages across those two definitions as equivalent evidence.

Permission to collect fresh test: state MAE improves versus persistence in both
validation episodes, joint mean action error beats BOTH identity and visual-only
in both episodes, and joint beats identity in at least64/96 validation cases.
If this fails, preserve the result; any next development protocol must be a new
explicit experiment, without opening the reserved test to pick a model.

## Independent test, only after development qualification

Collect six fresh600-action/601-observation trajectories under unchanged frozen
WSAGI synchronous feasible_v1, standard cameras, CPU PhysX/RTX,30Hz and simulator
tensor threads1. Seeds20270320–20270325, policy3120–3125 are RNG integers, NOT
future scheduled dates. Dataset creation is non-realtime and not task evaluation.
Preserve every dispatch/error; no seed substitution or automatic retry. New
namespace `m54l12_joint_context_test_v1`. No old test files are referenced.

Use12 anchors and four delays above, giving288 prespecified cases. Same frozen
state/visual predictors, offline coherent teacher and all six comparison arms;
paired noise320000+episode*10000+anchor*10+d. The primary metric is first25
feasible normalized-action L1; full50 and latent diagnostics are secondary.

Stable joint-versus-identity AND joint-versus-visual-only requires complete288
cases, improvement in >=192 cases for each contrast, positive mean reductions
in >=5/6 episodes, and positive lower bounds of episode-cluster bootstrap95%
relative reductions (50000 draws, RNG3101). Also report joint-versus-state-only
as a mandatory attribution contrast, without hiding negative visual contribution.
These are small correlated trajectory clusters, not288 independent trials.

Only after a frozen test report may an independently bound joint runtime and new
task cohort be planned. No online future-state binding is enabled by this initial
offline protocol. Old risk thresholds remain null. Failed earlier conclusions
and the negative L11 stable-effect gate are not rewritten.
