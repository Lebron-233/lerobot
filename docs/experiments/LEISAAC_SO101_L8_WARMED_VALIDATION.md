# L8 warmed execution successor: initialization is not task efficacy

Date: 2026-09-07. The initial frozen L8 campaign and independent action audit are
complete at `bec8cc944169e9f70edc29acb61e1e806ab3e62a`; all results stay immutable.

## Closed initial results

The synchronous reference reached the predeclared first-settled-region endpoint
in8/12 new scenes. Identity reached1/12, with11 technical failures; predicted
reached0/12, with12 technical failures. Mean restricted times are55.1444s,
110.4528s and120s respectively. The predicted-minus-identity mean is+9.5472s,
paired-bootstrap95% interval[0,28.6417]s. Its stable task-benefit gate fails.
These operational failures cannot establish that altered visual context harmed
grasping: several predicted trials never entered control. The first two startup
probes had first predictor-forward durations86.85/88.19ms and total240.48/247.74ms,
requiring9 steps, beyond the unchanged8-step cap. Numerous identity timing
failures also occur on the first physical action after loading the model.

The independent six-episode/216-case audit is also closed: first25 feasible
normalized-action L1 is0.09352348(identity) versus0.09164213(predicted), a2.0116%
reduction. Five of six episode means improve, but only112/216 cases improve;
episode-bootstrap95% reduction interval[-6.4660%,8.8827%]. The stable action gate
fails. Do not reopen these cases to fit or select any weights or risk threshold.

## Narrow initialization correction

Introduce `warmed_v2`, an explicitly recorded setup profile, not a change to
forecasting math or policy weights. All matched modes execute30 measured-in-logs
hold-position simulator steps after policy allocation and (for async modes)
after the existing engine startup finishes and is paused/reset. Then reset the
same environment seed before the task clock. These are setup physics calls,
not task actions, hidden warmup trimming, or a changed simulation timestep.

For predicted only, make exactly one discarded predictor forward on the first
actual observation's encoded tokens/current model-ready state, with a fixed
zero-valued eight-row prefix and valid d8 mask, before the production startup
probe. This is kernel initialization only, not a forecast used for control; no
action is dispatched or inserted into the queue. Record its wall time and call
count separately. Re-seed the measured policy RNG after all setup as before.
The subsequent real startup probe must still pass the original cap8/margin1
test. No repeated probes until success, changed late rule, FPS, or lost-slot
threshold is introduced. Native/strict defaults and earlier results stay intact.

The first-settled-region endpoint remains exactly as frozen previously. It tests
an orange staying within the declared plate region at low speed for ten ticks;
it is not a three-orange+rest result or a contact-sensor proof of release.

## Bounded qualification and independent outcomes

First run identity then predicted, each at envseed20261220/policy2420 and600
task ticks (60s environment bound), with task evidence and the common feasible
action contract. No first-placement early stop in these two engineering checks.
Both must complete their bound or native success with no technical failure;
the predicted arm must show actual planned predictor calls and takeovers.

Only if both pass, run a new12-block three-mode comparison: env20261201–20261212,
policy2401–2412, same shuffled six permutations twice (2088), same120s endpoint,
same uncertainty/gates and all failure handling. The only new execution change
is the explicitly described setup profile. No old trial is rerun or replaced;
do not pool the two startup protocols as one benchmark. A failure of this
qualification is a boundary to report, not permission to relax real-time gates.

The user has already authorized project work. No external review or new
authorization is needed for this targeted initialization correction. No neural
training, new predictor, hardware purchase or physical robot is involved.
