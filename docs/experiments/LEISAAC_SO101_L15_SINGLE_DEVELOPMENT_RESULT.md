# L15 single-case development closed: lower mean, insufficient case coverage

Date:2026-09-07. Collection and training source:
`1931d379dffb924edce04b60eb22efd339c29ca7`. This is development, not a held-out
claim or a task comparison. The L15 test scenes20270630--35 remain unopened.

Eight new state-only controlled-d7 trajectories completed9600 task actions plus
240 separately recorded hold/setup actions. Each completed1200 actions and44
current/future request pairs. The352 pairs split264 training/88 validation;
two fixed noise draws give528 training and176 validation action cases. No
technical collection failures, missing targets or terminal-prefix censoring
occurred. All episodes ended at the40-second collection bound, not native
task termination; these are not eight failed full-task trials.

A read-only audit matched every recorded normalized prefix to the actual seven
old actions executed before the future target and takeover. No visual predictor
controlled collection, and no future-state target enters the visual student.
Teacher, state-only, old-parent and student action decoding share the SAME
causally predicted state. Only their visual input differs.

The visual student began exactly at the state-only reference: the measured
zero-residual maximum action difference was0.0. All1600 fixed updates completed.
The real first backward produced student gradient norm0.93511355 and zero
policy gradients; VLM/action expert and state/parent predictors stayed frozen.

| Update | Validation action MAE | Improving cases /176 | Eligible |
|---:|---:|---:|---|
|0|0.076867422|0|no|
|200|0.072300960|95|no|
|400|0.075654017|84|no|
|600|0.075626011|92|no|
|800|0.072415122|97|no|
|1000|0.075921704|88|no|
|1200|0.073993231|87|no|
|1400|0.072529644|90|no|
|1600|0.069882075|83|no|

State-only reference MAE is0.076867421507 and old L6-joint is0.075359314622.
The diagnostic best1600 improves the equal-episode mean9.0875% over state-only;
both validation episode means improve. However only83/176 cases improve, and
no checkpoint reaches the required118/176. Maximum coverage is97/176 at800.
The frozen eligible-first selection was independently recomputed and agrees.
**Qualification fails; this diagnostic best is not enabled for test or tasks.**

Local artifacts: `m54l15_onpolicy_development_v1`,
`m54l15_visual_increment_training_v1` (all losses/checkpoints and selection),
and `m54l15_closed_development_audit_v1.json` under the project artifacts root.
Training wall time1099.233s includes reference preparation and validation;
peak allocated CUDA memory1970355200bytes is not an inference benchmark.

The separately specified [L15b profile](LEISAAC_SO101_L15B_BALANCED_UPDATES_PLAN.md)
is now eligible to execute because the original fixed run is complete with no
qualified model. It averages four distinct training episodes per optimizer step
with the same1600 forward/backward-example budget, not additional data or a
relaxed success threshold. It must preserve this result and keep the reserved
test unopened unless its own unchanged eligibility requirements are met.
