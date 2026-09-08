# L20 closed: shorter feedback span helped three scenes but did not qualify the native-task baseline

Date: 2026-09-08. Execution source: `2afee34e55da3ed08bfe2670e4a14252bb7ec2cc`.
The previous tool-session handle 831 is no longer retained. The original output
directory contains a complete, normally closed sixteen-condition summary and
all per-condition artifacts. A new read-only reconstruction verified every row,
source, seed, task literal, action span, pre-reset native witness, and simulator
exit. No experiment was restarted, replaced, or resumed from guessed state.
The previous access failure is not counted as a task or technical failure.

## Frozen comparison

Both arms use the same WSAGI policy, original task text, zero initial pose,
50-action generation, ten flow steps, feasible execution contract, full cameras,
and 120-s simulated-time bound. The only registered control-setting difference
is consuming ten rather than fifty generated actions before querying again.
That also changes stochastic query timing: it is not a single-factor proof of
the effect of observation freshness. No forecaster, RTC or oracle is used.
Physics pauses during inference; this is not wall-clock realtime qualification.

| Consumed span | Native three-oranges-plus-rest success | First settled-region subgoal | Technical failures | Mean restricted native time |
|---|---:|---:|---:|---:|
|50|0/8|5/8|0/8|120.000000 s|
|10|3/8|5/8|0/8|87.741667 s|

Failures are retained with the registered 120-s cost. All eight paired starting
state/object/camera geometries match; rendered images and later trajectories
are not asserted bitwise identical.

| Scene | Span50 actions / native result | Span10 actions / native result |
|---|---|---|
|20271110|3600 / timeout|3600 / timeout|
|20271111|3600 / timeout|3600 / timeout|
|20271112|3600 / timeout|872 / success (29.066667 s)|
|20271113|3600 / timeout|1494 / success (49.800000 s)|
|20271114|3600 / timeout|692 / success (23.066667 s)|
|20271115|3600 / timeout|3600 / timeout|
|20271116|3600 / timeout|3600 / timeout|
|20271117|3600 / timeout|3600 / timeout|

Span10 minus span50 paired native-time differences are
`[0,0,-90.9333333333,-70.2,-96.9333333333,0,0,0]` s. The mean is
-32.258333 s (26.881944% lower restricted mean). The preregistered descriptive
eight-scene bootstrap, 50,000 draws/RNG4001, gives [-63.766667,-8.775000] s.
Only three scene pairs have a nonzero difference. This small development
bootstrap is not independent confirmation or a guarantee of a population effect.
Most importantly, the unchanged eligibility requirement is >=7/8 native successes
and no technical failures: **neither arm qualifies**. The bounded feedback-span
investigation is closed, not extended into a span search to obtain a pass.

## Verified physical accounting

49,858 measured actions and 480 separately recorded setup actions; sixteen
independent simulator exits and metric closures, all normal. Native success was
reconstructed at every executed tick with zero disagreements. The thirteen
native failures never simultaneously placed all three oranges in the native
region; final arm rest alone cannot rescue them. Projected scalar components:
span50=309, span10=26 (not numbers of whole actions). No reserved L10/L15 test
was opened, and no model or normalization statistics were changed.

Artifacts: `artifacts/m54l20_feedback_span_v1`, including original
manifest/summary and all sixteen result/ticks/setup/log/image directories.
The closure reuses `run_so101_sync_capability.describe`, its native rule check,
and the exact registered paired-statistic computation; no model rerun is needed.

## Additional source-alignment check, not outcome fitting

A separate read-only calculation on all 36,293 rows of the pinned public
training source `LightwheelAI/leisaac-pick-orange@fa6e0625d814352b8e6ee1c6d2482194e4da8ed3`
compared state/action means and population standard deviations with the frozen
WSAGI processor statistics. Maximum mean discrepancies are 4.295e-6 (state)
and 1.700e-5 (action) in native motor coordinates; maximum relative std
discrepancies are 1.230e-7 and 1.180e-7. The action statistics in the preprocessor
and postprocessor match exactly. This supports numerical statistical alignment,
not byte-exact identity of the author's unpinned historical dataset revision.
No normalizer correction is justified by this check and none was made.

## Research decision

Preserve the three actual native successes and the favorable development mean;
do not call them reliable full-task competence or future-latent efficacy. The
registered pose, prompt and feedback-span investigations have not supplied the
required reliable native-task reference. Further residual fitting or another
PickOrange parameter grid is not the next step. Qualifying a separately bound,
task-matched SmolVLA benchmark candidate is preferable to rewriting these failed
gates. All existing LeIsaac results and exact candidate identities remain intact.
