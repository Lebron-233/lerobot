# L19b closed: native-rest initialization did not qualify complete-task control

Date: 2026-09-08. Execution source
`3e1b3a919607af0b04736c6c316b7893e26e4d1e`.
The once-only contingency was registered before L19 closed and executed only
after the sixteen-condition medoid comparison failed its native-task prerequisite.
All eight new conditions have now terminated, with no replacements or retries.

## What was held fixed

The original source-defined rest pose is (0, -100, 90, 50, 0, -10) physical
joint degrees. Each task-start measured state was checked against that pose;
the gripper transport value is 0 rather than -10. This is initial preparation,
not a forced return-to-rest during task execution. Thirty hold setup actions
were separately recorded before the same-seed reset. Objects and camera
randomization, physical step size, feasible action mapping, WSAGI weights and
original dataset instruction were unchanged.

Each condition uses synchronous fifty-action execution and runs to native
three-oranges-plus-rest success or 120 simulated seconds. There is no forecaster,
oracle, RTC or new training. Physics pauses for model computation, so this is
not a wall-clock real-time qualification. These scenes differ from L19: do not
interpret cross-cohort success differences as a causal pose-effect estimate.

## Complete results

| Scene seed | Native success | First-region subgoal | Executed actions | Maximum simultaneous native-region oranges |
|---|---|---|---:|---:|
|20271030|no|yes|3600|2|
|20271031|no|yes|3600|1|
|20271032|no|yes|3600|1|
|20271033|no|no|3600|0|
|20271034|no|yes|3600|1|
|20271035|no|no|3600|0|
|20271036|no|yes|3600|2|
|20271037|no|no|3600|0|

Native success is **0/8**, first-region occupancy is **5/8**, and technical
failures are **0/8**. Restricted native time is 120 seconds in every condition.
The fixed >=7/8 native-success / zero-technical-failure gate therefore fails.
No trajectory reached three oranges simultaneously inside the native region;
issuing a final rest command would not convert these failures into successes.

## Accounting and decision

Read-only reconstruction matched all eight terminal reports to their original
pre-reset witnesses and expected prepared states: 28,800 measurement actions,
240 setup actions, zero native-rule mismatches, and eight independently clean
simulator exits/metric closures. The feasible projector changed 47 scalar action
components, not 47 entire actions. Every recorded condition remains in the
denominator. No reserved L10/L15 student test was opened.

This closes the bounded source-defined initialization investigation. Neither
the recorded-first-state medoid nor native rest established a reliable complete
task baseline. It does not show that all possible initializations are equivalent
or rule out broader distribution mismatch. Do not continue a pose/prompt grid.

The separately registered [L20 feedback-span contingency](LEISAAC_SO101_L20_FEEDBACK_SPAN_PLAN.md)
was prepared before this cohort finished (Issue comment 5574340458). Its
prerequisite now holds. It compares one fixed ten-action consumption alternative
with fifty-action consumption on new scenes, keeping zero initialization and
all frozen models. It cannot overwrite the present failed result or establish
forecasting efficacy merely by qualifying a synchronous baseline.

Local artifacts: `artifacts/m54l19b_native_rest_v1` and
`artifacts/m54l19b_closed_rest_audit_v1.json` under the project root.
