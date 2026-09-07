# L19 closed: a source-defined initial pose did not qualify native-task control

Date: 2026-09-08. Execution source:
`74a0e9670eeadc204df9592b69c129d83b56dbcc`.
All sixteen registered conditions finished once with that source unchanged.
This is a frozen-policy initialization-development study, not a forecasting
effect test or real-time qualification. The L18 and earlier results are intact.

## Intervention actually executed

The runner recomputed the medoid from all sixty first-state rows of the pinned
published dataset and confirmed episode 9 before any environment dispatch.
`training_medoid_v1` changes only the native robot initial joint configuration.
Every reset verifies the measured joint positions; the adapter does not rewrite
a mismatching observed state or force the robot back to rest during task control.
No demonstration action sequence, object relocation, new policy weights or
normalization statistics were used. The normal 30 hold setup actions were
logged separately before the same-seed reset.

The medoid uses physical joint degrees approximately
[-12.2058693, -47.5487595, 47.2539429, 47.5971367, 0.1644653, -0.7079082].
The last value is a joint angle; its transport/native motor gripper coordinate
is 8.4473562, not a second degree-to-motor conversion.

All eight paired initial object positions and front-camera poses matched.
Robot state and wrist-camera pose deliberately differ between preparations.
There is no claim of bit-identical images, identical later dynamics, or that
matching one recorded position reproduces demonstration velocity/contact history.
The checkpoint training config does not pin the dataset revision, so the
published source is not asserted to be the author's byte-exact training snapshot.

## Complete native-task outcomes

Same frozen WSAGI checkpoint, original dataset instruction, synchronous
generated/executed chunk length 50, native three-orange-plus-rest success rule,
CPU PhysX / GPU RTX, standard dual 640x480 cameras and 120 simulated-second bound.
Model computation pauses physics; 50-action execution still has within-chunk
observation aging. Failed conditions receive restricted native cost 120 seconds.

| Preparation | Conditions | Native successes | First-region subgoals | Technical failures | Mean restricted native time |
|---|---:|---:|---:|---:|---:|
|Zero joints|8|1|7|0|109.808333 s|
|Training-first-state medoid|8|1|5|0|113.487500 s|

Both successful conditions were scene 20271011: zero completed in 1,154 actions
(38.466667 simulated seconds), medoid in 2,037 actions (67.900000 seconds).
Every other native outcome was a timeout after 3,600 actions. None was replaced.

The prepared-minus-zero native time differences are
[0, 29.4333333333, 0, 0, 0, 0, 0, 0] seconds. Their mean is +3.679167 seconds;
the predeclared descriptive paired-bootstrap 95% interval is [0, 11.037500] s.
This small development cohort does not establish a universal harmful effect
of prepared states. It does not establish an advantage or reliable native-task
performance for this specific medoid either. Both conditions fail the fixed
>=7/8 native-success / zero-technical-failure requirement.

All fourteen failed trajectories never simultaneously put all three oranges
inside the original task's placement region. These failures cannot be converted
into native successes by counting separate historical placements or merely
issuing a final return-to-rest command. First-region occupancy remains a
descriptive intermediate endpoint, not native task completion.

## Accounting and checks

A closed read-only reconstruction matched all sixteen per-condition records,
the complete native/subgoal endpoints, expected prepared positions and source
identities, all paired statistics, and the original native rule at every tick.

| Item | Verified result |
|---|---:|
|Measurement actions|53,591|
|Separately recorded hold setup actions|480|
|Native-rule reconstruction mismatches|0|
|Independent simulator exits / closed metric sinks|16 / 16|
|Projected action components, zero / medoid|19 / 219|

Projection counts are scalar components, not numbers of entire actions. Both
conditions retain the same explicit feasible-action execution contract. The
19 targeted initialization, coordinate, original-camera-reset and synchronous
literal tests passed before execution, together with lint/format checks. No
previously opened forecaster test set or new neural-network fit was used.

Artifacts under `/home/rp/Workspace/SmolVLA_RTC/artifacts/`:
`m54l19_prepared_start_v1` (all manifests/results/ticks/setup logs/images), and
`m54l19_closed_initialization_audit_v1.json` (closed raw-evidence reconstruction).

## Narrow continuation, already registered before L19 closed

The [L19b native-rest contingency](LEISAAC_SO101_L19B_NATIVE_REST_PLAN.md) was
published as Issue comment 5574061682 before all L19 outcomes were available.
Its prerequisite now holds. It tests the existing source-defined native-rest
pose on eight new scenes only, with the same complete native endpoint. It is
not a replacement of this cohort and cannot make these failed gates pass.
After that bounded contingency, do not perform an open-ended pose/prompt search.
L10/L15 reserved tests remain unopened and no forecasting benefit is claimed.
