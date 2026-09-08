# LIBERO training source confirmed and execution contract traced

Date: 2026-09-08. Follow-up to the completed independent preparation, at source
`e88e83a5b990f65f22eddd2bc1266742fc1bf4f4`.

**The training dataset identity is confirmed as `HuggingFaceVLA/libero`.** Its
pre-checkpoint v2.1 metadata reproduces all 30 saved state/action mean and standard
deviation values exactly after float32 conversion. The checkpoint author's
evaluation guidance, the contemporaneous LeRobot port and the installed native
controller establish the camera/state/action contract for this independent
reference. This resolves the previous blanket training-source and execution-
semantics unknowns. The complete historical training recipe and author evaluation
manifest remain unavailable; their limits are described below.

The one registered 20-action smoke remains the only model-driven run. No new
simulator, policy inference, qualification, confirmation or training run was
started. `baseline_qualified=false`, `realtime_qualified=false`,
`predictor_benefit_tested=false`, `risk_thresholds=null`.

## Training source and revision evidence

On 2025-09-26, uploader and LeRobot contributor Jade Choghari explicitly linked
this checkpoint to the [HuggingFaceVLA/libero training dataset](https://github.com/huggingface/lerobot/issues/1369#issuecomment-3337263535)
and directed users to the official LeRobot LIBERO evaluation guide. The earlier
[checkpoint announcement](https://github.com/huggingface/lerobot/issues/1369#issuecomment-3323183721)
also names the same model. These are author statements, independent of the generic
model card. The reported success range has no accompanying episode manifest and
is not used as a local qualification result or threshold.

The Hub's [fixed policy history](https://huggingface.co/HuggingFaceVLA/smolvla_libero/commits/6721902bc4d61e50a3bfdb11dfb4cb626f05d102)
contains the initial repository creation and the processor migration/upload on
2025-09-17. The initial commit's file list contains only `.gitattributes`; it does
not provide a hidden older training configuration. The fixed candidate remains
`6721902bc4d61e50a3bfdb11dfb4cb626f05d102`.

| Dataset metadata inspected | Revision | Published format / size |
|---|---|---|
| Latest dataset revision before the checkpoint upload | `affa19c0de0f6bce2a7edd26dddef8a532e7e6f6` | v2.1; 1,693 episodes, 273,465 frames, 40 tasks; metadata FPS 10 |
| Current `HuggingFaceVLA/libero` | `86958911c0f959db2bbbdb107eb3e17c5f9c798e` | v3.0; same reported episode/frame/task counts; metadata FPS 10 |
| Separate `HuggingFaceVLA/smol-libero` | `24b61cce80e40919ab5fd8f221708947634ab42c` | v2.1; 50 episodes, 13,021 frames, 1 task; metadata FPS 20 |

The v2.1 to v3.0 replacement happened on 2025-09-19, after the checkpoint upload;
further statistics changes followed through 2025-09-30. A
[maintainer follow-up](https://github.com/huggingface/lerobot/issues/1369#issuecomment-3363214415)
acknowledges conversion problems and subsequent repairs. Consequently the current
dataset revision cannot be labeled the checkpoint's original training revision.

Only metadata was downloaded. For v2.1, each feature's episode means and variances
were combined with the frame counts using LeRobot's published parallel-variance
equations. The saved policy statistics were not changed.

| Saved tensor | v2.1 maximum absolute difference before casting | Difference after float32 casting |
|---|---:|---:|
| state mean, 8 values | 4.3982741893e-8 | 0 |
| state std, 8 values | 1.3702235568e-8 | 0 |
| action mean, 7 values | 2.2278189016e-9 | 0 |
| action std, 7 values | 1.5374774787e-8 | 0 |

This is a measured numerical link between the saved normalization and the
pre-upload public corpus. It supports that corpus's normalization provenance;
it does not recover the training job's revision argument or sampling schedule.
The separate `smol-libero` metadata disagrees on all four tensors (maximum
differences 0.16720, 0.69722, 0.05460 and 0.06827). Current v3.0 state std has a
small nonzero difference, at most 2.6822090149e-7 after float32 casting. There is
no reason to substitute either dataset's statistics for the checkpoint's own.

## Camera, state and native control

The author-linked [initial LeRobot LIBERO port](https://github.com/huggingface/lerobot/blob/25384727812de60ff6e7a5e705cc016ec5def552/src/lerobot/envs/libero.py)
already specifies both camera mappings, a 180-degree input rotation, concatenated
EEF position/axis-angle/gripper state, and direct action forwarding. The current
processor preserves these semantics while moving the image rotation and state
construction into `LiberoProcessorStep`. The previous preparation's targeted
tests and actual saved-processor forward already exercised that current chain.

| Field | Contract used by the fixed independent reference |
|---|---|
| Cameras | `agentview_image` → `observation.images.image`; `robot0_eye_in_hand_image` → `observation.images.image2` |
| Image processing | Two 256×256 HWC uint8 RGB observations; BCHW float32 /255; exactly one flip of both spatial axes in the environment processor; saved model resize/pad to 512×512 and scale to [-1,1] |
| State 0:3 | Native EEF site position in world coordinates, metres |
| State 3:6 | Native EEF-body world quaternion, converted from xyzw to axis-angle, radians |
| State 6:8 | Native two gripper joint positions, metres, in their simulator order |
| State normalization | Saved MEAN_STD once on 8 values, then model padding with 24 zeros to 32 |
| Action output | Saved unnormalizer once on the seven output values; finite output passed unchanged to the native environment |
| Action 0:3 | Relative Cartesian controller input; native clipping to [-1,1], then per-axis scale to ±0.05 m |
| Action 3:6 | Relative axis-angle controller input; native clipping to [-1,1], then per-axis scale to ±0.5 rad |
| Action 6 | Panda gripper sign: negative opens, positive closes; internal opposing-finger integration with speed 0.01 and actuator-command clipping |
| Policy cadence | Saved `chunk_size=50`, `n_action_steps=1`, `num_steps=10`; one selected action per environment step |
| Episode start | Current hard reset, registered native initial-state row, 10 separate `[0,0,0,0,0,0,-1]` settling actions, fresh policy/processor queues; first policy observation after settling |
| Episode end | Native BDDL success via `info.is_success`; explicit TimeLimit 280 for Object; preparation's separate 20-action bound is not a task-failure verdict |

The installed robosuite 1.4.0 source makes the coordinate convention precise:
`SingleArm.setup_observables` reads MuJoCo world site position and body quaternion;
`Controller.update` reads the controlled EEF site's world pose. `OSC_POSE` sets
the position goal from current position plus the scaled translation. For a
nonzero rotation command it left-multiplies the current orientation by the
axis-angle increment's rotation matrix, so the increment is expressed about
world axes. With zero rotational input it retains the prior orientation goal.
Position and orientation limits are null in the saved native controller defaults;
the first-six input clipping/scaling still applies.

An action is therefore a scaled controller command, not an observed finite
difference `state[t+1] - state[t]`. State orientation comes from the native EEF
body observable, while control uses the native EEF site. Preserve these original
definitions instead of substituting joint angles or Euler-angle differences.
Controller and observable sources are copied in the audit artifacts.

## Dataset timestamps and physical control rate

The 10 FPS metadata is real and predates this checkpoint. It does not establish
10 Hz simulator control. The contemporaneous
[OpenPI converter](https://github.com/Physical-Intelligence/openpi/blob/871f120a7af9e6439354d39292c17331e0b14699/examples/libero/convert_libero_data_to_lerobot.py)
sets `fps=10` while iterating every RLDS step and copying its state/action to the
new dataset; it does not halve actions or resample them by two. That upstream
script combines the four `*_no_noops` suites. The earlier
[OpenVLA regeneration source](https://github.com/openvla/openvla/blob/c8f03f48af692657d3060c19588038c7220e9af9/experiments/robot/libero/regenerate_libero_dataset.py)
records original retained actions and pre-action observations, removes no-ops,
and documents the 180-degree image rotation in its subsequent RLDS conversion.
These sources explain the upstream format; they are not an unpublished exact
HuggingFaceVLA conversion command.

The initial LeRobot evaluator used the native `OffScreenRenderEnv` default of
20 Hz. Upstream commit
[`cfd9ff96`, PR #4124](https://github.com/huggingface/lerobot/commit/cfd9ff969ca91acf22a68c9f85d3bc2e05c1bc92)
explicitly records that historical behavior, aligns LeRobot's default with 20 Hz
and forwards the configured rate. The current independent environment already
uses this implementation. Keep 20 Hz, giving 0.05 s of simulator time per action;
do not double actions, change controller scaling or switch to 10 Hz from metadata.

SmolVLA requests consecutive action row indices `[0,...,49]`; the dataset loader
converts indices to timestamp offsets using dataset FPS and back to row offsets.
The 50 values are consecutive saved actions. Neither the metadata timestamps nor
this synchronous runner establish real-time operation; the prior measured mean
policy invocation time remains about 0.279 s.

## Task identity and qualification scope

All ten native Object task language strings occur in the confirmed dataset's
metadata. Native task IDs 0–9 map to dataset task indices
`[24,22,26,23,21,28,27,25,29,20]`; the orders differ. Keep the registered native
suite order. Matching uses the task strings, not dataset task numbers. The
per-task metadata counts and mapping are saved in
`object_task_dataset_mapping.json`.

The proposed confirmation cohort is independent of local preparation/development
initial-state IDs and random seeds. It is not a held-out training-task benchmark:
these task names occur in the training-source corpus, and no metadata maps the
training episodes to the qualification initial-state row IDs. Initial-state
disjointness from the original training data is not established.

The prior prerequisite to obtain traceable author/source evidence is satisfied
for the independent reference's operational camera/state/action contract. At
this audit's closure, the qualification proposal was still a draft and this
source audit did not approve or start it. The subsequently approved
[qualification protocol](LIBERO_REFERENCE_QUALIFICATION_PLAN.md) was executed
under a separate registration; its [closed result](LIBERO_REFERENCE_QUALIFICATION_RESULT.md)
reports 185/200 development successes, task 5 at 14/20, and confirmation not_started.

## Limitations

- Exact training job arguments, initialization checkpoint, optimizer schedule,
  number of updates, random seed, dataset revision argument and sampled episode
  subset are not recovered. The matching v2.1 revision is a reproducible evidence
  revision, not an asserted historical training-job revision.
- The author's exact HDF5/RLDS/LeRobot conversion command and all historical
  filters/timestamp handling are not published in the sources read. Upstream
  converter behavior and the matched statistics provide supporting evidence.
- Historical evaluator reset behavior and dependency versions differ from the
  current fixed reference. In particular, the first LeRobot port set an initial
  state before a later reset; current source resets before setting it. The author
  also recommended MuJoCo 3.3.2 in September 2025; this independently prepared
  environment is locked to 3.8.1. No exact reproduction of the author's scores
  is claimed, and neither setting is silently changed.
- The newly authorized fixed-revision README web retry again returned
  `not safe to open (non-retryable error)`. This affected operation was stopped;
  no alternate endpoint was used to fetch that README. Its prior generic-card
  contents remain attributed to the taskbook. The author statements, dataset
  metadata and source reads above succeeded independently. The README failure
  is not an episode or model-load failure.

## Evidence and delivery

Artifact root:
`/home/rp/Workspace/SmolVLA_RTC/artifacts/libero_contract_audit_20260908T034709Z`.
It contains raw Hub API responses and fixed metadata, author comments, historical
and installed source copies, source receipts, `compare_dataset_statistics.py`,
`dataset_statistics_comparison.json`, the task mapping and README retry evidence.
The first attempted reuse of LeRobot's statistics import hit its optional
`datasets` dependency guard. The same published aggregation equations were then
applied directly with existing NumPy; no environment package was installed.

The machine-readable follow-up is
[LIBERO_REFERENCE_TRAINING_CONTROL_AUDIT.json](LIBERO_REFERENCE_TRAINING_CONTROL_AUDIT.json).
Original preparation artifacts and the frozen smoke protocol remain intact.

Result commit: `72bc7654726445663f6ea1b3023b92a4f9502024`, pushed to
`codex/smolvla-future-latent-m3`. The separate
[Issue comment 5578989883](https://github.com/Lebron-233/lerobot/issues/1#issuecomment-5578989883)
uses marker `LIBERO_TRAINING_CONTROL_AUDIT_20260908`; it was published once after
completed duplicate lookup and read back by ID with an identical body. Publication
and readback receipts are retained. All follow-up metadata/source-read processes
have exited; no experiment process was opened by this follow-up.
