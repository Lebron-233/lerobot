# M5.4-L4: independent task-matched SO101 candidate

Date: 2026-09-07. Starting source: `982c897038ed5d094d9dc8eeff8995045bb524a1`.

The user explicitly authorizes the proposed independent-candidate work and
project-related implementation/experimentation in this session, without another
Pro review. This supersedes the L3 candidate-investigation authorization boundary.
Record experiments and decisions on Issue #1 and push source changes. Preserve
all old candidates, failed attempts and held-out scientific results.

## Candidate selection before outcomes

First candidate: `edge-inference/smolvla-so101-pick-orange` at
`71cf4a9d35ce317f6706efe1a9f9d4cbb2b8fb4d` (current main, metadata read before runs).
Its published model/config/processors specify the existing LeIsaac PickOrange
task, front/wrist 480x640 cameras, six actions, 50-step chunks, ten flow steps,
and its own state/action mean/std. Download this exact snapshot, not moving main.

Metadata-only alternative retained, not selected by our outcomes:
`wsagi/SmolVLA-PickOrange@c8c3318dba152b0ba671ff07b4314418d5aa4b4a`.
Its card and config disagree about training/full-parameter and architecture
settings. Do not silently repair those fields or treat publisher success as ours.

Sources:
- https://huggingface.co/edge-inference/smolvla-so101-pick-orange/tree/71cf4a9d35ce317f6706efe1a9f9d4cbb2b8fb4d
- https://huggingface.co/wsagi/SmolVLA-PickOrange/tree/c8c3318dba152b0ba671ff07b4314418d5aa4b4a
- https://huggingface.co/datasets/LightwheelAI/leisaac-pick-orange
- Pinned LeIsaac `utils/robot_utils.py`, `policy/service_policy_clients.py`
  and `tasks/template/single_arm_env_cfg.py` at `24d3bcd3f1e4585740fc79921782c41617237812`.

## Independent numerical contract

The task-matched candidate uses LeIsaac's motor-range representation, unlike the
old physical-degree SO100 candidate. Native model camera names remain front/wrist.
Verify the training metadata and official inference/export conversion before
running. Use the candidate's own serialized processors and statistics, with only
local tokenizer/device/camera transport overrides. No replacement with SO100 stats.

Reuse the existing simulator's physical-degree/gripper transport and its range
check; explicitly convert at the candidate boundary. Arm motor values [-100,100]
map linearly onto each existing USD degree range; gripper remains [0,100]. Measured
state uses the inverse transformation. Do not clip actions or measured states.
This is a new named candidate, not a workaround to the frozen old validator.

## Bounded execution

1. Pin/download candidate and required training **metadata**. Inspect tensor and
   processor compatibility; test actual mapping/order/normalization seams.
2. Use the accepted CPU-PhysX/GPU-RTX execution subprofile, existing assets,
   30 Hz, dt=1/60, decimation=2, default 25-second environment termination,
   and original success predicate. Run the independent synchronous capability
   episode with environment seed 20260911, policy seed 1801, at most 750 steps.
   This is development/qualification, not an unbiased task success-rate estimate.
   Stop and retain technical failures; repair evidenced interface defects rather
   than changing seeds or limits to obtain an apparent pass.
3. If task execution is valid, diagnose actual task outcomes before choosing any
   further development. Only actual success qualifies this candidate for the
   previously proposed task-comparison pilot. No conditional experiment is queued.
4. Production identity async may be validated with this new policy after its
   qualification. Old predictor weights are not automatically compatible: state
   coordinates/statistics, action normalization, cameras and weights changed.
   Keep predicted disabled until a separately recorded compatibility/training
   protocol establishes it. Risk thresholds remain null.

Each real run has a new directory, pinned code commit, seed, candidate manifest,
attempted-versus-completed dispatch accounting and post-shutdown telemetry.
No physical robot, cloud spending, global package changes, force pushes or merges.
Use the existing local compute and do not modify unrelated user processes.

## Verified before the first model run

The official training `meta/tasks.jsonl` contains exactly
`Grab orange and place into plate`. Use this string, not the different inference
example wording in the model card. The pinned upstream `LeRobotServicePolicyClient`
converts measured radians through `convert_leisaac_action_to_lerobot` and applies
`convert_lerobot_action_to_leisaac` to policy outputs, confirming the motor-range
contract. Six new CPU tests passed for batched/chunk coordinate conversion,
native-statistics ordering, policy-space preservation and unchanged range failure.
The new candidate's state is normalized by its own processor then padded by
SmolVLA; the old predictor's raw-state contract therefore does not carry over.
