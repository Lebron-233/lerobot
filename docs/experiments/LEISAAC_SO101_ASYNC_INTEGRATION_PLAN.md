# L5: reproducible resets and real production-async integration

Date: 2026-09-07. The user's project-wide authorization remains active.

## Scientific status, not a renamed success

L4 has established legal task-conditioned model execution, including complete
episodes, but no full task success under the tested development protocols.
All zero-reset/native-AMP/CPU-versus-GPU/time-budget/feedback/initial-pose variants
remain individually recorded. Teacher-forced training-frame agreement is not a
generalization score. No new predictor has been trained or qualified.

The original full-success prerequisite still prevents claiming a task-quality
identity/predicted comparison. It does not justify indefinitely blocking a
separately labelled **engineering integration** of the real production worker,
queue and environment. L5 authorizes that integration without relabelling a
timeout as success or a slow run as realtime. Full task success remains the
original all-three-fruits-and-rest termination predicate.

## Two evidenced environment issues

1. CPU-PhysX profiling identified camera observation/CPU Warp untile work as
   a material remaining cost. For this single environment, compare the existing
   supported `Camera` implementation against `TiledCamera`, keeping resolution,
   RGB format, camera poses, update/render rates and physics unchanged. Record
   `camera_backend=standard`; it is not assumed bit-identical to tiled rendering.
   Run one standard-camera 30-step realtime smoke, seed20260907. No timing gate
   is relaxed. A pass only qualifies this bounded environment-only load.
2. Pinned LeIsaac `enhance/envs/mdp/events.py::randomize_camera_uniform` adds
   deltas to current camera poses, despite its default-pose wording. Native
   `reset_scene_to_default` restores bodies/joints, not camera transforms.
   Production startup performs another explicit env reset, so this reachable
   behavior matters for matched initial observations. Record actual camera world
   transforms, then compare three same-seed resets in one real environment with
   no policy or control actions. If poses drift, restore the captured nominal
   front pose before explicit reset, preserving the upstream randomization
   distribution. Repeat only this targeted reset check on new source/artifacts.

## Engineering run after those seams are usable

Use `wsagi/SmolVLA-PickOrange@c8c3318dba152b0ba671ff07b4314418d5aa4b4a`:
its native use_amp=false matches the existing production worker precision; keep
the saved complete task weights/processors and motor/physical conversion. This
selection is for integration, not a claim that it won the L4 task comparison.
Predictor remains absent and context_mode is identity. No old frozen-candidate
validation or scientific result is borrowed.

Freeze a single real asynchronous episode, environment seed20260912, policy
seed1802, at most120s/3600steps, CPU PhysX, zero initialization, control30/camera30,
standard cameras if the smoke passes. Use the existing startup probe, original
q=.9/margin1/guard2/late2/cap8 and whole-discard, memory telemetry and original
lost-slot gate. Run only after the code is committed and record startup/actual
dispatches/requests/underflows/late discards/task termination separately.

A hardware/control failure stops this engineering trial and remains reported;
it is not permission to increase cap, hide dropped slots or call a virtual-clock
experiment real-time. A successful engineering run demonstrates real queue/env
coupling, **not** improved task success or a trained future-latent predictor.

## Initial engineering result

At `29a9a25a`, the tiled-camera three-reset check returned identical actual poses
and object positions. No reset fix was made on that evidence. The standard-camera
30-step environment smoke passed the existing full-slot criterion, without an
inference worker.

The first real identity episode reached actual production startup: its calibrated
probe was150.75ms /6 required steps, under cap8. The measured section then completed
two hold-target steps (65.77ms and40.57ms) before the unchanged full-slot gate
stopped it. No planned takeover occurred; four requests have actual terminals,
including the stopped section's bootstrap. This is not a real-time or task PASS.

Follow-up is limited to actual concurrent-load preparation and the selected
standard-camera reset seam. The first reset test covered tiled cameras only;
`Camera.reset` refreshes pose data unlike `TiledCamera.reset`, so the original
same-seed evidence must not be generalized across the two backends. Run the
same three-reset diagnostic with standard cameras before any remedy.
