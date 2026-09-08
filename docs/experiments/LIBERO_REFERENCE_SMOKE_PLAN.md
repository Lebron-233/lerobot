# Frozen independent LIBERO reference wiring run

Date: 2026-09-08. This protocol is committed before its first model-driven run.
The preparation uses a separate Python 3.12 environment. This is a technical
check; `baseline_qualified=false`, `realtime_qualified=false`,
`predictor_benefit_tested=false`, and `risk_thresholds=null` throughout.

## Fixed identity and execution boundary

- Policy: `HuggingFaceVLA/smolvla_libero@6721902bc4d61e50a3bfdb11dfb4cb626f05d102`.
- Declared VLM: `HuggingFaceTB/SmolVLM2-500M-Instruct`; server-resolved VLM:
  `HuggingFaceTB/SmolVLM2-500M-Video-Instruct@7b375e1b73b11138ff12fe22c8f2822d8fe03467`.
- Package: `hf-libero==0.1.4`; task BDDL and initial-state files come from its
  fixed official PyPI wheel. Assets: dataset repository
  `lerobot/libero-assets@0b3ea86be5fe169d0fd036ae63d1070ec09e90f6`.
- Full LIBERO-Object suite in source order (`task_order_index=0`), task **0**:
  `pick_up_the_alphabet_soup_and_place_it_in_the_basket`. Language:
  `pick up the alphabet soup and place it in the basket`. Initial state ID **0**.
- Environment seed **4200**; policy/noise seed **4201**, set after construction
  and reset. One run, one environment, no seed or task selection from outcomes.
- Hard reset; ten existing `[0,0,0,0,0,0,-1]` settling actions. The first model
  observation is after all ten settling actions and before measured action 0.
  Policy and both processors reset explicitly. Next initial-state index is 1;
  this invocation does not reset into or execute that next episode.
- Native relative Panda `OSC_POSE`, 20 Hz, agentview and wrist 256×256.
  Generate **50**, consume **1** through actual `SmolVLAPolicy.select_action`,
  ten denoising steps, saved `use_amp=false`, saved VLM initialization enabled.
- Stop after at most **20 measured model actions** or a native terminal signal.
  The suite's 280-action limit is represented by Gymnasium `TimeLimit`; the
  technical 20-action bound has its own label and is not a task timeout.

## Interface and observation records

Use `LiberoEnv._format_raw_obs`, `preprocess_observation`, `LiberoProcessorStep`,
the checkpoint's saved pre/post pipelines, and the existing strict policy loader.
Raw agentview maps to `observation.images.image`; raw wrist maps to `image2`.
HWC uint8 becomes BCHW float32 divided by 255, followed by the environment
processor's one 180-degree rotation. The `render()` display image is not used
as policy input. The model then uses its saved 512×512 resize/pad and [-1,1]
visual scaling.

State is `[eef_xyz_m, eef_axis_angle_rad, gripper_qpos_m]`, dimensions 3+3+2.
Quaternion input is xyzw. Saved MEAN_STD normalization occurs once on 8 values,
then the model zero-pads to 32. Seven output coordinates are relative EEF xyz,
axis-angle rotation commands, and the Panda gripper command. No motor-angle
conversion or SO101 statistics enter this chain.

The installed robosuite controller already clips the first six commands to
[-1,1] and scales them to ±0.05 m / ±0.5 rad before forming controller goals.
Panda gripper uses the command's sign, integrates opposing fingers at speed
0.01, then clips its internal actuator command to [-1,1]; negative opens and
positive closes. Preserve and report finite out-of-box policy outputs and pass
them unchanged to this native handling. Reject non-finite or malformed actions
before `env.step`; retain original values in diagnostics. Add no extra clipping.

Log state, normalized state, normalized/unnormalized action, out-of-box indices,
the actual step call and returned native success/done/terminated/truncated,
controller goals, generation projection shapes and remaining queue length.
`info.is_success` determines native success. Never infer success from reset
imagery or termination alone. Save a small pair of policy-input images.

## Failure and cleanup

Require committed, clean source and fixed local/offline model and asset paths.
Strict loading must reject missing/unexpected weights or incompatible shapes;
standard safetensors tied-storage handling is allowed, missing independent
weights are not. No predictor, oracle, RTC, training, or outcome sweep runs.

Each invocation creates a new output directory; preserve an unsuccessful run.
Only a documented concrete repair permits a separately labeled validation run
under a new source commit. Always record stage, error, completed action count,
and environment cleanup. A short success or failure does not select the task
or confer baseline eligibility. All closed SO101 results and reserved L10/L15
tests remain untouched.

## Local launch contract

Use `/home/rp/Workspace/SmolVLA_RTC/libero-reference-venv` and the saved
`requirements.lock.txt` / `pip_freeze.txt` in the preparation artifacts.
Launch each process with inherited `PYTHONPATH` removed, `MUJOCO_GL=egl`,
`PYOPENGL_PLATFORM=egl`, `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`, and
`LIBERO_CONFIG_PATH=/home/rp/Workspace/SmolVLA_RTC/libero-reference-cache/config`.
The new environment's package-local `libero/libero/assets` link targets the
fixed asset snapshot, so the package takes its existing local-assets branch.

This host's Conda-based interpreter otherwise mixes Conda GLdispatch/GLX with
system EGL/OpenGL. The isolated probe observed null GL strings and a subsequent
probe exited with SIGSEGV. Its process-library diagnosis verified the minimal
launch repair: `LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libGLdispatch.so.0:/usr/lib/x86_64-linux-gnu/libGLX.so.0`.
This selects one system GLVND stack for this process, without editing either
old environment or the system driver. The repaired standalone EGL probe reports
the NVIDIA RTX 4070 Ti SUPER. All unsuccessful probe records remain preserved;
they are environment diagnostics with zero model-driven actions.

## Limitations

The model card's historical dataset field is unknown and its training/evaluation
contract is not supplied. The chain above is the current source-supported
LeRobot LIBERO convention, a pending assumption about this checkpoint's training
provenance. The technical run checks this explicitly recorded wiring; a full
qualification campaign requires that provenance/semantic contract to converge.
Synchronous simulator stepping does not establish real-time asynchronous control.
