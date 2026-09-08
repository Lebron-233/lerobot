# Independent LIBERO reference: technical preparation completed

Date: 2026-09-08. The independent environment, fixed assets, complete strict
policy load, saved processors, and one registered 20-action wiring run all
completed. The smoke executed from committed source
`968c58603177a1888e4a6f117712bd4c84262bae`, which was pushed before execution.
The initial repository was clean at `8b9cd4e912036a0cd876140952419c95f09e562f`.

**Baseline qualified: false. Real-time qualified: false. Predictor benefit
tested: false. `risk_thresholds=null`.** Formal qualification and confirmation
were not run. Their separately specified draft is
[LIBERO_REFERENCE_QUALIFICATION_PLAN.md](LIBERO_REFERENCE_QUALIFICATION_PLAN.md).

## Candidate and saved evidence

| Component | Exact identity |
|---|---|
| Policy | `HuggingFaceVLA/smolvla_libero@6721902bc4d61e50a3bfdb11dfb4cb626f05d102` |
| Declared VLM/tokenizer | `HuggingFaceTB/SmolVLM2-500M-Instruct` |
| Server-resolved VLM | `HuggingFaceTB/SmolVLM2-500M-Video-Instruct@7b375e1b73b11138ff12fe22c8f2822d8fe03467` |
| Asset dataset | `lerobot/libero-assets@0b3ea86be5fe169d0fd036ae63d1070ec09e90f6` |
| Task BDDL / initial states | Official PyPI `hf_libero-0.1.4-py3-none-any.whl` |

The policy and VLM are public and ungated. The VLM's declared/resolved names
preserve the previously confirmed server alias; no substitute backbone was
selected. Fixed local snapshot paths are in
[LIBERO_REFERENCE_PREPARATION_SUMMARY.json](LIBERO_REFERENCE_PREPARATION_SUMMARY.json).
Only required model/configuration/tokenizer files and simulation assets were
downloaded. The final policy weights come from the LIBERO checkpoint.

The actual policy file contains 788 tensors and is 1,218,047,032 bytes. The VLM
initialization file contains 489 tensors and is 2,029,990,624 bytes. Raw policy
configuration, both processor JSON files, tokenizer/construction files, full
weight-key/shape inventories and source metadata are retained in the dedicated
cache and artifacts.

Both referenced normalization safetensors were read. State mean/std are shape
`[8]`; action mean/std are shape `[7]`; every value is finite and every standard
deviation is positive. Pre/post action mean and std match exactly, with maximum
absolute difference 0. The post file additionally contains state statistics;
its saved postprocessor features apply only to actions. Full values are in
`statistics.json`, with no SO101 statistics involved.

## Independent environment and rendering

Environment: `/home/rp/Workspace/SmolVLA_RTC/libero-reference-venv`, Python 3.12.14.
The original model Conda environment and Isaac environment received no package
installation or configuration edits. All simulator/policy processes ran in the
new environment. Its interpreter uses the original Python runtime as its venv
base; the GL library launch setting below handles that host-specific detail.

| Installed component | Version |
|---|---|
| LeRobot | 0.6.2, editable current source |
| torch / torchvision | 2.11.0+cu128 / 0.26.0+cu128 |
| transformers / huggingface-hub | 5.5.4 / 1.30.0 |
| hf-libero / robosuite / robomimic | 0.1.4 / 1.4.0 / 0.2.0 |
| MuJoCo / bddl | 3.8.1 / 1.0.1 |
| NumPy / SciPy | 2.2.6 / 1.18.1 |
| PyOpenGL / Gymnasium | 3.1.10 / 1.3.0 |
| OpenCV regular / headless distributions | both 4.13.0.92; imported cv2 4.13.0 |
| safetensors / accelerate | 0.8.0 / 1.14.0 |

`uv pip check` found all 139 installed packages compatible. Installation inputs,
resolved lock, complete freeze and logs are retained. The final isolated LIBERO
probe successfully performed hard reset, ten settling actions and two 256×256
uint8 observations, and closed normally. Renderer: **NVIDIA GeForce RTX 4070 Ti
SUPER/PCIe/SSE2**, OpenGL **4.6.0 NVIDIA 580.173.02**, `EGLGLContext`.

The fixed package normally searches for package-local assets before falling back
to an unpinned Hub download. The new environment's `libero/libero/assets` link
points to the exact cached asset snapshot. A task-specific `LIBERO_CONFIG_PATH`
pins the installed wheel's BDDL and initial-state paths. Final runtime processes
use local/offline model and asset loading. No task asset or success rule changed.
All ten suite initial-state files were read from the inspected official wheel
paths; each has shape `[50,110]`. Full source-order names and native goals are
recorded in `suite_inventory.json` and the qualification draft.

## Strict load and actual interface

The existing `SmolVLAPolicy.from_pretrained(..., strict=True)` succeeded with no
missing keys, unexpected keys or shape mismatches. The standard safetensors
loader's tied-storage handling was inspected. This checkpoint contains both
embedding and LM-head weights; their constructed parameters do not share
storage. No missing independent layer was filled from another policy.

The actual model has **32 VLM layers and 32 expert layers**, expert hidden size
480. Saved `num_vlm_layers=0` means no truncation at that source branch.
Key projections are state `[960,32]`, action-in `[480,32]` and action-out `[32,480]`.
There are 600,902,304 bfloat16 parameter elements and 4,031,872 float32 elements.
The saved `use_amp=false` is retained; it does not imply every parameter is
float32. All 604,934,176 parameter elements are frozen for inference.

The native input chain maps raw agentview to `observation.images.image` and
wrist to `image2`, converts HWC uint8 to BCHW float32 /255, then uses the existing
LIBERO processor's one 180-degree image rotation. Policy inputs do not come from
`render()` display output. The model applies its saved 512×512 resize/pad and
[-1,1] visual scale.

State is EEF position in metres, xyzw quaternion converted to a three-component
axis-angle in radians, then two gripper qpos values in metres. The saved
MEAN_STD processor runs once on eight values; model padding appends 24 zeros.
The saved postprocessor unnormalizes seven action values once. The synthetic
interface probe's discrepancies from the explicit single-normalization and
single-unnormalization equations were 2.20e-8 and 1.54e-8 respectively.
Its actual policy forward returned finite `[1,50,7]` actions; its input was
explicitly synthetic and executed no environment actions.

The execution endpoint is relative Panda `OSC_POSE`: three EEF translation
commands, three axis-angle rotation commands, one gripper command. The native
controller clips/scales the first six commands from [-1,1] to ±0.05 m / ±0.5 rad.
Panda gripper uses sign, opposing-finger integration at speed 0.01 and clipping
of its internal actuator command. Negative opens; positive closes. The runner
passes original finite actions unchanged to this endpoint and adds no clamp or
motor-angle conversion.

## Registered wiring result

Protocol: [LIBERO_REFERENCE_SMOKE_PLAN.md](LIBERO_REFERENCE_SMOKE_PLAN.md).
Task 0 was selected from the full source-order suite before execution:
`pick_up_the_alphabet_soup_and_place_it_in_the_basket`. Native language is
`pick up the alphabet soup and place it in the basket`. Initial state ID 0,
environment seed 4200, policy/noise seed 4201; hard reset and ten separately
counted settling actions. The first model observation follows those settling
actions. Policy/processor reset left the action queue empty; the next native
initial-state index was 1, and no second episode was opened.

| Measurement | Actual result |
|---|---|
| Model-driven actions / settling actions | 20 / 10 |
| Generation / consumption / denoising | 50 / 1 / 10 throughout |
| Native success | false |
| Native done / terminated / truncated | false on all 20 returned steps |
| Stop reason | registered technical 20-action bound |
| Smoke technical exceptions | 0 |
| Finite input states / output actions | all 20 |
| Original actions with out-of-box components | 11 actions, 11 scalar components, all gripper index 6 |
| Maximum absolute gripper output | 1.0290333032608032 |
| Runner-side clipping | none; native gripper handling retained |
| Process exit / environment close | 0 / confirmed |

Each recorded selection produced ten action-projection outputs of `[1,50,32]`
and left zero queued actions after returning the first seven-dimensional action.
All 20 sequential observation/action indices and actual `env.step` returns agree
with the completed-step count. No pending step remains. The full per-step ledger
includes original normalized/unnormalized actions, state, controller goals and
native terminal flags; a pair of actual policy-input images is saved.

Mean measured policy/pre-post invocation wall time was 0.279 s, median 0.251 s,
maximum 0.769 s; summed invocation time was 5.580 s. These are descriptive local
smoke timings. Physics pauses for synchronous inference. The absence of success
within the preparation bound does not remove this task from the proposed suite.

## Resolved preparation issues and retained unsuccessful probes

The first resolver command sent ordinary packages to the PyTorch extra index,
which shadowed compatible PyPI `requests` versions. Scoping CUDA resolution with
`--torch-backend cu128` resolved it without changing package safety settings.
The first full resolution also selected OpenCV 5 beside the project's mandatory
OpenCV headless 4. Both distributions supply `cv2`; the independent input now
pins both to 4.13.0.92. Final dependency and actual import checks passed.

The initial pytest launch inherited ROS's Python path and loaded an unrelated
plugin lacking `lark`. Removing inherited `PYTHONPATH` isolated the intended
environment. Thirteen targeted tests then passed; ruff lint and formatting passed.

The first simulator probe completed its ten settling actions but failed to read
null OpenGL vendor strings; it closed normally. A second probe that explicitly
bound the context exited with SIGSEGV (139). Its native cleanup and exact settling
completion are unknown; the process exited, and its directory is preserved.
Neither probe executed any model-driven action.

Library diagnosis showed Conda GLdispatch/GLX loaded alongside system EGL/OpenGL.
An EGL-resolved GL function returned NVIDIA correctly while the mixed PyOpenGL
path returned null. A process-local preload of system `libGLdispatch.so.0` and
`libGLX.so.0` made all four libraries resolve to the system stack and fixed the
standalone GL probe. Under that same setting, a newly labeled LIBERO probe and
the first model-driven smoke both completed and closed. The source protocol and
`run_smoke.sh` retain the exact launch settings. No old environment or global
driver configuration was edited.

## Limitations and next gate

The checkpoint's training dataset and complete action/state/camera evaluation
contract remain undocumented in the available model-card evidence. The recorded
LeRobot interface is a source-supported wiring convention and a pending assumption
about training provenance. Technical preparation is complete; formal baseline
qualification requires that contract to converge and the draft to be approved
and frozen. The draft proposes 200 development and 200 disjoint confirmation
episodes, with at least 180/200 overall, at least 16/20 on every task, and zero
technical failures in each cohort. These are new proposed project gates.

A fresh fixed-revision README web-page read returned `not safe to open
(non-retryable error)`. That read was stopped, with no alternate README fetch.
The model-card description/unknown dataset observation is attributed to the
user taskbook's prior successful read; no fresh local README is claimed. This
web-page error did not prevent independently authorized file downloads or local
strict loading, and is not an experimental failure.

No formal qualification, confirmation, training, predictor transfer, asynchronous
effect queue or real-time qualification was executed. Closed L20 and earlier
SO101 results remain unchanged; reserved L10/L15 tests were not opened.

## Artifacts and GitHub record

Artifact root:
`/home/rp/Workspace/SmolVLA_RTC/artifacts/libero_reference_preparation_20260908T030051Z`.
Dedicated cache: `/home/rp/Workspace/SmolVLA_RTC/libero-reference-cache`.
The root contains dependency inputs/locks/logs, fixed identities, statistics,
weight inventories, all probe diagnostics, `policy_check.json`,
`processor_numeric_check.json`, `smoke_accounting.json` and `smoke_v1/`.
`run_smoke.sh` records the exact executed command; its existing output must be
preserved. Weights and raw logs are outside Git.

The existing [L20 comment 5578043892](https://github.com/Lebron-233/lerobot/issues/1#issuecomment-5578043892)
was synchronously read back and confirmed; it was not republished. This result's
separate Issue 1 record uses marker `LIBERO_REFERENCE_PREPARATION_RECOVERY_V2`.
After result push, its actual publication response and known-ID readback are
saved as `publication.json` / `publication_readback.json` in the artifact root
and linked from the local handover.
