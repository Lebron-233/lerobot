# M5.4-L2 result: implementation complete; simulator launch blocked at EULA

Date: 2026-09-07. This report distinguishes implemented code, interface tests,
installed dependencies, and actual environment execution.

## Verdict

The minimal SO101 transfer interface and evaluation entry point are implemented
and pushed. The separate simulator environment and pinned PickOrange assets are
installed. The first source-bound environment-only run was **attempted**, but
returned **zero observations and executed zero control ticks**: NVIDIA's first
import waited for license acceptance. It is a retained technical failure, not a
task failure, a 0% success rate, or evidence against future-latent prediction.

No policy or predictor weights were loaded in this run. No task-level closed-loop
comparison was performed. The simulator child exited after bounded cleanup; the
installation command has also completed. Nothing is queued to run automatically.

## Delivered code and engineering evidence

- Implementation commit: `64aa443f3b5a5b958b3329322a281b3180ac0d28`.
- Run source / timestamp semantics correction:
  `56a07d045d739f233f8440a4753afd2d66d00316`.
- Branch: `codex/smolvla-future-latent-m3`.
- [Scope and execution order](LEISAAC_SO101_MINIMAL_EXPERIMENT.md).
- [Prior L1 design](LEISAAC_SO101_MINIMAL_ADAPTATION_PLAN.md).

The targeted suite passed **16 tests in 3.42 seconds** on the run source. This
includes a real Python 3.12 controller / Python 3.11 child transport fixture,
coordinate and camera contracts, terminal/reset accounting, lost-slot failure,
and the actual production predicted engine with CPU fixtures. The latter verifies
normalized committed prefixes, whole-discard after a deliberately late result,
and joined reset. Lint passed. These fixtures are not LeIsaac task rollouts.

The post-run correction redirects child stdin to `DEVNULL`, preventing the
launcher from leaving interactive input open in a captured subprocess. The
existing dual-interpreter test now checks this property. This correction does
not accept or bypass the EULA, change the task, or alter timing/model parameters.
The corrected launcher has not been used for a new real environment attempt.
After this correction, the two affected client/startup tests passed in 0.92
seconds (14 unaffected tests deselected); lint and formatting checks passed.

## Installed simulator environment

| Component | Actual installed version |
| --- | --- |
| Python | 3.11.16 |
| LeIsaac | 0.4.0, editable exact source `24d3bcd3f1e4585740fc79921782c41617237812` |
| Isaac Lab | 2.3.0 |
| Isaac Sim | 5.1.0.0 |
| torch / torchvision | 2.7.0+cu128 / 0.22.0+cu128 |
| NumPy / packaging | 1.26.0 / 23.0 |
| Installed distributions | 212; `uv pip check` reported all compatible |

Paths on the DevSpace-connected machine:

```text
model Python (unchanged):
  /home/rp/miniconda3/envs/smolvla-rtc/bin/python
simulator Python:
  /home/rp/Workspace/SmolVLA_RTC/leisaac-sim-venv/bin/python
LeIsaac source:
  /home/rp/Workspace/SmolVLA_RTC/leisaac-source
assets:
  /home/rp/Workspace/SmolVLA_RTC/simulator-assets/models--LightwheelAI--leisaac_env/snapshots/6c35af0af55506eb75c5592930134d4af44e8341/assets
```

Preparation resolved three observed issues without changing the model runtime:
the unused LeRobot export extra's packaging conflict; flatdict's legacy
pkg_resources build dependency (setuptools 80.9.0 build constraint); and Isaac
Sim kernel's exact NumPy 1.26.0 requirement. The final install exited zero.
The original failed install log, successful install log, installed-package
freeze, and asset download receipt remain under `artifacts/m54l2_preparation_v1`.

## Actual run and retained result

```text
directory: artifacts/m54l2_56a07d04_env_smoke_v1
source: 56a07d045d739f233f8440a4753afd2d66d00316
mode: smoke (environment only)
seed: 20260907
planned maximum: 30 steps
actual ticks: 0
environment metadata/observations received: none
result: technical_failure
failure: LeIsaac did not respond within 180 seconds
cleanup: simulator required termination after close timeout
subprocess return code: -15
metrics sink closed: true
success: null
model / predictor / paired pilot: NOT RUN
```

The retained `simulator.log` ends at the exact prompt:

```text
Do you accept the EULA? (Yes/No):
```

No answer was sent. The waiting process was terminated by the already bounded
cleanup path. `manifest.json`, `result.json`, empty `ticks.jsonl` / `events.jsonl`,
and `simulator.log` preserve this attempt; they are not overwritten by a retry.

NVIDIA's [official installation instructions](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/installation/install_python.html#running-isaac-sim)
confirm that first import requests EULA acceptance and describe operator-controlled
interactive or environment-variable acceptance. The operator's consent is still
required; the assistant did not set `OMNI_KIT_ACCEPT_EULA` or alter acceptance files.

## Next executable boundary and limitations

After the operator accepts the applicable license, run the same bounded smoke
using a fresh output directory and a clean source commit. Only a successful
real reset/step/RGB/timing check permits the predeclared same-weight capability
episode; the tiny paired pilot remains conditional on actual task ability.
No additional Pro review is required by the user's current instruction.

Package compatibility and CPU fixtures do not prove that the simulator boots,
that actual camera frames satisfy the contract, that this host maintains 30 Hz,
or that the frozen policy can solve PickOrange. These remain unmeasured.
The observed license prompt is the first blocker, not proof there can be no later
integration issues. GPU availability before launch was 10,257 MiB free out of
16,376 MiB with 35% utilization; other user processes were not stopped.

The linked official documentation now labels Isaac Sim 5.1.0 unsupported. This
phase retains the version pinned by LeIsaac's inspected dependency contract; it
does not silently upgrade the simulator or claim current vendor support.

Existing M3/B4 and M5.3 results remain unchanged. No physical robot, training,
new checkpoint, risk gating, or old scientific-cache reread was performed.
Risk thresholds remain null. The old model environment was not downgraded.
