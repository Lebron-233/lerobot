# M5.4-L2 continuation after operator license acceptance

Date: 2026-09-07. This report continues, and does not overwrite, the preceding
[L2 result](LEISAAC_SO101_MINIMAL_RESULT.md).

## Acceptance and first resumed attempt

The operator explicitly accepted NVIDIA's agreement and requested continuation.
[Issue #1 acceptance record](https://github.com/Lebron-233/lerobot/issues/1#issuecomment-5563844750)
records the resumed scope. Invocations inherit `OMNI_KIT_ACCEPT_EULA=YES`; the
runner does not install a global acceptance setting. This uses NVIDIA's
[documented mechanism](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/installation/install_python.html#running-isaac-sim).

Attempt `m54l2_007696a9_env_smoke_eula_v1` ran clean source
`007696a900220c5d00d552aea1300929282747a2`, environment seed 20260907, smoke mode,
at most 30 steps. Result: **technical_failure, startup timeout 180 s, zero ticks,
success=null, metrics_closed=true, simulator return code 0**. No model was loaded.

The simulator log shows actual RTX initialization, not a license prompt:
`Simulation App Startup Complete` at 173.528 s, then PickOrange asset/environment
creation. NVIDIA's [installation notes](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/installation/install_workstation.html)
describe shader-cache warmup. The local observed initialization did not fit the
original budget; it is not evidence of a model or task failure.

## Bounded correction before the next attempt

Allow 600 seconds for environment process initialization only. Preserve 30-second
per-request IPC, all control lost-slot checks, seed, task, 30-step limit, rendering
resolution and physics. Record the actual budget and the explicitly supplied EULA
environment value in new manifests. The old failure remains immutable.

The dual-interpreter transport fixture checks acceptance-variable inheritance,
600/30/30 startup/reset/close receive budgets, noninteractive stdin and clean exit.
It does not load Isaac or report task evidence.

## Artifacts

All new run directories are under `/home/rp/Workspace/SmolVLA_RTC/artifacts/`.
The first resumed attempt contains `manifest.json`, `result.json`, `ticks.jsonl`,
`events.jsonl` and `simulator.log`, written by the existing runner after cleanup.
Previous M3/M5 and the earlier EULA-blocked attempt are unchanged.

## Real environment smoke: initialization resolved, cadence FAIL

`m54l2_fe9119c0_env_smoke_eula_v2`, source
`fe9119c063ce7844b2214f0857cbb0873ab465ad`, reached the real environment. Actual
joint order matched all six names. Both cameras returned uint8 `(1,480,640,3)`
and their frame counters advanced from 1 to 2 before the second consumed action.
The real configuration reported physics dt 1/60, step dt 1/30, and only
`time_out`/`success` termination terms.

Two actual hold-target steps completed, with work times **61.491946 ms** and
**55.807602 ms**. The unchanged accumulated lost-slot check rejected the second
tick. Result: **technical_failure / lost_control_slot_during_tick**, success null,
zero model calls, sink closed and simulator exit code 0. This is not 0% task
success and is not a failed predictor experiment.

### Diagnostic scope before further deployment decisions

Add host phase timings and run one separately labelled `env-profile` with the
same seed, at most 30 steps and no model. This is unpaced throughput diagnosis,
not real-time evidence; no hidden warmup removal or permissive smoke flag is used.
The original real-time requirement and conditional capability/pilot gates remain.

### Native asset warnings

The pinned kitchen also emits PhysX warnings/errors: some dynamic triangle mesh
colliders fall back to convex hulls, some cabinet posts fail mesh cooking, some
outlet/light-switch shapes fail creation, and static-body joints cannot be created.
The real task still constructed and stepped. These warnings are retained in the
native log; their effect on task fidelity or runtime is not established. No meshes,
colliders or success conditions were edited to suppress them.

## Completed 30-step throughput diagnostic

Artifact `m54l2_f8b230fb_env_profile_v1` ran source
`f8b230fbd2342e1dec75eca3f9e587bd593db91d`, mode `env-profile`, environment seed
20260907 and 30 steps, without any policy. All 30 dispatches completed; camera
counters in consumed observations advanced from 1 through 30 for both streams.
The existing step validator also checked each returned successor. State came from
actual measured joints, not the held action target. No terminal success/timeout
was reached within the bound: **censored_step_limit, success=null, reward=0**.
The metrics sink closed and the simulator exited normally with code 0.

Thirty simulation control steps (1 second of simulation time) took
**1.508375767 s** from first tick start to final tick completion: observed
throughput **19.888943 Hz**, real-time factor **0.662965**. This was unpaced and
does not pass the real-time smoke. All 30 work intervals exceeded 33.333 ms.

| Host wall interval | Mean, all 30 (ms) | P90, all 30 (ms) |
| --- | ---: | ---: |
| Complete control work | 50.233357 | 54.936049 |
| Client reset/step connection: step round trip | 49.961834 | 54.710502 |
| Server `env.step` plus outcome flag copy | 46.082840 | 50.023078 |
| Server gripper effort update | 1.205631 | 1.600406 |
| Server target creation | 0.125415 | 0.184735 |
| Server observation packet | 0.814183 | 1.077855 |
| Round trip minus measured server phases | 1.733766 | 2.225586 |
| Controller work before step dispatch | 0.258770 | 0.369569 |

Quantiles use NumPy's default linear percentile over the specified samples.
Intervals are host wall observations, including CUDA waits. Quantiles are not
additive. The transport/dispatch residual is a subtraction, not a pure network
latency measurement; `env.step` is not yet split into physics/render/observation
manager subphases.

The last 20 samples, reported descriptively rather than treated as a replacement
benchmark, still averaged **49.799579 ms** overall and **46.080653 ms** in
`env.step` plus flags. All 20 also exceeded 33.333 ms. There is no evidence here
that discarding initial cold samples would restore 30 Hz.

Machine-readable summary: [LEISAAC_SO101_ENV_PROFILE_SUMMARY.json](LEISAAC_SO101_ENV_PROFILE_SUMMARY.json).
Full, unrounded events/ticks/native logs remain in the source-bound artifact
directories. The diagnostic's `events.jsonl` is empty because no inference engine
was instantiated; it is not missing policy telemetry.

## Decision and next useful work

**The current measured local deployment does not meet the 30 Hz real-time
requirement even before adding policy inference.** Environment initialization,
real observations, coordinate/IPC plumbing and real physics stepping have now
been exercised, but the real-time prerequisite remains FAIL. This is scoped to
the measured configuration and workload, not a universal claim about SO101,
Isaac or the GPU model.

Do not run the previously conditional synchronous capability diagnostic or the
identity/predicted pilot under a fictitious smoke PASS. Neither has run. No new
task success rate, predictor benefit or paired outcome is available.

The useful next performance investigation is **inside the existing simulator's
step**, separating physics, rendering and observation-manager work before changing
anything. Removing all measured IPC residual alone cannot make a roughly 46 ms
server step fit 33.333 ms. Do not replace the queue/IPC framework, raise horizon,
lower FPS, drop camera frames, simplify assets or change weights to conceal this
result. Any actual rendering/physics-profile change needs its own recorded
configuration and a fresh bounded real-time check; repeat attempts with unchanged
configuration are not justified by the current evidence.

The robot/camera transfer limitations and native asset warnings remain. Operator
acceptance is resolved, and installed dependencies remain available; neither is
the next blocker. Old scientific/runtime results and null risk thresholds remain
unchanged.

## Checks and cleanup

The startup-budget/acceptance propagation and failure-cleanup checks passed
(2 tests). The modified interface suite then passed its 16 prior cases; the new
no-model/non-real-time CLI fixture initially omitted the startup constant, and
passed its targeted rerun after correcting that test fixture. Ruff lint and
format checks passed for all three modified Python files. No unmodified M3/B4/M5
scientific experiment was rerun.

All three new execution sessions completed and their simulator processes exited.
No model, training job, simulator or queued pilot remains running from this work.

### Execution invocation

All three attempts used the following launcher environment and fixed resource
arguments; mode, output and source were the separately recorded values above:

```bash
MODEL_PY=/home/rp/miniconda3/envs/smolvla-rtc/bin/python
OMNI_KIT_ACCEPT_EULA=YES PYTHONPATH=src:examples/advanced/predictive_async \
  "$MODEL_PY" -m uv run --no-project --python "$MODEL_PY" python \
  examples/advanced/predictive_async/eval_leisaac_so101.py \
  --mode env-profile --seed 20260907 --max-steps 30 \
  --sim-python ../leisaac-sim-venv/bin/python \
  --leisaac-root ../leisaac-source \
  --assets-root ../simulator-assets/models--LightwheelAI--leisaac_env/snapshots/6c35af0af55506eb75c5592930134d4af44e8341/assets \
  --output ../artifacts/m54l2_f8b230fb_env_profile_v1
```

This is the recorded diagnostic invocation, not a command to rerun against an
existing output directory or to bypass the conditional policy-run gates.
