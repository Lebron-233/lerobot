# M5.4-L3 milestone: environment timing unblocked; frozen candidate feasibility stops

Date: 2026-09-07. This continues the
[L3 execution record](LEISAAC_SO101_STEP_OPTIMIZATION.md) and preserves all L2 failures.
Work was executed locally through DevSpace, without another Pro review.

## Verdict

**The CPU-PhysX / GPU-RTX execution subprofile passes the existing bounded
30-step real-time environment gate. The frozen base policy then produces an
out-of-range first action in the prescribed synchronous capability diagnostic.**

The environment bottleneck is no longer the immediate blocker. The remaining
boundary is candidate/action feasibility in the SO101 transfer, not installation,
license acceptance, IPC, or an outstanding review. The paired identity/predicted
pilot is **NOT RUN**, because its actual-task-success prerequisite is not met.
No policy weights, normalization statistics, predictor weights, action limits,
task success rules, camera fidelity, timestep, or risk thresholds were changed.

## Actual attempts

Artifacts are under `/home/rp/Workspace/SmolVLA_RTC/artifacts/`.

| Artifact directory | Source | Executed environment steps | Outcome |
| --- | --- | ---: | --- |
| `m54l3_60b35786_native_step_profile_v1` | `60b357866ddbe6664e9decc97498c7ea12116d1a` | 30 | GPU native-call diagnosis completed; not real-time evidence |
| `m54l3_a9ce5651_six_threads_smoke_v1` | `a9ce56518d80f54dbbfe93d41f7efed6e182be1e` | 2 | GPU real-time FAIL; six host workers did not fix lost slots |
| `m54l3_630feea1_cpu_physics_profile_v1` | `630feea126ef5b7db15693f912191f93109b2b17` | 30 | CPU-PhysX native-call diagnosis completed |
| `m54l3_630feea1_cpu_physics_smoke_v1` | same `630feea1...` | 30 | Unprofiled bounded real-time gate PASS |
| `m54l3_1ffda461_sync_capability_v1` | `1ffda461158e1fd992fcb83f44a4197c575b5072` | **0** | Real model output rejected before the first dispatch |

The last run records one **attempted** tick, not one executed step. Its tick has
`dispatch="not_sent"`; the exception is the first joint-range rejection. All five
simulator subprocesses exited with code zero, all metrics sinks closed, and each
attempt retains its manifest, raw ticks, events and simulator log. Diagnostic
return codes and task outcomes must not be conflated with subprocess cleanup.

## What changed and what was measured

Standard-library cProfile is enabled only in `env-profile`. Statistics are sent
in the close response after control has stopped. Timed smoke and model runs have
no cProfile instrumentation. Five relevant profiler/CLI/transport tests passed;
two modified independent-device CLI cases subsequently passed. Ruff checks passed.
These are targeted interface checks, not a claim that the whole repository suite
or old scientific experiments were rerun.

The GPU diagnosis attributed approximately 18.813 ms/control step to two native
physics advances, including 17.927 ms in PhysX completion waits, 11.431 ms to
rendering and 3.978 ms to observation computation. Rate limiting and the unused
viewport were already disabled. Restricting Carbonite/TBB pools to the host's six
physical cores was insufficient: the fresh GPU smoke still failed at two steps.

The next recorded deployment decision separated simulation compute from model
compute with `--sim-device cpu --device cuda:0`. The pinned AppLauncher still
renders on GPU zero. PhysX solver type 1, dt 1/60, decimation two, original USD
assets, both 640x480 RGB streams at 30 Hz, randomization and task remain fixed.
This is named `cpu_physx_rtx_v1`, not a claim of bitwise CPU/GPU equivalence.
Any future paired experiment must use the same execution subprofile in both arms.

CPU diagnosis attributed 3.147 ms/control step to physics, 11.313 ms to rendering
and 12.293 ms to observation computation. These instrumented host intervals may
include CUDA waits and instrumentation overhead; inclusive times overlap. The
independent unprofiled smoke, rather than these numbers, establishes the bounded gate.

### Unprofiled CPU-PhysX smoke

Seed 20260907, 30 steps, cProfile disabled:

| Metric | Observed value |
| --- | ---: |
| Completed dispatches / requested steps | 30 / 30 |
| Full lost control slots | 0 |
| Mean full work | 28.5957903058 ms |
| P90 full work (linear quantile) | 34.4493167126 ms |
| Maximum full work | 36.7960439762 ms |
| Work intervals over 33.333 ms | 4 / 30 |
| Maximum accumulated tick-start lateness | 11.2985543751 ms |
| First tick start to final completion | 1.00506184599 s |

The existing gate tolerates ordinary bounded lateness but rejects full lost slots;
it was not relaxed. This is not strict zero-jitter timing, long-duration stability,
or proof of timing under concurrent model load. Both cameras progressed; the run
ended at its fixed step cap with `success=null`, not a task-success claim.

## Real frozen-policy capability diagnostic

The predefined next diagnostic ran seed 20260908, policy seed 1701, maximum 750
steps, mode `sync`, CPU PhysX and CUDA model. The existing local-only frozen
loader successfully loaded:

- `lerobot/smolvla_base@c83c3163b8ca9b7e67c509fffd9121e66cb96205`;
- `HuggingFaceTB/SmolVLM2-500M-Video-Instruct@7b375e1b73b11138ff12fe22c8f2822d8fe03467`;
- the same checkpoint `so100.buffer.action` statistics and processors.

The first measured state was `[0,0,0,0,0,9.0909090042]`. The first postprocessed
action, in the frozen six-joint order and before simulator conversion, was:

```text
[-0.8353388309, 106.4734344482, 93.2369384766,
 65.5829925537, -29.2343635559, 161.4134216309]
```

| Component | Candidate output | Allowed transfer target |
| --- | ---: | --- |
| shoulder_lift | 106.4734344482 degrees | [-100, 100] degrees |
| elbow_flex | 93.2369384766 degrees | [-100, 90] degrees |
| gripper | 161.4134216309 | [0, 100] in RANGE_0_100 |

The exception reports shoulder_lift, the first violation; the retained action
also demonstrates elbow/gripper violations. Source inspection confirms this is
the unchanged policy -> one postprocessor -> explicit degree/gripper conversion
path, not the official SO101 teleoperation re-normalizer. No radians were fed
back into the predictor, no target was clipped, and no simulator action was sent.
The predictor was not loaded in this synchronous diagnostic.

Result: **technical_failure / out-of-range action, success=null, zero executed
task steps**. This does not establish a 0% task success rate, inability on every
seed, the cause of the transfer mismatch, or lack of benefit from future latents.
It does establish that this candidate does not pass the specified first feasibility
diagnostic under the frozen mapping. No alternate seeds were tried to bypass it.

## Next authorization boundary

Do not rerun the same diagnostic, relax joint limits, silently clip outputs,
renormalize physical degrees as motor percentages, or launch the paired pilot.
Such changes would alter candidate/action semantics and, for clipping, require
explicit treatment of the predictor's committed policy-space prefix.

Recommended next scope for user approval: prepare an **independent SO101 task-matched
SmolVLA candidate**, first verifying an existing checkpoint's task, joint units,
camera contract and processors. Selecting different weights is a new candidate;
it must not inherit the old predictor/scientific PASS automatically. Predictor
compatibility must be assessed anew. Any needed training is a separate explicit
scope, not automatically included in candidate investigation.

The existing M3/B4/M5 evidence remains unchanged, `risk_thresholds=null`, and no
old dataset/test cache was accessed. This milestone includes a real environment
runtime improvement and a real frozen-model feasibility result, while preserving
the negative evidence that bounds the next scientific step.
