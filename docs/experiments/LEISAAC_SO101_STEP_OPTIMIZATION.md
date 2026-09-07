# M5.4-L3: measured step diagnosis and semantics-preserving optimization

Date: 2026-09-07. Starting source: `d762c235a3e87813617eb84cae018104063c9771`.

The user requests continued local DevSpace work until a substantive milestone or
a genuinely new authorization boundary, with code and key results on GitHub.
NVIDIA acceptance is already explicit; use process-local
`OMNI_KIT_ACCEPT_EULA=YES`. No external Pro review is required.

## Fixed experiment

Keep PickOrange, both original 640x480 RGB streams at 30 Hz, physics dt 1/60,
decimation two, original assets, randomization, success predicate, action units,
model/predictor weights, processors, production queue and delay cap eight.
Do not reinterpret old real-time failures as passes. No physical robot, training,
different checkpoint, new scientific dataset, or old M3/B4/M5 rerun is included.

## Execution and decisions

1. Add standard-library cProfile around the actual `env.step` only in the existing
   explicitly non-real-time `env-profile` mode. Preserve the native implementation.
   Collect 30 steps at seed 20260907 in a fresh source-bound directory. Materialize
   statistics in the close response, after all steps, not to a file in the loop.
   Inclusive times overlap and include instrumentation/CUDA waits: use them to
   locate work, not as kernel timings or a replacement real-time benchmark.
2. Inspect the dominant native call and its pinned source. Implement the narrowest
   supported execution-setting correction that leaves the above semantics intact,
   if one exists. Do not blindly rewrite IPC or remove camera/physics work.
3. Validate each concrete correction with targeted tests and a fresh unprofiled
   30-step real-time smoke at seed 20260907. A failed result remains a failure;
   repeat only after an evidence-supported change, not to select a passing seed.
4. A smoke PASS permits the existing synchronous capability diagnostic: seed
   20260908, policy seed 1701, at most 750 steps, original frozen weights and no
   action clipping. True task success is still required for the two paired async
   seeds already fixed in the L2 protocol. Do not widen this scientific gate.

Changing renderer fidelity, cameras, physics/assets, control FPS, weights or task
would need a distinct scientific/deployment decision; stop and state the concrete
tradeoff instead of silently doing so. Host execution settings that remove wasted
work without such a change may be corrected, recorded and tested here.

## Validation intent

Test profiler serialization and disabled-by-default execution, local IPC close
receipt and no-policy diagnostic selection. These detect instrumentation leaking
into a timed smoke or losing statistics on shutdown; fix those interfaces before
running. Do not repeat unchanged queue/science tests.

Results and exact source/run identities are appended as they occur.

## Native profile and first bounded correction

`m54l3_60b35786_native_step_profile_v1` completed 30 real steps, no policy,
source `60b357866ddbe6664e9decc97498c7ea12116d1a`, with normal cleanup.
Host cumulative times per control step: native environment 38.309 ms;
two physics steps 18.813 ms (including PhysX `fetch_results` 17.927 ms);
render 11.431 ms (Kit update 10.884 ms); observation manager 3.978 ms.
These include profiling overhead and are not compared as a performance gain
against the previous run. Inclusive child times must not be added to parents.
The live settings confirmed rate limiting, viewport and eco mode were already
disabled. Do not propose disabling them again as an optimization.

The host has six physical / twelve logical cores. The installed SimulationApp
defaults to min(32, os.cpu_count()) host workers, hence twelve here. The
[NVIDIA 5.1 performance handbook](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/reference_material/sim_performance_optimization_handbook.html#cpu-thread-count-optimizations)
documents limiting Carbonite/TBB worker counts to avoid CPU oversubscription.
Test six workers for those two pools through the existing AppLauncher `kit_args`
interface (this pinned AppLauncher does not forward `limit_cpu_threads`).
Record the effective counts. This is a hypothesis-driven host execution change,
not an established fix; physics device, solver, dt, assets and render fidelity
stay unchanged. Next: one unprofiled real-time smoke, same seed and 30-step cap.

### Six-worker result and independent simulation device

`m54l3_a9ce5651_six_threads_smoke_v1` reached two real steps, then failed the
unchanged lost-slot check. Effective Carbonite/TBB counts were both six;
PhysX worker setting was eight. The profiler was disabled. This is not a fix or
a smoke PASS. No model loaded; the simulator exited normally.

The largest measured native interval remains the GPU PhysX completion wait.
For this single environment, next test the supported CPU PhysX compute device
while retaining RTX cameras on GPU 0. Add an explicit `--sim-device` distinct
from the model's `--device`, preserving the default GPU path. Record a named
`cpu_physx_rtx_v1` execution subprofile and the unchanged solver type. This is a
deployment decision under the user's continued execution instruction: not a new
simulator, different timestep, asset simplification, changed renderer fidelity,
or changed task. CPU/GPU floating-point trajectories are not asserted identical;
future paired comparisons must use one frozen execution subprofile throughout.

Run a 30-step native profile on CPU first, then one unprofiled real-time smoke
only if its observations/actions/physics remain valid. Use seed 20260907, new
source-bound directories, same 600 s startup/30 s IPC and original lost-slot gate.
The current pinned AppLauncher explicitly keeps rendering GPU 0 for `device=cpu`;
`parse_env_cfg` selects the simulator tensor/PhysX device. No upstream patch,
package reinstall, model CPU fallback or new dependency is needed.
