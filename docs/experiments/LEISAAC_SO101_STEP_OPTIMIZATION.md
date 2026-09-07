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
