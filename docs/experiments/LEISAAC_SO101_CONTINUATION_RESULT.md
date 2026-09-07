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

Further observed outcomes will be appended here; no capability or paired-pilot
result is claimed by this preparatory correction.
