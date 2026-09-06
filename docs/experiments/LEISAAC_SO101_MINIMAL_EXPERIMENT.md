# M5.4-L2: minimal SO101 transfer experiment

Date: 2026-09-07. Starting HEAD: `e7dd731e6a12536b71e7fe0e2ca28292880050f4`.

The user has instructed this session to implement the preceding minimal adaptation
plan locally through DevSpace and continue publishing code and key findings to
GitHub. This supersedes L1's planning-only boundary for the implementation and
bounded validation below. No external Pro review is required.

## Fixed scope

Implement `leisaac_so101_transfer_v1`: the existing PickOrange manager environment
in a separate Python 3.11 process, a narrow local reset/step/close connection, and
the unchanged production predictive-async engine in the existing Python 3.12
model environment. Keep the policy/VLM/predictor revisions, processors, normalized
committed prefix, delay cap eight, whole-discard, and null risk thresholds from
the [L1 plan](LEISAAC_SO101_MINIMAL_ADAPTATION_PLAN.md).

This is an explicitly named SO100-to-SO101 transfer, not proof of equivalent robot
calibration or camera viewpoint. No physical robot, training, checkpoint change,
new dataset/test-cache/B4 access, or rerun of previous M5 profiles is included.

## Implementation and bounded execution order

1. Implement the coordinate/observation contract, isolated environment process,
   and a dedicated evaluation runner; leave public SO100 validators untouched.
2. Test only the new interface: units/order/limits, image byte transport, reset
   outcome accounting, shutdown, and actual production-engine integration with
   CPU fixtures. Fixtures are engineering tests, not task evidence.
3. Check simulator availability and package compatibility without changing the
   accepted model environment. Use an isolated Python 3.11 environment if needed.
4. Once dependencies and pinned assets are available, run one environment-only
   smoke: seed 20260907, at most 30 steps, maximum startup 180 seconds, each IPC
   response bounded by 30 seconds. Holding measured starting joint positions is
   the action; no model is loaded. This tests real observations/outcome/transport,
   not task ability. A full lost control slot ends this technical validation.
5. Only after that smoke passes, perform a same-weight capability diagnostic:
   one synchronous episode, environment seed 20260908, policy seed 1701, at most
   750 steps. This cannot establish asynchronous benefit. Out-of-range targets
   end the run as a technical failure, without clipping.
6. Only if the diagnostic reaches the actual task success predicate, execute
   the minimal paired pilot: seeds 20260909 and 20260910, policy seed 1702,
   identity/predicted then predicted/identity, at most 750 steps per episode.
   Report individual outcomes and timings; two pairs cannot support a population
   success-rate improvement claim. A failed preparation/smoke/diagnostic is a
   reported outcome, not permission to silently change the model or task.

Each actual run uses a new output directory and records its source commit before
execution. Commands and terminal results are published after execution. Telemetry
stays in memory until control stops, the policy worker joins, and the environment
process closes. The current phase is implementation; later rows are conditional,
not claims of completed execution.

## Current local observations

The checkout's `.venv` contains package entry points but no executable Python.
The existing `/home/rp/miniconda3/envs/smolvla-rtc` environment is available with
Python 3.12.14, torch 2.11.0+cu128, transformers 5.5.4, NumPy 2.2.6, pytest 9.1.1,
and uv. Its metadata has no Isaac Lab or LeIsaac installation. Do not overwrite
this environment or confuse repository `.venv` remnants with a working runtime.

Results will be recorded separately from this pre-execution scope.

## Preparation findings

The first dependency resolution included LeIsaac's optional `lerobot` export
extra and failed: `lerobot==0.4.2` requires `packaging>=24.2,<26.0`, whereas
`isaaclab==2.3.0` requires `packaging<24`. The pinned
`leisaac/enhance/datasets/lerobot_dataset_handler.py` explicitly catches missing
LeRobot imports, and device imports are lazy. This experiment disables recorders
and does not use teleoperation/export. Therefore the correct minimal simulator
installation omits that unused extra; no constraint override or forced install
is needed. The model-side LeRobot remains unchanged.

The narrower resolution reached a real legacy build failure: `flatdict==4.0.1`
imports `pkg_resources`, absent from the newly selected setuptools. Constrain its
isolated build tool to setuptools 80.9.0 using the supplied build-constraints
file. This changes neither Isaac's `packaging` pin nor the model environment.

Isaac Sim's kernel package also requires **NumPy 1.26.0 exactly**; the simulator
requirements use that version, not 1.26.4. The failed 1.26.4 resolution is retained
in the preparation log. No incompatible environment was installed.

## Implemented interfaces and checks

The three planned Python modules are now implemented. Public SO100 validators,
the production engine, queue, predictor and checkpoint loaders are unchanged.
The controller reuses those components and distinguishes technical failures,
actual success/timeout, and step-limit censoring. After asynchronous startup it
quiesces the worker before resetting processors, explicitly resets the environment
seed, and seeds the measured policy RNG **after** loading/startup in both modes.
No runtime event file is written before worker/process shutdown.

`snapshot_ready_at_s` is the host time at which the CPU observation packet is
materialized; it is not claimed to be a camera exposure timestamp. The packet
also records logical simulation time, actual camera frame counters and receive
time. The current adapter does not measure exact host exposure time inside the
renderer; packet-ready-to-consumption timing must not be presented as full
camera-to-action latency.

The targeted suite passed **16 tests** using the existing model interpreter and
an actual Python 3.11 child interpreter. It covers units/endpoints/joint order,
raw image transport/provenance, actual local IPC reset/step/close, failed startup
cleanup, notify/get/step order, underflow holding, terminal observation exclusion,
lost-slot failure, and the production predicted engine's normalized prefix,
late whole-discard and joined reset. Ruff lint and format checks passed.
These are interface results, not task outcomes or a repeat of M3/M5 experiments.

### Reproduce preparation

Run from the LeRobot checkout. `leisaac-source` must be the sibling checkout at
`24d3bcd3f1e4585740fc79921782c41617237812`; no upstream source modification is required.

```bash
MODEL_PY=/home/rp/miniconda3/envs/smolvla-rtc/bin/python
SIM_PY=/home/rp/Workspace/SmolVLA_RTC/leisaac-sim-venv/bin/python
$MODEL_PY -m uv pip install --python "$SIM_PY" \
  --extra-index-url https://pypi.nvidia.com \
  --extra-index-url https://download.pytorch.org/whl/cu128 \
  --index-strategy unsafe-best-match \
  --editable ../leisaac-source/source/leisaac \
  --build-constraint examples/advanced/predictive_async/leisaac_so101_build_constraints.txt \
  -r examples/advanced/predictive_async/leisaac_so101_sim_requirements.txt
```

The simulator's separate dependency graph uses the two official vendor indexes;
the model environment is not a target of this command. The fixed Hub snapshot
download is limited to `assets/robots/so101_follower.usd` and
`assets/scenes/kitchen_with_orange/**` at the revision in L1, in the separate
`../simulator-assets` cache. It does not access old scientific data/cache artifacts.

### Source-bound environment smoke

The following command is the prepared first run, not a claim it has passed.
Use a fresh output path for every attempt and retain the preceding result.

```bash
ASSETS=../simulator-assets/models--LightwheelAI--leisaac_env/snapshots/6c35af0af55506eb75c5592930134d4af44e8341/assets
PYTHONPATH=src:examples/advanced/predictive_async \
  "$MODEL_PY" -m uv run --no-project --python "$MODEL_PY" python \
  examples/advanced/predictive_async/eval_leisaac_so101.py \
  --mode smoke --seed 20260907 --max-steps 30 \
  --sim-python "$SIM_PY" --leisaac-root ../leisaac-source --assets-root "$ASSETS" \
  --output "../artifacts/m54l2_$(git rev-parse --short HEAD)_env_smoke_v1"
```

The runner refuses a dirty source checkout, records its commit and arguments
before launch, and writes result/ticks/events/simulator output after shutdown.
`smoke` does not load policy weights. `sync` is a capability diagnostic, not an
async comparator. `identity` and `predicted` use the actual production engine.
Conditional seeds, mode ordering and limits remain as specified above.
