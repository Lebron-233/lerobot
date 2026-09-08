# LIBERO qualification runner implementation validation

Date: 2026-09-08. Validation precedes any formal qualification tuple.

`tests/rollout/test_libero_reference_qualification.py`: **20 passed in 2.23 s**
using the unchanged dedicated LIBERO interpreter and `uv run --no-project`.
Tests use synthetic observations and the production `SmolVLAPolicy.select_action`
queue method; they load no checkpoint and create no native simulator.

Coverage includes all 400 registered identities and ordering, altered or omitted
manifest fields, success at action 280 with simultaneous timeout, cleanup failure,
reset/settling/seed ordering, one action consumed per query, non-finite state,
per-task and total gates, blocked confirmation after development failure, native
process exit accounting, unrun slots versus observed failures, incomplete-sample
intervals, and disagreement between predicted and native action records.

The first test run had 16 passes and four failures because the synthetic cameras
were 2×2, below the existing channel-last preprocessing interface's minimum
spatial shape. Changing the fixture to 4×4 resolved all four; no production
preprocessing or policy behavior was changed. Both logs are preserved:

```text
/home/rp/Workspace/SmolVLA_RTC/artifacts/libero_reference_qualification_20260908T064953Z/qualification_tests_v1.log
/home/rp/Workspace/SmolVLA_RTC/artifacts/libero_reference_qualification_20260908T064953Z/qualification_tests_v2.log
```

Targeted Ruff lint passed, Ruff format check accepted both new Python files,
and `git diff --check` passed (all exit 0). Logs are captured as
`qualification_lint.log`, `qualification_format.log` and
`qualification_diff_check.log` in the same artifact directory before the
execution commit. No old smoke or unaffected historical test suite was rerun.
