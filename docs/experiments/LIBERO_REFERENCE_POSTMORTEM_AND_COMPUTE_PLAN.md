# LIBERO reference closure and bounded compute diagnosis

Date: 2026-09-08. Authority: the user's continued implementation and experiment
authorization. This document does not reopen the qualification protocol.

Execution closed: the one registered microbenchmark completed at source
`9699ef2307156918b1b7cae4fa829cb1647a0766`, with 25/25 exactly equal paired
actions and all 20 measured cached-token calls above 50 ms. See
[the result](LIBERO_REFERENCE_COMPUTE_RESULT.md). The design below is preserved.

## Decision on the closed experiment

Accept the completed qualification record at execution source
`1cd7d222c9c49d89c6d385006e7967f825f64a90`: 185/200 native successes, zero
technical failures, and task 5 at 14/20. The overall 180/200 gate passed;
the independent 16/20 task floor failed. Keep `baseline_qualified=false` and
confirmation `not_started`. A confirmation slot is not an observed failure.
No lower threshold, favorable-task subset, replacement seed, extended horizon,
rerun, policy change or predictor training is authorized by this diagnosis.

The successful nine other tasks total 171/180 (95%); task 5 contributes 6 of
the 15 failures (40%). This is a descriptive concentration, not a discovered
causal explanation. All failures exhausted the registered 280-action bound.
The committed tuple outcomes give 103 successes by action 150, 174 by 200,
184 by 250 and 185 by 280. Successful trajectories finish at median action
147 and maximum action 257. The summaries cannot establish what the unsuccessful
trajectories would do beyond 280 actions, nor whether they failed at grasp,
transport or placement.

## Separate timing question

The recorded policy-selection/postprocessing time is 8127.818778563407 seconds
for 32301 selected actions: mean 251.627466 ms, or 5.032549 nominal 20 Hz ticks.
This is not a measured asynchronous observation age or a latency percentile.
Policy time occupies 86.3383% of the sum of tuple wall times. Generate 50,
consume 1 uses 2% of the predicted action positions; this does not imply that
changing chunk size would save 98% of runtime or preserve the policy.

Before proposing a new baseline campaign, answer one technical question:
**Can removing only the current-image encoder bring this unchanged selector
within the nominal 50 ms action interval?** The answer informs whether visual
encoding alone is a plausible compute fix. It cannot qualify a controller or
test future-latent prediction benefit.

## Frozen bounded microbenchmark

Entry: `examples/advanced/predictive_async/profile_libero_reference_compute.py`.
Use the existing dedicated interpreter, strict whole-policy loader, policy/VLM
revisions, saved processors, saved precision and unchanged 50/1/10 selector.
No simulator, dataset, demonstration, stored rollout, qualification state or
confirmation sample is read. Inputs are two deterministic 256x256 coordinate
ramps, an identity quaternion, zero position/gripper values and the fixed task-0
language string. They are synthetic shape fixtures, not robot observations.

Capture the native current-image tokens once. Compare the ordinary selector
against the same selector with only `encode_image_tokens` replaced by a return
of those identical captured tokens. All preparation, prefix/VLM, action expert,
denoising, selection and postprocessing outside that call remain in place.
This is a component-removal diagnostic, not a deployable cached-image policy.

Run five paired warmups and twenty measured pairs. Pair i uses policy noise
seed 920000+i, with a reset and the same seed for both variants. Alternate
variant order by pair index. Synchronize CUDA around total timings, and check
exact equality of each paired normalized and postprocessed action. Separately
measure twenty calls to the original encoder on the captured arguments.
Keep every sample; no seed selection, precision change, compilation, denoising
sweep, chunk-size change or extra trials based on the result.

Report native/cached/encoder samples, empirical median/p95, mean and range,
paired exact-equality counts, token metadata and the strict-load report. The
small synthetic benchmark does not estimate deployment-tail reliability.
If pairing fails, stop and preserve the mismatch; do not interpret timing
differences as a semantics-preserving comparison. If the cached-token path
still exceeds 50 ms, current-image encoder removal alone does not meet that
budget for this fixture and implementation.

Commit and register the exact benchmark source before its one execution.
Store a new local output under the repository's ignored outputs directory;
do not write into the closed qualification artifacts. Source and dependencies
remain unchanged during execution. An execution/tool failure is a diagnostic
failure, never an additional task outcome. Do not evade rejected tool operations.

## Research boundary

The baseline, real-time and predictor-benefit flags remain false regardless of
the profile. Qualification outcomes are not a tuning set for a future candidate.
Further policy/candidate changes need an independent source-supported proposal,
fresh sample identities and a separate registered qualification design. The
original confirmation cohort stays unopened. Existing SO101 predictors and
reserved tests remain untouched.

## Evidence-access limitation in this continuation

The committed report, machine-readable result, all 200 committed tuple outcomes
and GitHub closure comment were read. DevSpace.read rejected the external
qualification launch script because its path is outside the allowed read roots;
that file was not fetched through another mechanism. A distinct source-interface
search command was blocked by the platform safety-state check and was not
retried. No fresh raw-step or image-level failure diagnosis is claimed.
