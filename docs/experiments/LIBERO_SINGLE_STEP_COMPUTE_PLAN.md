# One independently motivated denoising candidate: compute feasibility only

Date: 2026-09-08. The user's continuing execution authorization covers this
separate bounded probe. The original 50/1/10 qualification is closed and failed;
its thresholds, episodes, candidate and results remain unchanged.

## Independent source and question

The public primary experiment archive
[zuoxingdong/smolvla-libero-eval](https://github.com/zuoxingdong/smolvla-libero-eval/tree/114d19c51bf7655e61fb994f3b344a0257ceb20b)
predates these local outcomes (last commit 2026-06-28). It identifies the exact
same policy and VLM revisions. Its matched LIBERO-Spatial comparison changes
`num_steps=10` to `num_steps=1` with `n_action_steps=1` fixed, reporting 75.5% to
81.5% on 200 episodes. Those are another experimenter's results, not our Object
qualification. Its simulator version, renderer, image resolution, runtime and
evaluation design differ. This is evidence for examining one fixed candidate,
not a guarantee of improvement or a reason to inherit its score.

The current official LeRobot LIBERO page's `n_action_steps=10` recommendation
belongs to Pi0.5 reproduction; it does not establish a SmolVLA consumption change.
Keep our consumption at one. Do not sweep steps 2/4/5, consumption spans, precision,
MuJoCo versions, tasks, prompts or initial states.

Question: with the same weights and complete original image path, how much wall
time remains when only the supported Euler sampler's step count changes 10 to 1?
The sampler change can change every action; it is not an exact optimization of
the frozen controller and cannot rescue its failed qualification.

## Fixed execution

Entry: `examples/advanced/predictive_async/profile_libero_denoising_candidate.py`.
Use the existing dedicated environment and strictly loaded fixed snapshots.
Keep two 256x256 cameras, saved resize/statistics, precision, chunk 50, consumption
1, one thread, and all parameters frozen. No dependency change or compilation.
Use the same deterministic coordinate-ramp/zero-state synthetic shape fixture as
the prior compute diagnosis; no stored robot observation, dataset or simulator.

Five paired warmups plus twenty measured pairs, alternating 10/1 execution order,
equal seed within each pair (`930000+i`, i=0..24). Reset both processors and the
selector queue before every call. CUDA-synchronize the selector/postprocessor
wall timer. Verify actual projection counts/shapes are 10 or 1 times [1,50,32],
one finite seven-dimensional selected action, and an empty queue. Save both
action vectors and their difference; do not require equality across candidates.

After the paired calls, one instrumented call per candidate at seed 930024 uses
the sampler's existing opt-in vision/prefix/flow timings. Its action must equal
the corresponding already measured same-mode action. These two instrumented
calls describe phases only and do not enter the twenty-call timing statistics.
Restore num_steps=10 in memory at exit; write no checkpoint/configuration files.

Commit/push the implementation and this plan, then publish exact execution HEAD
and the fresh output directory in Issue #1 before this probe. Preserve all samples
and any failed attempt. A shape/finite/count/instrumentation mismatch stops the
interpretation; no automatic retries. Results include all timing samples, empirical
P95, maximum, 50-ms exceedance count and the explicit non-equivalence of actions.

## Boundaries

This probe authorizes 52 synthetic policy calls and zero task transitions. It
does not authorize reopening the old confirmation queue, rerunning development,
selecting a favorable subset, tuning on those episodes, training a predictor,
or calling the one-step sampler qualified or real-time-ready. Any subsequent
native-outcome study needs a separately frozen data/identity and qualification
design; untouched old samples are not silently repurposed.

Regardless of timing: baseline_qualified=false, realtime_qualified=false,
predictor_benefit_tested=false, risk_thresholds=null.
