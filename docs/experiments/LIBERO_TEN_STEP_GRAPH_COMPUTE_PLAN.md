# Preserve ten-step generation: a bounded CUDA-graph compute probe

Date: 2026-09-08. Implementation is prepared in an isolated worktree while the
registered 90-case matched control runs. Do not run this GPU probe until that
control has closed; do not modify its pinned checkout or any dependency.

## Question and independent basis

Can CPU submission overhead be reduced without removing the original sampler's
ten denoising evaluations, changing precision, or reusing stale visual context?
This is not a new denoising-step candidate or another native outcome screen.

PyTorch's [CUDA graph documentation](https://docs.pytorch.org/docs/main/notes/cuda.html#cuda-graphs)
describes replaying the same GPU operations with reduced Python/driver submission
overhead, after side-stream warmup, using long-lived input/output storage.
NVIDIA's [graph guidance](https://docs.nvidia.com/dl-cuda-graph/torch-cuda-graph/best-practices.html)
requires fixed shapes/control flow, no CPU-GPU synchronization inside capture,
and refreshing all inputs rather than relying on Python side effects during replay.
These sources were read before this probe; they do not promise a local speedup.

## Frozen experiment

Use the existing dedicated interpreter and strict pinned LIBERO policy/VLM,
original saved processors, precision and dependencies. Keep 50/1/10, full current
two-camera encoding and the native selector/postprocessor. No compiler, TF32
setting, weights, action clipping, horizon, controller or predictor changes.

Capture only the existing `sample_actions` path supplied with current native
tokens through its existing override API, including prefix prefill and all ten
denoising steps. Recompute both cameras normally on **every** measured replay;
copy both current token arrays, both masks, language IDs/mask, current state and
explicit flow noise into long-lived buffers. Clone the output before handing it
to the ordinary selector. Do not cache images/tokens across observations.

Capture preparation uses synthetic fixture 0, three side-stream warmup calls and
one capture. Actual capture must contain ten action projections [1,50,32]. No
Python hook count during replay is misrepresented as replayed Python execution.

Five paired warmups then twenty measured pairs, indices 0-24, alternating
eager/graph order. Each pair uses identical inputs and noise seed 980000+index.
Unlike the previous constant-image fixture, both synthetic cameras, state and
noise change at every index; language cycles through all ten fixed task strings.
Inputs are coordinate ramps and artificial robot states, not recorded task data.
Reject changing shapes instead of silently keeping old inputs or recapturing.

Before timing each selector, reset policy and processors and prepare its batch
and explicit noise. CUDA-synchronize total selector/postprocessor timing. Graph
timing includes current vision, input-buffer copies, replay and output cloning.
All 50 padded action vectors, the selected normalized action and postprocessed
action must be **exactly equal** in each pair. Save actual outputs as well as
maximum differences. Stop on the first mismatch or capture error; no numerical
tolerance relaxation, fallback strategy or additional candidate search.

Keep every paired timing. Report twenty-sample mean/median/nearest-rank P95,
min/max, counts exceeding 50ms, equality counts, input metadata and setup time.
Capture/setup is separate from warmed runtime; only full equality permits an
equal-output speed interpretation. A fixed-input graph experiment cannot confer
native task, real-time, or asynchronous qualification.

## Execution and decision

Commit/push the probe and register exact HEAD/output before any CUDA capture.
Run once, only after the pending matched control exits. No simulator, dataset,
qualification images, new initial states, optimizer, runtime queue changes or
SO101 predictors belong in this experiment. Keep original and failed evidence.

If capture and changing-input equality pass with meaningful timing improvement,
the next question is broader recorded-input equivalence and deployment overhead,
not declaring a qualified controller. Otherwise preserve the failed probe and
do not change the closed native outcomes. All qualification/benefit flags remain
false and risk_thresholds=null.
