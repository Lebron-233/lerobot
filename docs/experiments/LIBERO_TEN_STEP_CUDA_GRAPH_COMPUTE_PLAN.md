# Preserve all ten denoising steps while testing CUDA launch-overhead removal

Execution update: the original f3894bc4 capture failed on an unpinned CPU constant
copy. After the separately registered minimal device-constant repair,84e38259
completed all synthetic pairs exactly, with formal means219.914ms eager and63.652ms
graph. See [the consolidated result](LIBERO_TEN_STEP_GRAPH_RESULT.md), including
the incomplete subsequent recorded-input validation and all preserved failures.

Date: 2026-09-08. Bounded synthetic-input engineering experiment, not a task cohort.
The single-step native screen failed at 64/90. Its complete matched ten-step control
is being executed separately at its unchanged registered source. This experiment's
code is prepared in an isolated worktree; it does not modify that running checkout.
Do not execute this GPU profile until the matched-control process has finished.

## Question and independent implementation basis

Can CUDA graph replay reduce overhead while retaining the original checkpoint's
full 50/1/10 numerical computation? Unlike reducing the Euler step count, this
candidate changes scheduling, not the action generator's mathematical operations.
An equality failure forbids claiming it is numerically equivalent.

The official [PyTorch CUDA-graphs documentation](https://docs.pytorch.org/docs/main/notes/cuda.html#cuda-graphs)
describes removing Python/C++/driver dispatch overhead through fixed-address
capture/replay, side-stream warmup, and in-place input updates. It also prohibits
CPU/GPU synchronization inside capture. The independent
[NVIDIA integration guidance](https://docs.nvidia.com/dl-cuda-graph/latest/torch-cuda-graph/torch-integration.html)
describes input lifetime and static topology obligations. These sources motivate
the engineering test, not an assumed speedup or a local performance claim.

## Frozen implementation and input contract

Keep the same strict pinned LIBERO policy/VLM, saved processors, precision,
two-camera image encoding, 50-action generation, one-action consumption and all
ten Euler denoising evaluations. No compilation, attention-kernel substitution,
fine-tuning, future tokens, dataset, simulator or recorded rollout is involved.
No changes to src/lerobot, checkpoint files or environment dependencies.

Capture the original prefix/ten-step action computation after **fresh current-image
encoding on every call**, using the existing native-token input API. The installed
Idefics3VisionEmbeddings.forward uses GPU boolean indexing in its position-ID
construction; this data-dependent-size operation is outside the capture region.
This boundary was chosen by CPU-only source inspection before GPU execution,
not by a timing search. The image encoder still executes and is timed on every
selector call; no previous observation's tokens are reused. The existing keyword
name future_image_tokens supplies freshly encoded current tokens in this test.

The ordinary selector retains its existing image preparation, state preparation,
slicing, queue handling and postprocessor. One experiment-local graph owns
fixed-address current-image tokens, token masks, language tokens/masks, model-ready
state and noise buffers. Refresh every buffer before every replay. The graph output
must retain full [1,50,32]; verify all ten original projection calls during capture.
Warm up the capture workload three times on a side stream, then capture once.
Restore the original callable on exit. A capture error ends the experiment and
is recorded; there is no automatic precision, shape, step-count or fallback search.

Four fixed synthetic fixtures use coordinate-ramp dual cameras shifted by 17*i
columns and 11*i rows; four predetermined position/gripper/quaternion values;
language for alphabet soup, cream cheese, tomato sauce and orange juice. All are
shape-level synthetic inputs, not sampled qualification observations. They exercise
refresh of images, language, state and masks at a single supported padded shape.
The complete definitions are in the runner; no fixture is chosen from outcomes.

## Fixed measurements and acceptance

First compare an eager call and graph replay at fixture 0 / seed979999. Then five
paired warmups followed by 24 formal pairs (six per fixture), fixture round-robin,
and equal seed980000+i within every pair. Alternate eager/graph order within each
four-fixture cycle and reverse that assignment on the next cycle, giving three
eager-first and three graph-first formal pairs for each fixture. Reset the
policy and processors each call. Generate fresh model-native noise inside the
timed region and feed that explicit noise to the existing selector. Include all
graph input copies, normal selector work, postprocessing and GPU completion in
CUDA-synchronized host timing. Preprocessing before select_action remains outside
the primary timer, exactly as in the previous compute comparisons. Record that the
graph path's fresh visual-encoding call count equals replay count plus its single
setup encoding; visual encoding is not skipped to obtain the measured speed.

Compare every padded action in [1,50,32], the selected normalized [1,7] action,
and its actual postprocessed [1,7] output, all with exact equality. Stop at the
first mismatch; preserve its arrays and do not interpret subsequent speed numbers.
Retain all warmup/measured pairs, full output arrays, means, empirical nearest-rank
P95, minima/maxima, count above50ms, capture setup cost and allocated/reserved memory.
No claim about deployed 20Hz from these synthetic timings, even if all are <50ms.

Before execution, merge only these experiment/test/protocol files into the main
project branch after the running matched control closes. Commit/push, register
the exact execution HEAD and a fresh output directory in Issue1, and read back
the body before starting. Use the existing dedicated interpreter and offline
launch environment. Run one fresh process, bounded to180s; a timeout is a technical
profile failure, not a task outcome.

## Research decision and limitations

Exact-equal acceleration would justify a separately registered integration check
on dynamic observations, not automatic replacement in an already closed cohort.
Slow or unequal results reject this particular graph implementation. Neither
outcome changes the single-step screen, the old failed baseline qualification,
or the untouched old confirmation. No predictor training or new native cohort is
authorized by this compute protocol.

baseline_qualified=false; realtime_qualified=false; predictor_benefit_tested=false;
risk_thresholds=null. This does not establish future-latent asynchronous benefit.

## Readiness update

The matched-control supervisor has now exited with code0. Its subsequent combined
result/provenance/process read was blocked and was not rerouted; exact matched
scores remain unread. This graph experiment's fixed implementation and fixture
schedule were committed in isolation before that completion and do not depend
on the unavailable matched outcomes. Five CPU-only contract tests passed; Ruff
lint/format and diff checks passed. The graph profile itself has not yet run.
