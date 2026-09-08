# Closed LIBERO trajectory diagnosis

Date: 2026-09-08. This is post-hoc description, not qualification or model selection.

The closed reference remains 185/200, task 5 at 14/20, confirmation not_started.
Read all 200 existing development event journals once to extract actual selector
latency distributions, continuous EEF/gripper/action descriptors and command
out-of-box component counts. Physical orientation differences use quaternion
distance, not a naive axis-angle vector difference near the representation cut.
No descriptor replaces native success or a native grasp/contact predicate.

Inspect actual saved dual cameras for all 15 failures and all 20 task-5 episodes
(29 distinct episodes). Generate contact sheets at observation indices
0/40/80/120/160/200/240/280, clipped to each terminal index, plus that terminal
index. Temporal detail may be read from the same existing journals/images to
resolve a visible event; never generate a new transition. Describe visual stage
observations separately from causal hypotheses. Do not infer object poses or
contacts from EEF state alone.

Artifact source: `/home/rp/Workspace/SmolVLA_RTC/artifacts/libero_reference_qualification_20260908T064953Z`.
Its normal DevSpace workspace registration returned `ws_dd9599e15a`; an actual
event-journal read succeeded. The prior rejected launch-script read is not needed
or retried. Source files remain read-only; derived results go to a fresh directory
under the code checkout's `outputs/`.

No simulator, policy, optimizer, confirmation row, dataset download, changed
threshold, replacement seed, additional task rollout or prior SO101 reserved test
is involved. No new qualification campaign is authorized by a diagnostic pattern.
Subsequent changes require a concrete implementation defect or independently
supported candidate/contract, not a search on these qualification outcomes.

The direct GitHub connector returned `FORBIDDEN: This conversation is restricted
to developer MCPs` on its read action. The existing permitted DevSpace GitHub CLI
read had already retrieved the branch and discussion before that rejection.
