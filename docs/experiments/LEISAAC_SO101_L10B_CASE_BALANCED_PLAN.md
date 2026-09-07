# L10b: bounded case-balanced development before opening any new test

Date:2026-09-07. L10 absolute-loss fitting is complete at13c27554. No test scene
20270130–35 has been collected or opened. All prior L6/L8 tests remain closed.

## Closed L10 development result

Eight new trajectories completed4800 actions/4808 observations.400 updates on
1152 causal training cases completed with a real gradient through the frozen
policy; first predictor gradient norm0.295004, policy gradients0. Peak CUDA
allocation was1651332096 bytes. The validation-selected update300 gives:

* identity first25-action L1:0.0979260081;
* L6 parent:0.0984113293;
* absolute-loss student:0.0896918945 (about8.4% below identity),37/72 improving.

This satisfies L10's minimal permission to open a test, but it is not evidence
of the two-thirds case coverage required for a stable action result. Preserve
the selection and every validation/update record in `m54l10_action_training_v1`.
Defer its planned test, rather than spend a blind test on another thin mean-only
signal. This is a documented development-stage protocol amendment, not a change
to an observed test result or relabeling of L10's validation gate as failed.

## New candidate, same development data, still no test feedback

Hypothesis: absolute action error emphasizes large-error contexts; normalize each
training example by its identity-to-oracle error so small-error contexts also
matter. Start again from the immutable L6 parent (not the selected L10 student).

    L = MAE(student[:25],teacher[:25]) / (MAE(identity[:25],teacher[:25]) + 0.02)
        + 0.001*latent_SmoothL1 + 0.01*latent_cosine_distance

Teacher and identity are stop-gradient, using exactly the same current state,
language and noise. The0.02 denominator offset is a fixed training stabilizer,
not an online risk threshold or action clamp. All model/processor/feasible
projection semantics are unchanged. No VLA/VLM/expert parameter is trained.

Fixed800 updates, batch1, AdamW1e-4/weight_decay1e-4, grad clip1, RNG2500, same
train anchors/delays. Checkpoints at0/100/.../800. This new objective and fixed
larger optimization budget form one new candidate; it is not an isolated causal
ablation of objective versus training duration. No further tuning within L10b.

Validation is explicitly reused development data, not independent new evidence.
Prefer checkpoints with at least48/72 cases improving over identity and lower
mean action L1 than both identity and the L6 parent; among them choose lowest
mean. If none qualify, retain the best development checkpoint diagnostically
but leave the new test unopened. This stronger development prerequisite is set
before L10b fitting; original L10's recorded eligibility remains unchanged.

If qualified, publish/freeze that selection before collecting the original
reserved six test scenes. Evaluate this single selected candidate once with the
same216 cases and unchanged stable-action gate (144 improvements,5/6 improved
episodes, positive episode-bootstrap95% lower bound). Identity and the immutable
L6 parent are reported as controls. Do not choose between candidates using test
results. No claim about task success follows from this action-oracle comparison.
