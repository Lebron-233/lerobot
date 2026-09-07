# L18: zero-delay full-task capability before another forecasting campaign

Registered 2026-09-08 after L17's complete32-condition closure. No L17 scene or
old student test will be rerun. L17's native-task follow-through found one
privileged success and no original-task success in either causal state arm;
this does not reveal how the unaugmented synchronous policy performs on new
scenes. This finite DEVELOPMENT diagnostic answers that missing prerequisite.

## Source-grounded candidate alignment

Keep `wsagi/SmolVLA-PickOrange@c8c3318dba152b0ba671ff07b4314418d5aa4b4a` with its
entire pinned weights, statistics and camera schema. The local pinned model
card identifies LightwheelAI/leisaac-pick-orange and recommends action_horizon50.
It describes three-orange manipulation but reports only2/5 strict rounds under
its own benchmark. That author-reported result is not our native predicate or
a reliability guarantee. Do not infer a7/8 original-task baseline from the card.

The existing runtime literal is `Grab orange and place into plate`; the pinned
model card's actual inference command uses `Pick up the orange and put it in
the plate`. Test exactly these TWO documented strings, not a prompt search.
Both are recorded in each candidate manifest and actually supplied to its
preprocessor/policy. No language instruction is synthesized from simulator
object positions or changed after a scene outcome is observed.

## Fixed diagnostic

Eight entirely new development scenes20270910--20270917, policy seeds3810--3817.
Two prompt profiles per scene, sixteen conditions, each once. Four of each
two-profile ordering are shuffled with3800 before any outcome. No replacements
or extra seeds when a condition fails. The seed numbers are RNG identifiers.

Use the existing synchronous eval path: `select_action` with50 generated and
50 executed actions before replanning. NO visual predictor, learned future
state, delayed queue takeover, oracle input, coupled noise, or RTC. Reuse the
same CPU PhysX/RTX, one tensor worker, dual standard640x480@30Hz cameras,
feasible_v1 action projection, zero initial pose, and warmed_v2's thirty logged
hold actions followed by same-seed reset. Policy and processors remain frozen.

Each condition runs until the ORIGINAL native three-oranges-plus-rest success
or120 simulated seconds/3600 steps. There is no early subgoal stop. The first
ten-tick stable occupancy is retained only as a secondary diagnostic. Inference
pauses physics, so neither this experiment nor a good outcome qualifies real-time
asynchronous control. There are sixteen independent simulator process exits,
not one shared exit, because the already established eval entrypoint is reused.

Record native successes, all technical failures, restricted native time with
failures costing120s, first occupancy, and the maximum simultaneous occupancy
of the existing three native boxes. Independently reconstruct the native rule
from saved pre-auto-reset witnesses. Verify task text, execution length, lack
of predictors/engine, source identity and geometry for each pair. Do not relabel
a one-orange subgoal as complete task success.

Development qualification for EACH prompt independently is>=7/8 native successes
and zero technical failures. Report both, even if neither qualifies. If a profile
qualifies it still needs independent confirmation before a held-out forecasting
efficacy claim; no profile is chosen or tuned during this diagnostic. If neither
qualifies, do not start another same-candidate visual-forecasting outcome campaign
or blame the remaining failures solely on observation latency. Inspect candidate
and task compatibility before registering a successor.

These outcomes are not pooled with L17/L16: different scenes, no delayed context,
and independent prompt-specific bootstrap actions. Same scene/model seeds do
not imply pixelwise or trajectory identity. A prompt comparison is descriptive
development evidence, not an independently confirmed effect of prompt wording.

## Implementation scope and checks

Add a synchronous-only task-text argument to the existing eval entrypoint, with
the two literal choices above. With no argument all old behavior is unchanged;
asynchronous and old unmatched candidate paths reject it. Test the actual
preprocessor/policy task input using the existing synchronous loader interface.
Reuse the native-outcome and box/rest audit functions; no new queue, training
framework, asset or simulator is introduced. Push this plan and source before
running the sixteen conditions. L10/L15 reserved tests remain unopened.
