# L20 contingency: shorten open-loop execution without changing the base policy

Prepared while L19b is running. Execute only after L19b fully closes with failed
native-task qualification; otherwise prioritize independent confirmation of the
qualified source-defined initialization. This is not another pose or prompt search.
Do not modify the L19b execution HEAD while it is running.

## Motivation and scope

Zero additional inference delay is not zero observation aging: the existing
synchronous controller consumes 50 actions (1.667 simulated seconds at 30Hz)
before making a new policy query. L19/L19b change initial position, not this
feedback interval. Test the one separately registered alternative of consuming
10 actions (0.333 simulated seconds) before reobserving. Both still generate
50-action chunks with the same frozen policy and ten flow steps. There is no
new checkpoint, forecaster, training, RTC, privileged input or normalization.

Changing query frequency also changes the number/timing of stochastic policy
draws. This experiment tests the complete control setting, not a unique causal
explanation of old failures or proof of a future-latent mechanism.

## Fixed once-only design

Eight new scenes 20271110--20271117, policy seeds 4010--4017. Two settings per
scene: consume 50 versus consume 10, four forward/four reverse orders shuffled
by 4000, sixteen total conditions. Start at the original zero joint pose in
both settings; do not choose a pose from the preceding results. Thirty logged
hold setup actions followed by same-seed reset, current standard dual cameras,
same original task literal, same physical dt/assets, same feasible action mapping.
The different wrist-start poses in L19 are not part of this comparison.

Use only the existing synchronous SmolVLA select_action queue. Verify with the
actual select_action method that generation still returns fifty actions while
new policy calls occur at ten/ fifty consumed actions respectively. Record the
configured execution span in both candidate and run manifests. Default span50
and old async bindings remain unchanged; span10 is not silently enabled for the
production predictive engine or old frozen candidate.

Continue every condition to the original native three-orange-plus-rest success
or 120 simulated seconds /3600 actions. All native and technical failures stay
in the denominator with restricted native time120s. First plate-region occupancy
is secondary descriptive data only. No early stop, failed-trial replacement or
additional span grid is permitted. Current physics still pauses for inference:
no wall-clock30Hz deployment claim is possible.

Span10 development eligibility requires >=7/8 native successes and zero technical
failures, unchanged from the preceding native-task prerequisite. Report span50
regardless of direction, all paired native times, matched starting geometry,
and a descriptive eight-scene paired bootstrap (50000 draws/RNG4001). This is
development, not forecaster efficacy or independent confirmation. A positive
result requires a new confirmation before reopening a forecasting campaign.
Failure closes this bounded feedback-span investigation; do not keep trying
spans or train visual residuals on these opened outcomes. Reserved L10/L15
tests remain unopened. No unrelated local job is modified.
