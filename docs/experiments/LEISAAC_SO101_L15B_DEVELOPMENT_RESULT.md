# L15b closed: averaging gradients did not qualify the incremental visual student

Date:2026-09-07. Training source `9a6b89020d646732196bdc1b9e405d83e0728b75`.
The same L15 state-only development data were reused, explicitly as development.
No new test or task scenes were read to fit either student.

All400 optimizer updates of four distinct training episodes completed; this
is1600 single-query forward/backward examples, the same example budget as L15's
1600 individual updates. Policy decoder numerical batch size stayed1. Original
L15 loss, architecture, initialization, learning rate and eligibility threshold
were unchanged. No EMA or runtime risk gate was introduced.

| Update | Validation MAE | Improving cases /176 | Eligible |
|---:|---:|---:|---|
|0|0.076867422|0|no|
|50|0.076030454|95|no|
|100|0.076155783|93|no|
|150|0.073704210|97|no|
|200|0.074484800|97|no|
|250|0.074898591|93|no|
|300|0.073364195|93|no|
|350|0.071946297|92|no|
|400|0.071438625|94|no|

The diagnostic best is400. It improves the equal-episode state-only reference
MAE0.076867421507 by7.0625%, and both validation episode means improve. However,
94/176 improved cases remain below118. Maximum checkpoint coverage is97/176.
No eligible checkpoint exists, so **L15b qualification FAILS**. The stronger
mean improvement of the single-case fit (9.0875%) is also not a qualification.
Reused validation, objective differences from earlier studies, and optimizer
step-count differences prevent these means from being called blind efficacy.

Zero-residual initialization again matched state-only actions exactly (max0.0).
First accumulated student gradient norm1.06776297, policy parameter gradients0.
Training plus reference/validation wall time1073.630s, peak allocated CUDA memory
1971452416bytes. These are training diagnostics, not inference measurements.
The read-only audit checks every actual prefix, the two-noise case grid, all400
updates' four distinct training episode identifiers, and eligible-first selection.

Artifacts: `m54l15b_balanced_training_v1` and
`m54l15b_closed_development_audit_v1.json`, under the project artifacts root.
Both visual students remain offline and reserved test scenes20270630--35 remain
uncollected. No failed gate was relaxed to obtain a new task candidate.

The next independently registered [L16 information-reference study](LEISAAC_SO101_L16_PRIVILEGED_REFERENCE_PLAN.md)
therefore contains exactly THREE arms: state-only, true target-time visual with
causal predicted state, and true target-time visual+state. The optional learned
student arm is absent because neither model qualified. This study asks whether
accurate future information itself offers useful task improvement, rather than
assuming further visual training is guaranteed to help. Its privileged arms
cannot be deployed or presented as learned future predictions. All L15 tests
stay sealed; this is a new scientific reference question on new scenes.
