# L15b: conditional balanced-gradient development, same forward/backward budget

Date:2026-09-07. This is a second, explicitly bounded development candidate,
not a reopening of any test and not a replacement of the first L15 fit.
Run ONLY if the completed1600-update L15 single-case fit has no eligible
checkpoint. If L15 qualifies, this candidate is not executed or selected.

The first L15 development fit showed fluctuating validation means at its
early checkpoints, despite a lower mean at update200. This suggests testing
gradient aggregation; it does not prove gradient variance caused the errors.
The initial L15 fixed run must finish and every checkpoint remains retained.

Use precisely the same L15 six training/two validation trajectories, teachers,
state-only reference, paired noise draws, frozen models, zero final-projection
initialization, learning rate1e-4, loss and gradient clipping. The only optimizer
profile change is400 updates of four single-query microbatches per update.
Choose four distinct training episodes for each update, then one uniformly
sampled query/noise reference per chosen episode. Average their losses before
the optimizer step; clip the accumulated gradient once. Keep numerical decoder
batch size1, no EMA, new architecture, risk gate or extra dataset.

Both candidates consume1600 training forward/backward examples. L15b has400
optimizer steps rather than1600, so this is an optimizer/batch profile contrast,
not a claim to isolate every individual optimization effect. Seed3500,
AdamW weight_decay1e-4; validate at0 and every50 optimizer steps (the same nine
exposure checkpoints). Record the sampled episode/query/noise identifiers.

The eligibility rule is UNCHANGED: at least2/3 validation cases improve over
state-only, both episode means improve, and equal-episode average error is
lower than both state-only and immutable L6-joint. Eligible-first selection
then minimizes average error. A diagnostic minimum without eligibility is not
deployment permission. Reusing development validation is disclosed; no such
mean is called an independent result.

Only a qualified model may open the already reserved, still unseen L15 test
scenes20270630--35. Freeze/publish that one selection first, use its exact
checkpoint throughout testing, and preserve all L15 test gates and denominators.
If neither candidate qualifies, do not collect those scenes to choose a model.
Neither candidate is automatically enabled in a task or real-time controller.
