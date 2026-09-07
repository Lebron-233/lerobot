# L12 development selection frozen before any new test scene

Execution source `bf4e025dbf0cacf4777289b90498a8b70d4974fa`.
The small state residual model ran all30 scheduled epochs using only the six
L10 development-train episodes. Epoch29 was selected by validation STATE MAE,
before any decoder comparison. It has25,478 parameters. No vision/policy/action
expert parameters were fit; L6 visual weights remain the original epoch3.

Validation mean active-state MAE: persistence0.09821573950 -> predicted0.03595098458.
Episode6:0.10391567827 ->0.03719280384; episode7:0.09251580073 ->0.03470916532.
These are normalized-state errors, not degrees or task scores.

After that selection,96 fixed validation cases used the NEW coherent teacher
policy(Z[t+d],S[t+d],L,paired noise). All candidate online inputs remain causal.

| Context | Mean first25 feasible normalized-action L1 |
|---|---:|
|Current visual/current state (identity)|0.15503290057|
|L6 predicted visual/current state|0.16033042162|
|Current visual/predicted state|0.11306411059|
|L6 predicted visual/predicted state (joint)|0.11089841453|
|True future visual/current state (offline only)|0.11562720772|
|Current visual/true future state (offline only)|0.10633068938|

Joint improves79/96 cases versus identity and87/96 versus visual-only; both
episode means improve in both comparisons. State prediction improves in both
episodes, so the previously declared development permission is **true**.
Fresh test collection is now allowed with the predeclared seeds and case list.

The mandatory attribution contrast is NOT uniformly positive: joint beats
state-only in just32/96 cases, improves one episode mean and worsens the other.
The1.9155% joint-versus-state-only aggregate improvement is not stable visual
benefit. Most improvement may be due to state compensation; that remains a
hypothesis to test independently, not a victory for visual-only M3.

Frozen artifacts: `artifacts/m54l12_state_development_v1/state_best.pt`,
`state_selection.json`, all31 state-validation records (including epoch0), and
`selection.json` with all96 decoder cases. New test is
`m54l12_joint_context_test_v1`, six previously unused seeds20270320–25.
No model/threshold choice follows test collection; no old test is reopened.
No joint-state production binding or task-outcome comparison has run yet.
