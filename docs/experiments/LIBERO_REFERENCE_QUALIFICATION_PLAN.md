# Independent LIBERO reference qualification proposal

Date: 2026-09-08. **Draft, not approved for execution. No qualification episode
has been run.** These sample sizes, seeds and numerical gates are proposed here
before opening any formal task results. They do not inherit the SO101 7/8 gate.

## Prerequisites

The registered technical preparation is complete. The follow-up
[training/control audit](LIBERO_REFERENCE_TRAINING_CONTROL_AUDIT.md) establishes
the author-confirmed `HuggingFaceVLA/libero` dataset, exact float32 equality of all
30 saved normalization statistics with its pre-checkpoint v2.1 metadata, and the
camera/state/action convention in the author's linked official evaluator. The
source/semantics prerequisite is satisfied for this independent reference.
Keep the recorded relative OSC_POSE controller, camera rotation, axis-angle
state, saved normalization, 20 Hz control, current reset/settling and 50/1/10
generation/consumption/denoising. Dataset metadata FPS 10 does not warrant a
control-rate override. The audit records the supporting source and limitations
of historical training/evaluation reproduction.

Then approve and commit the final protocol, environment lock, checkpoint and asset
identities, runner and initial-state/seed manifest before starting the development
qualification. No outcome-dependent edits are allowed within a campaign.

The complete historical training recipe and author score reproduction are not
established. This proposal qualifies the fixed independent reference. All ten
Object task names occur in the confirmed training-source corpus; confirmation
IDs/seeds are disjoint from this project's preparation/development samples,
not proven disjoint from original training demonstrations. Dataset task indices
differ from native suite task IDs; preserve the native order specified below.

## Frozen candidate and complete suite

Use the candidate/VLM revisions, unmodified saved processors, hf-libero 0.1.4
wheel, pinned asset revision and local execution contract recorded in
`LIBERO_REFERENCE_SMOKE_PLAN.md`. Load the entire policy strictly. Use the full
LIBERO-Object suite with `task_order_index=0`, without filtering or reordering.
All ten official initial-state files actually contain 50 rows of 110 simulation
state values. No demonstrations are downloaded for this preparation.

The full task names are listed below. All have the suffix
`_and_place_it_in_the_basket`. The language is the full task name with underscores
replaced by spaces, exactly as the native suite supplies it.

| ID | Full task name | Native goal |
|---:|---|---|
| 0 | pick_up_the_alphabet_soup_and_place_it_in_the_basket | In alphabet_soup_1 basket_1_contain_region |
| 1 | pick_up_the_cream_cheese_and_place_it_in_the_basket | In cream_cheese_1 basket_1_contain_region |
| 2 | pick_up_the_salad_dressing_and_place_it_in_the_basket | In salad_dressing_1 basket_1_contain_region |
| 3 | pick_up_the_bbq_sauce_and_place_it_in_the_basket | In bbq_sauce_1 basket_1_contain_region |
| 4 | pick_up_the_ketchup_and_place_it_in_the_basket | In ketchup_1 basket_1_contain_region |
| 5 | pick_up_the_tomato_sauce_and_place_it_in_the_basket | In tomato_sauce_1 basket_1_contain_region |
| 6 | pick_up_the_butter_and_place_it_in_the_basket | In butter_1 basket_1_contain_region |
| 7 | pick_up_the_milk_and_place_it_in_the_basket | In milk_1 basket_1_contain_region |
| 8 | pick_up_the_chocolate_pudding_and_place_it_in_the_basket | In chocolate_pudding_1 basket_1_contain_region |
| 9 | pick_up_the_orange_juice_and_place_it_in_the_basket | In orange_juice_1 basket_1_contain_region |

The native `LIBERO_Floor_Manipulation._check_success` evaluates each BDDL goal
through its predicates. Here `In` requires the target region's native contact
and containment checks. Use `info.is_success` from the actual task step. Do not
replace this with an image judgment, distance threshold, reward shaping or a
separate approximate geometry test.

## Proposed samples and execution

Use one episode per task/initial-state/seed tuple, with no repeated attempt at a
tuple. Twenty distinct states per task cover more starting configurations than
the preparation example while keeping each cohort at 200 episodes.

- Development IDs, on **every** task:
  `[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]`.
- Independent confirmation IDs, on **every** task:
  `[21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40]`.
- For task `t` and state `s`, development environment seed is
  `510000 + 100*t + s`, policy/noise seed `610000 + 100*t + s`.
  Confirmation environment seed is `710000 + 100*t + s`, policy/noise seed
  `810000 + 100*t + s`. These formulas uniquely enumerate all proposed seeds;
  materialize them into the final manifest before execution.
- Order: task IDs 0 through 9; within each task, ascending listed initial-state
  ID. Run one environment at a time. Complete the development cohort and its
  decision before opening any confirmation outcome.
- Hard reset each episode to its registered state row, using the native initial
  state loader. Apply exactly ten `[0,0,0,0,0,0,-1]` settling actions, counted
  separately. Clear policy and processor state on every reset. Set policy/noise
  seeds after construction/reset and before the first model query.
- Relative Panda OSC_POSE at 20 Hz, two 256×256 cameras, saved 512×512 resize/pad,
  generation 50, execution 1, ten denoising steps and saved precision. Native
  EEF clipping/scaling and gripper sign/integration remain as documented.
- Maximum 280 measured actions per episode, with explicit time-limit accounting.
  Success at the final allowed action counts as success; retain simultaneous
  terminated/truncated flags. Stop an episode immediately on native terminal
  return. A reset/settling failure is a technical failure for that tuple.

State ID 0 on all tasks is excluded from both cohorts. Preparation used task 0,
ID 0, environment seed 4200 and policy seed 4201; the synthetic interface probe
used noise seed 4202 and no simulator. IDs 41–49 remain unused. Neither cohort
may be substituted for new predictor training data or treated as a tuning set
for a future candidate.

## Proposed eligibility decision and statistics

A cohort qualifies only if it has **at least 180/200 native successes overall,
at least 16/20 on each of the ten tasks, and zero technical failures**. Apply
the same gate separately to confirmation. A baseline is qualified only when
both complete cohorts pass under one unchanged candidate and contract.

The proposed 90% overall gate requires a broadly reliable synchronous reference;
the 80% floor prevents strong tasks from hiding a weak object task. Twenty
episodes per task give five-percentage-point resolution while using disjoint
initial-state sets for confirmation. These are project decision requirements,
not published model performance claims or an assertion of 90% population success.

Report native successes / all 20 scheduled tuples for each task, and the
unweighted mean of the ten task proportions. Equal task denominators make that
mean equal to the pooled 200-episode success proportion. Report cohort totals
separately; do not pool confirmation with development to rescue a failed gate.

Provide 95% Wilson descriptive intervals per task (n=20). For an overall interval,
resample the 20 tuples **within each task**, compute the macro mean over the ten
fixed tasks, and report the 2.5/97.5 percentiles of 20,000 draws; use bootstrap
seed 910001 for development and 910002 for confirmation. These intervals describe
uncertainty under the chosen sampling model; the registered gate uses observed
counts. The suite and initial states are fixed, so this is not a claim about
arbitrary unseen tasks or an official reproduction of the checkpoint's training.

Also report technical failures, task timeouts, first-success measured action and
restricted simulated completion time. Non-success episodes receive 280/20=14 s
for that descriptive restricted-time summary; record actual wall time separately.
Neither simulated frequency nor wall-time summaries confer real-time eligibility.

## Failure accounting and early stopping

Every scheduled tuple stays in the denominator. Python exceptions, simulator
crashes, invalid/non-finite observations/actions, unsuccessful required cleanup,
or missing final records after a launched tuple count as technical failures and
native non-successes. Record finite out-of-box commands separately; the declared
native action transformation handles them without a second clamp.

Task failure or timeout alone does not stop the cohort. Do not replace a failed
task, initial state or seed, and do not stop early because a pass or fail seems
likely. On a technical failure, stop the campaign to diagnose it rather than
running more tuples through a broken runtime. All scheduled but unexecuted tuples
remain unsuccessful in the conservative 200-slot decision denominator, explicitly
labeled **not run**, not falsely labeled observed technical failures. Such a cohort
cannot qualify, and its incomplete outcome data are reported as incomplete.

If a repair changes the runtime, source or candidate, preserve and close the
failed campaign. A separately approved, preregistered campaign is required;
do not quietly rerun the affected tuples inside the original one. A tool access
denial before a tuple launches is a campaign blocker, not an observed episode
failure. Do not open confirmation unless development qualifies and its records
are closed. No additional favorable-task queue is allowed.

## Subsequent research boundary

No future-latent training, old SO101 predictor transfer, RTC/oracle/risk-gated
comparison or production asynchronous migration starts under this draft. After
baseline confirmation, separately bind the new embodiment's state/visual
predictors, causal-prefix contract and training/validation/test identities.
The eventual native-outcome question still requires old committed actions to
execute during background inference and new chunks to take over or be discarded
under production semantics.
