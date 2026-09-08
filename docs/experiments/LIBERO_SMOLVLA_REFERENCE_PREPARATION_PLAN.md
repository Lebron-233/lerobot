# Separate matched SmolVLA reference preparation after L20

## 2026-09-08 execution update

Independent preparation has now completed: fixed policy/VLM and assets,
Python 3.12 environment with compatible dependencies, actual EGL rendering,
strict full policy loading, saved processors, and the preregistered 20-action
technical smoke. See
[LIBERO_REFERENCE_PREPARATION_RECOVERY_RESULT.md](LIBERO_REFERENCE_PREPARATION_RECOVERY_RESULT.md)
and [LIBERO_REFERENCE_QUALIFICATION_PLAN.md](LIBERO_REFERENCE_QUALIFICATION_PLAN.md).
The complete task/confirmation queues remain unrun, and all qualification flags
remain false. Candidate identity and prior GitHub/optional-package unknowns below
are historical; the checkpoint's training/control provenance still needs evidence.

## Original preparation proposal

Date: 2026-09-08. Status: preparation proposal / exact candidate unresolved.
This is not an executed or source-frozen outcome campaign. The user continues
to authorize project implementation and experiments without another Pro review.
All SO101 cohorts, including L20's failed native-reference qualification, remain
closed and immutable. No more pose, prompt or feedback-span search is scheduled.

## Reason for the separate candidate

L20 verifies actual native full-task improvement in three development scenes,
but only 3/8 native successes for the shorter span. Earlier source-defined
initializations and two documented prompts did not qualify reliable full-task
control either. The state/action normalizer now has a positive numerical
alignment check against the pinned published training source. There is no
evidence justifying another normalization adjustment or visual-residual fit
on these task outcomes.

Evaluate an existing, separately task-matched SmolVLA benchmark checkpoint,
rather than presenting SO100 or SO101 weights as another embodiment's policy.
The concrete public preparation target is `HuggingFaceVLA/smolvla_libero`,
with the repository's existing `LiberoEnv` and official `hf-libero==0.1.4` extra.
The current public model card exists, but its exact revision, processors,
architecture settings and task performance have NOT been verified locally.
No claim that it meets a success threshold is made.

## Source-supported interface differences

The exact local LeRobot `src/lerobot/envs/libero.py` defines an eight-dimensional
EEF/rotation/gripper observation, seven-dimensional relative EEF/gripper action,
20 Hz control by default, two 256x256 camera streams, fixed initial-state support,
and ten settling actions on reset. Its object suite limit is 280 actions.
`docs/source/libero.mdx` documents the relative/absolute control distinction,
image renaming, and hard resets for benchmark reproduction. These are not the
SO101 six-motor / 30 Hz / 480x640 contract.

Retain SO101 engines and validators unchanged. Do not load any SO101 visual or
state predictor into the new embodiment. In particular, changing six to seven
dimensions is not sufficient evidence for latent/predictor compatibility.
First qualify the frozen task policy through the existing synchronous interface;
only then design a separately bound causal-prefix and latent experiment.

## Bounded preparation and subsequent execution boundary

1. Resolve and record the exact public checkpoint revision and its configuration,
   tokenizer, preprocessing/postprocessing statistics and license. Inspect source
   files before allowing any downloaded custom code to run; do not assume that
   the old candidate's model configuration can load the new weights.
2. Prepare a distinct project environment without changing the original model
   Conda environment or Isaac Python environment. Pin LIBERO/MuJoCo/robosuite
   dependencies and assets; do not rely on a moving asset repository default.
   Record actual installation and smoke outcomes, not inferred availability.
3. Validate the declared camera orientation/key mapping, eight-dimensional state,
   seven-dimensional action semantics, native reset/settling and success signal.
   Use a bounded wiring run, not a task-selection sweep.
4. Before any outcome cohort, freeze the full, unfiltered LIBERO-Object suite,
   exact initial-state IDs, canonical control configuration and fixed policy
   noise seeds. Report native tasks, not a newly invented progress endpoint.
   Predeclare reference qualification and a disjoint confirmation before results
   are opened. Do not select only tasks on which the model happened to succeed.
5. A new task baseline cannot inherit real-time qualification or scientific passes
   from SO101. Any future prediction stage needs new training/validation/test
   identities; the old unopened L10/L15 tests remain untouched.

No automatic training, alternative-model sweep, asset edit, shared-job interruption
or remote service is part of this preparation.

## Actual tool boundary during this continuation

A DevSpace command that would inspect optional LIBERO/MuJoCo/robosuite package
availability in the current environment was blocked by the platform before
execution. A distinct read-only public-candidate metadata/configuration fetch
was also blocked before execution. Neither was retried by an alternate command,
tool or endpoint. Thus installed optional-package status and exact new-model
revision remain unknown. No new environment, package, checkpoint or simulator
was installed or launched. Static repository/documentation reads succeeded,
as did the existing SO101 result/statistics closure work.

Resume with public metadata/source qualification once that tool operation is
available; do not treat this preparation document as a passed smoke or a frozen
ready-to-run benchmark. No further project experiment process was started.

## Public primary sources consulted

- [LeRobot LIBERO documentation](https://huggingface.co/docs/lerobot/libero)
- [Published SmolVLA LIBERO model card](https://huggingface.co/HuggingFaceVLA/smolvla_libero)
- [hf-libero 0.1.4 package description](https://pypi.org/project/hf-libero/0.1.4/)

The package page describes assets loaded from `lerobot/libero-assets`; its
historical upstream installation example is not an instruction to downgrade
the current project Python or torch environment.
