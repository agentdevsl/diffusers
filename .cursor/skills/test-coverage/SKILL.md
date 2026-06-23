---
name: test-coverage
description: QA validation step (read-only). Confirm the test surface is real before review — correct base class, fast coverage of the public surface, nothing fast mis-marked @slow, no weakened assertions. Runs inside one subagent; dispatched by implement (shift-left) and as the standalone QA stage. Use after implement or on any diff.
---

# test-coverage (QA validation)

A read-only validation pass that runs **inside a single subagent — not an orchestrator**. `implement` dispatches it
(step 4, shift-left) as a cheap `composer-2.5[]` agent against the diff; it also runs as the standalone QA stage
between `implement` and `diffu-review`. Either way it does one job in its own context: judge whether the tests are *real*,
not just *present*. It writes no tests — coverage gaps fold back to the writer (`implementer`); this skill validates,
reports, and gates, then returns its verdict.

Ground the judgement in the library's own rubric **by role** — cite, never restate. The `diffusers-tests` rule
defines the base class for every subsystem (schedulers, pipelines, models). For models and pipelines,
`.ai/skills/model-integration` ("Testing") is the authoritative scaffolding detail — dummy configs, what to
auto-generate, the fast-coverage shape. Schedulers have no `.ai/` test doc, so the rule is the sole authority there
(the pack's own value).

**Input.** The diff, the touched test files, and the `plan` doc — its Test Plan names the base class + exemplar, its
impact set names the public surface to cover. **Output.** A coverage verdict — a before/after gap note for the plan —
plus any convention caught-but-unwritten tagged `CANDIDATE-RULE`, which routes *backward* to `improve`, never forward
into review.

Validate, in one read-only pass:
1. **Right base class — and its scaffold actually wired.** The base class alone is inert; the mixin needs its
   required hooks present and correctly shaped (the rubric carries the full contract — cite it, don't recopy it).
   The high-value traps: a scheduler on `SchedulerCommonTest` must override `get_scheduler_config(**kwargs)` (a
   scheduler test built on `PipelineTesterMixin` is the most common error of all); a pipeline on
   `PipelineTesterMixin` + `unittest.TestCase` needs `get_dummy_components()` and `get_dummy_inputs(device, seed=0)`
   at that exact signature, plus the `params` / `batch_params` / `image_params` frozensets that drive
   parametrization; a model test must be **generated** with `utils/generate_model_tests.py` (not hand-written) and
   its `get_init_dict` / `get_dummy_inputs` / `input_shape` / `output_shape` TODOs filled. A new component with *no*
   test at all is a blocker, not a gap.
2. **Fast coverage of the public surface** — a non-`@slow` test exercises the new public behavior (public methods,
   config options, `prediction_type`s, edge cases), and it stays fast because the dummy components are *tiny* (small
   `num_layers`/dims, a fixed `torch.manual_seed(0)`). Cross-check the plan's impact set so nothing public is left
   untested.
3. **No coverage silently dropped** — two ways it leaks: a fast test hidden behind `@slow` (scheduler common tests
   are fast by default) falls out of the gate; and an inherited mixin test that doesn't apply must be
   `@unittest.skip("…")`, never deleted or omitted — deletion drops the contract invisibly. Flag either.
4. **Assertions are real** — no tautologies, no assertion weakened to force green. A test that cannot fail proves
   nothing.

**Gate.** Run the fast pytest slice over the touched test file to confirm green — running tests edits nothing, so it
stays within the read-only remit. Never propose weakening an assertion to pass. When the slice is red, keep the
assertion real and report EITHER the path-under-test fix OR the red as a blocker — never `@slow` / `@skip` /
tautologize it away. The gaps and fixes go back to `implementer`; this skill returns the verdict, not a diff.
