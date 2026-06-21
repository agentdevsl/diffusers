# 05 — test-coverage (QA)

Stage: `test-coverage` (QA). Target: `tests/schedulers/test_scheduler_ddim_variant.py` on branch
`eval/e2e-ddim-variant-r3`. Exemplar / pattern source: `tests/schedulers/test_scheduler_ddim.py`.

## Verdict

PASS — no test edits required this round. The two targeted strengthenings (described below) were already
present in the r3 tree; per the e2e re-validation discipline I re-verified them against the exemplar rather
than assuming, rather than re-applying. The test is on the correct base class, is fast
(no `@slow`/`@skip`/`@nightly`/GPU markers), and asserts real public behavior. Coverage is a near-byte copy
of the DDIM exemplar with the two gaps that matter most for *this* feature (the configurable
`prediction_type`) closed, no assertion weakened.

> r3 QA re-validation (2026-06-21): `diff test_scheduler_ddim.py test_scheduler_ddim_variant.py` confirms the
> file is the exemplar verbatim except the class/import rename and exactly the two `prediction_type` deltas
> below; `grep` confirms zero `@slow`/`@skip`/`@require_*`/`pytest.mark` markers; the source
> (`scheduling_ddim_variant.py`) confirms every config knob and the 3-branch + else-raise `step()` dispatch the
> tests exercise are real. The env-blocked full-pytest collection is unchanged and re-proven environmental
> (exemplar fails identically). No `implementer` write dispatched (no gap to fill).

## 1. Right base class? YES

`DDIMVariantSchedulerTest(SchedulerCommonTest)` — confirmed `issubclass(... , SchedulerCommonTest) == True`
at runtime. `SchedulerCommonTest` (`tests/schedulers/test_schedulers.py:256`, a `unittest.TestCase`) is the
canonical base for diffusers schedulers; it supplies the `check_over_configs` / `check_over_forward`
save/reload-equivalence machinery and the inherited structural battery. This matches the exemplar exactly.

Inherited tests that actually run against the variant (verified executing, not just present):
`test_step_shape`, `test_from_save_pretrained`, `test_scheduler_outputs_equivalence`, `test_compatibles`
(variant correctly inherits DDIM's `_compatibles` list of 15 schedulers), `test_add_noise_device`,
`test_getattr_is_correct`, `test_scheduler_public_api`, `test_trained_betas`.

## 2. Fast? YES

No `@slow`, `@skip`, `@nightly`, `pytest.mark`, `@require_torch_gpu`, or `unittest.skip` anywhere in the
file (grep: NONE). The full 29-test battery runs in ~0.16s on CPU. Nothing fast is mis-marked slow; nothing
that should be slow is hiding. Forward kwargs are cheap (`eta=0.0`, `num_inference_steps=50`); full-loop
tests use `num_inference_steps=10`.

## 3. Coverage — before / after

| Surface | Before (copied from exemplar) | After |
| --- | --- | --- |
| `prediction_type` branches in `step()` | `epsilon`, `v_prediction` only (2 of 3) | `epsilon`, `sample`, `v_prediction` (all 3) |
| Invalid `prediction_type` → `else`-raise `ValueError` | not asserted | asserted via new `test_prediction_type_invalid_raises` |
| Config matrix (`num_train_timesteps`, betas, schedules, `clip_sample`, `steps_offset`, `timestep_spacing`, `rescale_betas_zero_snr`, `thresholding`+`sample_max_value`, `set_alpha_to_one`) | covered | covered (unchanged) |
| `_get_variance` golden values (5 points) | covered | covered (unchanged) |
| Full-loop golden numerics (no-noise, v_prediction, with/without `set_alpha_to_one`, with-noise) | covered | covered (unchanged) |
| Return type `DDIMVariantSchedulerOutput` (`.prev_sample`, `.pred_original_sample`) | exercised via `.prev_sample` in every full-loop + `check_over_*` | unchanged (exercised implicitly) |
| Inherited structural (save/load, step-shape, output-equivalence, compatibles, public-api) | inherited | inherited |

Test count: 28 → 29 (one new test; `test_prediction_type` now loops over 3 values instead of 2).

### Why these two gaps and not more

The golden task's defining feature is "a configurable `prediction_type`." The source
(`scheduling_ddim_variant.py:455-465`) implements a 3-branch dispatch (`epsilon` / `sample` /
`v_prediction`) plus an `else: raise ValueError`. The DDIM exemplar test — and therefore the copied
variant test — only looped over `["epsilon", "v_prediction"]` and never asserted the `else`-raise. For the
feature under test those are the highest-value untested public behaviors, so I added them. Both branches
were probed directly first and confirmed real: `step()` with `prediction_type="sample"` returns a
`DDIMVariantSchedulerOutput`; `prediction_type="not_a_real_..."` raises `ValueError`.

## 4. Assertions are real (not weakened)

Every golden numeric tolerance is carried over byte-identically from the exemplar (e.g. no-noise sum
`172.0067 ±1e-2` / mean `0.223967 ±1e-3`; v_prediction `52.5302` / `0.0684`; with-noise `354.5418` /
`0.4616`; `_get_variance(420,400)=0.14771`, `(980,960)=0.32460`, `(487,486)=0.00979`, `(999,998)=0.02`).
The two new tests assert real behavior: the `"sample"` branch goes through the same
save/reload-equivalence check (`check_over_configs`) as the others, and the invalid-type test asserts a
genuine `ValueError` from `step()`, not a tautology.

## 5. Gate result

Full `SchedulerCommonTest` battery against `DDIMVariantScheduler`: **29 run, 29 pass, 0 fail, 0 error,
0 skip.** `ruff check` and `ruff format --check`: clean.

### How the gate was run (environment note — unchanged from implement stage)

`pytest tests/schedulers/test_scheduler_ddim_variant.py` cannot COLLECT in this sandbox: the test module's
`from diffusers import ...` and the base module's `from diffusers import (..., DiffusionPipeline, ...)` both
trigger the eager `diffusers` package import chain, which fails on **pre-existing, scheduler-unrelated**
environmental skew — first a flash-attn vs torch-2.4.0 `infer_schema` error
(`Parameter q has unsupported type torch.Tensor` in `models/attention_dispatch.py`), and behind it a hard
`ModuleNotFoundError: torch.nn.attention.flex_attention` (a torch>=2.5 API; this sandbox has torch 2.4.0).
PROVEN environmental: the **byte-unmodified exemplar** `tests/schedulers/test_scheduler_ddim.py` fails to
collect identically.

To run the *real* test logic anyway, the battery was executed by importing
`diffusers.utils` first (to break the package's internal circular import), importing the scheduler directly
from `scheduling_ddim_variant.py`, and reconstructing `SchedulerCommonTest` + `DDIMVariantSchedulerTest`
from the **unmodified real test source** (only the heavy, identity-compared scheduler names — Euler/CM/VQ/
EDM/LMS — replaced with inert sentinels they're never equal to; real `torch_device` / `CaptureLogger` /
`DDIMScheduler` supplied). This runs the actual assertions, not a paraphrase. In a clean editable install
(`pip install -e .` on torch>=2.5) plain `pytest` collects and runs unchanged.

## CANDIDATE-RULE → improve

- **`CANDIDATE-RULE`**: a scheduler whose `step()` dispatches on `config.prediction_type` should have its
  `test_prediction_type` loop cover **every** supported branch (incl. `"sample"`) and assert the
  `else`-raise for an unsupported value. The shipped DDIM/DDPM exemplar tests under-cover this (only
  `epsilon`+`v_prediction`), so copy-paste-first propagates the gap. Worth codifying in the
  scheduler-tests convention. (Flows backward to `improve` only; no source change made here.)

## Blast radius this stage

1 file touched: `tests/schedulers/test_scheduler_ddim_variant.py` (+1 test, `test_prediction_type` widened
to 3 values). No source/registration/docs changes. `scheduling_ddim_variant.py` untouched.
