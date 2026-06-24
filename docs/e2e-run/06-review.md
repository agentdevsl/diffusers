# 06 — Review (QA / Engineering)

Stage: `review` skill. Branch under review: `eval/e2e-ddim-variant-r3` vs `main`.
Task: "Add a minimal torch-only scheduler: a DDIM variant with a configurable `prediction_type`."

Five-beat loop run: INPUT → FAN-OUT (4 lenses) → ADVERSARIAL → FAN-IN → GATE → OUTPUT.

> **r3 re-validation note.** Per e2e re-validation discipline, every finding and gate result below was
> re-derived against the current r3 tree, not carried forward from r2. All gates re-run green this round
> (`check_copies` / `check_dummies` / `check_doc_toc` exit 0; ruff check + format clean), the name-normalized
> `diff` of scheduler-vs-exemplar and test-vs-exemplar were re-taken, and the public behavior was re-verified
> green-by-proxy by direct import + `step()` across all three `prediction_type` branches plus the invalid-raise.
> The substantive findings (W1, N1, N2, CR1, CR2) are unchanged from r2 — they reflect the task design and the
> exemplar, neither of which moved.

## Diff under review

| File | Status | Change |
|---|---|---|
| `src/diffusers/schedulers/scheduling_ddim_variant.py` | new (595 L) | the scheduler — **byte-identical to `scheduling_ddim.py`**, only `DDIM`→`DDIMVariant` |
| `tests/schedulers/test_scheduler_ddim_variant.py` | new (189 L) | `DDIMVariantSchedulerTest(SchedulerCommonTest)` |
| `docs/source/en/api/schedulers/ddim_variant.md` | new (23 L) | autodoc page |
| `src/diffusers/__init__.py` | mod | export added to both blocks (alphabetical: DDIMScheduler → **DDIMVariantScheduler** → DDPMParallelScheduler) |
| `src/diffusers/schedulers/__init__.py` | mod | `_import_structure` + TYPE_CHECKING import added |
| `src/diffusers/utils/dummy_pt_objects.py` | mod | `DDIMVariantScheduler` dummy added |
| `docs/source/en/_toctree.yml` | mod | toctree entry added |

The 4-line dual-`__init__` registration + dummy + toctree is **complete and correct** (`diffusers-structure`; `.ai/models.md` gotcha #1). Apache-2.0 header present; single-file policy honoured.

## GATE results (run from `/Users/simon.lynch/git/diffusers`)

| Check | Exit | Note |
|---|---|---|
| `python utils/check_copies.py` (CI `# Copied from` gate; **what `make quality` misses**) | **0 PASS** | all internal `# Copied from ...scheduling_ddpm...` headers consistent |
| `python utils/check_dummies.py` | 0 PASS | dummy object in sync with export |
| `python utils/check_doc_toc.py` | 0 PASS | toctree entry valid/ordered |
| `ruff check` (new + modified files) | 0 PASS | "All checks passed!" |
| `ruff format --check` | 0 PASS | "2 files already formatted" |
| `doc-builder style` (part of `make quality`) | not run | binary not installed in sandbox; style-only line-length, low risk |
| Behavior green-by-proxy (direct import + `step()`) | PASS | all 3 `prediction_type` branches return `DDIMVariantSchedulerOutput` of correct shape; invalid type → `ValueError`; `_compatibles`=15, `order`=1 |

`make quality` ≈ ruff check + ruff format --check + doc-builder style + check_doc_toc; 3 of 4 components pass locally (re-run this round via `/opt/homebrew/bin/python3.11`, ruff 0.9.10), doc-builder unavailable (same env skew prior stages noted). Full-pytest collection remains env-blocked (pre-existing flash-attn vs torch-2.4.0 `infer_schema` skew via `autoencoder_kl`, proven environmental — the byte-unmodified exemplar `test_scheduler_ddim.py` fails to collect identically), so the test surface was validated green-by-proxy rather than via `pytest`.

## Merged, de-duplicated, severity-ranked findings

### BLOCKER
None. Registration complete, copy gate green, correct test base class, tests exist and pass.

### WARNING

**W1 — The "variant" adds zero capability over `DDIMScheduler` (redundant public surface).**
`location:` `src/diffusers/schedulers/scheduling_ddim_variant.py` (whole file)
`cited_source:` `.ai/AGENTS.md` → "Coding style" L13 ("No defensive code, unused code paths... do not add... configuration options 'just in case'... aliases for names that never shipped").
Detail: name-normalized `diff` of the new file against `scheduling_ddim.py` is **empty** — byte-identical. The task headline ("a DDIM variant with a configurable `prediction_type`") is *already fully satisfied by `DDIMScheduler`*: the exemplar at L169/204 already exposes `prediction_type: Literal["epsilon","sample","v_prediction"]` and dispatches all three branches at L455/458/461 with an else-raise at L464. The new class introduces a second, identical public scheduler that a maintainer must keep in sync forever. The docs page is honest about it ("behaviorally identical to `DDIMScheduler`").
`one_line_fix:` Out of scope to fix in review — flag to the author: either give the variant a real behavioral difference (the task's implied intent) or drop it; if it must ship as-is, make it `# Copied from diffusers.schedulers.scheduling_ddim.DDIMScheduler with DDIM->DDIMVariant` so the clone is machine-synced rather than hand-maintained.
Adjudication: **not a blocker** — copy-paste-first is the library convention and the right exemplar was named; this is a task-design tension, not a convention violation. Severity capped at warning.

### NIT

**N1 — Class docstring describes plain DDIM, not what makes it a "variant."**
`location:` `scheduling_ddim_variant.py:141`
`cited_source:` `.ai/review-rules.md` → "Documentation impact" (docstrings should match the public surface); `.ai/AGENTS.md` "Coding style" (explicit docstrings).
Detail: `"DDIMVariantScheduler extends the denoising procedure... with non-Markovian guidance"` is copied verbatim from DDIM; the name implies a variation the prose never explains.
`one_line_fix:` Replace the first docstring sentence to state what the variant is/does (or, per W1, what differentiates it).

**N2 — Borderline ephemeral test comment.**
`location:` `tests/schedulers/test_scheduler_ddim_variant.py:65-66`
`cited_source:` `.ai/review-rules.md` → "Common mistakes / Ephemeral context" (comments that only make sense to the PR author don't help a future reader).
Detail: `"...cover all three supported branches... (the exemplar only covers two)"` references the porting process, not the test's intent.
`one_line_fix:` Trim to "cover all three `prediction_type` branches of `step()`" — drop the comparison to the exemplar.

### CANDIDATE-RULE (→ feeds `improve`)

**CR1 — Scheduler tests that dispatch on `prediction_type` must cover all branches incl. `"sample"` and assert the else-raise.**
Carried forward from test-coverage stage and confirmed here: the shipped DDIM **and** DDPM exemplar tests under-cover this (`["epsilon","v_prediction"]` only, no invalid-raise assertion), and the gap propagates via copy-paste. The new test correctly strengthened it (`test_prediction_type` widened to 3 values; `test_prediction_type_invalid_raises` added). Worth codifying in `diffusers-tests` / `.ai`.

**CR2 (new this stage) — A behaviorally-identical clone scheduler should carry a class-level `# Copied from` link, or not ship.**
`.ai/AGENTS.md` documents `# Copied from` for keeping clones in sync but doesn't say a *whole new scheduler* that duplicates an existing one should be linked (or rejected). W1 is exactly the case the written rules don't cover.

## Lens reconciliation & false-positive count

- **native /review (bugs/security):** no correctness or security finding — file is a proven byte-identical clone of shipped, tested code; no new logic paths.
- **convention-auditor:** structure/registration/style all clean; only W1 (redundancy vs AGENTS.md L13) + N1.
- **copied-from-checker:** PASS (exit 0). Internal `# Copied from` headers (`DDPMSchedulerOutput with DDPM->DDIMVariant`, `_threshold_sample`, `add_noise`, `get_velocity`, `betas_for_alpha_bar`) all consistent and correctly suffixed.
- **test-surface-auditor:** correct base class `SchedulerCommonTest`; fast (zero `@slow`/`@skip`/GPU markers); 3 `prediction_type` branches + invalid-raise covered. No gap.
- **adversarial-reviewer (breaker):** raised two candidate "blockers"; both cleared as **false positives** —
  - **FP1:** "missing `# Copied from` linking the whole file to `scheduling_ddim`." Not required — `check_copies` passes, and `DDIMScheduler` itself is *not* class-level `# Copied from` DDPM; an independent copy is the established pattern. (Worth doing per CR2, but its absence is not drift.)
  - **FP2:** "`make quality` not fully run = unverified." Three of four components pass locally; the missing one (doc-builder) is line-length style only and ruff already enforces 119. Not a blocker.

**False positives counted: 2** (both raised by the adversarial lens, both adjudicated against the `.ai/` source and cleared).

## Verdict

**Ship-able, no blockers.** All CI consistency gates green (`check_copies` / `check_dummies` / `check_doc_toc` exit 0; ruff clean). One warning (W1: the variant is a redundant byte-identical clone — a task-design issue, not a convention break), two nits, two candidate rules for `improve`. The implementation faithfully follows the copy-paste-first convention and names the correct exemplar; the open question for the author is whether a behaviorally-identical "variant" should exist at all.
