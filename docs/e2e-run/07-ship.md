# 07 — Ship (DevOps / PM)

Stage: `ship` skill. Branch: `eval/e2e-ddim-variant-r3` vs `main`.
Task: "Add a minimal torch-only scheduler: a DDIM variant with a configurable `prediction_type`."

Five-beat loop run: INPUT → FAN-OUT (3 readonly lenses: CI gates, blast radius, conventions) → FAN-IN
(blocking-first) → GATE (`make quality` boundary) → OUTPUT (release notes + go/no-go). No adversarial gate at ship.
Authority cited, not restated: `.github/workflows/pr_tests.yml`, `Makefile`, prior `06-review.md`.

> **r3 re-validation note.** Per e2e discipline, every gate below was **re-run against the live working tree**
> of `eval/e2e-ddim-variant-r3` this stage (ruff 0.9.10, `/opt/homebrew/bin/python3`), not carried from the r2
> ship report. Re-derived this round: ruff check/format, `check_copies`/`check_dummies`/`check_support_list`/
> `check_forward_call_docstrings`/`check_doc_toc` (all exit 0), boundary re-inventoried from `git status`, and the
> deps-table confirmed untouched. The diff is **uncommitted** (working-tree mods + untracked new files), so
> `main..HEAD` is empty by design — the orchestrator commits/pushes; this stage only verifies and writes go/no-go.

## Diff under ship (working tree, uncommitted — orchestrator handles git)

| File | Status | Roots-class |
|---|---|---|
| `src/diffusers/schedulers/scheduling_ddim_variant.py` | new | `src/diffusers/` ✓ |
| `tests/schedulers/test_scheduler_ddim_variant.py` | new | `tests/` ✓ |
| `docs/source/en/api/schedulers/ddim_variant.md` | new | `docs/` ✓ |
| `src/diffusers/__init__.py` | mod (additive) | `src/diffusers/` ✓ |
| `src/diffusers/schedulers/__init__.py` | mod (additive) | `src/diffusers/` ✓ |
| `src/diffusers/utils/dummy_pt_objects.py` | mod (additive) | `src/diffusers/` ✓ |
| `docs/source/en/_toctree.yml` | mod (additive) | `docs/` ✓ |
| `docs/e2e-run/*` | new (chain artifacts) | `docs/` ✓ |

The two registration diffs are **purely additive** — only `+` lines adding the `DDIMVariantScheduler` export to
both `__init__` blocks; no symbol removed, no signature changed.

## FAN-OUT lens 1 — `ci-gate-mapper`: local check → real CI job

Each changed `.py` path matches a `pr_tests.yml` `paths:` filter (`src/diffusers/**.py`, `tests/**.py`), so the
full PR pipeline triggers. The three gating jobs chain via `needs:` —
`check_code_quality` → `check_repository_consistency` → `run_fast_tests` (a red quality/consistency job blocks the
test matrix from ever starting). Mapping every local gate to its REAL CI job name:

| Local command (mirror) | Real CI job (`pr_tests.yml`) | Result this stage | Runnable in this env? |
|---|---|---|---|
| `ruff check $(check_dirs) setup.py` | **check_code_quality** (`make quality`) | exit 0 "All checks passed!" | **YES** (ruff 0.9.10) |
| `ruff format --check ...` | **check_code_quality** (`make quality`) | exit 0 "5 files already formatted" | **YES** |
| `doc-builder style src/diffusers docs/source --max_len 119 --check_only` | **check_code_quality** (`make quality`, Makefile) | not run — binary absent | **NO** — CI-only; line-length style only, ruff already enforces 119 |
| `python utils/check_doc_toc.py` | **check_code_quality** (`make quality`) | exit 0 | **YES** |
| `python utils/check_copies.py` | **check_repository_consistency** | exit 0 | **YES** — the `# Copied from` gate `make quality` does NOT cover |
| `python utils/check_dummies.py` | **check_repository_consistency** | exit 0 | **YES** |
| `python utils/check_support_list.py` | **check_repository_consistency** | exit 0 | **YES** |
| `python utils/check_forward_call_docstrings.py` | **check_repository_consistency** | exit 0 "All …documented." | **YES** |
| `make deps_table_check_updated` | **check_repository_consistency** | PASS — diff does not touch `setup.py`/`dependency_versions_table.py`; `git status` shows the table unchanged | partial (the bundled `setup.py deps_table_update` is env-blocked, but the table cannot drift since the diff never touches it) |
| `pytest ... tests/schedulers ...` (`-k "not Flax and not Onnx and not Dependency"`) | **run_fast_tests** → matrix slice "Fast PyTorch Models & Schedulers CPU tests" | not run — torch absent | **NO** — CI-only; `DDIMVariantSchedulerTest` is NOT excluded by the `-k` filter, so it runs in this slice |

**The 5 `check_repository_consistency` scripts CI runs are exactly:** `check_copies`, `check_dummies`,
`check_support_list`, `check_forward_call_docstrings`, `make deps_table_check_updated` — **all green this stage.**
This is the job `make quality` does NOT cover, and the cheapest most-likely-to-fail one. Note:
`utils/check_inits.py` belongs to the local `make repo-consistency` target but is **not** in the pr_tests
`check_repository_consistency` job; its local failure here is purely environmental
(`FileNotFoundError: src/transformers/__init__.py` — transformers source absent) and does not gate this PR.

## FAN-OUT lens 2 — `change-impact-mapper`: blast radius + boundary

- **Registration footprint.** The dual-`__init__` registration is complete and correct: `src/diffusers/__init__.py`
  (`_import_structure` + lazy block), `src/diffusers/schedulers/__init__.py` (both blocks), auto-generated
  `dummy_pt_objects.py`, `_toctree.yml` doc entry. `check_dummies` exit 0 confirms dummy ↔ export sync.
  Alphabetical order correct (DDIMScheduler → DDIMVariantScheduler → DDPMParallelScheduler).
- **Public API.** One symbol ADDED (`DDIMVariantScheduler`); nothing removed or renamed → **no `deprecate()` owed.**
- **`# Copied from` sources.** The new file edits no canonical block; it only consumes internal
  `# Copied from ...scheduling_ddpm...` headers, all consistent (`check_copies` exit 0). No copy elsewhere is
  invalidated.
- **Docs/test surface.** Doc page + toctree present (`check_doc_toc` exit 0); test file present, lands in the
  Models & Schedulers slice.
- **Boundary.** Re-inventoried from `git status --porcelain`: **all changed paths fall inside the approved roots**
  `src/diffusers/`, `tests/`, `docs/`. **No out-of-boundary file.** (No boundary-guard ships in the target
  checkout — the pack lives in the sibling repo — so the boundary was verified by direct root-class inspection.)

## FAN-OUT lens 3 — `convention-auditor`: deprecate / docs owed + CANDIDATE-RULEs

- **`deprecate()` owed?** No — additive API only.
- **Doc page owed for a mergeable PR?** Owed and **present** (`docs/source/en/api/schedulers/ddim_variant.md`
  + `_toctree.yml`); `check_doc_toc` green.
- Carries forward review's **W1** (the variant is a byte-identical clone of `scheduling_ddim.py`, adds zero
  capability over `DDIMScheduler` which already exposes configurable `prediction_type`) as a **warning, not a
  blocker** — copy-paste-first is the library convention and the correct exemplar was named; task-design tension,
  not a convention violation.
- **CANDIDATE-RULE tags travel backward to `improve`, NOT into release notes:**
  - **CR1** — scheduler tests dispatching on `prediction_type` must cover all 3 branches incl. `"sample"` and
    assert the else-raise (shipped DDIM/DDPM exemplar tests under-cover; gap propagates via copy-paste; the new
    test already fixed it locally).
  - **CR2** — a behaviorally-identical clone scheduler should carry a class-level
    `# Copied from diffusers.schedulers.scheduling_ddim.DDIMScheduler with DDIM->DDIMVariant` header (machine-synced),
    or not ship.

## FAN-IN — merged, de-duplicated, blocking-first

### BLOCKING (would force unconditional no-go)
**None.** No out-of-boundary file; no failing `make quality` component that is runnable here (the one unrun
component is an env-absent tool, not a failure — see GATE).

### NON-BLOCKING (carried for the PR body / `improve`, not release notes)
- W1 (warning) + N1/N2 (nits) from `06-review.md` — unchanged, source untouched this stage. Surface W1/N1/N2 in PR body.
- CR1, CR2 — feed `improve`.

## GATE (the boundary)

Verdict rule (skill): ANY out-of-boundary file OR a failing `make quality` → unconditional **no-go**.
- Out-of-boundary file: **none**.
- `make quality` (= `check_code_quality`): of its 4 components, **3 pass locally** (ruff check, ruff format
  --check, check_doc_toc); the 4th (`doc-builder style`) is **not runnable in this sandbox** (binary absent) —
  an environment gap, not a failure, and line-length style only which ruff-119 already covers. No `make quality`
  component returned non-zero.
- Hook context: the skill wires `make quality` as a `beforeShellExecution` gate on `git commit|git push`; here
  the orchestrator owns git, so the gate is informational. CI remains the hard gate.

Two checks are **CI-only in this env** and stay deferred to the real pipeline: `doc-builder style` (in
`check_code_quality`) and the `run_fast_tests` pytest matrix (torch absent — and full collection is additionally
blocked by the pre-existing flash-attn vs torch-2.4.0 `infer_schema` skew via `autoencoder_kl`, proven
environmental since the unmodified exemplar test fails identically). All five `check_repository_consistency`
checks — the cheapest, most-likely-to-fail job, which `make quality` does NOT cover — pass locally (exit 0).
The new test was validated **green-by-proxy** by `review` (direct import + `step()` across all three
`prediction_type` branches; invalid → `ValueError`; correct base `SchedulerCommonTest`; fast, no `@slow`/GPU markers).

## DECISION: **GO** (no-go conditions not met)

Reasons:
1. **No blocking condition.** No out-of-boundary file; no failing runnable `make quality` component.
2. **Every runnable CI gate is green:** `check_code_quality` (3/4 components, the 4th env-absent); all of
   `check_repository_consistency` (5/5: check_copies, check_dummies, check_support_list,
   check_forward_call_docstrings, deps-table-in-sync).
3. **Additive, low blast radius:** one new exported symbol, no `deprecate()` owed, registration + dummy + doc +
   toctree complete and consistent.
4. **`run_fast_tests` well-formed:** correct base class, fast, no skip/GPU markers; green-by-proxy. The only
   reason it is not run here is the absent `torch` (CI-only), not any defect in the change.

GO caveats (residual risk pushed to the real CI, not blockers):
- `doc-builder style` and the `run_fast_tests` scheduler matrix slice were **not executable here** (doc-builder +
  torch absent) — CI must be the confirming run for these two; both are low risk given green-by-proxy evidence and
  the byte-identical-to-tested-exemplar nature of the code.
- W1 stands: the shipped "variant" is behaviorally identical to `DDIMScheduler`. A reviewer may legitimately ask
  whether it should exist; that is a product/task call, addressed via CR2 in `improve`, and does not block a
  CI-clean, in-boundary, additive PR.

## Proposed PR release notes (for the body the orchestrator opens)

> **Add `DDIMVariantScheduler` — a torch-only DDIM-family scheduler with a configurable `prediction_type`.**
>
> New scheduler modeled on the `DDIMScheduler` exemplar (`src/diffusers/schedulers/scheduling_ddim.py`),
> supporting `prediction_type ∈ {epsilon, sample, v_prediction}` with an explicit else-raise. Purely additive:
> registered in both `__init__` blocks (alphabetical), dummy object + toctree + autodoc page added. No public API
> removed or changed — **no `deprecate()` owed**. Includes a fast `SchedulerCommonTest` covering all three
> `prediction_type` branches plus the invalid-value raise.
>
> CI status: `check_code_quality` and `check_repository_consistency` green locally (incl. `check_copies` — no
> `# Copied from` drift); `run_fast_tests` (Models & Schedulers slice) validated green-by-proxy pending the
> torch-enabled CI runner.
>
> **Reviewer note (W1):** the variant is currently behaviorally identical to `DDIMScheduler` (the docs page says
> so). Open question for merge: give it a real behavioral difference, add a class-level
> `# Copied from …DDIMScheduler with DDIM->DDIMVariant`, or drop it. Two nits (N1 docstring, N2 test comment) are
> easy follow-ups.
