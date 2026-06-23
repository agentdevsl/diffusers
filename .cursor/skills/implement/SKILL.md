---
name: implement
description: Engineering build step. Realize the approved plan in code, tests-first (TDD), via a single writer in a worktree. Use after plan, before test-coverage.
---

# implement (Engineer)

Thin orchestrator (orchestrate in root; agents execute). The crux: **two writers race** — keep **one writer**
(`implementer`); use concurrency only to *verify* the diff.

**Isolation.** Run Cursor's native **`/worktree`** before any write — the build lands on its own branch off the base,
not in-place on the default branch. One writer per tree; a second writer needs a second `/worktree`.

**Efficiency (the wall-time budget).** Wall time is dominated by serial round-trips, so: `Read` whole files **once**
(never `cat`/`head`/`sed`/`tail`), search with `Grep`/`Glob` (never ad-hoc `rg`), batch independent reads/searches into
one turn, scope `pytest` to the changed path. Each avoided slice-read or lone shell call is a round-trip removed.

**Input.** The `plan` doc's test plan (base class + exemplar) and named reference component. **Two pre-flight stops
before any write** — *already-present* (capability already in the tree → verify, say so, stop; don't reimplement a
no-op) and *plan-ambiguity* (spec too underspecified to build → route back to `plan`, don't invent the design). Full
criteria: `references/terminals.md`.

1. **WRITE (single writer).** Dispatch `implementer` — the sole writer, tests-first, in its own context; it owns the
   copy/adapt/registration mechanics, `# Copied from`, and the soft-dep dummy-objects stub. The TDD contract it must
   hold — test-first (new component scaffolds a base-class test; a change extends the existing test), **red for the right
   reason** (absent behavior: `ImportError` for a new class / assertion for a modified one, never a typo or
   `NotImplementedError`), implement to green without over-building — is in `references/build-contract.md`.
2. **VERIFY (read-only, ∥ one turn).** Dispatch the auditors **in parallel, single turn**, on the diff — narrow
   structural checks. **Scale to the change:** the trio is the ceiling, not a mandate — a trivial single-file copy may
   run only `copied-from-checker`; a new public component runs all three.
   - `test-surface-auditor` — base class + scaffold correctness (weighted; the test was written first).
   - `convention-auditor` — structure / registration / style vs the library's `.ai/` conventions.
   - `copied-from-checker` — `# Copied from` drift (`python utils/check_copies.py`, check mode).

   The glob-fired `diffusers-*.mdc` rules also attach by path. No adversarial pass here (`diffu-review` carries the next).
3. **FAN-IN → fix.** Synthesize the findings into de-duplicated gists; hand back to `implementer` to apply. Re-run the
   fast slice — still green.
4. **VALIDATE (nested test-coverage, ∥).** Before the gate, dispatch **composer agents running the `test-coverage`
   skill** (read-only, use `composer-2.5[]`, each its own context) against the diff — validating the test surface is
   *real*, not just present: right base class, fast coverage of the public surface, no fast test mis-marked `@slow`, no
   weakened assertions. Fold the coverage gaps back to `implementer` to fill, then re-run the slice. This shifts the QA
   rubric left so the handoff is already coverage-clean; the downstream `test-coverage` stage confirms rather than
   first-discovers.
5. **GATE (deterministic, pre-PR).** `implementer` runs `make style && make fix-copies && make quality &&
   make repo-consistency`, then `python utils/check_copies.py`, then the changed-path `pytest` slice — green before it
   returns (`references/build-contract.md`).
6. **COMMIT (incremental).** As each logical unit lands green and clean, `implementer` commits it in the worktree —
   conventional message, **no attribution footer**, never `WIP`/partial; one commit per unit. `ship` owns push + PR
   (`references/build-contract.md`).
7. **Output** — a committed, passing diff that ticks the plan's acceptance criteria → hand to `test-coverage`.

**Blocked / open-research terminal** (the fix genuinely cannot land): revert, keep a reproducer, document root cause,
record `--status blocked` → `ship` emits NO-GO. Never fake green / `@skip` / tautologize. Full steps:
`references/terminals.md`.
