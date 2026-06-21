# Bugbot review rules — diffusers

> **Authoritative source:** [`.ai/review-rules.md`](.ai/review-rules.md) — the exact correctness pass the
> `@claude` CI reviewer runs (it links onward to `AGENTS.md`, `models.md`, `pipelines.md`, `modular.md`, and
> `skills/model-integration/pitfalls.md`; Bugbot follows those too). The flag index below is a
> **non-authoritative resilience mirror**, not a second rulebook — if the two ever disagree,
> `.ai/review-rules.md` wins. This file is a *consumer* of the convention layer, never a writer to it.

Scope: **correctness, not style** — ruff / `make quality` own formatting, so do not flag style.
Bugbot is **non-blocking**; CI stays the hard gate.

## Blocking flags

- **Ephemeral context** — `# per reviewer comment on PR #NNNN`, `# as discussed in review`, `# TODO from
  offline chat`, debug printouts, and `scripts/` files importing the reference repo or hardcoding dev paths.
  State the *reason* so a comment stands alone, or drop it. (review-rules.md → "Common mistakes")
- **Defensive / dead code** — fallback paths, safety checks, or config options "just in case"; unused method
  parameters kept "for API consistency"; silently correcting user intent instead of raising a concise error.
  (`.ai/AGENTS.md` → coding style)
- **Unshipped-name aliases / unreleased-code deprecation shims** — no back-compat alias for a name that never
  shipped; no `deprecate()` tombstone for code that was never released. (`.ai/AGENTS.md`; `diffusers-deprecation.mdc`)
- **`# Copied from` drift** — edits inside a `# Copied from …` block without propagation. Do not edit a copied
  block directly. Deterministic check: `python utils/check_copies.py`.
- **Missing tests** — "no tests = no merge"; a new component must use the correct base class / factory
  (e.g. `PipelineTesterMixin`, `ModelTesterMixin`, `SchedulerCommonTest`). (the `test-surface-auditor`
  subagent; `diffusers-tests.mdc`)
- **Scheduler output dataclass** — scheduler PRs must honor the invariant in
  `.cursor/rules/diffusers-schedulers.mdc`: a scheduler's `step` returns a `BaseOutput` subclass (the shared
  `SchedulerOutput` or its own `*SchedulerOutput`).
  Check: `grep -L SchedulerOutput src/diffusers/schedulers/scheduling_*.py | grep -v _flax` → empty = all conform.

## Advisory / suggestions (not blocking — frame as suggestions)

- **Documentation impact** — new or changed public surface (a new pipeline/model, a new argument, changed
  defaults, a renamed API) that leaves `docs/`, docstrings, or examples stale. Flag as a suggestion.
  (review-rules.md → "Documentation impact" — advisory)
- **Dead-code trace (new models)** — trace the pipeline `__call__` → model under the **default** config, then
  flag *likely*-unreachable branches, unused `forward` params, and layers initialized but never used. Qualify
  every finding ("appears unreachable under the default config and the pipeline's call path — the author may
  know a config that exercises it"). (review-rules.md → "Dead code analysis" — advisory)

## Flywheel

Caught a convention that isn't written down yet? Tag it **`CANDIDATE-RULE`** so the `improve` skill proposes it
**upstream** to `.ai/` (and/or a thin pack rule) under human approval — the same documentation flywheel
`.ai/review-rules.md` runs. `@cursor remember <fact>` may be used as an **ephemeral** fast-path, but every
`remember` must also raise a `CANDIDATE-RULE`: Bugbot's learned-rule store is not a durable home, is not
governed by `improve`, and can be auto-disabled. Durable invariants belong in `.ai/` (and a thin pack rule),
not in a learned rule.
