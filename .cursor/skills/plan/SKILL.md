---
name: plan
description: Engineering design step. Turn a spec into an explicit, convention-conformant feature design so implement just executes it. The highest-leverage gate on a convention-heavy library. Use after brainstorm, before implement.
---

# plan (Engineer — the design stage)

Thin orchestrator (orchestrate in root; agents execute): dispatch agents, synthesise the design, own
the adversarial verdict, delegate the write. A wrong design propagates through `implement` + `test-coverage`
before any code runs, so the breaker runs **here, before code exists**. Ground every agent in the convention
authority **by role** (governing `.ai/` doc + glob-fired `diffusers-*.mdc` rule + source exemplar) — cite,
never restate. **Every path the design cites must resolve** — the citing agent reads it first; a fabricated
exemplar or `# Copied from` source is a blocker, not a nit (the cheapest design defect to catch, the costliest to ship).

**Grounding root — resolve it once, read it locally.** The diffusers source is checked out locally (`src/diffusers/**`
under the repo root, or a vendored copy of it) — that on-disk tree is the source of truth. Discover the one canonical
root once (don't assume a literal) and reuse it for every agent. Agents read source from disk; they do **not**
round-trip to `raw.githubusercontent.com` or `gh api .../contents` for files that exist locally. Ignore the gitignored
`.worktrees/*` scratch checkouts and `evals/**` artifacts — a bare `**/src/diffusers/**` glob matches many copies and
stalls; resolve the canonical root, scope reads to it, and never re-fetch source an agent already located.

The five-step loop `INPUT → FAN-OUT → (ADVERSARIAL) → FAN-IN → OUTPUT`:

1. **INPUT** — read the requirements doc (`docs/brainstorms/*-requirements.md`) if one exists; else plan from
   the request. Classify the change (scheduler / pipeline / model) and resolve true scope forks via the
   interaction rules before spending research.
2. **FAN-OUT — dispatch three readonly agents in parallel** (∥), each in its own context window, each grounding
   in the resolved local root (see *Grounding root* — read from disk, never re-fetch from GitHub), each
   returning a structured finding (not a file dump):
   - `prior-art-scout` — the closest existing component to copy (path, base classes, `# Copied from` sources,
     registration footprint) + a redundancy verdict.
   - `convention-auditor` — grounds the design in the governing `.ai/` doc + `diffusers-*.mdc` rule; flags
     deprecated-API / anti-patterns to avoid.
   - `test-surface-auditor` — designs the **test plan co-equally** here: the correct **per-type** test base
     (schedulers → `SchedulerCommonTest`, models → `ModelTesterMixin`, pipelines → `PipelineTesterMixin`) —
     naming the wrong subsystem's base is the most common scaffolding error — plus named fast tests + what each
     asserts, behaviors / config / `prediction_type`s / edge cases to cover, the reference test file, any `@slow` owed.
3. **ADVERSARIAL** — hand the synthesised design to `adversarial-reviewer` (refute-unless-proven): wrong base
   class? a `# Copied from` source that doesn't exist? missed optional-dep gating? a simpler exemplar missed?
   a registration footprint that breaks imports? **You own the verdict** — adjudicate each objection against
   the cited `.ai/` source and clear its blockers (or log them as Risks).
4. **FAN-IN** — merge the agent gists, adjudicate conflicts against the cited `.ai/` source, and structure the
   cleared design into dependency-ordered **implementation units** with stable **U-IDs** (decisions, not code).
5. **OUTPUT — single-writer write.** Hand the synthesis + section contract to **one** `doc-author` (the only
   writer this stage dispatches) to write `docs/plans/<date>-NNN-<type>-<topic>-plan.md`. It emits the hard
   floor — led by the **redundancy / scope verdict** (does this already exist? the cheapest correct change is
   possibly *none* — `prior-art-scout`'s most valuable finding, so it leads the doc, not buried in Risks):
   reference component, files touched (incl. the dual-`__init__` registration + test file — spec in
   `diffusers-structure`), base classes / decorators / output dataclass, `# Copied from` sources, the co-equal
   **Test Plan**, and acceptance criteria with U-IDs including the test-coverage bar (fast test green · correct
   base class · no-tests-no-merge). Agents return findings only — never two writers at once.

**Gate:** none here (no code yet) — the step-3 adversarial pass is the only breaker. **Output → `implement`.**
