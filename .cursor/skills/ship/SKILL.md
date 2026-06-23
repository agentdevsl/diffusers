---
name: ship
description: DevOps step. Answer "will this pass CI, is it in-boundary, and is it releasable?" — fan out parallel readonly agents (CI gates incl. unit tests, blast radius + boundary, conventions), gate a go/no-go, then have doc-author write release notes and, on a go, open the PR. Use before/at PR time.
---

# ship (DevOps)

Thin orchestrator (orchestrate in root; agents execute): dispatch, synthesise a go/no-go, delegate the write, and
on a go own the push + PR. Five-beat loop: INPUT → FAN-OUT → FAN-IN → GATE → OUTPUT. No adversarial gate here.

**Input.** The reviewed diff (+ the `diffu-review` report). Ground every agent in the library's own authority by
role — `.github/workflows/pr_tests.yml` + the `Makefile`, `.ai/AGENTS.md`, `.cursor/rules/diffusers-*.mdc` — cite,
never restate.

Steps:
1. **Fan out two readonly agents in parallel** (each its own context window):
   - `ci-gate-mapper` — the CI specialist: which jobs the changed paths trigger + the local mirror for each —
     `check_code_quality` (`make quality`), `check_repository_consistency` (the five checks quality skips), and the
     **`run_fast_tests` slice(s)** the change lands in (it's a matrix — Pipelines / Models & Schedulers / Examples;
     mirror the slice's exact bundled `pytest <dirs> -k <filter>` line, since a single-dir run can pass while the
     bundled slice fails). This agent owns the "do the unit tests pass?" answer.
   - `change-impact-mapper` — blast radius + the **boundary check**: does the diff stay inside the approved roots?
     flag any out-of-boundary file. (Leave CI mapping to `ci-gate-mapper` — don't duplicate it.)
2. **Fan in → verdict.** Merge into gists, de-duplicated, **blocking items first** (the deterministic checks ran in
   the agent contexts, never in root). **Any one of these is an unconditional no-go**, surfaced as a blocking item:
   - an out-of-boundary file;
   - a failing CI gate — `check_code_quality`, `check_repository_consistency`, or **any triggered `run_fast_tests`
     slice**. Read this from `ci-gate-mapper`, **not** from a clean commit: the commit hook fails open when the
     toolchain is absent (see Gate), so a clean commit is no proof CI's tests or quality passed;
   - acceptance criteria not fully met — stretch / open-research cases (real-issues eval with no known fix) ship
     **no-go even if `diffu-review` returned PASS**; document partial outcomes honestly, never emit GO unless every
     criterion is met;
   - a **blocked** terminal from `implement` (reverted fix + reproducer + `status: blocked`) — ships no-go with the
     reproducer + root cause documented, an honest blocked outcome, not a pass.

   Otherwise it's a **go** (a clean bill). Reserve **conditional** for the middle case — every *hard* gate is green
   (in-boundary, CI passing, all acceptance criteria met) but a *non-blocking* item is still owed — e.g. a follow-up
   that can land separately. A go or conditional opens the PR (conditional
   names the caveat in the release notes); only a no-go withholds it.
   Record the verdict via `finalize-result --ship-verdict go|no-go|conditional` (separate from case `--passed`).
3. **Single writer → `doc-author`** (the one execution agent): writes the **release notes** (PR body or
   `docs/releases/<date>-<topic>.md`) + the **go/no-go**. The others return findings only; never two writers at once.
4. **On a go or conditional, own the push + PR** (`implement` deliberately does not — ship does). Push the `implement` worktree
   branch to the **approved fork** (never the default branch, never upstream — `boundary-guard` enforces the fork
   target), then open the PR with doc-author's notes as the body via `gh pr create … --body-file <tmp>` (a piped or
   `$(…)` body can land empty while `gh` exits 0), **no attribution footer**. A no-go opens **no PR**: surface the
   blockers, record the verdict, never push past a failing gate. Exact commands + the fork/empty-body traps:
   `references/gates-and-pr.md`.

**Gate (the boundary).** The `make quality` commit/push hook **fails open** — it's skipped wherever the toolchain
is absent (the dev repo), so a clean commit/push is *no proof* CI passed. Take the `check_code_quality`,
`check_repository_consistency`, and `run_fast_tests` verdicts from `ci-gate-mapper`; CI stays the hard gate, never
bypass a failing one. Mechanics + the fail-open guard: `references/gates-and-pr.md`.
