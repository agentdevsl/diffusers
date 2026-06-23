---
title: <verbatim release title — matches the H1 below>
type: <feat | fix | refactor | chore | docs | perf | test>
date: YYYY-MM-DD
status: <draft | active | superseded>
verdict: <go | no-go | conditional>
---

# `<Title>`

[Authoring notes — delete this block before saving.
PURPOSE: the `ship` stage's durable record — the go/no-go and the gate evidence behind it. ship emits this same
shape either as `docs/releases/<date>-<topic>.md` or as the PR body; write it once, reuse it for both.
FLOOR (always): the Verdict line · Summary · Gates. Everything else is include-when — omit a section entirely when
it has no substance; padding is worse than omitting.
VERDICT (set the frontmatter key AND lead the body with it): **go** = clean bill. **conditional** = every *hard*
gate green but a *non-blocking* follow-up is owed — name it under Caveats. **no-go** = a hard gate failed — name it
under Blockers. The hard no-go triggers, any one of which forces no-go: an out-of-boundary file; a failing CI gate
(`check_code_quality` / `check_repository_consistency` / any triggered `run_fast_tests` slice); an acceptance
criterion not met; or a `blocked` terminal from `implement`. Keep the caveat tight, e.g. "GO (host pytest blocked
by missing torch — green in CI)".
GATES are the heart of the doc — one row per gate the diff triggers, its result, and the local mirror command (from
`ci-gate-mapper`). The `make quality` commit hook fails open, so a clean commit is NOT evidence — cite the mapper's
verdict, never the hook.
NOT HERE: no `CANDIDATE-RULE` (it routes backward to `improve`, never into release notes); no attribution footer in
the PR body; no process exhaust.]

**Verdict:** <GO | NO-GO | CONDITIONAL> — [one line: the basis, plus any caveat in parentheses].

## Summary
[What shipped, in 1-3 lines — the change and the problem it closes. Forward-looking.]

## Gates
[The releasability evidence — one row per gate the diff triggers, with its result and the local command that
mirrors the CI job. Take results from `ci-gate-mapper`, not from a clean commit (the hook fails open).]
| Gate | Result | Local mirror |
| --- | --- | --- |
| `check_code_quality` | pass / fail | `make quality` |
| `check_repository_consistency` | pass / fail / n/a | `python utils/check_copies.py` … |
| `run_fast_tests` (<slice>) | pass / fail | `pytest <dirs> -k <filter>` |
| boundary | in-boundary / out-of-boundary | — |
| acceptance criteria | N/N met | — |

## Files changed
[Include when more than a couple of files move, or when a reader scanning the PR benefits. Repo-relative paths; the
dual-`__init__` registration and the new test file count. Skip for a one-file change named in the Summary.]

## PR
[Include on a go or conditional. Target: the **fork** (`gh pr create --repo <fork>`), never upstream. Paste the PR
URL once opened. No attribution footer in the body.]

## Caveats / Follow-ups
[Include on a **conditional** verdict — the non-blocking item owed (a doc page that can land later, a known
env-only gap). One line each; say what's owed and why it doesn't block the merge.]

## Blockers
[Include on a **no-go** — the failing gate, the unmet criterion, or the `blocked` terminal's reproducer + root
cause. State plainly what must turn green before this can ship; don't soften a red into a pass.]

## Validation
[Include when a real run was made — the gate commands exercised and what they returned. If a gate couldn't run
locally (no torch, paid service, hardware), say so plainly here rather than implying it passed.]
