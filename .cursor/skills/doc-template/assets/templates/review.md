---
date: YYYY-MM-DD
status: <draft | active | superseded>
topic: <kebab-case-topic>
verdict: <ready | needs-changes>
---

# Review: `<Title>`

[Authoring notes — delete this block before saving.
PURPOSE: diffu-review's durable severity-ranked report — the catch-before-PR record. Same rubric as the CI review
gate (native `/review` + the diffusers convention layer in `.ai/review-rules.md`), but over the **whole** diff
— review tests, docs, and scripts too, not just `src/diffusers/` + `.ai/` (CI scopes itself; you don't). ship/the PR
may carry it inline; same shape either way.
FLOOR (always): the Verdict line + Summary. A clean diff returns **READY** with empty finding buckets — never pad
findings to look thorough; precision over volume, downgrade or drop when uncertain. Cite the rule, review the whole
diff, and don't invent issues or flag pure style.
DE-DUP: collapse the native and convention lenses — the same issue is one finding, not two.
NOT HERE: a `CANDIDATE-RULE` (a convention caught but unwritten) is named for routing to `improve`, not a finding to
fix in this diff; no process exhaust.]

**Verdict:** <READY | NEEDS CHANGES> — [one line; e.g. "fix 2 blockers, then ship"].

## Blocking issues
[Numbered. Each: **title** → explanation → `file:line` → impact; cite the rule, e.g. *Per `.ai/models.md`: "…only
keep the inference path."* Must be fixed before submitting. Omit the section if none.]
1. **[title]** — [explanation]. `path:line` — impact: [one line]. Rule: [cited `.ai/` source].

## Non-blocking issues
[Same format, lower severity (warning / nit). Raise with the reviewer rather than guessing — don't fix speculatively.
Omit the section if none.]
1. **[title]** — [explanation]. `path:line` — impact: [one line]. Rule: [cited source].

## Dead code (advisory)
[Call-path traced, per `.ai/review-rules.md` §"Dead code analysis": params declared in `forward`/helpers but never
passed, private methods never called, layers initialised but never used in `forward`. Findings are *likely* dead, not
certain — the author may know a config/use case that exercises the path. Omit the section if none.]

| `path:line` | Likely-dead / Used | reason |
| --- | --- | --- |
| `path:line` | Likely-dead | [under the default config + the pipeline's call path, appears unreachable] |

## Gate
[Include when run — the deterministic checks diffu-review owns, which are also local commit/push gates:
`python utils/check_copies.py` (CHECK mode) + `make quality`, pass/fail. These run regardless of how the fan-out
scaled. A clean commit is no proof CI passed when the toolchain is absent — say so.]

## Candidate rules
[Include when a convention was caught but isn't written down — name it; it routes to `improve`, it is not fixed in
this diff.]

## Summary
[Short synthesis, then the action split:
- **Fix before submitting** — every blocking issue, and remove the flagged dead code.
- **Leave for the actual review** — non-blocking issues that aren't obviously correct; raise them with the reviewer
  rather than guessing now.]
