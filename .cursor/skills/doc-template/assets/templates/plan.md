---
title: <verbatim plan title — matches the H1 below>
type: <feat | fix | refactor | chore | docs | perf | test>
date: YYYY-MM-DD
status: <draft | active | superseded>
---

# `<Title>`

[Authoring notes — delete this block before saving.
WARRANTED? Bias toward writing a plan. Skip the doc only when ALL hold: work is atomic (one commit, no unit
boundaries), no Key Technical Decisions, no scope worth pinning, no upstream artifact needing traceability. Stress-
test "looks atomic" — caching/migration/rate-limiting hide KTDs (write the plan); typo/rename/dep-bump don't (skip).
FLOOR (always): Summary · Problem Frame · Requirements · Key Technical Decisions · Implementation Units. Everything
else is include-when — present only when it carries content not covered elsewhere; padding is worse than omitting.
AGENCY: the catalog is a floor, not a ceiling — add a section when content fits none; merge Problem Frame into
Summary for compact plans; group Requirements/KTDs/Units when they span distinct concerns.
IDs: plain prefixes `R1.` `U1.` `A1.` `F1.` `AE1.` (not bold); stable across revisions, never renumber; IDs stay
continuous across groups. Repo-relative paths only, never absolute.
PROSE: one idea per sentence; a requirement/unit is one sentence of intent + at most one qualifier (send forks to
Open Questions); cut hedges; resolve in place (rewrite/remove superseded text — no strikethrough or "resolutions"
layer). No process exhaust.
OPTIONAL frontmatter, add when applicable: `origin:` (repo-relative path to the upstream brainstorm requirements
doc), `execution: knowledge-work` (non-code deliverable; default is code), `deepened:` (set by tooling).
VISUALIZATIONS (altitude for this doc; universal diagram rule is in the skill's authoring rules) live in High-Level
Technical Design or embedded next to the unit/KTD they illustrate — architecture topology, sequence, state machines.]

## Summary
[What the plan proposes, in 1-3 lines. Forward-looking; orients the reader before they invest in detail.]

## Problem Frame
[Why the work is being done — backward-looking / situational. May merge into Summary when the motivation is one sentence.]

## Requirements
[What must be true after the work ships — the reviewer's checklist; code review verifies against these. Group by concern when they span distinct logical areas.]
R1. [requirement]

## Key Technical Decisions
[The load-bearing choices that constrain implementation — `<decision>: <rationale>`. Without these the implementer can't tell which choices are open vs pinned.]
- **`<decision>`:** `<rationale>`

## High-Level Technical Design
[Include when the approach has shape prose alone doesn't carry — architecture across components, sequencing across processes, state machines, branching gates. Diagrams (component topology, sequence, swim lane, flowchart, data-flow) typically live here. Skip when the approach is a one-paragraph pattern application.]

## Implementation Units
[The discrete, independently landable units of work — what `implement` executes. May collapse into Summary prose for a trivial single-step plan (rare).]
U1. [unit of work]

## Scope Boundaries
[Include when scope is contested or tempting non-goals are worth naming; split "Deferred for later" vs "Outside the product's identity" when useful. Skip when scope is obvious from Requirements.]

## System-Wide Impact
[Include when the change touches cross-cutting concerns — data lifecycles, auth boundaries, performance posture, cardinal rules, shared infrastructure. Skip for localized changes with self-evident impact.]

## Risks & Dependencies
[Include when there are real risks (external service changes, version pins under churn, behavioral assumptions) or material upstream dependencies. Skip for low-risk localized work.]

## Acceptance Examples
[Include when any requirement is state-dependent or conditional ("When X, Y") where prose alone leaves edge-case ambiguity. Skip when all requirements are unconditional.]
AE1. (`R<n>`) When [condition], [expected outcome].

## Documentation / Operational Notes
[Include when documentation, monitoring, runbooks, or rollout steps need explicit notes. Skip for purely internal work on existing operational scaffolding.]

## Open Questions
[Include when there are genuinely unresolved items that block planning or implementation. Skip when complete — an empty "none" signals false uncertainty.]

## Sources / Research
[Research that orients the implementer or justifies load-bearing choices — code locations (`path/file.ts:174-176`), external docs, RFCs, constraints, prior plans (inclusive, not enumerated). Test: "would this breadcrumb help the implementer reading cold?" Omit process exhaust. May sit inline next to the KTD/unit it justifies instead.]
