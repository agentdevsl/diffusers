---
date: YYYY-MM-DD
status: <draft | active | superseded>
topic: <kebab-case-topic>
focus: <optional focus hint>
mode: <repo-grounded | elsewhere-software | elsewhere-non-software>
---

# Requirements: `<Title>`

[Authoring notes — delete this block before saving. Decide per brainstorm whether each section below carries
information not covered elsewhere; filling a section with placeholder prose is worse than omitting it. **Summary**
and **Requirements** are the hard floor (always present); every other section is include-when.
Visualizations (altitude for this doc; the universal diagram rule is in the skill's authoring rules): conceptual
diagrams next to the Key Decision / Requirements group / Flow they illustrate — data-shape before/after,
source-of-truth fan-out, state/lifecycle, multi-step flow, quantitative comparison. This conceptual-diagram
affordance is distinct from a wireframe (wireframes are for visual-product UI only).]

## Summary
[The proposal / remedy in a few sentences — what to build and the change it makes. Always present; other sections reference it and must not restate it.]

## Problem Frame
[Include when motivation isn't obvious from Summary alone (the why needs paragraphs, not a sentence). Backward-looking / situational. Does NOT restate the proposal — the remedy lives in Summary.]

## Key Decisions
[Include when the brainstorm produced opinionated framing choices (defaults, scope narrowings, foundational technical picks) that constrain Requirements / Flows / Scope below. One entry per decision, naming it in **bold** with prose rationale. Sits high so readers meet the framing before descending into detail.]

## Actors
[Include when the proposed thing has multi-party behavior (multiple humans, agents, or systems meaningfully involved). Skip for non-behavioral briefs — naming briefs, data-shape briefs, pure research, decision frameworks.]

## Key Flows
[Include when the proposed thing has multi-step behavior — expected by default for behavioral brainstorms. Omit only when genuinely non-flow-shaped (pure API surface, policy, artifact output) AND Actors / Requirements / Scope Boundaries / Acceptance Examples together prevent downstream invention of paths; when omitting from a behavioral brainstorm, note the reason here.]

## Requirements
[The load-bearing, IDed prose — one requirement per entry with a stable `R-<n>` id; complete and standalone. Use `U-<n>` for assumptions/unknowns referenced downstream.]
- **R-1** — [requirement]

## Change-Impact
[Include for code changes (this pack's existing brainstorm contract). Blast radius: files touched, public API, `# Copied from` sources affected/invalidated, docs/test surface. If a public signature/behavior of shipped code changes, call out `deprecate()` / back-compat. This map is the contract `plan` and `diffu-review` hold the diff to.]

## Acceptance Examples
[Include when any requirement is state-dependent or conditional ("When X, Y") where prose alone leaves edge-case ambiguity — always cover the behavioral-conditional requirements, where ambiguity bites hardest. Skip when every requirement is unconditional and unambiguous.]
- **AE-1 (`R-<n>`):** When [condition], [expected outcome].

## Success Criteria
[Include when there are quality / metric / handoff signals Requirements don't already carry — quantitative ("p95 latency under 200ms"), qualitative ("the output reads as one voice"), or process/handoff ("the planner/review can act on this without follow-ups"). Skip when Requirements ARE the success criteria.]

## Scope Boundaries
[Include when scope is contested or tempting non-goals are worth naming. For product-positioning brainstorms, split into "Deferred for later" (eventually, not v1) and "Outside this product's identity" (positioning decision). Otherwise a single non-goals list is fine.]

## Dependencies / Assumptions
[Include when material upstream dependencies exist or load-bearing assumptions need surfacing.]

## Outstanding Questions
[Include when there are unresolved items. Distinguish "Resolve Before Planning" (blocks planning) from "Deferred to Planning" (answered during planning or codebase exploration).]

## Sources / Research
[Surface research that orients the planner or justifies framing choices — code locations, external docs, RFCs, constraints, prior plans (the category is inclusive, not enumerated). Test: "if I were the planner reading this cold, would this breadcrumb help me make better choices?" Omit process exhaust (reading the prompt, glancing at obvious files).]
