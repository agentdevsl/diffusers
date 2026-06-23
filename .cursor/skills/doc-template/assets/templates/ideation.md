---
date: YYYY-MM-DD
status: <draft | active | superseded>
topic: <kebab-case-topic>
focus: <focus hint — omit the key when open-ended>
mode: <repo-grounded | elsewhere-software | elsewhere-non-software>
---

# Ideation: `<Title>`

[Authoring notes — delete this block before saving.
This is a human-facing DISCOVERY doc: a ranked, critiqued candidate set + the grounding the candidates were
qualified against + a record of what was cut. Keep it about the ideas and their basis — not a requirements doc or
plan, not implementation.
IDEA CARDS stay expanded — they are meant to be read in full to choose a direction; never hide their substance
behind default-closed `<details>`. When Ranked Ideas runs long (typically 5-7 cards), add a within-section jump-list
of the ranked titles (anchor links to each card) at the top of the section.
ILLUSTRATIVE VISUALS (altitude for this doc; the universal diagram rule is in the skill's authoring rules) — decide
per survivor on the idea's SHAPE, not on how clear the prose reads. "The prose is already clear" is never the reason
to skip: as a text-native reasoner you over-trust your own prose and under-produce useful visuals. Lean toward one
when the idea HINGES ON a relationship, flow, before/after contrast, structural arrangement, analogy (esp.
cross-domain), or a quantitative comparison; a point with nothing structural (rename, copy change, dep swap) gets
none — decoration is slop. Stay at the idea's altitude: illustrative & directional, NOT a spec — detailed
architecture, sequence diagrams, and wireframes belong downstream in brainstorm/plan once a direction is chosen.]

## Grounding Context
[The Phase 1 grounding the ideas were qualified against — label it "Codebase Context" in repo mode, "Topic Context" in elsewhere mode.]

## Topic Axes
[The 3-5 axes from Phase 1.5, one per line. When Phase 1.5 was skipped, a single line records why: `Decomposition skipped — atomic subject` or `Decomposition skipped — surprise-me mode`. Omit the section entirely when not applicable.]

## Ranked Ideas

[The surviving candidates, ranked, each card read in full. Add a jump-list of the ranked titles here when the section runs long.]

### 1. `<Idea Title>`
**Description:** [Concrete explanation]
**Axis:** [Topic axis this idea targets — omit when decomposition was skipped]
**Basis:** [`direct:` quoted evidence / `external:` named prior art / `reasoned:` written-out first-principles argument]
**Rationale:** [How the basis connects to the move's significance]
**Downsides:** [Tradeoffs or costs]
**Confidence:** [0-100%]
**Complexity:** [Low / Medium / High]

## Rejection Summary

[Considered-and-cut ideas, one line each. When an axis ended with zero survivors despite recovery, append it as its own row (the `| - | axis: … |` form below) so the coverage gap is visible rather than silently absent.]

| # | Idea | Reason Rejected |
|---|------|-----------------|
| 1 | `<Idea>` | `<Reason rejected>` |
| - | axis: `<name>` | recovery skipped (cap reached) — no survivors on this axis |
