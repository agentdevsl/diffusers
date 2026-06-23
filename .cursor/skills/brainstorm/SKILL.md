---
name: brainstorm
description: PM step. Turn one chosen idea into a scoped, convention-aware spec with a change-impact summary — collaborative dialogue, approaches, then a right-sized requirements doc the `plan` step designs against. Use once an option is picked and before design.
---

# brainstorm (PM)

**Current year: 2026** — use it when dating requirements docs.

Thin orchestrator — dispatch, run the dialogue, delegate the write. The skill writes no files and edits no code.
You answer **WHAT** to build: read the chosen option from `ideate`, resolve product-level decisions through
dialogue, and hand a scoped requirements spec to `plan` (which answers HOW). "In-convention" = the real
`src/diffusers/**` source plus the subsystem's governing convention doc — cite a real `file:line` / doc section,
never restate.

**Grounding root — resolve it once, read it locally.** The diffusers source is checked out locally (`src/diffusers/**`
under the repo root, or a vendored copy of it) — that on-disk tree is the source of truth. Discover the one canonical
root once (don't assume a literal) and reuse it for every read and every fan-out agent. Read source from disk; do
**not** round-trip to `raw.githubusercontent.com` or `gh api .../contents` for files that exist locally. Ignore the
gitignored `.worktrees/*` scratch checkouts and `evals/**` artifacts — a bare `**/src/diffusers/**` glob matches many
copies and stalls; resolve the canonical root, scope reads to it, and never re-fetch source an agent already located.

Chain: **ideate** (which ideas?) → **brainstorm** (what does the chosen one mean? — here) → **plan** (how is it
built?). Designing or editing code is `plan` / `implement`, not here.

**Work in one pass, not a sequence of gates.** The stages below are *lenses you apply while the readonly agents
run* — not phases to narrate one at a time. Fire the whole fan-out in the **same turn** you open the dialogue (the
agents work during the user's think-time), fold scope, rigor, and approaches into that one exchange, and stop the
moment the spec is clear. Every extra serial turn is wall-time; every re-read is tokens; **ask nothing when the
option is already clear.**

## Principles

1. **Resolve product decisions here, defer design to plan** — user-facing behavior, scope, and success criteria are
   yours; schemas, file layouts, and code-level design are `plan`'s. Carrying architecture forward forces the user
   to decide it on shallow research.
2. **Thinking partner, not interviewer** — bring alternatives, challenge the framing, name the non-obvious
   consequence.
3. **Right-size the artifact** — match ceremony to scope; padding a section is worse than omitting it.
4. **Adjudicate to one resolved view** — fan-in the gists, de-dup, and settle every conflict against the real
   source; the spec carries one position, not a menu.

**Interaction.** `AskUserQuestion` (`ToolSearch select:AskUserQuestion` if unloaded), **one question per turn** —
stacking dilutes answers. Single-select menu to choose a direction; drop the menu and go **open-ended** when the
answer is narrative or you can't write 3–4 distinct, plausible options without padding. Options come from what the
agents and dialogue surfaced, never a template.

**Model tiers** — retrieval / extraction (change-impact, prior-art, convention) → the cheap fast model
(`composer-2.5[]`); dialogue / synthesis / arbitration → inherit the orchestrator's model.

## Input + scope

Read the chosen option (argument, or the `ideate` options doc) and classify scope in a line, matching ceremony to
it: **Lightweight** (small, bounded, low-ambiguity — minimal probing, short doc or none) · **Standard** (a real
feature or bounded refactor with decisions to make) · **Deep** (cross-cutting, strategic, or ambiguous — the change
shape itself is in question). Resuming an existing `docs/brainstorms/<date>-<topic>-requirements.md` → have an agent
update it **in place**, not a duplicate. A trivial, already-decided alignment that flows straight to `plan` needs
**no doc** — say so and stop.

## Fan-out (∥, same turn the dialogue opens)

Dispatch the readonly agents in one turn (concurrent composer agents, each its own context, gist-only). Each grounds
in the resolved local root (see *Grounding root*) — read from disk, never re-fetch from GitHub. Scale to
the input — the trio is the ceiling, not a mandate. **A conditional run picks the agent the open question calls
for**, not a fixed default: an open-shaped option ("should we even build this?") leads with `prior-art-scout`; a
settled shape with prior art known runs `change-impact-mapper` for the most signal; a deeply ambiguous option (the
shape itself in question) makes a blast-radius map premature — favor the dialogue + `prior-art-scout` /
`convention-auditor` and defer the impact map until the shape lands.

- `prior-art-scout` — the closest existing component to copy (named for `plan`) + a redundancy verdict; on a
  copy-paste-first library the cheapest spec is often "this exists, adapt *that*" — so on an open-shaped option this
  leads.
- `change-impact-mapper` — blast radius: files touched (incl. the dual-`__init__` registration + test file for a
  new scheduler), public API, `# Copied from` sources affected/invalidated, docs/test surface. A change to a
  **public signature/behavior of shipped code** MUST call out `deprecate()` / back-compat (cite
  `deprecation_utils.py`). Highest-signal once a concrete shape exists; this map is the contract `plan` and
  `diffu-review` hold the diff to.
- `convention-auditor` — which subsystem convention doc governs the change, so "in-convention" is named.

## Dialogue (root) — pressure-test as you go

Ask what the user thinks first; probe the gaps the agents surface, broad → narrow; keep requirements concrete
enough that `plan` won't invent behavior. As you go, apply these rigor lenses and raise **only the ones that
actually bite** — folded into the conversation, never fired as a checklist:

- **Evidence** — is the need observable (an issue, a real failure, a workaround) or merely asserted? ·
  **Specificity** — is the beneficiary concrete enough to design for? · **Counterfactual** — what do users do today,
  and what breaks if nothing ships? · **Attachment** (raise last) — is a solution *shape* being treated as the goal
  instead of the value it delivers? Ask for the smallest version that still delivers.

Before exiting, run one **integration check**: combine what the user has said and surface any non-obvious
downstream consequence the one-at-a-time dialogue missed. Exit when the idea is clear and no integration question is
pending, or the user wants to proceed.

## Approaches (when the mechanism is still open)

When more than one plausible direction survives the research — **especially for Deep scope, where the change shape
itself was the question** — give **2–3 concrete approaches before recommending**; the menu is how the user
adjudicates an open mechanism, so don't collapse it to a single winner just because grounding surfaced a
front-runner. Each: a 2–3 sentence description + pros/cons + key risk + when it's best. Use at least one non-obvious
angle (inversion, constraint-removal, analogy); if an option would appear in a generic listicle for this problem,
sharpen it against the agent gists or drop it. State the recommendation **after**, with why. **Granularity is
mechanism / shape** — "config arg vs subclass vs new component" and their trade-offs, never paths, column names, or
signatures (that's `plan`). Flag **reuse / extend / build-new**. Skip the menu only when one path is genuinely
unambiguous (e.g. prior-art dictates a single in-convention shape) — not merely because you have a lean.

## Synthesis gate → output

Surface a scoping synthesis — the user's last chance to correct scope before the doc lands:

- **Path A** — Lightweight **and** no clarifying question fired: announce "what we're building" in 1–3 sentences,
  then write in the **same turn** — no confirmation wait.
- **Path B** — any clarifying question fired, **or** Standard/Deep: present goal · scope · non-goals · change-impact
  headline and **confirm** before writing.

Verify any checkable claim the doc will assert (absence claims, named files/config/deps) against the source first —
an unverified claim ships as a labeled assumption, not a fact.

Then hand the resolved goal/non-goals, agent gists, and decisions to **one** `doc-author` (the single writer; the
skill never writes). It uses the **`doc-template`** skill to select the **brainstorm** template
(`.cursor/skills/doc-template/assets/templates/brainstorm.md`) and fills its slots →
`docs/brainstorms/<date>-<topic>-requirements.md`. Hard floor **Summary** + **Requirements** (stable `R-`/`U-` IDs);
include-when-material Problem Frame · Key Decisions · Actors · Key Flows · Change-Impact · Acceptance Examples ·
Success Criteria · Scope Boundaries · Dependencies/Assumptions · Outstanding Questions · Sources. Omit, don't pad.
Confirm the written path (clickable), give a one-line summary, route to `plan`.
