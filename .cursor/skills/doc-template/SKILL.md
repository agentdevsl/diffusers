---
name: doc-template
description: Canonical Markdown templates for the pack's SDLC artifacts (ideation, brainstorm/requirements, plan, review, release). Use whenever a stage or doc-author is about to write a durable docs/ markdown artifact — fill the template's slots instead of inventing structure, so artifacts stay consistent, lean, and fast to produce. Reach for this any time you author a docs/ doc, even if the stage skill doesn't name it.
---

# doc-template — fill the template, don't reinvent the doc

Each durable SDLC artifact has **one canonical Markdown template** under `assets/templates/`. The writer
(`doc-author`, or whichever stage emits the doc) reads the template for the stage, fills every slot from the
already-decided synthesis, and writes it to the conventional `docs/` path. One fixed shape per artifact means
output is cheaper to produce (no structure invented per run), cheaper to read, and trivial to diff across runs.

This skill owns the **shape**; the stage skill owns the **content decision** and `doc-author` owns the **path** and
the write. It does not re-run analysis.

## How to use

1. **Pick the template** for the stage from the catalog below.
2. **Read it** (`.cursor/skills/doc-template/assets/templates/<name>.md`) and copy its skeleton.
3. **Fill every slot** from the synthesis. Each template self-documents: the `[bracketed]` / `<angle>` hints say
   what goes there and, for optional sections, *when to include them* — replace the hints, never echo them.
4. **Omit, don't pad.** A section whose hint says "include when …" comes out *entirely* when there's no
   substance — an empty or padded section is worse than a missing one.
5. **Write** to the stage's conventional `docs/` path (see `doc-author`), creating parent dirs as needed.

## Template catalog

| Stage | Template | Output path |
| --- | --- | --- |
| ideate | `assets/templates/ideation.md` | `docs/ideation/<date>-<topic>.md` |
| brainstorm | `assets/templates/brainstorm.md` | `docs/brainstorms/<date>-<topic>-requirements.md` |
| plan | `assets/templates/plan.md` | `docs/plans/<date>-NNN-<type>-<topic>-plan.md` |
| diffu-review | `assets/templates/review.md` | `docs/reviews/<date>-<topic>.md` (or PR inline) |
| ship | `assets/templates/release.md` | `docs/releases/<date>-<topic>.md` (or the PR body) |
| improve | `assets/templates/solutions.md` | `docs/solutions/<category>/<slug>.md` |

Every doc-emitting stage now has a template. Adding a new artifact type means adding its template here and one
catalog row — same slot-filling pattern.

## Authoring rules (every template)

- **Fill from the synthesis; cite, don't restate.** One idea per sentence; point at the real `file:line` / source
  rather than re-explaining it. Keep stable IDs (`R-`/`U-`) where downstream stages reference them.
- **Visualizations are cross-cutting, not a section.** Place a diagram next to the concept it illustrates; add one
  only when it lets a reader grasp a load-bearing concept faster than prose — one per concept, never for ceremony.
  It complements prose, never replaces it: the IDed prose stays complete and standalone, so a reader who ignores
  every diagram still gets the full content. Each template adds only its stage-specific diagram cue — the altitude
  they sit at, and when one earns its place.
- **Honour the frontmatter.** Fill every key; `date` is `YYYY-MM-DD`, `topic` is kebab-case. Every template carries
  a `status`: `draft` while it's being written, `active` once it's the current artifact for its topic, `superseded`
  when a newer doc replaces it (e.g. an `r2-` re-run, or a later plan for the same work). `status` is the doc's
  lifecycle and is independent of any verdict the doc records (a release can be `verdict: no-go, status: active`).
- **No process exhaust.** Keep engineering-process metadata out of the artifact — no "captured at Phase X" notes,
  no skill-pointer "next steps", no italic provenance lines. The reader wants the content and its basis. (The one
  provenance element that belongs is the single visible composition-signal footer in an HTML render, per the
  html-rendering invariant.)
