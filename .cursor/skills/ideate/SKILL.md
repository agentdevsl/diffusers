---
name: ideate
description: PM step. Turn a fuzzy request into a ranked, critiqued shortlist of grounded diffusers change options, then route the strongest into brainstorm. Use at the very start — "what should I improve", "give me options", "surprise me", "triage from github", "what would you change" — not for refining an idea you already hold.
---

# ideate (PM)

**Current year: 2026** — use it when dating ideation docs.

Thin orchestrator — dispatch, critique what agents return, delegate the write. The skill writes no files and edits
no code. You turn ambiguity into a **ranked, critiqued shortlist of change options** grounded in real
`src/diffusers/**`, then route the strongest into `brainstorm`. Ground every option in the actual source and
`PHILOSOPHY.md` (design fit) — cite a real `file:line`, never restate.

**Grounding root — resolve it once, read it locally.** The diffusers source is checked out locally (`src/diffusers/**`
under the repo root, or a vendored copy of it) — that on-disk tree is the source of truth. Resolve its path once at the
start of grounding (discover it — don't assume a literal) and reuse that same root for every read and every fan-out
agent. Read source from disk; do **not** round-trip to `raw.githubusercontent.com` or `gh api .../contents` for files
that exist locally — that is the single biggest walltime/token sink this skill hits. The repo may also hold duplicate
trees — gitignored `.worktrees/*` scratch checkouts and `evals/**` run artifacts — so a bare `**/src/diffusers/**`
glob can match many copies and stall; resolve the one canonical root, scope every grounding read to it, ignore the rest.

Position in the chain — don't skip ahead:

- **ideate** — what are the strongest ideas worth exploring?
- **brainstorm** — what exactly does one chosen idea mean?
- **plan** — how is it built? (designing or editing code is `plan` / `implement`, not here.)

## Core principles

1. **Ground before ideating** — scan the real tree first, from the local root (see *Grounding root*); no
   abstract advice detached from the code, and no network fetch for source already on disk.
2. **Generate many → critique all → keep survivors only** — the quality mechanism is *explicit rejection with
   reasons*, not optimistic ranking. The rejection IS the value.
3. **Route action into brainstorm** — ideate names promising directions; `brainstorm` defines the chosen one.
4. **Checkpoints, not interrogation** — pause at fixed gates so the user steers; never run a serial Q&A chain.

**Interactive by default.** Use `AskQuestion` (`ToolSearch select:AskUserQuestion` if unloaded) at each checkpoint
below — **one question per turn**, options derived from what you actually found (not a generic menu). Skip a
checkpoint only when the user's message already resolved it (see Fast path).

**Be fast behind the gates.** Batch independent reads and the whole fan-out in **one turn after the user confirms
direction** at the relevant checkpoint; scale agent count to the input; have agents cite narrow `file:line` sections,
never re-read (or re-fetch) whole files. Don't pre-ground the whole surface yourself *and* have the agents redo it —
pick one: either let the fan-out agents read the local root, or pass them the `file:line` ranges you already pulled.
Avoid re-running an identical glob/grep or re-fetching the same file by a second mechanism — resolve it once, reuse.

**Model tiers** — route by task shape, not a hardcoded name: **retrieval / extraction** (evidence, prior-art) → the
cheap fast model (`composer-2.5[]`); **synthesis / critique / arbitration** → inherit the orchestrator's model.
`go deep` raises the whole fleet to the ceiling tier.

Loop: `INPUT → [CP0] → DECOMPOSE → [CP2] → FAN-OUT → FAN-IN → GATE → [CP3] → OUTPUT`.

### Checkpoints

| Gate | When | Ask (one turn) |
|------|------|----------------|
| **CP0 — Mode** | Subject/mode not fully specified | How to focus: named subject · surprise me · triage GitHub · cancel |
| **CP1 — Issue pick** | GitHub triage mode only | Which issue(s) to ideate on (top-5 preview table + "Other issue #") |
| **CP2 — Axes** | After decomposition (3+ axes) | Confirm proposed axes · adjust one · skip axes (atomic) |
| **CP3 — Pre-write** | Survivors ranked, before `doc-author` | Write ideation doc · change recommendation · iterate depth · route to brainstorm (no doc) |

**Fast path** — skip CP0–CP2 when the user already named subject + scope (`/ideate schedulers top 3`,
`/ideate triage from github #8673`). Still run **CP3** unless they said `no doc`, `just options`, or
`route to brainstorm`.

## 1. INPUT — settle subject, scope, volume

- **Subject gate.** Is the subject identifiable? A prompt that names or plausibly names a feature/flow/component
  (`callback API`, `schedulers`) is identifiable — proceed. A catch-all quality (`improvements`, bare `bugs`,
  `quick wins`) is vague — run **CP0** (name a subject / "surprise me" / triage GitHub / cancel). More than 2
  questions in a row means ideate is the wrong workflow — suggest `brainstorm`.
- **Surprise-me mode.** The user delegates the focus; discover subjects from the grounding rather than carry a named
  one. Skip DECOMPOSE (cross-cutting synthesis covers the surface) and note it in the doc.
- **Triage from GitHub Issues** — focus on open feature not bug, two phases; do not fan-out until **CP1** resolves:
  1. **Preview (cheap)** — `gh issue list --repo huggingface/diffusers --state open --limit 40 --json number,title,labels,createdAt,comments,assignees,updatedAt` (metadata only; no bulk `body`). That one payload already has title/labels/comments/assignees — build the CP1 table straight from it; do **not** re-`gh issue view` each candidate. Apply the no-linked-PR filter with a single `gh search prs` over the shortlist, not a per-issue loop. Rank top 5 **open, unassigned, no linked PR, low comment noise**; show a compact table (# · title · labels · age). Run **CP1**.
  2. **Deep ideate** — fetch `body` only for the chosen issue(s); ground in the local root (*Grounding root*); continue the loop.
- **Ground** the request in the real source (*Grounding root*) in one batched **local** read (after CP0/CP1 lock direction).
- **Volume / depth.** Default ~8–12 candidates → top 3–5 survivors. Honour overrides: `top 3`, `100 ideas`,
  `raise the bar`, `go deep` (max depth — ceiling-tier fleet + a second critic). If not specified, include depth
  as an option in **CP0** or **CP3**.

## 2. DECOMPOSE — topic axes (skip when atomic or surprise-me)

Decompose the subject into **3–5 orthogonal axes** — *what aspects to think on* (frames are *how* to think). Without
explicit axes, parallel agents converge on the most salient reading and leave the rest of the surface unexamined.
Axes must be orthogonal, at the same level, derived from the grounding (not a generic template), and named in the
topic's language — e.g. for schedulers: step/output contract · timestep handling · config validation · numerical
edge cases. Fewer than 3 viable axes → the subject is atomic; skip and note `Decomposition skipped — atomic subject`
(or `— surprise-me mode`). When 3+ axes, present them and run **CP2** before fan-out.

## 3. FAN-OUT — grounding agents in parallel, in ONE message

Each runs in its own context, reads its authority *by role* from the resolved local root (see *Grounding root* — never
re-fetch from GitHub source the orchestrator already located), returns findings only, edits nothing. Scale to the
input: default the three below; one for a single-component focus; none when the authority is obvious. When axes
exist, pass each agent the axis list so coverage spans the whole surface.

Use concurrent composer agents

- `prior-art-scout` — closest component to copy + the **redundancy verdict** (already exists? adapt *that*).
- `philosophy-fit-reviewer` — score each surface against `PHILOSOPHY.md` (abstraction altitude, self-containment,
  copy-paste over abstraction, simple over magic).


## 4. FAN-IN — generate, critique, cut, rank

List the **full** candidate set first (≈8–12; aim for coverage across the axes), then critique **every** candidate
against the grounded findings. **Cut hard and record each cut in one line** — the rejection is the value. Rank the
survivors. Tag each survivor's **basis**: `direct:` (quoted code evidence) / `external:` (named prior art) /
`reasoned:` (first-principles argument). Tag any unwritten convention `CANDIDATE-RULE` for `improve`.

Present survivors in chat (ranked titles + one-line hook each + named recommendation) **before CP3**.

## 5. GATE

Keep only survivors anchored to a real `file:line` (not invented), past the philosophy bar, and not redundant with
prior art. When an axis ends with zero survivors, record the coverage gap rather than letting it vanish silently.

## 6. OUTPUT

Run **CP3** — user picks write doc / swap recommendation / go deeper on one survivor / route straight to `brainstorm`.

When writing: hand the ranked synthesis to **`doc-author`** (the single writer), which fills the ideation template
(`.cursor/skills/doc-template/assets/templates/ideation.md`) → `docs/ideation/<date>-<topic>.md`. The template owns
the shape — each survivor carries description · axis · basis · rationale · downsides · confidence · complexity;
**exactly one** is the recommendation with a one-line "why over others"; the Rejection Summary records every cut and
any zero-survivor axis. Keep process exhaust out of the artifact.

After the doc lands (or on a no-doc route): confirm the path when written, give a one-line summary, offer
**brainstorm** on the recommendation or **iterate** (re-run fan-in with `go deep` / different issue / new subject).
