---
name: diffu-review
description: QA/Engineering review step. Wrap Cursor's native /review for generic bugs/security and fan out parallel readonly convention agents, run an adversarial breaker, then merge into one severity-ranked report. Use before opening a PR.
---

# diffu-review (QA + Engineer)

Thin orchestrator (orchestrate in root; agents execute): dispatch, synthesise, delegate the write. You produce one
**severity-ranked** report combining native review with the diffusers convention layer — the catch-before-PR moment.
Ground every agent in the library's own rubric by **role** (`.ai/review-rules.md`, the `.ai/` subsystem docs, the
pack's `diffusers-*.mdc`) — cite, never restate.

The five-beat loop: **INPUT → FAN-OUT → (ADVERSARIAL) → FAN-IN → GATE → OUTPUT**.

1. **INPUT** — read the diff + the tests + the plan doc.
2. **FAN-OUT (∥)** — dispatch the convention agents concurrently, each readonly in its own context, findings only.
   **Scale to risk:** a substantive/high-risk change (public API, a new component, scheduler/pipeline internals) runs
   both + the breaker; a zero-risk diff (docstring, comment, pure rename) runs native `/review` only. The pair is
   a ceiling, not a mandate — but scaling trims the fan-out, **not** the GATE (step 5 always runs; a "trivial" edit can
   still drift `# Copied from`).
   - native Cursor `/review` — generic bugs + security (Bugbot); always on.
   - `convention-auditor` — the library's `.ai/` conventions + the pack's rules; deprecated-API / anti-patterns, and a
     breaking change to public/shipped behavior that owes `deprecate()` / back-compat.
   - `copied-from-checker` — runs `python utils/check_copies.py` (CHECK mode = the exact CI gate).
3. **ADVERSARIAL** — dispatch `adversarial-reviewer` as the breaker: *what did the agents miss? which "blocker" is
   actually a nit? where's the false positive?* The skill **owns the verdict** — it clears or logs each blocker the
   breaker raises before proceeding.
4. **FAN-IN** — merge into one list, ranked **blocker / warning / nit**, de-duplicated (the native and convention
   lenses overlap — collapse the same issue once; conflicts adjudicated against the `.ai/` source), each with
   **file:line + cited rule + one-line fix**. **Precision over volume:** downgrade or drop when uncertain — a clean
   diff returns a plain *no blockers*, never padded. Tag a convention caught but unwritten as `CANDIDATE-RULE` →
   feeds `improve`.
5. **GATE** — `python utils/check_copies.py` (the CI gate `make quality` misses) + `make quality`.
6. **OUTPUT** — hand the synthesis to **one** execution agent, `doc-author`, which writes the report to
   `docs/reviews/<date>-<topic>.md` (or PR inline). Agents report only; no code is edited here.
