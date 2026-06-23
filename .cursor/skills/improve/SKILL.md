---
name: improve
description: The flywheel (team / closer). Codify a caught lesson into a docs/solutions/ learning + a thin .cursor/rules/*.mdc pack rule (improve is the only approved writer of that path) and an upstream .ai/ proposal. Use when any stage surfaces a CANDIDATE-RULE, or on a human correction.
---

# improve (team — the closer / the flywheel)

Thin orchestrator (orchestrate in root; agents execute): dispatch,
synthesise, delegate the write. This is the self-improving step: a lesson caught in review becomes a sharper
convention the next cycle reads. `improve` is the **only** approved writer to `.cursor/rules/*.mdc`, and it
keeps **no parallel convention doc** — authority is the library's own `.ai/` guides + source; cite, never
restate.

The loop for this stage — **INPUT → COVERAGE+ROUTING → ADVERSARIAL → GATE → OUTPUT**:

1. **INPUT** — take a `CANDIDATE-RULE` from any stage that emits one (`ideate` / `test-coverage` / `diffu-review` /
   `ship` / `workflow-test-e2e`) — a convention caught but not written down — or a human correction, plus the
   report it rode in on.
2. **COVERAGE + ROUTING (root, before dispatch)** — decide in root whether a rule should exist and where it lives;
   this is the skill's call, not an agent's:
   - **Already covered?** `grep -ri <topic> .cursor/rules .ai` (the library's `.ai/` at the target checkout +
     existing `.cursor/rules/*.mdc`; in this vendored dev repo rebase `.ai/` as `vendor/diffusers/.ai`). If covered,
     **stop** — a no-op is a success, it prevents a duplicate rule.
   - **Where does it belong?** A library-owned convention/gotcha → propose it **upstream `.ai/`** (a diffusers PR)
     *and* a **temporary** thin pack rule as its durable home until the PR lands. A pack-specific enforcement nuance
     (a glob, a subsystem `.ai/` doesn't cover) → a thin pack rule only.
   - **Draft the proposed rule spec** in root from the routed lesson (type · glob · `.ai/` source pointer · the
     falsifiable invariant + the check it ties to) — steps 3–4 and `doc-author` run on this draft.
3. **ADVERSARIAL — dispatch `adversarial-reviewer`** (its own context window). It attacks the proposed rule from
   *outside* the authoring checklist (second-order effects): will a low-value rule train the team to ignore rules?
   will the "temporary" pack rule be permanent because the upstream PR never lands? It need not re-run the authoring
   mechanics ("restates `.ai/`? / glob too broad? / falsifiable?") — `doc-author` enforces those when it writes.
4. **GATE (adversarial + human)** — you own the breaker's verdict: clear or log each blocker before proceeding; a
   survived blocker stops the write. Then **present the proposed rule spec** (type · glob · `.ai/` source pointer ·
   the falsifiable invariant + the check it ties to) **and ask the human to approve** via the blocking question tool
   (single-select: approve / revise / reject — `ToolSearch select:AskUserQuestion` first if unloaded). No
   deterministic hook gate here; never silently commit — the team owns the rulebook.
5. **OUTPUT** — on approval, delegate the write to the **single** execution agent `doc-author`, which — holding the
   draft to the **`doc-template`** skill's authoring rules and **verifying the invariant TRUE against the live tree first** — writes the
   `docs/solutions/<category>/<slug>.md` learning (metadata frontmatter: category · tags · problem-type; date in the
   entry, not the filename) **and** the thin `.cursor/rules/<name>.mdc` pack rule, and stages the proposed upstream
   `.ai/` addition (a diffusers PR). Agents return findings only; only `doc-author` writes — never two writers at
   once. Note in the summary that next cycle's `plan` / `implement` / `diffu-review` now load the rule.
