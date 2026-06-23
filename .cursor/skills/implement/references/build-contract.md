# implement — build contract (TDD, gate, commits)

Loaded by `implement` step **WRITE / GATE / COMMIT**. The `implementer` agent owns the copy/adapt/registration
mechanics (`implementer.md`, `diffusers-structure`); this is the contract the orchestrator holds it to.

## TDD contract

- **Test first.**
  - *New component* — scaffold next to a neighbor with the base class the plan names (`SchedulerCommonTest` for
    schedulers; pipelines/models/LoRA use their own — the `.ai/` testing doc). Supply only the factory methods the
    contract needs so the inherited suite comes free.
  - *Change to existing code* — extend that component's existing test, not only a fresh file.
- **Red for the right reason.** The fast slice fails because the behavior is genuinely *absent* — a collection-time
  `ImportError` for a new/unregistered class, or a targeted assertion/`AttributeError` for a modified one — never a
  typo, `NotImplementedError`, or wiring slip. Never implement on a green or wrongly-red test.
- **Implement to green.** Adapt the exemplar until the slice passes; don't over-build past it. Stay within the plan's
  scope — adjacent cleanups are a follow-up, not this diff.

## Gate (deterministic, pre-PR)

`implementer` runs it and fixes what it flags before returning:

```bash
make style && make fix-copies && make quality && make repo-consistency
python utils/check_copies.py                   # standalone check-mode copied-from verify
pytest tests/<subsystem>/test_<name>.py -q      # fast slice, changed path only — must be green
```

## Incremental commits

As each logical unit lands green and gate-clean, `implementer` commits it **in the worktree**:

- Conventional message (`feat(schedulers): add FooScheduler`), **no attribution footer**.
- Never a `WIP`/partial commit — heuristic: commit only a message describing a complete, valuable change; if it'd
  read "WIP", keep building.
- One commit per logical unit (multi-unit plan → multiple commits; a single small change → one).
- Do **not** `git push` or open a PR — `ship` owns push + PR + release notes. The `make quality` commit hook is the
  gate (fails open in the dev repo).

Incremental commits checkpoint the build, keep a clean branch for `test-coverage`/`diffu-review`, and make the
blocked-terminal revert precise (drop the last commit).
