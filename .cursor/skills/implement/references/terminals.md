# implement — pre-flight stops & terminals

Loaded by `implement` when a build either should not start or cannot finish. Three states distinct from a normal
pass — recognize the right one; don't thrash, and don't fake a green.

## Pre-flight stops (before any write)

- **Already-present.** If the plan's capability is already in the tree (named symbol/config exists, acceptance
  already holds), it shipped on a prior branch. Verify it matches the plan's intent, say so, and stop — don't
  reimplement a no-op. No writer is dispatched.
- **Plan-ambiguity.** If the plan is too underspecified to build (no named exemplar, untestable/contradictory
  acceptance), route back to `plan` — don't invent the design here. This is an unbuildable *spec*, distinct from the
  blocked terminal below (a *fix* that can't land).

## Blocked (the fix genuinely cannot land)

One sanctioned exit — do not thrash, fake green, `@skip`, or tautologize:

1. **Revert** the non-working attempt (worktree left free of a broken fix; with incremental commits, drop the last
   commit).
2. **Keep a reproducer** — a red test or `@unittest.expectedFailure` pinning the gap and what a correct fix must
   satisfy. The reproducer *is* the deliverable.
3. **Document** root cause + hypotheses tried (plan / review docs).
4. **Record `--status blocked`** (first-class, distinct from pass/fail). `ship` emits **NO-GO**. Never report
   `pass`, never stub over the reproducer.
