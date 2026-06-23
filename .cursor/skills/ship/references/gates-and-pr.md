# ship — gate + PR mechanics

Detail behind two lines in `SKILL.md`. Read when you need the *why*, not on every run.

## The `make quality` commit hook fails open — why the verdict comes from `ci-gate-mapper`

`make quality` is wired as a `beforeShellExecution` hook on `git commit|git push` in `.cursor/hooks.json`; it runs
the exact `check_code_quality` CI job. But it **fails open** by design:

```
if command -v doc-builder && [ -f Makefile ]; then make quality; else exit 0; fi
```

So it only blocks where the toolchain is present (`doc-builder` + the `[quality]` extras + a root `Makefile`). In
the dev repo neither is present (`.ai/AGENTS.md`), so the hook is skipped and the commit/push succeeds regardless.

Consequence: a clean commit/push is **no proof** CI passed. Take the verdict from `ci-gate-mapper`, which reads the
live `pr_tests.yml` + `Makefile` and returns, per triggered job, the local mirror command and what failure looks
like:
- `check_code_quality` → `make quality`
- `check_repository_consistency` → the five checks `make quality` skips (`check_copies.py`, `check_dummies.py`,
  `check_support_list.py`, `check_forward_call_docstrings.py`, `make deps_table_check_updated`)
- `run_fast_tests` → the matrix slice's exact `pytest <dirs> -k <filter>` line for the changed paths (do **not**
  reduce it to one dir — a single-dir run can pass while the bundled slice fails)

`check_copies` / `check_dummies` / `pytest` stay CI-side; the hook only moves the `check_code_quality` signal
earlier when the env can verify it. CI remains the hard gate — never bypass a failing one.

## Push + PR (on a GO or conditional)

`implement` deliberately does **not** push or open a PR (`build-contract.md`) — ship owns it. The build already
sits on its own worktree branch off the base (from `implement`'s `/worktree`), so:

1. **Never push the default branch.** Push the worktree branch to the **fork** remote (`srlynch1/diffusers`). The
   push is where the `make quality` hook fires — the boundary.
2. **Target the fork, not upstream.** `boundary-guard.py` asks to confirm any `gh pr create` with no `--repo` (it
   can default to upstream `huggingface/diffusers`) and blocks an explicit upstream target. Always pass
   `--repo srlynch1/diffusers`.
3. **Body via temp file.** Write doc-author's release notes to a temp file and pass `--body-file`:

   ```bash
   BODY_FILE=$(mktemp "${TMPDIR:-/tmp}/ship-pr-body.XXXXXX")
   cat > "$BODY_FILE" <<'__BODY__'
   <doc-author's release notes, verbatim — NO attribution footer>
   __BODY__
   gh pr create --repo srlynch1/diffusers --title "<title>" --body-file "$BODY_FILE"
   ```

   Never use `--body "$(cat …)"`, a stdin pipe, or `--body-file -`: those can land an **empty** PR body while `gh`
   still exits 0 and returns a URL.
4. **No attribution footer** in the body (global rule) — no "Generated with…" line, no `Co-Authored-By` trailer.
5. **A no-go opens no PR.** Surface the blockers, record the verdict, and never push past a failing gate.

Eval runs only record the verdict (`finalize-result --ship-verdict …`); the harness owns branch/PR there.
