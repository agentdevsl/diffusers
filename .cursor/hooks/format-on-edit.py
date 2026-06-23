#!/usr/bin/env python3
"""Per-file formatter for the diffusers Self-Improving Pack.

Cursor `afterFileEdit` hook. Reads the event on stdin, and if the edited file is a `*.py`, runs
`ruff format <file>` on **only that file**.

This is a deliberately bounded, NON-blocking side-effect runner — NOT a gate:
  * `afterFileEdit` is observe-only; its stdout is discarded ("No output fields currently supported"),
    so this can clean the file on disk but can never report a result back to the agent or block.
  * It is **format only**. It must NOT run `ruff check --fix`: this repo's ruff selects `F` (pyflakes)
    and `I` (isort), so `--fix` removes/reorders imports — semantic edits applied silently with no
    agent visibility. Leave `--fix` to an explicit `make style` the agent can review.
  * `ruff format` is idempotent: it writes only when the file actually changes, so a re-fire of the
    edit pipeline converges immediately (no formatting loop).

Fails quiet: any error (no ruff, bad event) exits 0 without touching anything. The agent's loop does
not depend on this hook.
"""
import json
import shutil
import subprocess
import sys


def main():
    try:
        event = json.load(sys.stdin)
    except Exception:
        sys.exit(0)

    file_path = event.get("file_path") or event.get("input", {}).get("file_path") or ""
    if not file_path.endswith(".py"):
        sys.exit(0)

    ruff = shutil.which("ruff")
    if not ruff:
        sys.exit(0)

    try:
        # Format ONLY. Never `ruff check --fix` (it rewrites imports — see module docstring).
        subprocess.run(
            [ruff, "format", file_path],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=8,
            check=False,
        )
    except Exception:
        pass  # fire-and-forget; never surface an error into the agent loop
    sys.exit(0)


if __name__ == "__main__":
    main()
