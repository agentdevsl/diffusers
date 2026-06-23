#!/usr/bin/env python3
"""Boundary guard for the diffusers Cursor Pack — two hooks, one script.

  * `beforeShellExecution` (default mode) — scrutinize mutating SHELL commands, and gate eval
    PRs/pushes so they never target the huggingface/diffusers UPSTREAM (eval ships go to the user
    fork only).
  * `preToolUse` with `--tool-edit` (matcher `Write|Delete` — Cursor's real write tool-types; it does
    not expose `Edit`/`MultiEdit`) — scrutinize the agent's NATIVE file writes, which never reach the
    shell hook. The internal tool-name allow-list stays broad so an over-broad matcher fails open.

Both read a JSON event on stdin and print a JSON decision on stdout (snake_case keys, per spec):
  {"permission": "allow" | "deny" | "ask", "agent_message": "...", "user_message": "..."}

This is the pack's deterministic WRITE-LOCATION boundary. It is a **best-effort accident guard, not a
sandbox**: it filters WHERE a write lands, not whether it is destructive (an `rm -rf` of an in-boundary
path is *allowed*), and command parsing is heuristic — a determined bypass (laundering a write through a
runtime it doesn't recognize) is possible. Defense in depth = this guard + readonly subagents + CI (the
hard gate). `.cursorignore` only scopes read-tools/index; `permissions.json` only reduces prompts.

Out of boundary = parent-dir escapes (`../`), home (`~`), absolute paths outside the repo (except
`/tmp`), and the excluded repo dirs (benchmarks/ examples/ docker/ .github/ build/ dist/). Everything
in-repo and not excluded is allowed (the source tree, tests/, docs/, .cursor/, …). The excluded-dir and
`src/diffusers/`-style roots are **target-diffusers-repo-relative** — i.e. the layout the pack guards
when deployed into a diffusers checkout. In this vendored dev repo the library lives under
`vendor/diffusers/`, so the boundary resolves relative to wherever `.cursor/` sits (see find_repo_root).

The repo root is **derived from the event's `cwd`** (walking up, preferring a `.cursor` marker over a
bare `.git`), NOT hardcoded — so absolute in-repo paths are recognized on both local and cloud
checkouts, and a nested git repo below the pack root can't silently re-root the boundary. Fails OPEN on
any parse error or unrecognized shape: a guard bug must never wedge the agent.
"""
import json
import os
import re
import sys

# Repo subdirs that are out of boundary even though they live in the tree.
EXCLUDED_DIRS = {"benchmarks", "examples", "docker", ".github", "build", "dist"}

# Temp roots that are always writable (cover macOS's real temp locations too).
TMP_PREFIXES = ("/tmp", "/private/tmp", "/var/folders", "/private/var/folders")

# Tokens that indicate a mutation / write. Broadened to cover interpreter wrappers that can launder a
# write (sh/bash/zsh -c, perl/ruby/node -e, python -c/-m) — their inner paths still get scanned below.
MUTATING = re.compile(
    r"(^|\s)(rm|mv|cp|tee|dd|truncate|install|ln|chmod|chown|mkdir|rmdir|touch)\b"
    r"|>>?|sed\s+-i"
    r"|(^|\s)(sh|bash|zsh|perl|ruby|node)\s+-[ce]\b"
    r"|python[0-9.]*\s+-[cm]\b"
)

# Candidate "interesting" path tokens in a raw command: home, parent-escape, absolute, or an
# excluded-dir-relative path. `:` is deliberately NOT a leading delimiter so URLs (https://…) don't match.
SUSPECT = re.compile(
    r"(?:^|[\s=><(\"'])"
    r"(~[^\s\"';|&)]*"
    r"|\.\./[^\s\"';|&)]*"
    r"|/[^\s\"';|&)]*"
    r"|(?:benchmarks|examples|docker|\.github|build|dist)/[^\s\"';|&)]*)"
)

# Keys a Cursor tool_input might carry the target file under. `file_path` is Cursor's documented key
# (used by beforeReadFile/afterFileEdit); the rest are defensive belt-and-braces in case the
# undocumented Write/Delete tool_input differs. (`target_file` is an alternate editor-agent key, kept just in case.)
PATH_KEYS = ("file_path", "path", "target_file", "relativePath", "relative_path", "filename", "uri")

# Eval ship-boundary: eval fixes ship to the user FORK only.
# Never open a PR or push to the huggingface/diffusers UPSTREAM. This is a hard rule that a prose
# rule alone failed to hold (a run opened 8 upstream PRs before they were caught + closed).
UPSTREAM_SLUG = "huggingface/diffusers"
FORK_SLUG = "srlynch1/diffusers"
UPSTREAM_MSG = (
    "Eval ship-boundary: refused a PR/push targeting huggingface/diffusers (UPSTREAM). Eval ships go to "
    "the user fork ONLY — use `gh pr create --repo srlynch1/diffusers …` or push to the fork remote. "
    "A PR to upstream is a separate, explicit, human-confirmed step."
)
UPSTREAM_ASK_MSG = (
    "`gh pr create` without `--repo` can default to huggingface/diffusers (UPSTREAM). Confirm the target — "
    "eval PRs must use `--repo srlynch1/diffusers` (the user fork)."
)


def find_repo_root(start):
    # Walk up from `start` to the pack root. Prefer the `.cursor` marker (the pack lives there) over a
    # bare `.git`, so a nested git repo / embedded clone *below* the pack root can't silently re-root the
    # boundary. `.git` may be a dir (normal repo) or a file (submodule), so probe with exists, not isdir.
    cur = os.path.abspath(start) if start else os.getcwd()
    git_root = None
    probe = cur
    for _ in range(40):
        if os.path.isdir(os.path.join(probe, ".cursor")):
            return probe
        if git_root is None and os.path.exists(os.path.join(probe, ".git")):
            git_root = probe
        parent = os.path.dirname(probe)
        if parent == probe:
            break
        probe = parent
    return git_root or cur


def classify(target, repo_root, cwd):
    """Return a deny-reason string if `target` is out of boundary, else None (allow)."""
    t = (target or "").strip().strip("\"'")
    if t.startswith("file://"):
        t = t[len("file://"):]
    if not t:
        return None

    if t.startswith("~"):
        return f"home path '{t}'"

    if t.startswith("/"):
        norm = os.path.normpath(t)
        if any(norm == p or norm.startswith(p + "/") for p in TMP_PREFIXES):
            return None
        if repo_root and (norm == repo_root or norm.startswith(repo_root + os.sep)):
            return _classify_rel(os.path.relpath(norm, repo_root))
        return f"absolute path '{t}' outside the repo"

    base = cwd or repo_root or os.getcwd()
    norm = os.path.normpath(os.path.join(base, t))
    if repo_root and not (norm == repo_root or norm.startswith(repo_root + os.sep)):
        return f"path '{t}' that escapes the repo boundary"
    rel = os.path.relpath(norm, repo_root) if repo_root else os.path.normpath(t)
    return _classify_rel(rel)


def _classify_rel(rel):
    seg = rel.split(os.sep, 1)[0]
    if seg in EXCLUDED_DIRS:
        return f"excluded directory '{seg}/'"
    return None  # in-repo and not excluded -> allow


def _first_tool_path(tool_input):
    if not isinstance(tool_input, dict):
        return None
    for key in PATH_KEYS:
        val = tool_input.get(key)
        if isinstance(val, str) and val.strip():
            return val
    return None


def decision(permission, agent_msg=""):
    # Cursor's before-class output contract is snake_case (`permission`, `agent_message`,
    # `user_message`, `updated_input`). camelCase keys are silently dropped, so a deny reason
    # would never reach the agent/user. `permission` is the load-bearing field either way.
    out = {"permission": permission}
    if agent_msg:
        out["agent_message"] = agent_msg
        out["user_message"] = agent_msg
    print(json.dumps(out))
    sys.exit(0)


def boundary_msg(reason):
    return (
        f"Write-location boundary: refused a write touching {reason}. Allowed = in-repo paths outside "
        f"{', '.join(sorted(EXCLUDED_DIRS))}/ (e.g. the source tree, tests/, docs/, .cursor/) plus /tmp. "
        f"This guards write LOCATION, not destructiveness — edit only approved areas; CI is the hard gate."
    )


# git global options that may sit between `git` and the `push` subcommand. An agent (or a
# normal worktree push, `git -C <dir> push …`) that uses any of these must NOT slip past the gate.
_GIT_GOPT = (
    r"(?:(?:-C|-c)\s+\S+"
    r"|--(?:git-dir|work-tree|namespace|super-prefix)(?:=\S+|\s+\S+)"
    r"|--exec-path(?:=\S+)?"
    r"|--no-pager|--bare|--paginate|--no-replace-objects|--literal-pathspecs|-p)"
)
_PUSH_RE = r"git\s+(?:" + _GIT_GOPT + r"\s+)*push\b"


def _git(args, cwd):
    """Bounded `git <args>`; returns trimmed stdout or '' on any error. Fails OPEN by design."""
    try:
        import subprocess
        r = subprocess.run(
            ["git", *args],
            cwd=cwd if (cwd and os.path.isdir(cwd)) else None,
            capture_output=True, text=True, timeout=3,
        )
        return (r.stdout or "").strip() if r.returncode == 0 else ""
    except Exception:
        return ""


def _resolve_remote_url(remote, cwd):
    """Resolve the PUSH url of a named remote (`git remote get-url --push`) — the url `git push`
    actually writes to, which can differ from the fetch url. Fails OPEN (returns '')."""
    if not remote or not re.match(r"^[A-Za-z0-9._/-]+$", remote):
        return ""
    return _git(["remote", "get-url", "--push", remote], cwd)


def _resolve_push_target_url(cwd):
    """For a bare `git push` (no remote arg), resolve the current branch's push/upstream remote
    and return its push url, so a branch that tracks the upstream remote is still gated. Tries
    @{push}/@{upstream} (works once remote-tracking refs exist), then falls back to the branch's
    configured remote (works without a fetch). Fails OPEN (returns '')."""
    ref = (_git(["rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{push}"], cwd)
           or _git(["rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{upstream}"], cwd))
    if ref and "/" in ref:
        return _resolve_remote_url(ref.split("/", 1)[0], cwd)
    branch = _git(["symbolic-ref", "--quiet", "--short", "HEAD"], cwd)
    remote = ""
    if branch:
        remote = (_git(["config", "--get", f"branch.{branch}.pushRemote"], cwd)
                  or _git(["config", "--get", f"branch.{branch}.remote"], cwd))
    remote = remote or _git(["config", "--get", "remote.pushDefault"], cwd)
    return _resolve_remote_url(remote, cwd) if remote else ""


def upstream_guard(command, cwd):
    """Gate eval PRs/pushes away from the huggingface/diffusers UPSTREAM. Returns a
    (permission, message) tuple to act on, or None to fall through to the write-location scan."""
    c = (command or "").strip()
    if not c:
        return None
    low = c.lower()
    sep = r"(?:^|[\s;&|(])"
    # Gate only WRITE ops that can ship to upstream — PR-mutating `gh pr` and `git push`. Reads
    # (`gh pr list/view`), fetch-remote setup (`git remote add upstream …`) and fork ops are left alone.
    # Both detections tolerate global options/flags before the subcommand (`git -C <dir> push`,
    # `gh --repo X pr create`) — those forms previously slipped through the gate entirely.
    is_gh_pr_write = re.search(sep + r"gh\s+(?:[^\s;&|]+\s+)*pr\s+(?:create|merge|edit|ready|close|reopen)\b", low) is not None
    is_push = re.search(sep + _PUSH_RE, low) is not None
    if not (is_gh_pr_write or is_push):
        return None

    # 1) Explicit upstream slug in a PR-write/push command -> hard deny. Covers
    #    `gh pr create --repo huggingface/diffusers` and URL-form `git push …huggingface/diffusers…`.
    if UPSTREAM_SLUG in low:
        return ("deny", UPSTREAM_MSG)

    # 2) `git push …`: resolve the destination's PUSH url; deny if it points at upstream. Handles an
    #    explicit bare remote (`git push origin …`) AND a bare `git push` whose branch tracks upstream.
    if is_push:
        m = re.search(_PUSH_RE + r"\s+((?:-\S+\s+)*)([^\s\"'|;&]+)", low)
        remote = m.group(2) if m else None
        if remote and not remote.startswith("-") and "/" not in remote and ":" not in remote:
            url = _resolve_remote_url(remote, cwd)
        else:
            url = _resolve_push_target_url(cwd)  # bare `git push` / flags-only push
        if UPSTREAM_SLUG in url.lower():
            return ("deny", UPSTREAM_MSG)

    # 3) `gh pr create|merge|…` with no explicit --repo/-R can default to upstream -> ask to confirm.
    #    `-R\S*` matches both the spaced (`-R repo`) and glued (`-Rrepo`) short-flag forms.
    if is_gh_pr_write and not re.search(r"--repo\b|(?:^|\s)-R\S*", c):
        return ("ask", UPSTREAM_ASK_MSG)

    return None


def main():
    tool_mode = "--tool-edit" in sys.argv
    try:
        event = json.load(sys.stdin)
    except Exception:
        decision("allow")  # fail open on parse error; never wedge the agent
        return

    cwd = event.get("cwd") or (event.get("tool_input") or {}).get("cwd") or os.getcwd()
    repo_root = find_repo_root(cwd)

    if tool_mode:
        # Only enforce on write-ish tools; everything else (a too-broad matcher, a Read) fails open.
        tname = (event.get("tool_name") or "").lower()
        if tname and not any(k in tname for k in ("write", "edit", "delete", "create", "move", "rename")):
            decision("allow")
            return
        target = _first_tool_path(event.get("tool_input") or event.get("input") or {})
        if not target:
            decision("allow")
            return
        reason = classify(target, repo_root, cwd)
        decision("deny", boundary_msg(reason)) if reason else decision("allow")
        return

    command = (event.get("command") or event.get("input", {}).get("command") or "").strip()
    # Eval ship-boundary first: `gh pr create` / `git push` are not file mutations, so they would
    # otherwise sail past the write-location scan. Never PR/push to huggingface/diffusers upstream.
    verdict = upstream_guard(command, cwd)
    if verdict:
        decision(*verdict)
        return
    if not command or not MUTATING.search(command):
        decision("allow")
        return
    for m in SUSPECT.finditer(command):
        reason = classify(m.group(1), repo_root, cwd)
        if reason:
            decision("deny", boundary_msg(reason))
            return
    decision("allow")


if __name__ == "__main__":
    main()
