# AGENTS.md — unionai-examples

For coding agents. Claude Code reads this file through `CLAUDE.md`.

## Read first

`CONTRIBUTING.md` covers what belongs in this repository, how examples are structured and tested,
and how they reach the docs. It describes the `v2/` tree. `v1/` is legacy, so don't add to it.

## With Docsy, changes on GitHub go through Docsy

If Docsy's MCP server is connected (a server named `docsy`, whose tools include `open_pr`, `merge_pr`
and `whoami`), every change on GitHub to `unionai-docs`, `unionai-docs-infra` or `unionai-examples`
goes through its tools, never `gh` or `git push`. That covers opening, pushing to, commenting on and
merging pull requests. Reading is fine (`git fetch`, `gh pr view`). Do the work itself locally: read
from `origin/main`, edit in a worktree, run the checks, then hand Docsy the finished files.

Without Docsy, use git and `gh` as usual, following `CONTRIBUTING.md`.
