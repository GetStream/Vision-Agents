---
name: pr
description: Create a draft pull request for the Vision-Agents repo using gh CLI.
---

# Pull Request (Vision-Agents)

## Before creating

- Run `git log main..HEAD --oneline`. If the branch contains more than one independent logical change, STOP and ask the user whether to split it before proceeding.
- Run `uv run --no-sync dev.py check`. Skip if the diff does not touch Python code (`*.py`) or `pyproject.toml` — e.g. docs-only, `.gitignore`, `.github/`, or `.claude/` changes.
- Do not run integration tests locally, CI handles them.
- If the change is user-facing (public API break, new feature, bug fix), update `CHANGELOG.md` per the rules in `CLAUDE.md`.

## Creating

- Always `gh pr create --draft`. Push the branch first.
- Follow `.github/pull_request_template.md`. Read every commit on the branch, do not summarise from the latest commit alone.
- `## Why` is motivation + context. `## Changes`, if included, is high-level; never per-bullet justifications, those belong in `## Why`.
- Link public GitHub issues inline within `## Why` (e.g. "users reported X (#478)"), not as a trailing `Fixes #N`.
- Do not paste CI, lint, or tool output in the body.
