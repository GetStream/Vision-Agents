---
name: commit
description: Create a git commit for the Vision-Agents repo.
---

# Commit (Vision-Agents)

## Rules

- If on `main`, warn and ask whether to create a new branch first.
- Use conventional commits. Keep the subject line ≤72 chars.
- One logical change per commit. Stage files individually by name, never `git add .` / `-A`.

## Pre-commit hooks

- The repo runs ruff (check + format), mypy, trailing-whitespace and eof fixes on commit.
- Never pass `--no-verify`. If a hook modifies files, re-stage and retry. If mypy or another check fails, fix the root cause before committing.
