---
name: commit
description: Create a git commit for the Vision-Agents repo.
---

# Commit (Vision-Agents)

## Rules

- Before committing, check the current branch. If on `main`, warn and ask whether to create a new branch first.
- Use conventional commits format: `feat:`, `fix:`, `refactor:`, `test:`, `docs:`, `chore:`, etc.
- Keep the subject line (first line) at 72 characters or fewer. If it does not fit, the commit is probably doing too much.
- One logical change per commit. Do not bundle unrelated changes.
- Stage files individually by name. Never use `git add .` or `git add -A`.
- Commit message body (when needed) focuses on why, not what.

## Pre-commit hooks

- The repo runs ruff (check + format), mypy, trailing-whitespace and eof fixes on commit. Let them run.
- Never pass `--no-verify`. If a hook modifies files, re-stage the changed files and retry the commit. If mypy or another check fails, fix the underlying issue before committing.
