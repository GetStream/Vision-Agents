---
name: pr
description: Create a draft pull request for the Vision-Agents repo using gh CLI.
---

# Pull Request (Vision-Agents)

## Before creating

- Run `uv run --no-sync dev.py check` (ruff + mypy + unit tests). Fix anything it reports.
- Do not run integration tests locally. CI handles them.
- If the change is user-facing (public API break, new feature, bug fix), update `CHANGELOG.md` per the rules in `CLAUDE.md` before opening the PR.

## Creating

- Use `gh pr create --draft`.
- Follow `.github/pull_request_template.md`. It is a single `## Why` section.
- Read every commit on the branch before writing `## Why`. Do not summarise from the latest commit alone.
- `## Why` must cover the motivation and enough context about the current state that a reviewer can evaluate the change. It is not a description of what changed, the diff already does that.
- If the change relates to a public GitHub issue, link it inline within `## Why` (for example, "users reported X (#478), which happens because Y"), not as a trailing `Fixes #N`.

## Rules

- Always draft. The human publishes.
- Push the branch before creating the PR.
- Do not paste CI output, lint output, or tool logs into the body.
