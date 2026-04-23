---
name: pr
description: Create a draft pull request for the Vision-Agents repo using gh CLI.
---

# Pull Request (Vision-Agents)

## Before creating

- One logical change per PR. If the branch covers more than one independent change, split it into separate PRs before writing the body.
- Run `uv run --no-sync dev.py check` (ruff + mypy + unit tests). Fix anything it reports.
- Do not run integration tests locally. CI handles them.
- If the change is user-facing (public API break, new feature, bug fix), update `CHANGELOG.md` per the rules in `CLAUDE.md` before opening the PR.

## Creating

- Use `gh pr create --draft`.
- Follow `.github/pull_request_template.md`. It has two sections: `## Why` (required) and `## Changes` (optional).
- Read every commit on the branch before writing the body. Do not summarise from the latest commit alone.
- `## Why` must cover the motivation and enough context about the current state that a reviewer can evaluate the change. It is not a description of what changed, the diff already does that.
- `## Changes` is a short, high-level bullet list of the main changes. Omit it entirely when the diff is small or self-evident (e.g. a dependency bump). Never let `## Changes` dissolve into a file-by-file list or turn into per-bullet justifications — that belongs in `## Why`.
- If the change relates to a public GitHub issue, link it inline within `## Why` (for example, "users reported X (#478), which happens because Y"), not as a trailing `Fixes #N`.

## Rules

- Always draft. The human publishes.
- Push the branch before creating the PR.
- Do not paste CI output, lint output, or tool logs into the body.
