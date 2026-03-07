## Motivation
<!-- Explain why this change is needed. What problem does it solve? Provide context. -->
<!-- If it fixes an open issue, please link it here (e.g., `Fixes #123`). -->

## What's included
<!-- Describe the changes introduced in this pull request in bullet points. -->
- 
- 
- 

## Design decisions
<!-- Explain any significant design or architectural choices made locally within the scope of this PR. -->
<!-- Why was it done this way? Are there alternative approaches considered? -->

## Checklist
<!-- Make sure all of these are checked before submitting your PR. -->

- [ ] `uv run py.test -m "not integration" -n auto` — all unit tests pass
- [ ] `uv run python dev.py check` — ruff, mypy, and non-integration tests pass
- [ ] `uv run ruff check --fix` — no lint issues
- [ ] `uv run mypy --install-types --non-interactive -p vision_agents` — no type errors
- [ ] Added or updated tests to cover my changes
- [ ] Documentation has been updated (if applicable)
