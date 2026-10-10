# Wave 8 brief (AI-1049+1048, AI-1052, AI-958, AI-1053+F52+F67, plugins deprecation), read after waves/3d-brief.md
- Ledger: waves/8.md. Your PR's row and the Decisions block are your spec; Linear ticket text is background. Decisions were taken by Kanat; do not re-open them, raise a Question instead.
- Base: origin/accelerate (fetch at start; 599298c6 or newer). Newest migration on base: 20261014120000. Use your assigned slot only if a migration is unavoidable; expand-only; no new column on a table the previous tag reads through bun — use a side table.
- Test DB: `source <scratchpad>/w3c/testdb.sh <your worktree> model_router_test_w8_<t>`; never commit `acceleration/internal/config/testing.yaml`.
- Local router (container vision-agents-router-1, DB model_router) is OFF LIMITS: never restart it or write to model_router. Never touch staging.
- Known base failures (do not diagnose): waves/8.md «Known base failures».
- Hardcoded values (limits, codes) get a comment with their source and a line in the PR body.
- Other PRs in this wave touch other files; stay inside your scope; anything else is a Question. Generated files (openapi.yaml, SDKs) are regenerated, never hand-edited.
- Never print secrets. No Slack posts. Never kill processes broadly.
- Commits end with «Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>»; PR body ends with «🤖 Generated with [Claude Code](https://claude.com/claude-code)»; PR title «AI-<id>: …». Branch from the ledger.
