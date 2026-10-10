# Wave 5 brief (AI-990 follow-ups + AI-993), read after waves/3d-brief.md
- Ticket: Linear AI-990 (finding ids F15–F40, text in the ticket and in <S>/e2e/findings.md), AI-993. Evidence files: <S>/e2e/*.md.
- Base: origin/accelerate (fetch at start). Newest migration on base: 20261011180000. Use your assigned slot only if a migration is unavoidable; expand-only; no new column on an existing table that the previous tag reads through bun — use a side table.
- Local e2e router (container vision-agents-router-1, DB model_router) is OFF LIMITS: another agent runs a migrate test there. Never restart it or write to model_router.
- Known base failures (do not diagnose): TestTurnRecordingSuite, TestSIPTrunksSuite (2), TestDataMoveSuite (order-dependent, passes alone), TestSessionUpdateSuite/TestAUserRenamesASessionThatEnded, TestKnowledgeUrlsSuite (order-dependent). gofmt flags internal/api/audit_test.go and streams_test.go:382 on base.
- Other PRs in this wave touch other files; stay inside your scope; anything else is a Question.
- Never print secrets. No Slack posts. Never touch staging.
- Commits end with «Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>»; PR body ends with «🤖 Generated with [Claude Code](https://claude.com/claude-code)»; PR title «AI-990: …» (or «AI-993: …»).
