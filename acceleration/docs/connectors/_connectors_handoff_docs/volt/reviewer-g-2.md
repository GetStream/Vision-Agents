# Reviewer G, round 2 (delta) — volt-dashboard PR #1000 (2026-10-09)
VERDICT: GO
REVIEWED: 45123e6acb (one commit on db69ed017d) merged onto connectors/agents-ui 2a5e3f0a19 ("Already up to date"). Delta: agent-delete-dialog.tsx copy, plus tests in agent-connectors, connection-create and connectors-page.
PRIOR:
- R1.1 (R26, a read error is not Broken): fixed. The it.each server/network test fails under the mutation.
- R1.2 (R28, a 401 on create blocks): fixed. The needs_reauthorization create test fails under the mutation.
- R1.3 (R30, the list's revision guard): fixed. The "is no longer listed once the credential it checked was replaced" test fails under the mutation.
- R1.4 (R31, operator-only OAuth client read): fixed. The operator-only test fails under the mutation.
- R1.5 (R34, F68 reads only while open): fixed. The "reads no connection while the dialog is closed" test fails under the mutation.
- Nit (F68 wording): fixed. The copy is now «No other agent config binds X. If this connection receives messages (a Slack bot), they will no longer be answered.» It is conditional and sentence case.
NEW FINDINGS: none.
MUTATIONS: 5 checked, 0 survived ($S/volt/rg2-mut.py, rg2-mutations.log).
CHECKS: unit suite with .env.local: 236 files, 2745 tests, all pass (rg2-unit.log). bun lint 0 (rg2-lint.log).
Worktree wt-rg2 removed. .env.local and certs removed with it.
