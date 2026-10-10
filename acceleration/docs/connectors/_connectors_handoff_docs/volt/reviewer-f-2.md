# Reviewer F, round 2 (delta): volt-dashboard PR #999 (2026-10-09)

VERDICT: GO

REVIEWED: head d90e49b0bc merged with origin/connectors/agents-ui 329fc17161 ("Already up to date"). Delta 466e491492..d90e49b0bc touches only tests/unit/agents/connectors-page.test.tsx (+84/-10). Worktree $S/volt/wt-f-rev2, removed at the end.

R1.1 (F-1, 6 surviving mutations): fixed.
- New assertions: the full dialog body before the 409 (`unforced`) and after it (`forced`); a test where DELETE answers 500 (dialog stays open, no success toast, no navigation); a test that the oauth-client read is dropped, then read again for a re-created id while the old client is not shown (clientGate).
- Mutations re-run ($S/volt/rvf2-mut.py, log $S/volt/rvf2-mutations.log): R23, R24, R25, R28, R30, R31 are all killed (6/6).

Checks (with .env.local copied in, 1577 bytes; removed with the worktree):
- unit: 236 files, 2716 tests pass, 0 VITE_AMPERE_API failures ($S/volt/rvf2.unit.log). That is 2715 plus the 1 new test.
- lint 0 (tree clean after --fix), knip:check 0 (exports 317) ($S/volt/rvf2.{lint,knip}.log).

Note: vi.mock('sonner') is file-wide; the suite still passes, so no other test in the file relied on real toasts.
The round-1 Nits F-2, F-3 and F-4 stay as tickets.
