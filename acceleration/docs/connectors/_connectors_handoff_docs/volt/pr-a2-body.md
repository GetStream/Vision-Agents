Fixes the design-audit findings 2-5 on phase A (#994). Finding 1 (nav move to Library) is left to phase B.

## Changes
```
widgets/connection-detail-page.tsx (694 lines)
  -> connection-overview.tsx, connection-tools.tsx, connection-invocations.tsx,
     connection-audit.tsx, validation-result.tsx      (no behaviour change)
connections-page.tsx   Owner/User/Connector row -> TableFilters in DataTable.Header
lib/voices.ts          listNames -> lib/format.ts (voices importers updated)
connection-delete-dialog.tsx   confirmation={name}
```
- Filters: "End user ID" (text; setting it means owner=user, removing it means the app's own) and "Connector" (one at a time, like sessions). Signed-in-user default for `?owner=user` is unchanged.
- `AgentsResourcesSelectors.connectionsUserInput` removed (the input is now the filter popover); the repo's pre-commit hook requires it. Phase B edits the same selector block, so expect a one-line merge.

## Checks
bun lint, knip:check, build, unit suite (231 files, 2615 tests) pass; `tsc -p tsconfig.app.json` error set identical to base.
Mutations: drop `confirmation` -> 3 delete tests fail; filter commit drops user_id -> "lists another end user" fails.

## Design standards
- [x] One component per file, function declarations: `.claude/rules/code-style.md`; the 5 new files in `dashboard/agents/widgets/`.
- [x] Volt explorer tables: filters in `TableFilters` inside `DataTable.Header`, as `sessions-table.tsx:413`; `connections-page.tsx:~100-165` and the header.
- [x] Destructive delete types the name: `connection-delete-dialog.tsx:38`, as `voices-delete-dialog.tsx:44` (design-review "irreversible actions").
- [x] Cross-feature helpers in a shared lib, not another feature's module: `lib/format.ts:50`.
- [x] Tier: `components/dashboard/` only, no `custom/` import, no Remixicon (AGENTS.md); icons from design-system (unchanged).
- [x] Sentence case: "End user ID", "Add connection filter".
- [x] Test ids: one removed in `src/test-ids.ts`; no new ones (BEM unchanged).
- [x] Tests drive the UI and stub at the `agentRequest` boundary: `tests/unit/agents/connections-page.test.tsx`.
- n/a: sidebar/route rules (finding 1, phase B); forms/state/routing rules untouched.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
