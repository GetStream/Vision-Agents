# Reviewer round 1 — volt PR #995 (head c5ac2eb27a, base connectors/agents-ui 744c52f775)
Date 2026-10-09. Worktrees: $S/volt/rv-a2 (head, detached), $S/volt/rv-a2-base (744c52f775). Base == origin/connectors/agents-ui at review time, so no merge needed.

VERDICT: NO-GO (2 Should fix)

## Checks (logs in $S/volt/)
| check | head | base 744c52f775 | log |
|---|---|---|---|
| bun lint | 0 | 0 | rv-a2.lint.log / rv-a2-base.lint.log |
| bun run knip:check | 0 | 0 | rv-a2.knip.log / rv-a2-base.knip.log |
| bun run build | 0 | 0 | rv-a2.build.log / rv-a2-base.build.log |
| unit (`bun run vitest run --project unit`) | 231 files, 2615 pass | 231 files, 2615 pass | rv-a2.unit.log / rv-a2-base.unit.log |
| npx tsc -p tsconfig.app.json --noEmit | 1121 errors | 1121 errors | identical set after stripping (line,col); none in touched files |
CI on head: lint, knip, unit, Vercel all SUCCESS (`gh pr view 995 --json statusCheckRollup`). No review comments.
Script: $S/volt/rv2a-checks.sh. Worktrees clean after runs (rv-a2.status.log empty).

## Finding by finding
2 (one component per file): fixed. Old bodies (744c52f775 connection-detail-page.tsx:266-292, 293-417, 418-468, 469-596, 597-end) diffed against new files with only `function`->`export function`: identical except trailing blank line. ConnectionDetailPage body (old :80-265) identical to new. Types Invocation/AuditEvent moved with their users. No behaviour change. Diff dir: $S/volt/rv2a-split.
3 (TableFilters in DataTable.Header): fixed, mirrors sessions-table.tsx:192-260,414-428 (pending state, applied, commit, remove, clear, refresh).
4 (listNames -> lib/format.ts): fixed; all 3 importers updated; knip clean.
5 (confirmation): fixed, connection-delete-dialog.tsx:38 `confirmation={name}`, same as voices-delete-dialog.tsx:44.

## Filter semantics vs router (Vision-Agents origin/accelerate 26051062, acceleration/internal/api/connections.go:169-173)
owner_type app|user required, connector_id one, user via X-Stream-User-Id. UI: user chip -> owner=user + header user id; no chip -> owner_type=app; connector chip -> one connector_id. Matches.
Loss 1 (Should fix): before, Owner=End user with no id listed the signed-in person's connections (`agentUser()` = 'volt-'+dashboardUserId, agents-credentials.ts:60). Now the only UI path is typing that synthetic id; adding "End user ID" opens an empty editor (filterOptions placeholder only). `?owner=user` default still works by URL only; no link in src sets owner=user without user_id (grep). The component doc (connections-page.tsx:54-59) still says the signed-in person is the default.
Loss 2 (Nit): old Input had maxLength=256; TableFilters text kind has no maxLength (core/index.d.ts:5120). A >256-char id fails connectionsSearchSchema user_id max(256) `.catch(undefined)` (connections.ts:32) and the page falls back to the signed-in user; the chip then shows volt-<id>, not what was typed.
Not lost: explicit App (= no user chip), All connectors (= remove chip).
Multi-tick in one Apply picks the first un-current tick, not "newest"; same as sessions-table precedent, not a finding.

## Mutations (script $S/volt/rv2a-mut.py, run from $S/volt/rv-a2: `python3 ../rv2a-mut.py [M3 ...]`; log rv2a-mut.log)
M1 drop confirmation -> KILLED (3 delete tests)
M2 commit drops user_id -> KILLED (lists another end user)
M3 commit `{owner:'user',user_id}` -> `{user_id}` -> SURVIVED (only test starts at ?owner=user)
M4 empty commit keeps owner=user -> SURVIVED
M5 remove user chip keeps owner=user -> SURVIVED
M6 connector commit ignored -> SURVIVED
M7 connector newest-tick -> picked[0] -> SURVIVED
M8 clear keeps owner -> SURVIVED
M9 remove connector ignored -> SURVIVED
Base had no UI test of the owner Select or connector Select either, but commit/remove/clear are new code in this PR.

## Findings
[Should fix] connections-page.tsx:100-154 + tests/unit/agents/connections-page.test.tsx — owner switch and connector filter untested (M3-M9 survive) — add UI tests: from the app list add "End user ID", type an id -> request owner_type=user with that header; remove the chip / commit empty -> owner_type=app; pick a connector -> connector_id=...; tick a second -> the new one; remove -> no connector_id; Clear -> app, no connector.
[Should fix] connections-page.tsx:101,126-134 — signed-in user's connections no longer reachable without typing 'volt-<dashboardUserId>' — when "End user ID" is added and agentUser() exists, set owner=user (the existing default then shows "End user ID: volt-42", editable), or prefill it; test it.
[Nit] connections-page.tsx:101 — no maxLength on the id; >256 silently shows the signed-in user. Ticket.

## Not run / unverified
- /pr-review skill not run: it targets the Vision-Agents repo; review done by hand against the same checklist.
- No browser run of the new filter popover (unit tests drive it via TableFilters in jsdom only).
- Finding 1 (nav) out of scope.
