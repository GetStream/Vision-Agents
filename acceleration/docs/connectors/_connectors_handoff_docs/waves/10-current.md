# Wave 10 current state (handoff detail), 2026-10-13

Do not commit this file. Wave 10 (Volt phase J) is done. No router tag. Branch `ai-team/agent-dashboard` tip `dce53973`. Exit tickets AI-1146..AI-1162 are under AI-816. The ledger is `10.md` (Decisions, rulings J1-J9, PR table, Log, Tickets to file). The orchestrator owns `10.md`. Do not edit it. This note is the short copy for the handoff. Plan: handoff section 7 item 2.

Checked 2026-10-13 against `10.md` log 05:08Z (smoke, status done) and `9.md` log 03:05Z-03:06Z (B1/B3). No fresh `gh pr view` this sync.

Base before the wave: `1c66de1ad9` (origin/ai-team/agent-dashboard). Router for the UI: staging `accelerate-v0.6.35` = `8160c4cd` (has `tool_patterns`).

## What shipped

Five PRs (J1), not one. Kanat 01:45Z: Q1 yes, Q2 block with no `*` default, Q3 defer `last_validation` to wave 11, Q4 drop (not swap), deploy and smoke OK in advance. Q5 wording: orchestrator took the recommendation.

- Connectors is the first section on Agent > Tools (w10-top).
- A binding grants chosen tools or patterns, never both. Opening a pattern binding shows its patterns and saves them back unless the user switches mode. Switching sends exactly one of `tools` / `tool_patterns` (`tools: []` with patterns). An empty grant is blocked. Names and patterns split on commas and whitespace. No silent `*` (J3, AI-1120, AI-1057).
- Duplicate and Save as new drop a fixed binding the router would 409 (`channel: true`, connection used by a live config) and say so before submit. Test copies are kept: the router accepts them (J4 amended 03:31Z, AI-1083).
- The delete dialog names only a fixed channel connection no other config uses (J7, AI-1069).
- A `connector_unavailable` frame is a notice: info for login reasons, warning otherwise, one per alias (J5, AI-1056).
- New connection starts consent before the dialog closes. A failure stays open, shows the router message, and deletes the pending connection (J6, AI-1058).
- A binding row is a skeleton until the connector name and, for fixed, the connection label are read. A failed read falls back to the id (J8, AI-1062).
- Spec: `openapi.yaml` copied from Vision-Agents `8160c4cd` into `api/agents/openapi.yaml`, then `bun gen:agents`, in w10-grant only (J2). Diff was `tool_patterns` only.

## PRs (ledger, 2026-10-13)

Merge order: #1013, #1015, #1016, #1017, #1014. SHA is the squash on `ai-team/agent-dashboard`.

| PR | State | SHA | What |
|---|---|---|---|
| volt-dashboard#1013 | MERGED | `55b3afc9` | w10-top. Connectors first; no raw id (AI-1062). GO head `2fa164a3`. |
| #1015 | MERGED | `07e6a4e3` | w10-unavail. `connector_unavailable` notice (AI-1056). GO head `dd3cd0f8`. |
| #1016 | MERGED | `0b9fc9b0` | w10-copy. Drop channel binding on copy; narrower delete (AI-1083, AI-1069). GO head `30db9a7a`. |
| #1017 | MERGED | `ef80bae3` | w10-grant. `tool_patterns` editor + spec regen (AI-1120, AI-1057). Head before merge `703e5ffe`; reviewed `ac497387`. |
| #1014 | MERGED | `dce53973` | w10-consent. Failed consent stays open (AI-1058). GO head `6d413061`. Branch tip. |

## Deploy and smoke

Vercel run 38348753114 succeeded at `dce53973` (log 04:57Z). Alias step emitted no skip warning. Deploy url `volt-dashboard-czsf2gzxm`. Alias last-modified 04:58:31Z. Preview: https://volt-dashboard-git-ai-team-agent-dashboard-getstreamio.vercel.app/organization/1181507/1257545/agents/

Smoke pass 05:08Z on org 1181507 / app 1257545, staging v0.6.35:

- `e2e-connectors` Tools: Connectors first, names shown, "Tools matching list_teams/get_me", not saved.
- `ui-test-w10-smoke-0100`: "All tools" saved and survives reload. Empty grant blocked.
- Duplicate: no channel warning. Both test agents deleted.

Unverified: the commit is not visible in the HTML. JS console errors were not hooked.

## Local router

Rebuilt 03:05Z from worktree `<scratchpad>/w10/va-accel` at origin/accelerate `2ab444fd` (includes `8160c4cd`). Goose `20261018120000`, `/health` 200. Dump before the rebuild: `<scratchpad>/w10/local-db-before.sql`.

Lesson (log 03:29Z): the local router socket CORS allows only `https://local.getstream.io:3011` (`ROUTER_CORS_ORIGINS`). Socket and playground browser checks use port 3011, one agent at a time. The :3000 server (`<scratchpad>/volt/wt-int`) is Kanat's and was behind base; do not move or stop it.

Volt known base failure on `1c66de1ad9`: `tsc -p tsconfig.app.json` exits 1, 1122 errors (17 under agents). Do not fix them. Unit agents: 69 files / 823 passed.

## Deferred (wave 11)

- `Connection.last_validation` in volt (F70, AI-1052 caller). Q3, Kanat: defer. "Last check" is still cache-only after reload.
- F73: the info hint renders as `role=alert aria-live=assertive`. Not in scope.

## Tickets

Done: AI-1062, AI-1120, AI-1057, AI-1058, AI-1083, AI-1069, AI-1056.

Exit AI-1146..AI-1162 under AI-816. Notable: AI-1155 (openapi says `credential_rejected` offers no login; router `loginReasons` includes it), AI-1161 (Cursor attribution on PR bodies), AI-1162 ("Connections are not copied" text). The nits live in `10.md` "Tickets to file at exit". Do not refile them.

## Rulings (do not re-open)

J1-J9 are the Decisions and Planner rulings at the top of `10.md`. J4 was amended in the log (03:31Z): the client drop rule mirrors the router's 409 (keep a binding the router would accept, including test copies). Q1-Q5 are answered (log 01:45Z).
