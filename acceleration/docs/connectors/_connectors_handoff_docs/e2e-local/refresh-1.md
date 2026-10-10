# Slack user-token refresh, run 1 (scenario «refresh», 2026-10-09)

The local router was recreated by the coordinator at 13:14:01.7Z (`docker inspect … StartedAt`). `ROUTER_PUBLIC_URL` is `https://carol-elliptic-uncloak.ngrok-free.dev`.
`PS` is `docker exec vision-agents-postgres-1 psql -U postgres -d model_router -Atc`. No token value was read or printed.

## Before (13:28:50Z)

`PS "select id, owner_id, status, revision, expires_at, connected_at, updated_at, last_error, length(credentials_sealed), md5(credentials_sealed) from connector_connections where id like '0d476f22%' or id like '6d92545f%'"`

| connection | user | status | revision | expires_at | updated_at | sealed len | sealed md5 |
|---|---|---|---|---|---|---|---|
| 0d476f22… | e2e-user-1 | connected | 2 | **2026-10-09 02:27:11.811955Z** (expired) | 2026-10-08 15:20:41Z | 1363 | 5f41175e… |
| 6d92545f… | e2e-user-2 | connected | 2 | **2026-10-09 02:27:12.492522Z** (expired) | 2026-10-08 16:59:19Z | 1363 | b125bbb1… |

`connector_audit` for both connections had only `grant_created/consent` (15:20:02Z and 16:59:19Z).

## The calls

Each user got one turn through `turn.sh`, on config `e2e-slack`: «Call slack_read_channel on channel C0C8MKNUNBA with limit 3 … Do not post anything.»

| user | session | turn sent | tool_call → result | answer | file |
|---|---|---|---|---|---|
| e2e-user-1 | 01a11bde… | 13:29:30Z | `slack_read_channel {"channel_id":"C0C8MKNUNBA","limit":3}` 13:29:34.598Z → messages from #kanat-test, 13:29:34.707Z | «Exactly 3 messages came back.» | `refresh-1-user1.out` |
| e2e-user-2 | 01a11c6f… | 13:30:14Z | the same call, 13:30:17.532Z → messages, 13:30:17.650Z | «3 messages came back.» | `refresh-1-user2.out` |

In `connector_invocations`, the user-1 call is at 13:29:34.598Z (109 ms) and the user-2 call is at 13:30:17.532Z (117 ms). Both have `error_type` null.

## After

| connection | status | revision | expires_at | updated_at | sealed len | sealed md5 | last_error |
|---|---|---|---|---|---|---|---|
| 0d476f22… | connected | **3** | **2026-10-10 01:29:31.912268Z** | 13:29:31.912Z | 1374 | d5e82265… | empty |
| 6d92545f… | connected | **3** | **2026-10-10 01:30:14.938643Z** | 13:30:14.938Z | 1374 | `unverified` (not read) | empty |

- In each case, `expires_at` minus `updated_at` is 43200 s, which is 12 h. The access token was refreshed.
- `connector_audit` has a new row for each connection:
  - `grant_refreshed`, revision 3, request `74876d72…`, session `01a11bde…`, 13:29:31.914Z (user-1);
  - `grant_refreshed`, revision 3, request `30ca12dc…`, session `01a11c6f…`, 13:30:14.940Z (user-2).
  The action is written at `internal/connectors/resolver/resolver.go:357` (`store.AuditGrantRefreshed`).
- The refresh came **when the session opened**, about 2.5 s before the tool call. That was the `said` turn's session join; see the timestamps above. It did not come on the call itself.
- No `refresh`, `credential` or `error` line is in the router log for 13:29:30-13:30:20Z (`docker logs --since … | grep -iE 'refresh|credential|token|resolv|error'`). The search-provider warnings are not related.

## Rotation of the refresh token: `unverified`

- The sealed blob changed: its md5 changed and its length went from 1363 to 1374. Every save re-seals with a new nonce and with the revision as AAD (`internal/store/connections.go:120-124`, `internal/auth/secret.go:85-91`). So a changed blob does not prove a changed refresh token. The 11-byte growth also covers other fields, such as `expires_at`.
- Telling whether the refresh token changed would need the old plaintext. It was overwritten in place, and opening it needs the KEK. So the rotation is `unverified`.
- To verify it next time, before a refresh: open both blobs with the KEK in a tool that prints only `sha256(refresh_token)[:8]` and its length.
- Observation: F11 suggested that Kanat's two connections hold the same Slack token (their `expires_at` 02:27:11Z and 02:27:12Z were 0.7 s apart). User-1 refreshed first. User-2 then refreshed 43 s later with its own stored refresh token, and that worked too. If the two refresh tokens were the same and Slack rotated it on user-1's refresh, Slack still took the old one. Both premises are `unverified`.
