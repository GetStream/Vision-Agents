# Linear connector e2e via DCR, run 1 (2026-10-08)

Router `$R` at `daa71764`. Started while slack-5 waits; no router recreate was needed.

## Manifest

`GET /v1/agents/connectors/linear` → `revision 2`, `schemes [oauth2_code]`, `client.registration [dcr]`, `scopes [read, write]`.
`providers/linear.yaml` has `mcp: https://mcp.linear.app/mcp`. Its authorize and token endpoints come from discovery (RFC 9728/8414).

## Tool names

- Neither the manifest nor https://linear.app/docs/mcp names the tools. `curl` at 17:20Z returned 200 and 777 KB; grep found no `list_*`/`get_*` names.
- The names granted, `list_issues` and `get_issue`, come from the claude.ai Linear connector's tool names in this agent session (`mcp__claude_ai_Linear__list_issues`, `…__get_issue`). That this is the same server, `mcp.linear.app`, is `unverified`.
- What a wrong name would do: a name the provider does not list is not pinned and not offered (`internal/session/connector_tools.go:323-325`, `:366-368`). In that case the fix is to grant after the connection, from `GET …/tools`.

## Setup (done)

- Config `e2e-linear` = `34fb827cc6b0d0566ab95959fcb4c9b5` (POST 201 at 17:20:04Z, `config-linear-create.out`).
  It is text mode with `gemini/gemini-3.8-flash`. Its binding `linear` is `selection: session` and grants `[{name: list_issues}, {name: get_issue}]` with no digest.
- Session `01a11c87-a9bf-7d17-b38a-8cfbbcf4aff6` for `e2e-user-1`, created 17:20:15Z (`state-linear.json`).
- Turn at 17:20:22Z: `linear__list_tools` → `authorization_required` (`turn-linear-a.out`).
  Consent `bd24c052…`, connection `524c6685328fbf3533b29779d4e6681a`, expires 17:30:24Z.
- `connector_oauth_clients` has 0 linear rows, and still has none after the connect. A DCR client is not kept there; see "DCR client" below.
- Helper: `https://gordon-search-chairman-advisor.trycloudflare.com/?state=state-linear` → 200, «Consent bd24c052…», Connect button (17:20:48Z).
  Through localhost:3091 it did not work at the time: `DASHBOARD_BASE_URL` was the helper-tunnel origin, so the launch page would have refused a handoff from localhost.

## Results (Kanat logged in at about 17:23:43Z)

The launch page loaded `GET /v1/agents/connectors/oauth/launch/bd24c052…` (200, 17:23:43.016Z), then the handoff POST returned 200. The callback returned 302 at 17:23:50.092Z, and the helper logged `?connection_id=524c6685…&status=connected`.

### Connection

`GET /v1/agents/connections/524c6685…` as e2e-user-1:
- `status: connected`, `definition_revision 2`, `definition_status current`, `granted_scopes [read, write]`.
- `metadata {}`: the manifest captures no identity.
- `expires_at 2026-10-09T17:18:48.834167Z`. That is the 17:23:48.831Z exchange plus 23 h 55 min, 5 min short of the manifest's `access_ttl: 24h`. Whether Linear sent `expires_in` 86100 or the router takes off a margin is `unverified`.

### DCR client

- **No row exists** for a DCR client: `connector_oauth_clients` (where `registration` is customer, managed or operator) has no linear row.
- A client this scheme registers is held only inside the sealed attempt and then the sealed credential. Its secret is kept "only for a client this scheme registered (dcr), which has nowhere else to live" (`internal/connectors/schemes/oauth2code/client.go:64-74`).
- The client is picked in `pickClient` (`client.go:94-130`). linear lists only `dcr`, so the client must be one `register` (`:202`) made by RFC 7591 at the discovered `registration_endpoint`. That is derived from the code: the client id was not read, because that would mean unsealing.
- DB evidence:
  - `connector_authorization_attempts` row `bd24c052…`, `kind consent`, `attempt_sealed` 1231 bytes, created 17:20:24.025Z, `consumed_at` 17:23:48.831Z.
  - `connector_connections` row `524c6685…`, `credentials_sealed` 730 bytes, `credentials_kek_version 1`, `connected_at` 17:23:49.421Z.
- When it registered: inferred from timing, at consent creation. `linear__list_tools` was called at 17:20:23.529Z, and the attempt row was created 0.5 s later, at 17:20:24.025Z. That is `unverified` against a log line, since the router logs no DCR line.
- So the API and DB cannot show "client id present" without unsealing. See F16.

### Carry-on turn (`items-linear.out`)

No validate, grant or restart came before this turn:
```
17:23:50.09Z said        Linear is connected now. Carry on with what I asked for before you needed it.
17:23:52.58Z tool_result linear__list_tools → ["get_issue","list_issues"]
17:23:54.71Z tool_call   linear__call_tool {"tool":"list_issues","arguments":{"orderBy":"updatedAt","limit":3,"assignee":"me"}}
17:23:56.75Z answer      Here are your 3 most recently updated Linear issues:
                         1. AI-971 – Review follow-ups for #822, #823, #826 (Slack capture, broken revisions, tool pins)
                         2. AI-970 – Connectors E2E follow-ups: tools list, error kinds, consent messages, broken_revisions in API
                         3. AI-969 – OpenAI tool calls fill every optional argument with empty values (Slack invalid_thread_ts)
```
The guessed names `list_issues` and `get_issue` were right. Both were offered, so no regrant or rerun was needed.

### Pins and audit

- `connector_tool_pins` for `524c6685`:
  - `get_issue` `d4be51f4f70a…`
  - `list_issues` `80a3765d22ff…`
  - both with `connected_at` 17:23:49.421Z and `pinned_at` 17:23:49.889Z.
- `GET /connections/524c6685…/invocations` → 1 row: `tool list_issues`, 17:23:54.711Z, 224 ms, no error, session `01a11c87`, binding `linear`.
- F9 holds here too: the connection's `tools_checked_at` is NULL, so `GET …/tools` is empty while the session listed live.

## Verdict

- DCR, consent and the automatic carry-on all work end to end for Linear with by-name grants.
- The DCR client cannot be seen without unsealing (F16).
