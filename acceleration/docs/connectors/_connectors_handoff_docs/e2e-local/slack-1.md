# Slack connector e2e, step 1: consent link (2026-10-08, local router)

Router: `vision-agents-router-1`, image built from `$R` = origin/accelerate at `0bc2fe89`.
Public URL: `https://game-mines-perfect-render.trycloudflare.com`.
Secrets: none below. The key secret, tokens and handoff tokens are redacted.
Rerun anything with `$S/e2e/slack.sh <cmd>`. `lib.sh` reads the key from `$S/e2e-key.txt`.

## Ids

| What | Value | Source |
|---|---|---|
| organization / app | `8c5cbdf0…a8b0` / `4dc634af3f0edb5876243c3c04ef4279` | `$S/e2e-key.txt` lines 3-4 |
| api key id | `vak_live_2b413e20a568116871656854` (secret not shown) | `$S/e2e-key.txt` line 9 |
| config `e2e-slack` | `1b21facf50e34e8b6a5180cea1406ebd` | `config-create.out` |
| session (text) | `01a11bde-0c6f-75dc-a86c-470bc7c3a820` | `session-create.out` |
| conversation / agent_id | `agent:support-c9f3fb76-ce78-40dd-ae78-fb0f4b95affc` / `9fe7e066d58c33663898f6d098ea785f` | `session-create.out` |
| end user | `e2e-user-1` (header `X-Stream-User-Id`) | |
| connection (pending) | `9c3d34a08dcdcceb682ace9b9cc09a77`, owner user `e2e-user-1`, status `pending` | `slack.sh connection` at 14:18Z |
| first consent (expired 14:25:08Z) | `b62f3ec8b37c12da6db971c0cda77eb4` | `messages.raw` |
| consent after the restart | `7bb64824f978cae74fe64b1496b8b626`, expires 14:27:51Z | `slack.sh consent` |

## 1. How a server-side caller authenticates (api_key mode)

The caller sends four headers:

| Header | Value | Code |
|---|---|---|
| `X-Api-Key` | the key id (`vak_live_…`) | `internal/auth/auth.go:67`, `:459` |
| `Authorization: Bearer <jwt>` | HS256 JWT signed with the key's secret. Claims `{"server":true,"exp":…}` and no `user_id` | `auth.go:350-355` (HS256, exp required), `:441-454` (`serverToken`) |
| `Stream-Auth-Type: server` | marks a server call. Both this header and the claim must agree | `auth.go:71`, `:366` |
| `X-Stream-User-Id: e2e-user-1` | the end user the backend acts for. Read only for a server caller | `auth.go:89`, `:381-389` |

`lib.sh` builds the JWT with `openssl` (`server_token`).

```
GET /v1/agents/connectors           -> HTTP 200, 18 ids incl. slack, slack_bot
GET /v1/agents/connectors/slack     -> HTTP 200
  {"id":"slack","revision":4,"schemes":["oauth2_code"],
   "client":{"registration":["operator"],"auth_method":"client_secret_post"}, scopes: 29}
```

The API does not report whether an operator client is set. The router reads it from
`<client.env>_MCP_CLIENT_ID/_SECRET` (`internal/connectors/schemes/oauth2code/client.go:53-61`).
The manifest sets `env: SLACK` (`internal/connectors/providers/slack.yaml`). Container check, names only:
`SLACK_MCP_CLIENT_ID set len=25`, `SLACK_MCP_CLIENT_SECRET set len=32`.

## 2. Config with a per-user binding

A binding is `AgentConnectorBinding` (`internal/api/configs.go:1628-1637`).
- `connection.type: session` means the session's verified user picks their own connection (`configs.go:466-470`).
- `tools` is required. Each grant needs `name` and a 64-hex `schema_digest` (`configs.go:1718-1721`).
  An empty list is accepted and grants nothing.

LLM keys set in the container (names only): `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GEMINI_API_KEY`,
`GOOGLE_API_KEY`, `XAI_API_KEY`, `BASETEN_API_KEY`. Chosen: `openai/gpt-5.6-sol`.

```
POST /v1/agents/configs   (body: $S/e2e/config.json)
{"name":"e2e-slack","mode":"text","llm":"openai/gpt-5.6-sol","instructions":"…",
 "connectors":[{"name":"slack","connector_id":"slack","connection":{"type":"session"},"tools":[]}]}
-> HTTP 201 {"id":"1b21facf50e34e8b6a5180cea1406ebd", …,
   "connectors":[{"name":"slack","connector_id":"slack","connection":{"type":"session"},"tools":[],"required":false}]}
```

## 3. Text session as `e2e-user-1`

`cmd/chat` does not fit. It calls the LLM router in-process, with no configs, sessions or connectors
(`cmd/chat/main.go:1-50`). This test uses the HTTP API instead.

```
POST /v1/agents/sessions  X-Stream-User-Id: e2e-user-1
{"config_id":"1b21facf50e34e8b6a5180cea1406ebd","text":true,"title":"e2e slack 1"}
-> HTTP 201 {"id":"01a11bde-0c6f-75dc-a86c-470bc7c3a820","conversation_id":"agent:support-c9f3…affc","state":"live",…}
log 14:14:59.902Z WARN "opening the session without a connector" connector=slack reason=no_selection

POST /v1/agents/sessions/01a11bde-…/responses  X-Stream-User-Id: e2e-user-1
{"text":"Post the message \"hello from accelerate connectors e2e\" to the #general channel in Slack","command_id":"e2e-slack-1-msg-1"}
-> HTTP 202 {"id":"8a8a4a1f6c305a4a78812bbdae4bf22e","status":"running"}
```

`no_selection` on an optional `session` binding in a persistent text chat makes the binding a
login. The model then gets the tools `slack__list_tools` and `slack__call_tool`
(`internal/session/connector_login.go:128-170`).

## 4. The consent

Turn items (`GET /v1/agents/sessions/{id}/responses/items`):

```
said        Post the message "hello from accelerate connectors e2e" to the #general channel in Slack
answer      One moment, I'm checking the available Slack actions.
tool_call   slack__list_tools
tool_result {"status":"authorization_required","message":"The user has not connected Slack. They have been shown a button …"}
answer      Please press the **Connect Slack** button. Once connected, I'll post the message to **#general**.
```

The button is on the assistant message in the conversation:

```
GET /v1/agents/conversations/agent%3Asupport-c9f3…/messages?agent_id=9fe7e066…  -> HTTP 200
{"type":"connector_authorization","name":"slack","connector_id":"slack",
 "connection_id":"9c3d34a08dcdcceb682ace9b9cc09a77","authorization_id":"b62f3ec8b37c12da6db971c0cda77eb4",
 "launch_url":"https://game-mines-perfect-render.trycloudflare.com/v1/agents/connectors/oauth/launch/b62f3ec8b37c12da6db971c0cda77eb4",
 "handoff_token":"<redacted>","expires_at":"2026-10-08T14:25:08.995062Z","title":"Connect Slack"}
```

`launch_url` is on the public tunnel host. It is built from `ROUTER_PUBLIC_URL` (`internal/api/authorizations.go:385`).

### You cannot open launch_url directly

The launch page takes the handoff token only by `postMessage` from `window.opener`, and only when the
opener's origin is `DASHBOARD_BASE_URL` (`authorizations.go:431-452`). With no opener, the page shows
"Start from the dashboard, so this browser can be checked."
- `DASHBOARD_BASE_URL` was unset, so the default `http://localhost:3000` applied (`internal/config/config.go:292`).
  Proof: `curl …/oauth/launch/x | grep dashboardOrigin` returned `"http://localhost:3000"`.
- Port 3000 is already in use: `lsof -iTCP:3000` showed `Obsidian … 127.0.0.1:3000 (LISTEN)`.
  No dashboard can run there.

What I changed (scratchpad only, no repo code):
1. Added `DASHBOARD_BASE_URL: http://localhost:3091` to `$S/compose.e2e.yaml` (original in `compose.e2e.yaml.orig`).
2. Recreated the router with the compose command from the task.
   The launch page now shows `dashboardOrigin = "http://localhost:3091"`.
   After consent, the callback also sends the popup to that origin (`authorizations.go:622-632`).
3. Started `$S/e2e/consent_helper.py` on `http://localhost:3091`. It stands in for the dashboard's popup:
   - It reads the newest live `connector_authorization` from the conversation.
   - It shows a **Connect Slack** button. The button opens `launch_url` in a popup and posts the handoff token on `va.connector.oauth.ready`.
   - If the consent expired (10 min), or with `?fresh=1`, it first sends the prompt again. The agent then begins a new consent in the same live session, so the consent still returns to that session.
4. The restart dropped the in-memory session. `?fresh=1` reopened it under the same id (log 14:17:49Z `session joined session=01a11bde-…`).
   The same call made consent `7bb64824…`, still on connection `9c3d34a0…`.

## How to continue

Kanat: open **http://localhost:3091/** (use `localhost`, not `127.0.0.1`), press **Connect Slack** and approve in Slack.
The popup ends on `http://localhost:3091/?connection_id=…&status=connected`.

After a `connected` status, the router tells the session "Slack is connected now. Carry on …"
(`internal/session/connector_login.go:434-456`). Then run:

```
$S/e2e/slack.sh connection    # expect "status":"connected"
$S/e2e/slack.sh items         # see what the agent did after the hand-back
$S/e2e/slack.sh continue      # tools -> grant slack_send_message -> stop -> send the prompt -> items
```

Why `continue` is needed: the binding grants no tools (`tools: []`). A grant needs the tool's
`schema_digest`, and the router can read it only from a connected account
(`GET /v1/agents/connections/{id}/tools`, today `{"tools":[]}`).
- The opened binding offers only granted tools (`internal/session/connector_tools.go:248-287`).
  So after the hand-back, `slack__list_tools` is expected to list nothing (`unverified`).
- `continue` PATCHes the grant, stops the session, then sends the prompt again. The session reopens on the
  updated config. It picks the connection the login recorded (`connector_login.go:272-274`, `chose`).
- The tool name `slack_send_message` is `unverified`. Check it against `slack.sh tools` first,
  and use `slack.sh grant <names…>` if Slack names it differently.

## Issues

1. The consent link cannot be opened as a plain link. It needs a `window.opener` at `DASHBOARD_BASE_URL`
   (`authorizations.go:431-452`). This is by design (RFC 9700 browser binding).
   A local e2e needs a dashboard, or the helper above, on a free port.
2. A `session` binding cannot grant tools before someone has connected. `tools[].schema_digest` is required
   (`configs.go:1720`) and comes only from a connected connection's tool list.
   So a config author needs one connected account before the grants can be written. Design note, not a bug.
3. Not a blocker: `search-fast` fails because `EXA_API_KEY`, `PERPLEXITY_API_KEY` and `TAVILY_API_KEY` are missing
   (log 14:14:59.913Z, "this agent cannot find out what is true today").

## Files

`lib.sh` (auth helpers), `slack.sh` (all commands), `config.json`, `state.json` (ids),
`consent_helper.py` (+ `.pid`, `.log`), `*.out` and `messages.raw` (responses, tokens redacted).
