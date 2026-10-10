# Slack bot (slack_bot) as the customer's own provider app, run 1 (2026-10-08)

Router `$R` at `daa71764`, local compose; no recreate for this task.
The secrets are read from the repo `.env` (`/Users/kanat/Projects/stream/Vision-Agents/.env`; `$R/.env` is a symlink to it) inside the shell, and never printed.
Lengths only: CLIENT_ID 25, CLIENT_SECRET 32, SIGNING_SECRET 32, all printable ASCII.
**App id: `A0C7N6LNZMH`** (not secret).

## 1. Customer provider app registered

- Read `internal/api/oauth_clients.go`:
  - The PUT stores `registration: customer` (`:183-190`).
  - It seals `client_secret` with AAD (customer, connector) (`:203-209`).
  - It seals `signing_secret` with AAD (customer, connector, provider_app_id) (`:211-218`).
  - It refuses a signing secret unless the manifest has `channel.verifier.secret: provider_app` (`:310-314`).
- Read `providers/slack_bot.yaml`:
  - It is revision 3, with `client.registration [operator, customer, managed]` and `auth_method client_secret_post`.
  - Its scopes are `chat:write, channels:history, im:history` (separator `,`).
  - Its verifier is `hmac_header`, `secret: provider_app`, `v0:{timestamp}:{body}`, `max_age 5m`.
  - Its `subscriptions` are `[message.channels, message.im, tokens_revoked]`.
- `PUT /v1/agents/connectors/slack_bot/oauth-client` with `{client_id, client_secret, provider_app_id, signing_secret}` at 17:21:51Z → **201**.
  The body is `{connector_id: slack_bot, registration: customer, client_id: <set, len 25>, provider_app_id: A0C7N6LNZMH, created_at, updated_at}`. It has no secret field (`slackbot-put.out`).
- A GET on the record has no endpoint: `GET /v1/agents/connectors/slack_bot/oauth-client` → **405**. Only PUT and DELETE are registered (`oauth_clients.go:116-160`), so the record was checked in the local DB instead.
  `select connector_id, registration, client_id<>'' , auth_method, secret_sealed is not null, kek_version, provider_app_id, signing_secret_sealed is not null, signing_kek_version, stream_app_pk is not null from connector_oauth_clients where connector_id='slack_bot'`
  → `slack_bot|customer|t||t|1|A0C7N6LNZMH|t|1|f`.
  - `auth_method` is empty, so the manifest's `client_secret_post` applies.
  - `stream_app_pk` is not pinned. That is expected locally (`s.stream` nil, `:194-200`; `unverified`).

## 2. Event Subscriptions request URL and url_verification

**Request URL to paste in Slack:**
`https://game-mines-perfect-render.trycloudflare.com/v1/connectors/events/slack_bot/A0C7N6LNZMH`
(route `providerAppEventsPath = "/v1/connectors/events/"`, `internal/api/connector_events.go:31`; handler `receiveProviderAppEvent` `:160`).

Probe (`slackbot-probe.out`, 17:22:15Z): the body was `{"token":"e2e-legacy-token","challenge":"e2e-challenge-…","type":"url_verification"}`. The signature was `v0=` + hex HMAC-SHA256 over `v0:{ts}:{body}`, keyed with the signing secret and computed in the shell.

| Request | localhost:8080 | public router tunnel |
|---|---|---|
| signed, ts now | **200 `text/plain`, body = the challenge** | **200 `text/plain`, body = the challenge** |
| no signature headers | 401 `unauthenticated` «the request is not signed by the provider» | 401, same |
| wrong signature (`v0=000…`) | 401, same | 401, same |
| correctly signed, ts −600 s | 401 | 401 |

Router log lines `refused an unverified connector event`, 17:22:15.828Z–17:22:16.969Z:
- reason `hmacheader: the request is not signed by the provider` for the unsigned and wrong-signature requests;
- reason `hmacheader: the request's signed timestamp is too far from now` for the stale one.

**Bot events to subscribe** (Event Subscriptions → Subscribe to bot events): `message.channels`, `message.im`, `tokens_revoked`.
`app_uninstalled` is delivered without a subscription (slack_bot.yaml, comment above `subscriptions`).

## 3. Install through our consent (fixed, app-owned binding)

What is prepared:
- An app-owned connection `37f8486ac3891197e0b8a2eee20f74cf`: `POST /v1/agents/connections {"connector_id":"slack_bot","owner":{"type":"app"}}` → 201, `status: pending`, `definition_revision 3` (17:23:08Z).
- Config `e2e-slackbot` = `4e57c7be723620696912ee48c826bf52`: text, `gemini/gemini-3.8-flash`, binding `slack_bot` with `connection: {type: fixed, connection_id: 37f8486a…}` and `tools: []` (201, 17:23:13Z, `config-slackbot-create.out`).
  The bridge routes an inbound message to the **one** config that binds the connection of the event's team (`internal/channelbridge/bridge.go:382-400`), so no other config may bind it.
- Helper: `?state=state-slackbot` starts a new consent on every load with `POST /v1/agents/connections/37f8486a…/authorizations` and serves the Connect button (`state-slackbot.json`).
  Checked at 17:24:15Z through the then-public helper: 200, consent `7e76cb75…` for connection `37f8486a…`, `launch_url` on the router tunnel, and a handoff token (redacted, 43 chars). That attempt was not used and expires at 17:34:15Z.

What Kanat does in the bot app's settings (api.slack.com):
- **OAuth & Permissions → Redirect URLs:** `https://game-mines-perfect-render.trycloudflare.com/v1/agents/connectors/oauth/callback` (`ConnectorCallbackPath`, `authorizations.go:37`).
  Whether it is already there is `unverified`.
- **Event Subscriptions:** the request URL and the three bot events above.
- **After IT approves,** open `http://localhost:3091/?state=state-slackbot` and click Connect. The helper tunnel was removed at 17:25:44Z, and `DASHBOARD_BASE_URL` is `http://localhost:3091` again. The popup goes to `https://slack.com/oauth/v2/authorize` with the bot scopes. The router exchanges the code at `oauth.v2.access` with the customer client, and captures `team_id` (`$.team.id`) as the identity.
- The install needs the approval first. While it is pending, Slack's authorize page asks to *request* the install rather than grant it. That is `unverified`: it was not opened.

Then to test inbound: invite the bot to #kanat-test and post a message. Expected: the bridge writes it into a thread channel, `e2e-slackbot` answers, and the reply goes back with `chat.postMessage` in the thread.
Whether the local router's Stream Chat client can do that is `unverified`.
