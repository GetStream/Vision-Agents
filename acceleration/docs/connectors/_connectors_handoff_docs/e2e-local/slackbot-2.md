# Slack bot (slack_bot) on the channel bridge, run 2 (2026-10-08)

Router `$R` at `daa71764`. The container was started at 17:25:53Z and was not recreated in this run.
`PS` is `docker exec vision-agents-postgres-1 psql -U postgres -d model_router -Atc`. Slack `ts` values are written as `<ts>`.

## 1. Connection

`PS "select * from connector_connections where id='37f8486ac3891197e0b8a2eee20f74cf'"` (17:39Z, secret columns left out):
- `status connected`, `owner_type app`, `definition_revision 3`, `auth_scheme oauth2_code`, `connected_at 17:38:04.406Z`.
- `metadata = {"team_id": "T02RM6X6B"}`. That is the only captured field (`capture: team_id from $.team.id`, providers/slack_bot.yaml:49-53).
- `account_id = T02RM6X6B`. The identity is `[team_id]` (slack_bot.yaml:56).
- `granted_scopes = [chat:write, channels:history, im:history]`. `expires_at` is null, so the bot token does not expire.
- The consent: `POST …/authorizations` 201 at 17:37:58Z, launch GET+POST 200 at 17:38:00-01Z, callback 302 at 17:38:04.417Z (router log).

## 2. Slack events on the router

`docker logs vision-agents-router-1 | grep connectors/events`:

| time (Z) | status | what it was |
|---|---|---|
| 17:31:29.659 | 200, bytes=52 | Slack's url_verification (the challenge echo; 52 bytes of text) |
| 17:38:50.282 | 200, no body | the bot's `channel_join` in #kanat-test (Slack shows it at 13:38:49 EDT) |
| 17:40:52.822 | 200, no body | Justin's post from retry A (§5) |

- The signatures verified. A failed verify answers 401 and logs `refused an unverified connector event` (`internal/api/connector_events.go:299-304`). No 401 and no such line appears after the 17:25:53Z start.
- The `channel_join` was skipped because of `skip_if_present: $.event.subtype` (slack_bot.yaml, `messages.skip_if_present`). No thread was written for it, and the skip logs nothing.
- **Kanat's channel message and DM are not in Slack.** `slack_read_channel C0C8MKNUNBA` at 17:41Z shows no message from Kanat after the bot joined. `slack_read_channel U0C7Y7NAZ60` (the DM with the bot, D0C7PCB2UER) is empty. No other events arrived up to 17:43:10Z. Kanat's own message is therefore `unverified`: it was not sent.

## 3. Bridge

Justin's post (retry A, 17:40:52Z) is the one inbound message that went through. It is a person's post: his user token posted it through the `slack` MCP, so it carries no `bot_id`.

What the bridge did, from the DB (17:41Z):
- `channel_threads`: 1 row. `thread-2f66f9b9-5110-472b-8576-4cfc96b27cef`, `slack_bot`, unit `T02RM6X6B`, key `C0C8MKNUNBA:<ts>`, `connection_id 37f8486a…`, `thread_parts {"channel":"C0C8MKNUNBA","thread_ts":"<ts>"}`, `stream_app_pk` null, `created_at 17:40:52.816Z`.
- `channel_thread_messages`: 1 row, `inbound`, 17:40:52.819Z. **There is no `reply` row.**
- In Stream Chat (`POST /channels/agent/thread-2f66…/query`, 17:42Z), the channel has `agent_config_id 4e57c7be…` (e2e-slackbot) and `support_agent_id thread-2f66…`. It has one message, from `slack_bot-728be842…`, at 17:40:53.236Z, with the text «hello from a second Slack user (accelerate connectors e2e) \*Sent using\* Accelerate connectors test».

**Did the agent's reply go back to Slack? No.** `slack_read_thread C0C8MKNUNBA <ts>` → «No thread messages». No session joined on the thread channel: the log has only the e2e-user-3 session (17:40:50Z).

**Exact cause: the Stream app does not send `message.new` to this router.**
- An agent answers a thread channel only when Stream Chat delivers `message.new` to `POST /v1/chat/hooks/stream` (`internal/api/threadhooks.go:38-49`, route `internal/api/server.go:549-550`, `chat.MessageHookPath` `internal/chat/hooks.go:20`).
- The router log has **no** request to `/v1/chat/hooks/stream` after 17:40:53Z.
- `GET https://chat.stream-io-api.com/app` for the deployment's app (1257545, the `STREAM_API_KEY` in the container; 17:42Z) shows `webhook_url ""` and one event hook. That hook is `webhook`, enabled, at `https://eec4-24-8-28-137.ngrok-free.app/v1/phone/hooks/stream`, with 2 event types and no `message.new`. **No hook points at `https://game-mines-perfect-render.trycloudflare.com/v1/chat/hooks/stream`.**
- This is by design: the router points the hook itself only for a customer's pinned Stream app. «A pin of zero, or the deployment's own app, names no app of the customer's: the deployment's hooks are the operator's to point» (`internal/api/connector_provider_apps.go:446-457`). The operator command is `router phone hooks -url <public>` (`cmd/phone/main.go:500-575`).
- The other conditions are met:
  - The binding is fixed to this connection. Exactly 1 config, `4e57c7be… e2e-slackbot`, matches the bridge's query (`store/channel_threads.go:209-224`).
  - The bridge is enabled. It is wired when there is a connector resolver (`cmd/router/main.go:1039-1047`), and it wrote the thread.
  - The episode source exists: `slack_bot` → `store.EpisodeSlack` (`internal/channelbridge/bridge.go:83`).
  - The opt-out and sandbox gates do not apply. `slack_bot` has no `optOuts`, so `allowed` and `mayReply` return true at once (`internal/channelbridge/gate.go:19-22, 36-39`).
- Fix for the local test (not run, because it changes the shared Stream app 1257545): `docker exec vision-agents-router-1 router phone hooks -url https://game-mines-perfect-render.trycloudflare.com`. That it works is `unverified`. It also points the call hook at the tunnel.
- What happens to this thread after the hook is pointed is `unverified`. The inbound message is already claimed, and Stream does not resend an old `message.new`. A new message in the thread is needed.

## 4. Episode card

- `episodes`: 1 row, `ad1970e3…`, `source slack`, `thread_channel agent:thread-2f66…`, `card_message_id episode-ad1970e3…`, `status in_progress`, 17:40:52.820Z.
- `contact_map 14a9bd98…`: `kind slack`, `address T02RM6X6B:U0AUY8NBBL5` (Justin Lei, from the join message in #kanat-test), `agent_config_id 4e57c7be…`, `conversation_id agent:omni-51f0c8d1-6308-482a-bfa2-8f755f5a9e2e`.
- In Stream Chat (`query agent:omni-51f0c8d1…`), the card message is `episode-ad1970e3…` with the text «Episode in progress», at 17:40:53.538Z.
- **The card belongs to Justin, not Kanat.** No card for Kanat's Slack user exists, because no message from Kanat reached the bot (§2).

## 5. Retry A: e2e-user-3 (Justin)

`E2E_STATE=state-3.json slack.sh send` at 17:40:49Z → 202, response `88ca6c85…` (`turn-6a.out`, items in `items-6a.raw`):
```
17:40:50.04Z said        Post "hello from a second Slack user (accelerate connectors e2e)" to #kanat-test (channel C0C8MKNUNBA)
17:40:52.12Z answer      One moment, I am posting your message to #kanat-test.
17:40:52.12Z tool_call   slack__slack_send_message {"channel_id":"C0C8MKNUNBA","message":"hello from a second Slack user (accelerate connectors e2e)"}
17:40:52.36Z tool_result {"message_link":"https://getstream.slack.com/archives/C0C8MKNUNBA/p<ts>","message_context":{"message_ts":"<ts>","channel_id":"C0C8MKNUNBA"}}
17:40:54.18Z answer      The message has been posted: https://getstream.slack.com/archives/C0C8MKNUNBA/p<ts>
```
- Slack shows «Justin Lei: hello from a second Slack user (accelerate connectors e2e) *Sent using* Accelerate connectors test» at 13:40:52 EDT (`slack_read_channel`).
- Audit: `GET /connections/b7a36262…/invocations` lists the newest `slack_send_message` with `error_type` null.
- The session was reopened by the send (log `session joined` 17:40:50.039Z). No validate, grant or restart came before it.
