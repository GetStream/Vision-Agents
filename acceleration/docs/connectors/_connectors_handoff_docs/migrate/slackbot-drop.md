# Staging slack_bot mention with no reply (2026-10-10 03:18:18Z)

Read-only investigation. Code at Vision-Agents 599298c6 (`git show 599298c6:...`). Raw artefacts in `migrate/slackbot-drop/`.

## Verdict
The router did not drop the message. The bridge took it, wrote it into the thread channel in Stream app 1257545, and opened the episode. Stream then sent the `message.new` hook to the **local router through ngrok**, not to staging. App 1257545's only message hook is `https://carol-elliptic-uncloak.ngrok-free.dev/v1/chat/hooks/stream`. The local router has no config a54fb800, so it logged ERROR and answered nothing. Staging never got a hook request, so it never ran a turn.

```
Slack -> staging /v1/connectors/events/slack_bot/A0C7N6LNZMH (200, 90 ms)
      -> channelbridge.Deliver: take OK -> thread linked, inbound claimed, episode opened
      -> write (async) -> Stream app 1257545, channel agent:thread-f9ac9bf6-...
Stream app 1257545 message.new hook -> ngrok -> LOCAL router (docker vision-agents-router-1)
      -> messagehooks.go:269 ERROR "config nobody in its app holds" -> no turn, no reply
staging: 0 requests on /v1/chat/hooks/stream* in the last hour
```

## Evidence
| Claim | Source |
|---|---|
| Staging served the event once, at 200 in 90 ms, with nothing logged after it | `kubectl ... logs chat-accelerate-staging-79c77db9f6-d56wq -c router --since=1h` -> `slackbot-drop/router-1h.log` line 1573, req 2043e2e1 |
| Staging got 0 message hooks in the last hour | `grep -c chat/hooks router-1h.log` = 0 |
| The bridge linked the thread | psql RO 03:21Z: `channel_threads` thread-f9ac9bf6-4276-410e-b72b-eadba0d18426, key `C0C8MKNUNBA:1791602298.571799`, connection 53815210, stream_app_pk 1257545, created 03:18:18.911Z (`slackbot-drop/q1.out`) |
| It claimed the message | `channel_thread_messages` inbound 1791602298.571799 at 03:18:18.973Z. There is no `turn` row and no `reply` row |
| It opened the episode | `episodes` f7bad8fc, source slack, thread_channel agent:thread-f9ac9bf6..., in_progress, 03:18:18.985Z |
| The write into Stream succeeded | no `could not write an inbound message` ERROR on staging (bridge.go:588), and Stream delivered a message.new for that channel (next row) |
| The hook went to the local router | `docker logs vision-agents-router-1 --since 03:18:00Z`: 03:18:19.336Z ERROR `an arriving message's channel names a config nobody in its app holds channel=thread-f9ac9bf6-... config=a54fb800... stream_app=1257545`, then POST /v1/chat/hooks/stream 200 (req 62dd862a) (`slackbot-drop/local-router-0318.log`) |
| App 1257545's hooks | read-only GET /app with the .env key, app id 1257545 «Video Demo App» (`slackbot-drop/gethooks.py`, `hooks.out`): message.new -> `https://carol-elliptic-uncloak.ngrok-free.dev/v1/chat/hooks/stream`; call hooks -> carol-elliptic ngrok and `eec4-24-8-28-137.ngrok-free.app`. No staging URL |
| The staging hook route passes the gateway | `curl -X POST https://accelerate.gcp.stream-io-api.com/v1/chat/hooks/stream/1257545 -d '{}'` -> 401 router envelope «that is not a message event from Stream» (03:23Z) |
| Staging runs per-app tenancy | deploy env `ROUTER_STREAM_TENANCY=app`, `ROUTER_CONNECTORS_ENABLED=true`, `ROUTER_PUBLIC_URL=https://accelerate.gcp.stream-io-api.com` |
| Local working run (00:11Z) | e2e-d3 row 6 and `e2e-d3/router-since-0011.log`: events 200 at 00:11:10.257Z, then `/v1/chat/hooks/stream` 200 at 00:11:10.899Z, then the session joined. The local run worked only because the hook points at ngrok |

## Why no router said anything
- `setConnectorOAuthClient` (`internal/api/oauth_clients.go:209-291`) stores a customer's own client with `provider_app_id` and a signing secret, which makes it a provider app, and pins the Stream app (line 246). It never calls `pointMessageHook`. Only the managed and operator provider-app PUTs call it (`connector_provider_apps.go:313,406`; function at :452). Kanat set the slack_bot client through this PUT at 02:57:07Z, so no hook was pointed at staging.
- `warnWithoutMessageHook` (`cmd/router/stream.go:124`) returns early when `clients.PerApp()` (line 126). Staging is in app tenancy, so the startup WARN never runs. The staging startup log has only 2 WARNs (trusted_proxies, auth.mode).

## Every return without a reply on the inbound path (599298c6)
| # | Where | Condition | Level | Staging 03:18Z |
|---|---|---|---|---|
| 1 | connector_events.go:165 | store, resolver or secrets is nil (connectors off) | none, 404 | not hit (200; ROUTER_CONNECTORS_ENABLED=true) |
| 2 | :171 | no provider app record for slack_bot/A0C7N6LNZMH (customer pin) | none, 404 | not hit: oauth client row for 1257545, registration customer, provider_app A0C7N6LNZMH, has_signing_secret t |
| 3 | :180 / :189 | no definition, no channel, or the verifier is not provider_app | none, 404 | not hit (200) |
| 4 | :287 | verifier not registered | WARN | not hit |
| 5 | :294 | bad signature or stale | INFO, 401 | not hit (200) |
| 6 | :300 | url_verification challenge | none, 200 | not this event (03:00:35Z was the challenge) |
| 7 | :305 | `skip_if_present` (`$.event.app_id`, F67) and other skipped rules | **DEBUG** | not hit: the message was taken (thread row exists) |
| 8 | :324 | event carries 0 messages (an event type the manifest does not read) | **no log at all** | not hit |
| 9 | bridge.go:411 | app has no customer, or no provider unit | INFO | not hit |
| 10 | bridge.go:417 | no app connection for team T02RM6X6B | INFO | not hit: 53815210 is owner app, account T02RM6X6B, connected |
| 11 | bridge.go:429 | `len(configs) != 1` | WARN | not hit: exactly 1 (a54fb800 e2e-connectors, live, not deleted, the only config naming 53815210) |
| 12 | bridge.go:443-455 | not addressed to the bot on a thread nobody linked (connection metadata bot_user_id) | **DEBUG** | not hit |
| 13 | bridge.go:476 | retried: dedupe claim | **DEBUG** | not hit (claim fresh) |
| 14 | bridge.go:268 / gate.go | keyword or sandbox gate | INFO | not applicable: slack_bot has no optOuts (bridge.go:85) |
| 15 | bridge.go:588 | Stream write fails | ERROR | not hit |
| 16 | after the write | Stream's message.new hook does not reach this router | **nothing on this router** | **HIT** |
| 17 | messagehooks.go / reply | hook, turn and reply path (Redis turn holder, bot not in channel -> `not_in_channel` refusal, bridge.go:713 logged at ERROR via Reply) | ERROR when it fails | not reached: no hook arrived |

Bot not in channel: `unverified` on staging, because the reply was never sent. Locally the bot replied in #kanat-test at 00:11Z, so it is in the channel (e2e-d3 row 6). Redis: `unverified`, not reached.

## Fix
1. Data step (Kanat approves; it changes app 1257545's shared hooks, the «Stream Public Demos» org): add a message.new hook to `https://accelerate.gcp.stream-io-api.com/v1/chat/hooks/stream/1257545`. Do it in the Stream dashboard (Webhooks, message.new only), or run `go run ./cmd/phone hooks -url https://accelerate.gcp.stream-io-api.com -app 1257545` from `acceleration/` with app 1257545's keys. That command also adds a staging *call* hook. Keep the ngrok hook only while the local router is still needed: both routers then get every message, and each logs ERROR (messagehooks.go:269) for the other's channels. `-remove` with `-app` appends `/1257545`, so it does not match the ngrok hooks, which have no segment. Stream may refuse the update because of the dead `eec4-24-8-28-137` call hook (AI-990 F22): `unverified`.
2. Code fix (ticket scope): `setConnectorOAuthClient` calls `s.pointMessageHook(ctx, customerID, record.StreamAppPK)` after the put when `ProviderAppID != ""` and the manifest has a channel, the same way the provider-app PUTs do, with a test in the connector_provider_apps style. Also make the app-tenancy gap visible: at the PUT (or on the first bridged write), check `chat.DeliversMessagesTo(public+MessageHookPath+"/"+pin)` and WARN when it is false. `warnWithoutMessageHook` skips app tenancy.

## Debug -> Info
This drop was not a Debug line: no router logs anything for #16. The F67 Debug (`connector_events.go:306`, skip_if_present) is a separate issue and was not hit here. AI-1053 is the connector `channel` flag (F68), not a log level, so this is not the same issue. Suggested separate ticket: raise `connector_events.go:306` (skipped rule) and `bridge.go:452` (not addressed) to INFO, and log at INFO when an event carries 0 messages (:324). These are the silent ways a mention vanishes.

## Cleanup
- The psql pod `kanat-slackdrop-psql-ro` was deleted at about 03:23Z. Nothing was written to staging, Slack or Stream. The hooks read was a GET. The curl probe was refused (401).
