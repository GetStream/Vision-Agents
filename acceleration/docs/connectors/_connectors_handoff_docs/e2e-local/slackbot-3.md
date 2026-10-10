# Slack bot (slack_bot) through ngrok, run 3 (2026-10-09)

The router was recreated by the coordinator; the container started at 13:14:01.7Z. `ROUTER_PUBLIC_URL` is `https://carol-elliptic-uncloak.ngrok-free.dev`.
The router log was captured to `router-sb3.log` (`docker logs -f --since 13:28:00Z`) from 13:30:0xZ to 13:40Z, and watched until 13:38:50Z.
`PS` is `docker exec vision-agents-postgres-1 psql -U postgres -d model_router -Atc`. Slack `ts` values are written as `<ts>`.

Note: `channel_threads` also holds 6 threads from 2026-10-08 (17:40Z to 19:19Z), from runs that are not in this file. Only the rows created at 13:30Z or later belong to this run.

## 1. Stream hooks (fixes F19)

`GET https://chat.stream-io-api.com/app` (app 1257545) at 13:30Z shows 3 enabled `webhook` hooks:
- `https://carol-elliptic-uncloak.ngrok-free.dev/v1/phone/hooks/stream`, with `call.session_started` and `call.session_ended`;
- `https://eec4-24-8-28-137.ngrok-free.app/v1/phone/hooks/stream`, the same two types. This is the dead hook that F22 is about, still there;
- `https://carol-elliptic-uncloak.ngrok-free.dev/v1/chat/hooks/stream`, with `message.new`.

The router log has `POST /v1/chat/hooks/stream 200` from 13:29:32Z on, so message hooks now reach the router.

## 2. Kanat's channel message

Router log (`router-sb3.log`):
```
13:28:12.627Z POST /v1/connectors/events/slack_bot/A0C7N6LNZMH 200 bytes=52   url_verification (challenge echo)
13:30:37.936Z POST /v1/connectors/events/slack_bot/A0C7N6LNZMH 200            Kanat's «Hello» in #kanat-test
13:30:38.412Z POST /v1/chat/hooks/stream 200                                  message.new in the thread channel
13:30:38.490Z session joined session=01a120db-cdd0-7dc1-8c02-e888db498a77
13:30:38.658Z ERROR nobody could answer an arriving message channel=omni-16cfde16-… error="dispatch: no worker is waiting for a call"   (F23)
13:30:39.936Z model call timing … gemini-3.8-flash success=true
13:30:40.596Z POST /v1/connectors/events/slack_bot/A0C7N6LNZMH 200            the bot's own reply, skipped by bot_id
```
- No `refused an unverified connector event` line appears, so every event verified (`connector_events.go:299-304`).

DB (`PS`, 13:31Z):
- `channel_threads`: `thread-177fcfff-a519-4e96-a5c5-78f64cc68399`, unit `T02RM6X6B`, key `C0C8MKNUNBA:<ts>`, `connection_id 37f8486a…`, `thread_parts {"channel":"C0C8MKNUNBA","thread_ts":"<ts>"}`, created 13:30:37.928Z.
- `channel_thread_messages` for it:
  - `inbound <ts>` 13:30:37.931Z;
  - `turn 85b0d6b1…` 13:30:38.412Z;
  - **`reply 7d8c02de…` 13:30:40.083Z**.
  The reply row is claimed before the send and released again if the send fails (`internal/channelbridge/bridge.go:328-363`). So a row that stays means Slack accepted the post with `$.ok == true` (`slack_bot.yaml` `reply.accepted`, `bridge.go:611-622`).
- Stream Chat `agent:thread-177fcfff…`:
  - `85b0d6b1…` from `slack_bot-b5a777f2…`: «Hello» (13:30:38.295Z);
  - `7d8c02de…` from the thread agent: **«Hello! How can I help you today?»** (13:30:38.669Z).
- Slack (`slack_read_thread C0C8MKNUNBA <ts>`): the parent is «Hello» from Kanat Kiialbaev, and the one reply is from **accelerate-bot-test**: «Hello! How can I help you today?».
- Episode `46b7e83c…`: `source slack`, `in_progress`, 13:30:37.934Z, contact `T02RM6X6B:U034NG4FPNG` (Kanat).
  The card is `episode-46b7e83c…` «Episode in progress» in `agent:omni-16cfde16-e22e-4388-ac86-8b7870d6df63`. Its fields are `source slack`, `status in_progress`, `thread_channel agent:thread-177fcfff…`.
  **This is the episode card for Kanat's Slack user.** The run-2 gap is closed.

## 3. Kanat's DM

Router log:
```
13:32:15.208Z POST /v1/connectors/events/slack_bot/A0C7N6LNZMH 200            Kanat's DM «Blah»
13:32:15.904Z POST /v1/chat/hooks/stream 200
13:32:16.009Z session joined session=01a120dd-4aa3-7e0e-ae42-d9325a7bd6ef
13:32:16.446Z ERROR nobody could answer an arriving message channel=omni-16cfde16-… (F23 again, with the new card)
13:32:18.652Z POST /v1/connectors/events/slack_bot/A0C7N6LNZMH 200            the bot's own reply, skipped
```
DB:
- `channel_threads`: `thread-74ff9adb-873a-435f-8a50-f5d5c89380f9`, key `D0C7PCB2UER:<ts>`, created 13:32:15.202Z.
- `channel_thread_messages`: `inbound` 13:32:15.204Z, `turn` 13:32:15.905Z, **`reply` 13:32:18.203Z**.
- Episode `474c6648…`, `in_progress`, contact `T02RM6X6B:U034NG4FPNG` (Kanat).

Slack (`slack_read_thread D0C7PCB2UER <ts>`): the parent is «Blah» from Kanat. The reply is from accelerate-bot-test: **«Feeling uninspired, or just having a "blah" day? Let me know if you need help with anything!»**

- In a DM the reply comes as a thread reply under Kanat's message, not as a new DM message. The reason: `thread_ts` falls back to `event.ts` (`slack_bot.yaml` `thread_key`). The reply template always sends `thread_ts` (F25).
- After 13:32:18.652Z no other event arrived up to 13:38:55Z.

## 4. Result

| check | result |
|---|---|
| events | 5 POSTs, all 200: 1 url_verification, 2 Kanat messages (channel and DM), 2 of the bot's own replies (skipped) |
| bridge thread (channel) | `thread-177fcfff…`, 1 inbound, 1 turn, 1 reply |
| reply sent (channel) | yes, «Hello! How can I help you today?» in Kanat's thread, posted by accelerate-bot-test |
| episode card | yes, Kanat (`U034NG4FPNG`), in `omni-16cfde16…` |
| bridge thread (DM) | `thread-74ff9adb…`, 1 inbound, 1 turn, 1 reply |
| reply sent (DM) | yes, «Feeling uninspired, or just having a "blah" day? …», as a thread reply |
| new errors | F23, twice (13:30:38.658Z and 13:32:16.446Z, both after a card was written) |
