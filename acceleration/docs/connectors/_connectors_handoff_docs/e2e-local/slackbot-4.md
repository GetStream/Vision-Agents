# slack_bot rev 4 live check (local router 1d34dc26 = v0.6.29), 2026-10-09
Kanat removed user scopes/user_events, revoked all tokens, re-consented (15:29:51Z, log `connector credential event event=grant_created … access_fingerprint=3190c417`), added `message.im` to bot events.
Source: ngrok inspector `localhost:4040/api/requests/http`, DB channel_thread_messages since 15:30Z.
| UTC | event | channel | authorizations[0].is_bot | bridge |
|---|---|---|---|---|
| 15:31:44 | message, mention | C0C8MKNUNBA top | true | inbound + reply |
| 15:32:04 | message, thread reply no mention | C0C8MKNUNBA thread | true | inbound + reply |
| 15:35:28 | message im | D0C7PCB2UER | true | inbound + reply (reply threaded: F25) |
| 15:35:52 | message, no mention | C0C8MKNUNBA top | true | skipped (no row) |
Bot echoes (app_id A0C7N6LNZMH) skipped. No events from other DMs. 0 level=ERROR in 15 min (also confirms #848: no "nobody could answer" after cards).
Before message.im was added to bot_events, DMs to the bot never arrived: earlier DMs came only through the user install's user_events.
