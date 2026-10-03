# On call (text, with Sentry and Google Calendar)

A text agent that reads the team's Sentry issues and your own Google Calendar.

```bash
cd examples/text_agents/on_call
uv sync
uv run on_call.py
```

Both are hosted MCP servers from the router's catalog, set up in `agent.yaml`, and they are
connected in two different ways:

```yaml
plugins:        # connected once by the company, on the dashboard
  - sentry
user_plugins:   # connected by each person, in the chat, when the agent needs it
  - google_calendar
```

## Sentry: once, for the company

Until somebody connects Sentry, the agent's plugins list it as `not_connected`, which is what
the dashboard shows as a reminder to finish setting it up:

```bash
curl -H "X-Customer-Id: $STREAM_ACCELERATION_CUSTOMER_ID" \
  $STREAM_ACCELERATION_URL/v1/agents/configs/$CONFIG_ID/plugins
# [{"plugin_id":"sentry","name":"Sentry","status":"not_connected",...}]
```

Connect it from the dashboard, or start the login yourself and open the URL it returns:

```bash
curl -X POST -H "X-Customer-Id: $STREAM_ACCELERATION_CUSTOMER_ID" \
  $STREAM_ACCELERATION_URL/v1/agents/configs/$CONFIG_ID/plugins/sentry/authorize
```

Sentry registers the router as a client by itself, so nothing needs configuring first.

## Google Calendar: each person, in the chat

A calendar is somebody's own, so the conversation is opened for an end user
(`ON_CALL_USER_ID`, default `on-call-engineer`). The agent gets
`google_calendar__list_tools` and `google_calendar__call_tool`. The first time it calls one
for somebody who has not connected their calendar, the reply carries a
`plugin_authorization` attachment:

```json
{"type": "plugin_authorization", "plugin_id": "google_calendar",
 "title": "Connect Google Calendar", "authorize_url": "https://accounts.google.com/..."}
```

A chat client renders it as a button. This example prints the URL, waits for you to connect,
and asks again. A login belongs to one person on one agent and is never used for anybody
else, so the router has to know who that person is: `api_key` mode with the app's key and
secret naming them, or `proxy` mode. A `noauth` router treats every caller as the app's own
backend, and an anonymous caller goes by a name nobody checked, so neither is offered the
calendar.

Google registers no client on the fly, so the router needs one of its own. In a Google Cloud
project, enable `calendarmcp.googleapis.com`, create an OAuth client of type *Web
application* whose redirect URI is the router's `public_url` followed by
`/v1/agents/plugins/callback`, and give the router:

```bash
GOOGLE_CALENDAR_MCP_CLIENT_ID=...
GOOGLE_CALENDAR_MCP_CLIENT_SECRET=...
```

Needs a router: see `acceleration/README.md`, then `STREAM_ACCELERATION_URL` and
`STREAM_ACCELERATION_CUSTOMER_ID`, plus the Stream app's `STREAM_API_KEY` and
`STREAM_API_SECRET` that every agent is built with.
