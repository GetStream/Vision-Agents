# MCP plugins (text, with the catalog, Blender and TableJourney)

A text agent that reads the team's issues in Sentry, GitHub and Linear, the customer in
HubSpot and Salesforce, and your own calendar, Drive, Docs, Calendly, Cal.com and Slack. It
makes 3D renders with Blender and knows where to eat from TableJourney's MCP server. The
conversation can carry on in Slack, Teams, WhatsApp, RCS, by text or in iMessage.

```bash
cd examples/text_agents/mcp_plugins
uv sync
uv run mcp_plugins.py                                      # issues and your calendar
uv run mcp_plugins.py "render a red teapot on a checkered floor"   # Blender
```

The directory is the agent, and nothing in `mcp_plugins.py` sets up a plugin. `agent.yaml`
connects each kind in a different way:

```yaml
plugins:        # connected once by the company, on the dashboard
  - sentry
  - github
  - linear
  - hubspot
  - salesforce
user_plugins:   # connected by each person, in the chat, when the agent needs it
  - google_calendar
  - google_drive
  - google_docs
  - calendly
  - calcom
  - slack
sandbox: daytona  # Blender, built into the router's sandbox for the render skill
```

Not every catalog plugin registers the router as a client by itself. Sentry, Linear,
Calendly and Cal.com do; the rest need an app of the deployment's own, whose redirect URI
is the router's `public_url` followed by `/v1/agents/plugins/callback`:

| Plugin | What the router needs |
| --- | --- |
| `github` | `GITHUB_MCP_CLIENT_ID`, `GITHUB_MCP_CLIENT_SECRET` from a GitHub OAuth app |
| `hubspot` | `HUBSPOT_MCP_CLIENT_ID`, `HUBSPOT_MCP_CLIENT_SECRET` from a HubSpot app |
| `slack` | `SLACK_MCP_CLIENT_ID`, `SLACK_MCP_CLIENT_SECRET` from a Slack app |
| `google_calendar` | `GOOGLE_CALENDAR_MCP_CLIENT_ID`, `GOOGLE_CALENDAR_MCP_CLIENT_SECRET`, `calendarmcp.googleapis.com` enabled |
| `google_drive` | `GOOGLE_DRIVE_MCP_CLIENT_ID`, `GOOGLE_DRIVE_MCP_CLIENT_SECRET`, `drivemcp.googleapis.com` enabled |
| `google_docs` | `GOOGLE_DOCS_MCP_CLIENT_ID`, `GOOGLE_DOCS_MCP_CLIENT_SECRET`, `docsmcp.googleapis.com` enabled |

The three Google entries can carry the same client id and secret: they are separate because
the router reads one pair per plugin, and the Cloud project has to have each API enabled.
`salesforce` has no single global host, so its login is given the org's `instance_url`.

## Sentry: once, for the company

Sentry is a hosted MCP server from the router's catalog. Until somebody connects it, the
agent's plugins list it as `not_connected`, which is what the dashboard shows as a reminder
to finish setting it up:

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

Google Calendar is a hosted MCP server too, but a calendar is somebody's own, so the
conversation is opened for an end user (`MCP_PLUGINS_USER_ID`, default `on-call-engineer`).
The agent gets `google_calendar__list_tools` and `google_calendar__call_tool`. The first
time it calls one for somebody who has not connected their calendar, the reply carries a
`plugin_authorization` attachment:

```json
{"type": "plugin_authorization", "plugin_id": "google_calendar",
 "title": "Connect Google Calendar", "authorize_url": "https://accounts.google.com/...",
 "text": "Read calendars, events and free time.",
 "thumb_url": "https://<public_url>/v1/agents/plugins/google_calendar/logo",
 "title_link": "https://accounts.google.com/..."}
```

A chat client renders it as a button. `text`, `thumb_url` and `title_link` are Chat's own
attachment fields, so a client that has never heard of `plugin_authorization` still shows a
card with the logo, what the plugin is for and a link somebody can press. This example
prints the URL, waits for you to connect, and asks again. A login belongs to one person on one agent and is never used for anybody
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

## Blender: in the sandbox

Ask for a picture and the agent hands it to the render skill, whose subagent writes a
Blender scene in Python and renders it in the router's Daytona sandbox:

- `sandbox_options` in `agent.yaml` says how the sandbox is built: `setup` installs
  Blender's libraries and `bpy==5.2.2` on a slim Python 3.13 image, `timeout` allows five
  minutes for a run, and `cpu` and `memory_gb` size it. Daytona keeps the built image, so
  only the first sandbox waits for the build, which takes a few minutes.
- `skills/render.md` is what the subagent writes under: a starting program that frames the
  scene and renders it with Cycles, and the instruction to pass
  `files=["/tmp/render.png"]` to `run_code`. Its `deadline: 20m` covers the first build.

The router downloads the file `run_code` names, uploads it to the conversation's channel,
and attaches it to the reply that settles the work, so it is still there when the
conversation is reopened. `task_settled` carries the file's URL as well, which is what this
script prints. The router needs `DAYTONA_API_KEY` for this.

## TableJourney: an MCP server the router does not know

Sentry and Google Calendar are in the router's catalog, which is what lets `agent.yaml` name
them under `plugins` and `user_plugins`. Any other MCP server goes under `mcp_servers`, by
its URL:

```yaml
mcp_servers:
  - name: tablejourney
    url: https://tablejourney.com/mcp
```

[TableJourney](https://tablejourney.com/agents/) is a public server of verified restaurants,
food festivals and food tours, and needs no login. The router opens it when the
conversation does, offers its tools as `tablejourney__<tool>`, and runs each call itself.
What the server says at initialize about using its tools is added to the agent's
instructions, which is how TableJourney's own terms (keep its booking links whole, say
they are affiliate links, cite its pages) reach the model.

```bash
uv run mcp_plugins.py "I'm in Rome tonight. Where should I go for cacio e pepe?"
```

What the catalog does and this does not: there is no login, so only a server that needs
none can be named (TableJourney's optional `X-API-Key` is not sent), and the URL has to be
https.

## Slack, Teams, WhatsApp, RCS, texting and iMessage: carrying the conversation over

A plugin is an account the agent reads. A channel is somewhere the conversation happens.
Slack is both, and they are not the same thing: `slack` under `user_plugins` lets the agent
search your workspace, while the Slack channel below is you talking to the agent in Slack.

With any of `SLACK_BOT_TOKEN`, `TEAMS_APP_ID`, `RBM_AGENT_ID`, `WHATSAPP_ACCESS_TOKEN`,
`TELNYX_SMS_NUMBER` or `LINQ_API_KEY` set, the script keeps the conversation open after its
first answer and prints how to reach it. Send the code it gives (`link <code>`) from that
channel. That ties you to this conversation, and from then on what you write there is asked
in it, as `MCP_PLUGINS_USER_ID`, and the answers come back to you. It is the same session,
so the dashboard shows every channel at once, and the calendar you connected in the terminal
is the one the agent reads in Slack. A login the agent asks for arrives as whatever that
channel has: a Block Kit button on Slack, a hero card with the plugin's logo in Teams, a
*Connect* button on WhatsApp, a suggestion over RCS and a link in a text.

`channels.py` is all of it. Each channel checks its provider's signature, reads the webhook
with the `omni` plugin (`SlackProvider`, `TeamsProvider`, `GoogleRBMProvider`,
`WhatsAppProvider`, `TelnyxProvider`, `LinqProvider`) and sends replies to the provider's
API. They are served on one port (`INBOX_PORT`, 8090), at `/slack`, `/teams`, `/rcs`,
`/whatsapp`, `/sms` and `/imessage`, so one tunnel carries them all.

### Setting up Slack's side

1. At [api.slack.com/apps](https://api.slack.com/apps), create an app. Under *OAuth &
   Permissions*, give the bot `chat:write`, `im:history`, `channels:history` and
   `app_mentions:read`, and install it.
2. Under *Event Subscriptions*, set the request URL to `https://<tunnel>/slack`. Slack
   checks it as you save, so start the script first. Subscribe the bot to `message.im` and
   `app_mention`.
3. Copy the bot token and the signing secret (*Basic Information → App Credentials*) into
   the repo's `.env`:

   ```bash
   SLACK_BOT_TOKEN=xoxb-...
   SLACK_SIGNING_SECRET=...
   ```

Then message the app directly, or mention it in a channel it is in.

### Setting up Teams' side

1. In the Azure portal, create an *Azure Bot* with a multi-tenant Microsoft App ID, and add
   the Microsoft Teams channel to it.
2. Set its messaging endpoint to `https://<tunnel>/teams`.
3. Create a client secret for the app registration and put both in the repo's `.env`. A
   single-tenant bot also needs its tenant:

   ```bash
   TEAMS_APP_ID=...
   TEAMS_APP_PASSWORD=...
   # TEAMS_TENANT_ID=...     # single-tenant bots only
   ```

Every delivery carries a Bot Framework token, which `channels.py` checks against
`login.botframework.com`'s keys. Where the reply is sent comes from that token's
`serviceurl` claim rather than from the body, so a delivery cannot point the agent's answers
at a host of its own. The keys are fetched once at startup, so a key rotation mid-run means
a restart.

### Setting up Google's side (RCS)

RCS Business Messaging needs a registered RBM agent, which Google approves per carrier.

1. In the RBM developer console, create an agent and note its id.
2. Set its webhook to `https://<tunnel>/rcs` with a client token of your choosing. RBM
   sends that token back on every delivery, which is what `channels.py` compares.
3. Download a service account key with the *RCS Business Messaging* role and point at the
   file:

   ```bash
   RBM_AGENT_ID=...
   RBM_SERVICE_ACCOUNT=/path/to/service-account.json
   RBM_CLIENT_TOKEN=pick-anything
   ```

The service account signs its own way to an access token, so there is nothing else to
install. Only a test device added to the agent can message it before it is launched.

### Setting up Meta's side

1. At [developers.facebook.com](https://developers.facebook.com/apps), create an app with the
   *Connect with customers through WhatsApp* use case. It comes with a test business number.
2. Under *WhatsApp → API Setup*, copy the **Phone number ID** (not the number) and generate
   an access token. Add your own phone as a recipient and confirm it with the code WhatsApp
   sends: a test number only talks to up to five numbers confirmed this way.
3. Under *App settings → Basic*, copy the **App secret**. Webhooks are signed with it.
4. Put them in the repo's `.env`, with any string you like as the verify token:

   ```bash
   WHATSAPP_ACCESS_TOKEN=...
   WHATSAPP_PHONE_NUMBER_ID=...
   WHATSAPP_APP_SECRET=...
   WHATSAPP_VERIFY_TOKEN=pick-anything
   # INBOX_PORT=8090
   ```

5. Give the webhook a public https URL. Meta has to reach it, so a laptop needs a tunnel,
   such as `ngrok http 8090` (its free static domain saves re-registering each run) or
   `cloudflared tunnel --url http://localhost:8090`.
6. Start the script, since Meta checks the URL while you save it. Then under
   *WhatsApp → Configuration → Webhook*, set the callback URL to `https://<tunnel>/whatsapp`
   and the verify token to `WHATSAPP_VERIFY_TOKEN`, then **Verify and save**. Under
   *Webhook fields*, subscribe to `messages`.

The token from *API Setup* lasts a day. For one that does not expire, add a system user in
*Business settings → Users → System users*, give it the app and the WhatsApp account, and
generate a token with `whatsapp_business_messaging` and `whatsapp_business_management`.

If messages arrive at nobody although the webhook verified, check that the WhatsApp
Business Account is subscribed to the app (`GET /{waba-id}/subscribed_apps`, and `POST` it
if the app is missing) and that the app's subscription lists the `messages` field.

### Setting up Telnyx's side

Texting uses the same Telnyx account as the router's phone numbers (`TELNYX_API_KEY`).

1. Create a messaging profile whose webhook is `https://<tunnel>/sms`, in the portal under
   *Messaging*, or with `POST /v2/messaging_profiles` (`webhook_url`, `webhook_api_version`
   `2`, `whitelisted_destinations` such as `["US"]`).
2. Assign one of the account's numbers to it, in the portal or with
   `PATCH /v2/phone_numbers/{id}/messaging` and the profile's id. A number can text and take
   calls at once: the messaging profile and the voice connection are separate.
3. Put the number and the account's webhook signing key (*Keys & Credentials → Public Key*,
   or `GET /v2/public_key`) in the repo's `.env`:

   ```bash
   TELNYX_SMS_NUMBER=+1...
   TELNYX_PUBLIC_KEY=...
   ```

**Texts to US phones need 10DLC registration.** Carriers block business texts from a US
local number that is not on a registered 10DLC campaign, so until the account has a brand
and a campaign with the number on it, texts reach the agent but its replies are refused.
Register both under *Messaging → 10DLC* in the Telnyx portal; approval takes days. A
toll-free number needs toll-free verification instead.

### Setting up Linq's side (iMessage)

Linq carries iMessage, and falls back to RCS or SMS for a phone Apple cannot reach. It
needs a Linq partner account.

1. Create an API token at
   [dashboard.linqapp.com/api-tooling](https://dashboard.linqapp.com/api-tooling), and note
   the line to message (`GET /v3/phone_numbers`).
2. Subscribe to `message.received`, pinning the payload version the `omni` provider reads:

   ```bash
   curl -X POST https://api.linqapp.com/api/partner/v3/webhook-subscriptions \
     -H "Authorization: Bearer $LINQ_API_KEY" -H "Content-Type: application/json" \
     -d '{"target_url": "https://<tunnel>/imessage?version=2026-02-03",
          "subscribed_events": ["message.received"]}'
   ```

   The response's `signing_secret` (`whsec_...`) is what deliveries are signed with.
3. Put them in the repo's `.env`:

   ```bash
   LINQ_API_KEY=...
   LINQ_NUMBER=+1...
   LINQ_WEBHOOK_SECRET=whsec_...
   ```

### Where this runs out

Each channel needs configuring at both levels the plugins do, and the router has neither for
a channel, so this example holds both itself:

- **Once for the app.** The Slack app, the Azure bot, the RBM agent, the WhatsApp business
  number and the Telnyx number all belong to the company, like the Sentry login. There is no
  `channels:` beside `plugins:` in `agent.yaml` and nowhere on the dashboard to connect one,
  so they live in this script's environment, and only this process can answer them.
- **Once per person.** Which end user a phone number is lives in this process's memory, for
  one conversation. The router has no record of it, so a second run needs a new code, and a
  number writing in when the script is not running reaches nobody. A Chat channel already
  starts a session by itself; a phone number cannot.
- **One direction.** Only answers to what was written on a channel are sent there. A reply
  to something typed on the dashboard stays on the dashboard.
- **Meta's 24-hour window.** A business may only write freely within 24 hours of the
  person's last message, and needs an approved template after that. The agent only ever
  answers, so this holds, but nothing could start a WhatsApp conversation.
- **Formatting.** The agent writes Markdown, which WhatsApp reads as its own `*bold*` and a
  text not at all, so a heading or a table arrives as typed.

Needs a router: see `acceleration/README.md`, then `STREAM_ACCELERATION_URL` and
`STREAM_ACCELERATION_CUSTOMER_ID`, plus the Stream app's `STREAM_API_KEY` and
`STREAM_API_SECRET` that every agent is built with. The router needs the same Stream
credentials, since the render goes to the conversation's channel.
