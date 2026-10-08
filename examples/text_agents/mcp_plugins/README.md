# MCP plugins (text, with the catalog, Blender and TableJourney)

A text agent that reads the team's issues in Sentry, GitHub and Linear, the customer in
HubSpot and Salesforce, and your own calendar, Drive, Docs, Calendly, Cal.com and Slack. It
makes 3D renders with Blender and knows where to eat from TableJourney's MCP server. The
same agent answers on WhatsApp, on a text message and in iMessage, with no code for it here:
`channels` in `agent.yaml` names the numbers and the router does the rest.

```bash
cd examples/text_agents/mcp_plugins
uv sync
uv run mcp_plugins.py                                      # issues and your calendar
uv run mcp_plugins.py "render a red teapot on a checkered floor"   # Blender
```

The directory is the agent, and nothing in `mcp_plugins.py` sets up a plugin. `agent.yaml`
connects each kind in a different way:

```yaml
agent_plugins:  # connected once by the company, on the dashboard
  - sentry
  - github
  - hubspot
  - salesforce
user_plugins:   # connected by each person, in the chat, when the agent needs it
  - linear
  - google_calendar
  - google_drive
  - google_docs
  - calendly
  - calcom
  - slack
sandbox: daytona  # Blender, built into the router's sandbox for the render skill
```

Not every catalog plugin registers the router as a client by itself. Sentry, Linear,
Calendly and Cal.com do; the rest need an OAuth app of your own, whose redirect URI is the
router's `public_url` followed by `/v1/agents/plugins/callback`. Each agent holds its own:
on the dashboard's Tools tab, press **Set up** on the plugin, paste the app's client id and
secret, and choose whether each user connects in the chat or the agent connects one account.
The same thing over the API, server-side only:

```bash
curl -X PUT "$ROUTER/v1/agents/configs/$CONFIG_ID/plugins/google_calendar/client" \
  -d '{"client_id": "...", "client_secret": "...", "user": true}'
```

The secret is sealed with the router's `auth.kek` and never returned. A sync warns about a
`user_plugins` entry that still has none, and a user who asks for it is told it is not
available. A deployment can instead give every agent the same app through the environment:

| Plugin | What the router needs |
| --- | --- |
| `github` | `GITHUB_MCP_CLIENT_ID`, `GITHUB_MCP_CLIENT_SECRET` from a GitHub OAuth app |
| `hubspot` | `HUBSPOT_MCP_CLIENT_ID`, `HUBSPOT_MCP_CLIENT_SECRET` from a HubSpot app |
| `slack` | `SLACK_MCP_CLIENT_ID`, `SLACK_MCP_CLIENT_SECRET` from a Slack app |
| `salesforce` | `SALESFORCE_MCP_CLIENT_ID`, `SALESFORCE_MCP_CLIENT_SECRET` from an External Client App |
| `google_calendar` | `GOOGLE_CALENDAR_MCP_CLIENT_ID`, `GOOGLE_CALENDAR_MCP_CLIENT_SECRET`, `calendarmcp.googleapis.com` enabled |
| `google_drive` | `GOOGLE_DRIVE_MCP_CLIENT_ID`, `GOOGLE_DRIVE_MCP_CLIENT_SECRET`, `drivemcp.googleapis.com` enabled |
| `google_docs` | `GOOGLE_DOCS_MCP_CLIENT_ID`, `GOOGLE_DOCS_MCP_CLIENT_SECRET`, `docsmcp.googleapis.com` enabled |

The three Google entries can carry the same client id and secret: they are separate because
the router reads one pair per plugin, and the Cloud project has to have each API enabled.

### Salesforce

`salesforce` is Salesforce's hosted `sobject-reads` server, which reads any object in the
org and changes none: ask it "what are the top 10 opportunities for Q4?" and it answers
with a SOQL query on `Opportunity`. One URL serves every production org, so the login
needs no org host. Set the org up once, as an admin
([Salesforce's guide](https://developer.salesforce.com/docs/platform/hosted-mcp-servers/guide/create-external-client-app.html)):

1. In Setup, open **MCP Servers**, then **Salesforce Servers**, and activate `sobject-reads`.
2. In **External Client App Manager**, create an app with OAuth on. Its callback URL is
   the router's `public_url` + `/v1/agents/plugins/callback`, such as
   `http://localhost:8080/v1/agents/plugins/callback`. Its scopes are `mcp_api` and
   `refresh_token`. Require PKCE, and issue JWT-based access tokens for named users.
3. Put the app's consumer key and secret in `.env` as `SALESFORCE_MCP_CLIENT_ID` and
   `SALESFORCE_MCP_CLIENT_SECRET`, restart the router and connect Salesforce on the dashboard.

The agent reads the org as the person who connected it, so it sees what they see.

### Slack

`slack` is [Slack's MCP server](https://docs.slack.dev/ai/slack-mcp-server/). Each person
connects their own account, and the agent searches and posts as them. Slack registers no
client by itself, and only an internal app or one published in the Slack Marketplace may use
MCP:

1. At [api.slack.com/apps](https://api.slack.com/apps), create an app in your workspace.
   Under **OAuth & Permissions**, add the redirect URL (the router's `public_url` +
   `/v1/agents/plugins/callback`) and the **user token** scopes the agent will ask for.
   Slack takes only an https redirect URL, so a local router needs an https tunnel as its
   `public_url`.
2. On the dashboard, open the agent's Tools, press **Set up** on Slack and paste the app's
   client id and secret, with "Each user, in the conversation". The page lists these
   steps too. `SLACK_MCP_CLIENT_ID` and `SLACK_MCP_CLIENT_SECRET` in `.env` still work, for
   every agent that sets no client of its own.

By default the login asks for `channels:history`, `channels:read`, `chat:write`,
`search:read.public` and `users:read`: search and read public channels, post, and look
people up. `scopes` replaces that list, say to search private channels and DMs as well, or
to drop `chat:write` so the agent can only read:

```yaml
user_plugins:
  - name: slack
    scopes:
      - search:read.public
      - search:read.private
      - search:read.im
      - channels:history
      - users:read
```

[Slack's docs](https://docs.slack.dev/ai/slack-mcp-server/#oauth-scopes) list which tool
needs which scope. The router refuses one Slack's server does not offer, and the app must
have every scope asked for. A tool whose scope was not granted fails when it is called, so
pair `scopes` with `tools` to keep the model from seeing it:

```yaml
    tools: [slack_search_*, slack_read_*]
```

### Read-only, and which scopes a login asks for

A plugin named by its id alone is reached as the catalog has it, which for Linear is
`https://mcp.linear.app/mcp` asking for `read` and `write`. Name it with a mapping instead
to change how it is reached and what its login asks for. Either form goes in either list:

```yaml
user_plugins:
  - name: linear
    readonly: true   # https://mcp.linear.app/mcp/readonly: no tool that writes, asks for read
    scopes: [read]   # replaces the scopes the login asks for
```

`readonly` is only accepted for a plugin whose vendor runs a read-only server (Linear, so
far), and the router refuses it for any other. A login made before a change keeps what it
was granted, so connect the plugin again after changing either.

`toolsets` limits a server to some of its tools, for a vendor that lets you choose. Cal.com
does, and offers every tool when none are picked:

```yaml
user_plugins:
  - name: calcom
    toolsets: [bookings, availability, event-types]
```

Its toolsets are `profile`, `event-types`, `bookings`, `availability`, `schedules`,
`calendars`, `teams`, `organizations`, `routing-forms` and `catalog`; `get_app_link` and
`search_docs` come whichever are picked, and the router refuses a name not on that list.
The login is for the server, not the toolsets, so changing them needs no new one.

`scopes` does nothing for Cal.com: its MCP server ignores the scopes it is asked for and
always asks your Cal.com account for the same fixed set. Toolsets are what narrow it.

Cal.com also only registers clients whose redirect host is on its own list. `localhost`
is, so a local router logs in; a router at any other host, getstream.io included, is
refused at registration until Cal.com adds it.

Google Drive asks for `drive.readonly` unless told otherwise. Its `create_file` and
`copy_file` tools need `drive.file` as well, which reaches only the files the agent made or
was given. `scopes` must be ones the server advertises; for Drive those are `drive`,
`drive.readonly` and `drive.file`:

```yaml
user_plugins:
  - name: google_drive
    scopes:
      - https://www.googleapis.com/auth/drive.readonly
      - https://www.googleapis.com/auth/drive.file
```

### Only some tools

Every tool a server offers goes to the model unless `tools` says otherwise. Fewer tools
take less context and give the agent less it may do. `tools` takes names or patterns such
as `get_*`, works on any plugin and on `mcp_servers`, and needs no new login:

```yaml
user_plugins:
  - name: linear
    tools: [list_issues, get_issue, save_comment]
mcp_servers:
  - name: tablejourney
    url: https://tablejourney.com/mcp
    tools: [search_places, place, dishes]
```

A tool left out is not listed, and the router refuses to run it.

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

Google registers no client on the fly, so the agent needs one of your own. In a Google Cloud
project, enable `calendarmcp.googleapis.com`, create an OAuth client of type *Web
application* whose redirect URI is the router's `public_url` followed by
`/v1/agents/plugins/callback`, and set it on the agent: **Set up** on Google Calendar in the
Tools tab, with "Each user, in the conversation".

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
them under `agent_plugins` and `user_plugins`. Any other MCP server goes under `mcp_servers`, by
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

The URL has to be https, and no headers are sent (TableJourney's optional `X-API-Key` is
not). The server says whether it needs an OAuth login, and the router asks it when the
config is saved: by its `/.well-known/oauth-protected-resource`, or by a 401 that says how
to authenticate. The answer is the read-only `needs_login`. The app connects such a server
once, on the dashboard, unless `user: true` has each person connect their own in the chat,
as `user_plugins` do for the catalog:

```yaml
mcp_servers:
  - name: crm                  # the app connects it once, on the dashboard
    url: https://crm.example.com/mcp
  - name: notes                # each person connects their own, in the chat
    url: https://notes.example.com/mcp
    user: true
    scopes: [notes.read]       # left out, asks for what the server advertises
```

The login is the one the MCP authorization spec describes: the authorization server's
metadata, a client the router registers there itself, PKCE and `resource`. A server that
needs a login the router cannot make is refused when the config is saved, and so is
`scopes` or `user` on one that needs none. Until the app connects it, the server's tools
fail with "connect crm on the dashboard". The app's login is listed and connected with the
same `/v1/agents/configs/{id}/plugins` endpoints as Sentry's below. A login is only good at
the URL it was made at, and one made before a change of scopes keeps what it was granted,
so connect again after changing either.

## WhatsApp, texting and iMessage: the agent on a phone number

A plugin is an account the agent reads. A channel is somewhere the conversation happens.
Slack is both, and they are not the same thing: `slack` under `user_plugins` lets the agent
search your workspace; the channels here are a person writing to the agent from their phone.

There is no code for this in the example. The router answers the lines, so it takes three
steps and the third is the only one in `agent.yaml`.

**1. Connect the line once, for the app.** This hands the router the provider's credentials
and answers with the URL that provider should deliver to. They are sealed under the router's
`auth.kek` and never read back, so a deployment without one refuses to hold them.

```bash
curl -X POST -H "X-Customer-Id: $STREAM_ACCELERATION_CUSTOMER_ID" \
  -H "Content-Type: application/json" $STREAM_ACCELERATION_URL/v1/agents/channels \
  -d '{"kind": "whatsapp", "number": "+15556325550",
       "account_id": "'$WHATSAPP_PHONE_NUMBER_ID'", "token": "'$WHATSAPP_ACCESS_TOKEN'",
       "signing": "'$WHATSAPP_APP_SECRET'", "challenge": "'$WHATSAPP_VERIFY_TOKEN'"}'
# {"id":"...","kind":"whatsapp","number":"+15556325550","delivering":false,
#  "webhook_url":"https://<public_url>/v1/agents/channels/hooks/<token>"}
```

What each credential is, per channel:

| Channel | `kind` | `token` | `signing` | `account_id` |
| --- | --- | --- | --- | --- |
| WhatsApp | `whatsapp` | a Meta access token | the app secret | the phone number id |
| Texting | `sms` | a Telnyx API key | the account's public key | the Telnyx number's id |
| iMessage | `imessage` | a Linq API key | the subscription's `whsec_...` | — |

WhatsApp also takes `challenge`, the verify token Meta echoes while saving a webhook.
Connecting a line that is already connected replaces its credentials and keeps its
`webhook_url`, so rotating a token does not mean setting the webhook up again. `GET
/v1/agents/channels` lists the lines; `DELETE /v1/agents/channels/{id}` drops one.

**2. Point the provider at that URL.** Paste it into Meta's or Linq's webhook setup, below.
A `sms` line is the exception: the number was bought through the router, so the router points
it at the URL itself and answers `"delivering": true` with nothing left to do.

**3. Name the number in `agent.yaml`.** This is what makes the agent the one that answers:

```yaml
channels:
  whatsapp:
    number: "+15556325550"
  sms:
    number: "+12187021098"
  identity: link
```

The number has to be a line the app connected, and only one agent may answer on it. A message
that arrives earns a turn in the writer's own persistent conversation, so it is in Stream Chat
like anything else -- the dashboard shows it, and `custom.channel`, `custom.channel_number`
and `custom.channel_from` say where it came from. A second message carries on the first one's
conversation rather than starting over.

### Who is writing

`identity` decides what a phone number means to the agent.

`phone`, the default, makes each number an end user of its own, `phone:+13475550100`. Anybody
who writes is answered and what they say is kept as theirs, which is what a support agent or
this example's TableJourney questions want.

`link` answers a number only once somebody already signed in has tied it to themselves. Ask
for a code, show it to them, and the number they text it from is theirs from then on:

```bash
curl -X POST -H "X-Customer-Id: $STREAM_ACCELERATION_CUSTOMER_ID" \
  -H "Content-Type: application/json" $STREAM_ACCELERATION_URL/v1/agents/channels/links \
  -d '{"config_id": "'$CONFIG_ID'", "user_id": "on-call-engineer"}'
# {"code":"418702","expires_at":"..."}
```

Text that code to the agent and it answers that the number is yours. This example uses `link`
because its agent reads your Linear issues and your calendar: the Google Calendar you
connected in the terminal is the one it reads over WhatsApp, because both are the same end
user. Until a number is linked, every message gets the same answer asking for a code.

### Setting up Meta's side

1. At [developers.facebook.com](https://developers.facebook.com/apps), create an app with the
   *Connect with customers through WhatsApp* use case. It comes with a test business number.
2. Under *WhatsApp → API Setup*, copy the **Phone number ID** (not the number) and generate
   an access token. Add your own phone as a recipient and confirm it with the code WhatsApp
   sends: a test number only talks to up to five numbers confirmed this way.
3. Under *App settings → Basic*, copy the **App secret**. Webhooks are signed with it.
4. Connect the line with those three and any string you like as the verify token, as above.
5. The router has to be reachable by Meta, so a laptop needs a tunnel: `ngrok http 8080` (its
   free static domain saves re-registering each run) or
   `cloudflared tunnel --url http://localhost:8080`. Set the router's `public_url` to it, so
   the `webhook_url` it answers with is the one Meta can reach.
6. Under *WhatsApp → Configuration → Webhook*, set the callback URL to the `webhook_url` and
   the verify token to what you connected the line with, then **Verify and save**. Meta
   checks it as you save, which is the router answering the GET. Under *Webhook fields*,
   subscribe to `messages`.

The token from *API Setup* lasts a day. For one that does not expire, add a system user in
*Business settings → Users → System users*, give it the app and the WhatsApp account, and
generate a token with `whatsapp_business_messaging` and `whatsapp_business_management`. Then
connect the line again with the new token; the webhook URL does not change.

If messages arrive at nobody although the webhook verified, check that the WhatsApp Business
Account is subscribed to the app (`GET /{waba-id}/subscribed_apps`, and `POST` it if the app
is missing) and that the app's subscription lists the `messages` field.

### Setting up Telnyx's side

Texting uses the same Telnyx account as the router's phone numbers, and the number has to be
one the app bought through the router (`POST /v1/phone/numbers`) with the `sms` capability.
Connecting the line points that number's messaging at the webhook, so there is nothing to do
in the portal. The `signing` credential is the account's webhook signing key (*Keys &
Credentials → Public Key*, or `GET /v2/public_key`).

**Texts to US phones need 10DLC registration.** Carriers block business texts from a US local
number that is not on a registered 10DLC campaign, so until the account has a brand and a
campaign with the number on it, texts reach the agent but its replies are refused. Register
both under *Messaging → 10DLC* in the Telnyx portal; approval takes days. A toll-free number
needs toll-free verification instead.

### Setting up Linq's side (iMessage)

Linq carries iMessage, and falls back to RCS or SMS for a phone Apple cannot reach. It needs
a Linq partner account.

1. Create an API token at
   [dashboard.linqapp.com/api-tooling](https://dashboard.linqapp.com/api-tooling), and note
   the line to message (`GET /v3/phone_numbers`).
2. Connect the line with that token, to get a `webhook_url`.
3. Subscribe to `message.received` at that URL, pinning the payload version the router reads:

   ```bash
   curl -X POST https://api.linqapp.com/api/partner/v3/webhook-subscriptions \
     -H "Authorization: Bearer $LINQ_API_KEY" -H "Content-Type: application/json" \
     -d '{"target_url": "<webhook_url>?version=2026-02-03",
          "subscribed_events": ["message.received"]}'
   ```

4. Connect the line again with the response's `signing_secret` (`whsec_...`) as `signing`.
   The `webhook_url` does not change, so the subscription stays good.

### Where this runs out

- **Meta's 24-hour window.** A business may only write freely within 24 hours of the person's
  last message, and needs an approved template after that. The agent only ever answers, so
  this holds, but nothing could start a WhatsApp conversation.
- **One direction.** Only what the agent says while answering a channel message goes back
  there. A reply to something typed on the dashboard stays on the dashboard.
- **Formatting.** The agent writes Markdown, which WhatsApp reads as its own `*bold*` and a
  text message not at all, so a heading or a table arrives as typed.
- **Media in.** A photo or a voice note sent to the agent is not read; only text, a pressed
  button and a list choice are.

Needs a router: see `acceleration/README.md`, then `STREAM_ACCELERATION_URL` and
`STREAM_ACCELERATION_CUSTOMER_ID`, plus the Stream app's `STREAM_API_KEY` and
`STREAM_API_SECRET` that every agent is built with. The router needs the same Stream
credentials, since the render goes to the conversation's channel.
