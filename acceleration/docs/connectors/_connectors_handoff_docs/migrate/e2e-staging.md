# E2E staging: connectors release through the Vercel preview

- Start: 2026-10-10T02:41:48Z (`e2e-staging/start.txt`). Browser tab: page 29 (preview, logged in).
- Dashboard: https://volt-dashboard-git-ai-team-agent-dashboard-getstreamio.vercel.app, org 1181507, app 1257545.
- Router: `chat-accelerate-staging-76667bf6bb-26n49` container `router`, image `accelerate:v0.1.3@sha256:d3f8462e…` (pod started 2026-10-10T02:33:37Z). Gateway pods `-gateway-86b665484b-dklsg/zxtdb`, same image.
- Logs: `K="kubectl --context gke_stream-gcp-2026-v2_us-east1_us-east1-acceleration -n accelerate-staging"; $K logs chat-accelerate-staging-76667bf6bb-26n49 -c router --since-time=2026-10-10T02:41:48Z`
- PAT: `E2E_GITHUB_PAT`, put into the page by `e2e-staging/secret_helper.py` (loopback, origin pinned to the preview, logs nothing, PID in `e2e-staging/helper.pid`). Value never printed; the page script returned only its length.

## Step log

### 1. Library › Connectors
- URL `/agents/library/connectors/`: 18 built-in connectors (Cal.com … WhatsApp). GET `/v1/agents/connectors?limit=25` 200 (reqid 320).
- Search `slack` → URL `?q=slack`, rows: Slack, Slack bot.
- GitHub detail: Revision 3, Auth «OAuth, Bearer token», Client registration «Stream’s, Your own», 11 scopes, «Set OAuth client», Redirect URI `https://accelerate.gcp.stream-io-api.com/v1/agents/connectors/oauth/callback`, «No OAuth client of your own».
- Linear: Revision 2, OAuth, «Registered at each consent», scopes read/write, same redirect URI, «This connector does not take your own OAuth client».
- Slack: Revision 6, OAuth, «Stream’s», 29 scopes, same redirect URI, does not take own client.
- Slack bot: Revision 4, «Stream’s, Your own, Created by Stream for this app», scopes chat:write/channels:history/im:history, «Set OAuth client», same redirect URI.
- **Staging redirect URI: `https://accelerate.gcp.stream-io-api.com/v1/agents/connectors/oauth/callback`.**

### 2. New connection: GitHub bearer, owner App
- Connections tab started empty for owner App (GET `/connections?owner_type=app` 200, 30 bytes, 02:43:06.604Z).
- New connection › GitHub › Bearer token › App › label `e2e-connectors-github` › token from helper (len 93) › Create.
- Network: POST /connections 201 (reqid 417), PUT credentials 200 (419), POST validate 200 (422), GET tools 200 (432).
- Page `/connections/3edb0db98466bb7d4d424fa2fbcfc156/`: «Validation: Works · The credential works and the provider listed its tools.», «Status Connected · Last check: Works», Owner App, Scopes None, Credential revision 2.
- Router: `02:44:39.305Z connector credential event event=grant_created connection=3edb0db98466bb7d4d424fa2fbcfc156 connector=github revision=2 reason=credentials access_fingerprint=3577528b`; validate 200 516ms (req e1b38045); tools 200 56860 bytes (req 03d4ebf7).

### 3. Agent `e2e-connectors` + GitHub fixed binding + playground
- New agent › Custom › name `e2e-connectors`, Calls off → `/agent/configs/a54fb8004dee62fd4d9020e80bdb15b0/behavior/` (Text).
- Tools tab: «Connectors» section, «No connectors yet». (No «Connected apps» section on staging.) Add connector › GitHub › One app connection › «e2e-connectors-github · Connected» → 49 tool checkboxes; ticked get_me + search_repositories. Row: «GitHub · github / App connection e2e-connectors-github · 2 tools». Save → toast «Agent saved», PUT /configs/a54fb800… 200 (reqid 558).
- While the dialog loaded, it also sent GET `/connections?owner_type=user&connector_id=github` → **400** (reqid 552; router 02:46:00.674Z req 0b006326, 252 bytes). Router code `acceleration/internal/api/connections.go:385-388` refuses owner_type=user when no `X-Stream-User-Id` reached it. The dashboard sets that header (`volt src/api/agents.ts:66-69`), so on staging it is dropped before the router (gateway overwrites it from the token; the server token has no user). Gateway logs nothing, so where it is dropped is `unverified`. → F80.
- Playground (Test): session `01a123b5-8dbe-70c0-920a-9953eedcf13a` (POST /sessions 201, reqid 561; router «session joined» 02:47:43.970Z). Prompt asked for get_me + search_repositories. Reply: «Your GitHub login is kanat. The repository GetStream/Vision-Agents currently has 8,156 stars.» Tools panel: `github__get_me` Done · 238 ms, `github__search_repositories` Done · 380 ms. Router: POST /sessions/01a123b5…/respond 200 at 02:48:34.786Z; gemini reply call success=false then fell back to deepseek success=true (02:48:36Z), tool round on gemini success=true.
- Test artefact: the first prompt typed with the MCP `fill` + Enter cleared the box without sending (no /respond in the router log); typing with real key events worked.

### 4. Linear as each user's own (session) — BLOCKED by F80
- Add connector › Linear › Each user's own: dialog shows info alert «Connect Linear to choose its tools. [Connect]». «Add connector» was accepted with no tools: row «Each user’s own account · 0 tools» (the router doc says an empty list grants none) → F82. Then Edit › Advanced › Tool names (tag input, Enter per name) `list_teams`, `list_issues` → «· 2 tools», Save → «Agent saved» (router PUT /configs 200 02:50:50.994Z req f436de58).
- Playground (Test) session `01a123b8-833c-7f10-b974-0d90c465cd3b`, prompt «call list_teams…». **No connector_authorization card.** Reply: «I don't have access to Linear tools to look up your teams. Would you like me to connect you with a representative…». Tools panel «No tool calls yet».
- Router: `02:50:57.701Z level=WARN msg="opening the session without a connector" connector=linear reason=caller_unverified`. `acceleration/internal/session/connector_tools.go:57-58,90`: caller_unverified = «the caller is anonymous, a guest or a backend acting for nobody». The dashboard opens sessions as a backend (server token); its `X-Stream-User-Id` did not reach the router.
- Confirmation: Connections › New connection › Linear › Signed-in user › label `e2e-connectors-linear` › Create → dialog error «Owner.user_id must be the user this backend acts for, named by X-Stream-User-Id.» Router `02:52:09.867Z POST /v1/agents/connections status=400` (req f3fde52d). Nothing created; dialog cancelled.
- CORS is not the cause: `curl -X OPTIONS https://accelerate.gcp.stream-io-api.com/v1/agents/connections …` with the preview Origin → 204, `access-control-allow-headers` includes `x-stream-user-id, x-stream-actor-id, x-stream-actor-name`. So the browser sends the header and the gateway (v0.1.3) drops or overwrites it before the router (gateway source/logs not checked: `unverified`).
- Linear consent popup, postMessage handoff and session resume: not reached (`unverified`).

### 4b. Linear consent popup and handoff, app-owned (in place of the blocked session flow)
- Connections › New connection › Linear › App › label `e2e-connectors-linear` › Create. A pop-up (page 31) opened and closed by itself within ~5 s; I clicked nothing in it. The list (`?connector_id=linear`) showed `e2e-connectors-linear 4fc42f2c1d4d2b695942c76d9139a376 · Linear · App · OAuth · Connected`.
- Router: 02:53:09.121Z POST /connections 201 (req d784b3d2) → 02:53:09.770Z `registered an OAuth client connector=linear connection=4fc42f2c… registration_host=mcp.linear.app auth_method=none` → POST /authorizations 201 (req c66d850b) → GET+POST `/v1/agents/connectors/oauth/launch/1b518f86…` 200 → **02:53:14.547Z `grant_created connection=4fc42f2c… connector=linear revision=2 reason=consent access_fingerprint=abdc8efa refresh_fingerprint=64e9134e access_expires_at=2026-10-11T02:48:14.086Z`** → GET /oauth/callback 302 (req b9f7f228).
- So the pop-up → Linear → staging callback → preview handoff works. Linear did not ask for login or approval (the browser profile was already signed in and had approved before; `unverified` which).

### 5. Slack user OAuth — BLOCKED (staging config)
- Connections › New connection › Slack › App › label `e2e-connectors-slack` › Create. POST /connections 201 (02:53:40.465Z req 930804dc), then POST `/connections/ad4c2731…/authorizations` **400** (02:53:40.750Z req 463eb8a7). The pop-up opened as about:blank and closed; the dialog closed with no visible error that I caught (I read toasts ~5 s later; a 4 s toast may have passed: `unverified`) → F83. Row left `Pending`.
- Connection page › Connect → toast «The provider's consent could not be started: oauth2code: no OAuth client available for this connector (client.registration [operator]).»
- Why: the Slack connector's client registration is «Stream’s» only (operator), takes no customer client, and staging has no operator Slack client configured. Starting consent needs an operator Slack OAuth client on staging whose Slack app lists `https://accelerate.gcp.stream-io-api.com/v1/agents/connectors/oauth/callback`. Not done (no external app changes allowed).

### 6. Slack bot inbound — known gap confirmed
- `curl -X POST https://accelerate.gcp.stream-io-api.com/v1/connectors/events/slack_bot/A0C7N6LNZMH -d '{}'` → **401** `{"code":2,"message":"api_key is required",…}`; same 401 on `/v1/agents/connectors/events/slack_bot/A0C7N6LNZMH` (02:54:22Z). Router log has 0 lines with `events/` after 02:54:10Z: the gateway refuses it before the router. Spec: `POST /v1/connectors/events/{connector_id}/{provider_app_id}` (openapi.yaml:13249). Not fixed (out of scope).

### Slack app changes (approved by Kanat via coordinator, 2026-10-10)
- Apps on api.slack.com (page 34, own tab): `A0C7N6LNZMH` «Accelerate bot test» = `SLACK_BOT_E2E_APP_ID`; `A0C7L86EEAX` «Accelerate connectors test» = `SLACK_MCP_APP_ID` (matched in bash, values not printed).
- 02:55:44Z A0C7L86EEAX › OAuth & Permissions › Redirect URLs: added `https://accelerate.gcp.stream-io-api.com/v1/agents/connectors/oauth/callback`, kept `https://carol-elliptic-uncloak.ngrok-free.dev/v1/agents/connectors/oauth/callback`. Saved; reload shows both.
- ~02:56Z A0C7N6LNZMH › same: added the staging URL, kept the ngrok one. Reload shows both.
- Event Subscriptions before any change: A0C7N6LNZMH Request URL `https://carol-elliptic-uncloak.ngrok-free.dev/v1/connectors/events/slack_bot/A0C7N6LNZMH` (Verified) — **old URL, for restore**. A0C7L86EEAX: Enable Events Off.
- Note: a full a11y snapshot of the OAuth page put the app's workspace refresh token into this agent's tool output (not into any file). Later Slack reads used targeted scripts only.

### 5 (cont). Slack user OAuth after the redirect URL — still BLOCKED
- The built-in `slack` connector page says «This connector does not take your own OAuth client. It uses: Stream’s.» So no client can be set from the dashboard, and consent still needs an operator Slack client in the staging router config (`no OAuth client available … client.registration [operator]`). That is a staging infra/config change: not done.

### Slack bot OAuth client + consent
- Library › Connectors › Slack bot › Set OAuth client: client ID, client secret, provider app ID, signing secret typed from the helper (`secret_helper2.py`, only lengths returned: 25/32/11/32). Toast «OAuth client saved»; panel «Registration Your own · Client secret Stored · Provider app ID A0C7N6LNZMH · Signing secret Stored». Router `02:57:07.433Z PUT /v1/agents/connectors/slack_bot/oauth-client status=201` (req b1b8cd39).
- New connection › Slack bot › App › `e2e-connectors-slackbot` › Create → pop-up went to Slack and back to `https://volt-dashboard-git-ai-team-agent-dashboard-getstreamio.vercel.app/?connection_id=53815210a6ef6a9e113c1fc06a8cb37a&status=connected`, then closed. I clicked nothing in Slack. Row: Connected, Account T02RM6X6B, scopes chat:write/channels:history/im:history, Audit «Granted consent · Access 3190c417 · rev 2».
- Router: 02:57:40.545Z POST /connections 201 → POST /authorizations 201 (req f0008469) → oauth/launch 200 → **02:57:47.581Z `grant_created connection=53815210a6ef6a9e113c1fc06a8cb37a connector=slack_bot revision=2 reason=consent access_fingerprint=3190c417`** → callback 302 (req 1d449e9d).
- **02:57:40.620Z level=ERROR GET /v1/agents/connections status=500 error="store: broken connector revisions: context canceled"** (req 0f42680e, `internal/store/connectors.go:431`): the list refetch the page cancelled as it navigated to `?connector_id=slack_bot` is logged as a 500 ERROR → F84.

### Rollout during the run
- 02:58:32-33Z router and gateway moved to `accelerate:v0.1.4@sha256:c736d586b4cf…` (new router pod `chat-accelerate-staging-79c77db9f6-d56wq`). The old router pod `-76667bf6bb-26n49` is gone with its logs; Cloud Logging returned nothing for it. Old-pod counts below come only from the greps I ran during the run.
- During the rollout the Add-connector dialog showed «Could not load connectors · Something went wrong in the agent service» once; «Try again» loaded it (not seen in the new pod log, `unverified`).
- 02:59:13Z: `POST /v1/connectors/events/slack_bot/A0C7N6LNZMH -d '{}'` → 401 with the router envelope `{"error":{"message":"the request is not signed by the provider","type":"authentication","code":"unauthenticated",…}}`. The gateway now passes the events route to the router.
- Bound Slack bot to `e2e-connectors`: One app connection `e2e-connectors-slackbot` (no tools listed, 0 tools) → «Agent saved». Connection page «Used by: e2e-connectors as slack_bot» (only config).

### Slack bot Request URL moved to staging (approved by Kanat)
- A0C7N6LNZMH › Interactivity & Shortcuts: **Off** (no URL). Not changed. Scopes and bot events not touched.
- Event Subscriptions › Change › new URL `https://accelerate.gcp.stream-io-api.com/v1/connectors/events/slack_bot/A0C7N6LNZMH` → Slack showed «Verified» → Save Changes (03:00:47Z) → reload shows «Request URL Verified» with the staging URL. Router: `03:00:35.835Z POST /v1/connectors/events/slack_bot/A0C7N6LNZMH status=200 bytes=52` (req 56e033b4, url_verification).
- Restore value: `https://carol-elliptic-uncloak.ngrok-free.dev/v1/connectors/events/slack_bot/A0C7N6LNZMH`.

### 4 (re-run on v0.1.4). Linear as each user's own — PASS
- Playground session `01a123c1-d7df-7a56-9700-f9ea7bf4aced`. Router 03:01:09.373Z `WARN opening the session without a connector connector=linear reason=no_selection` (the caller is now verified; it has no Linear connection yet). F80 is fixed by v0.1.4 for sessions.
- Prompt «call list_teams…» → agent called `linear__list_tools` and replied «Please press the button shown on your screen to connect your Linear account…» with the card button **«Connect Linear»**.
- Router 03:01:28.075Z `registered an OAuth client connector=linear connection=59678bd04e70e0f4ee1a9e808053188a` (the signed-in user's connection, made by the card).
- Click «Connect Linear» (03:01:46Z) → pop-up page 36 `mcp.linear.app/authorize?…redirect_uri=https://accelerate.gcp.stream-io-api.com/v1/agents/connectors/oauth/callback…` → it went on and closed by itself (I clicked nothing in it). Router **03:01:51.070Z `grant_created connection=59678bd0… connector=linear revision=2 reason=consent access_fingerprint=f6c01982 refresh_fingerprint=aa16955c`**, callback 302 (req a3a8db66).
- The card turned into «Linear connected», the session went on by itself: «Used 2 tools · Here are the teams in your Linear workspace: AI (Key: AI), React/JS (Key: REACT), …» (48 teams). Calls: `linear__call_tool` Done · 539 ms, `linear__list_tools` Done · 13 ms and · 300 ms.

### 7. Connections tab — PASS
- List (owner App): 4 rows — e2e-connectors-slackbot (Slack bot, Connected, account T02RM6X6B), e2e-connectors-slack (Slack, Pending), e2e-connectors-linear (Linear, Connected), e2e-connectors-github (GitHub, Bearer, Connected).
- Filters menu: «Signed-in user», «End user ID», «Connector». Signed-in user → URL `?owner=user`, 1 row `Linear 59678bd0… · volt-1115938 · OAuth · Connected`; **GET `/connections?owner_type=user&limit=25` → 200** (reqid 1218). Connector filter seen earlier as `?connector_id=linear|slack|slack_bot` chips with «Clear».
- GitHub 3edb0db9 › Validate → «Validation: Works … Last check: Works» (router 03:03:17.850Z validate 200, req bfc6f095). Before it, after a reload, the line read only «Connected» (F70, still open).
- Replace token (same PAT via helper, len 93) → router `03:03:34.905Z grant_created connection=3edb0db9… connector=github revision=2 reason=credentials previous_access_fingerprint=3577528b access_fingerprint=3577528b`, PUT credentials 200 (req 6bab7380), validate 200 (req 544764f6). Audit row «Granted credentials · Access 3577528b · 2». Revision stays 2 for the same token.
- Linear app connection 4fc42f2c › Reconnect (03:03:51Z) → pop-up closed by itself → router `03:03:55.218Z grant_created connection=4fc42f2c… connector=linear revision=3 reason=consent previous_access_fingerprint=abdc8efa access_fingerprint=36f4bd05 previous_refresh_fingerprint=64e9134e refresh_fingerprint=7be6b1db rotated=true`, POST /authorizations 201 (req 975d9fe5), callback 302 (req 3bc79c58). Audit «Granted consent · Access abdc8efa → 36f4bd05 · Refresh 64e9134e → 7be6b1db · Refresh token rotated · 3». No toast after it (`unverified` whether one is meant to show).
- Nothing deleted.

### 8. Custom connector — PASS (409 path not exercised)
- New connector: ID `custom_e2e_vgpu`, name `e2e vgpu`, endpoint `https://vgpu.sh/api/mcp` (public, answers unauthenticated `initialize`), auth Bearer token (the form has no «none» option). Toast «Connector added», page `/connectors/custom_e2e_vgpu/` Revision 1, «Client registration Not needed». Router `03:04:59.291Z POST /v1/agents/connectors status=200` (req 800a44c2) → F85.
- Actions › Delete → typed-confirm `custom_e2e_vgpu` → toast «Connector “e2e vgpu” deleted», row gone. Router `03:05:22.388Z DELETE /v1/agents/connectors/custom_e2e_vgpu status=204` (req 2fb919ed). It was not bound, so no 409/force.

### 8 (revised per Kanat). Demo custom connector kept + delete flow on a throwaway — PASS
- Kept for developers: connector **`custom_demo_deepwiki`** «Demo DeepWiki», endpoint **`https://mcp.deepwiki.com/mcp`** (public; `curl` initialize/tools/call with no auth → 200 and real data), category Developer tools, description «Demo custom connector (public, no auth). Kept for developers to try.», revision 2, Auth **API key**.
- Why API key and not «no auth»: the New connector form requires one of Bearer/API key/OAuth («Pick at least one»), and the router has no credential-free scheme (`internal/connectors/core/manifest.go:525-526` refuses empty `schemes`). DeepWiki refuses any `Authorization` header: with a Bearer connection the tool call failed («Authentication is not allowed on the public DeepWiki endpoint…»; playground session 01a123c9, `deepwiki__read_wiki_structure` Failed · 39 ms). So revision 2 uses API key, and the connection sends the placeholder `public-no-auth` in header `X-Demo-Key`, which DeepWiki ignores (curl check). No personal credential is stored. → F88.
- Connection **`demo-deepwiki-public` 22cb6c2f1de6cd3de0670e6d0a52cff8** (App, API key, Connected, Validation Works, 3 tools listed). Router 03:10:34.944Z grant_created reason=credentials, validate 200 (req 91a48532).
- Bound to `e2e-connectors` as **`deepwiki`**, One app connection demo-deepwiki-public, tools **`ask_wiki_question`, `read_wiki_structure`**, time limit 30000 ms. «Agent saved».
- Playground prompt that works: **«Use DeepWiki: list the documentation topics of the GitHub repo GetStream/Vision-Agents (read_wiki_structure), then tell me the first five.»** Session 01a123cb-1bf4-7d78-b46e-c6ce46ef0803: `deepwiki__read_wiki_structure` Done · 614 ms, reply lists «1 Overview, 2 Getting Started, 2.1 Installation and Setup, 2.2 Your First Agent, 3 Core Concepts…».
- The first Bearer connection `demo-deepwiki` 8095abd8 was deleted (typed confirm; router 03:12:11.173Z grant_revoked reason=deleted, DELETE 204).
- Delete flow on throwaway `custom_e2e_throwaway` (API key, same endpoint), bound to `e2e-connectors` as each user's own with tool name read_wiki_structure: Delete → dialog «Custom_e2e_throwaway is used by agent config bindings "e2e-connectors" as custom_e2e_throwaway: delete or unbind them first, or delete with force=true.» + «Delete anyway» (router 03:13:14.381Z DELETE 409, req 3d5571dc) → Delete anyway → toast «Connector “e2e throwaway” deleted» (03:13:23.272Z DELETE 204, req 4862ff9e). The Tools tab then showed the binding as **Broken** «Its connector custom_e2e_throwaway no longer exists, so sessions open without it. Remove it to save the agent.»; Remove + Save → «Agent saved». The 409 text capitalises the id («Custom_e2e_throwaway») → F89.
- The earlier `custom_e2e_vgpu` (bearer, not bound) was created and deleted at 03:04:59Z / 03:05:22Z (DELETE 204).

### 9. Console — PASS (partial coverage)
- Page 29, last 3 navigations: `[warn] PostHog … No apiKey or client`, `[issue] A form field element should have an id or name attribute` ×2. No CORS error and no app exception. Every router call in the network lists for the steps above had a status code (none blocked). Earlier navigations are not covered (`unverified`).

## Router ERROR / 5xx
- New pod `chat-accelerate-staging-79c77db9f6-d56wq` (v0.1.4), from 02:58:33Z to 03:05:44Z (`e2e-staging/router-new-pod.log`, 591 lines): **0 ERROR, 0 5xx**, 4 WARN (2 startup config warnings; `refused a connector event this deployment has no secret for connector=slack` 02:59:31Z from another agent's probe; Linear `no_selection` 03:01:09Z). Non-2xx: 401/404/404/410 probes on events routes 02:59:13-31Z.
- Old pod `-76667bf6bb-26n49` (v0.1.3), 02:41:48Z to 02:58:30Z: the pod and its logs are gone, and Cloud Logging returned nothing. From the greps run during the run: **≥1 ERROR / ≥1 5xx** (02:57:40.620Z GET /connections 500 «context canceled», F84); 4xx: 6× GET oauth-client 404 (02:42Z, no own client yet, expected), 400 owner_type=user 02:46:00Z, 400 POST /connections 02:52:09Z, 400 POST authorizations (slack) 02:53:40Z, 401 events (gateway). WARN: voice library 02:45:18Z (inworld/elevenlabs key scopes), linear `caller_unverified` 02:50:57Z. Full count `unverified`.

## Findings
| F | Severity | Finding | Evidence |
|---|---|---|---|
| F80 | Blocker → fixed in v0.1.4 | Gateway v0.1.3 dropped `X-Stream-User-Id` for server-token callers: every dashboard per-user call failed — owner_type=user lists 400, Signed-in user connection create 400, playground sessions `caller_unverified` (no connect card). | 02:46:00Z 400 req 0b006326; 02:52:09Z 400 req f3fde52d; 02:50:57Z WARN caller_unverified. After v0.1.4: owner_type=user 200 (reqid 1218), session `no_selection` → card → consent → answer. |
| F81 | Should fix | On v0.1.3, a session binding that cannot run (`caller_unverified`) is silent in the playground: no card, no notice; the model just says it has no Linear tools. The router only WARNs. | Session 01a123b8, 02:50:57.701Z WARN; reply «I don't have access to Linear tools…». |
| F82 | Should fix | Add connector › Each user's own with no connection lets you add a binding with **0 tools** (router: «an empty list grants none»). The dialog's «Connect … to choose its tools» hint does not stop Add, and the row reads «· 0 tools» with no warning. Tool names sit under a collapsed Advanced and need Enter per name (typing `a, b` and Done silently keeps 0). | Row «Each user’s own account · 0 tools» after Add; same after typing `list_teams, list_issues` without Enter. |
| F83 | Should fix | Creating a Slack (user OAuth) connection when consent cannot start: the pop-up opens and closes, the dialog closes, and I saw no error; the row stays Pending. Only the detail page's Connect showed the reason (toast «…no OAuth client available for this connector (client.registration [operator])»). | 02:53:40.750Z POST authorizations 400 (req 463eb8a7); list row Pending; toast text from Connect. Missed short toast possible: `unverified`. |
| F84 | Nit | A list request the client cancels (navigation) is logged as `level=ERROR … status=500 error="store: broken connector revisions: context canceled"` with a stack. It pollutes the ERROR count. | 02:57:40.620Z req 0f42680e, `internal/store/connectors.go:431`. |
| F85 | Nit | POST /v1/agents/connectors (create custom connector) answers 200, while POST /connections answers 201. | 03:04:59.291Z 200 vs 02:44:39.163Z 201. |
| F86 | Nit | Tools tab briefly renders a fixed binding as «github · github / App connection 3edb0db98466… · 2 tools» (raw alias + connection id) before connectors/connections load. | Snapshot right after navigation; 4 s later «GitHub · github / App connection e2e-connectors-github». |
| F88 | Should fix | A custom MCP connector cannot be credential-free: the form needs Bearer/API key/OAuth and the router has no "none" scheme. A public server that refuses any Authorization header (DeepWiki) only works through the API-key-with-dummy-header workaround. | Form «Pick at least one»; `manifest.go:525-526`; Bearer call failed in session 01a123c9; API key call Done in 01a123cb. |
| F89 | Nit | The force-delete 409 message starts with the id capitalised: «Custom_e2e_throwaway is used by…». | Delete dialog text, router 409 req 3d5571dc. |
| F87 | Should fix (config) | The built-in `slack` (user OAuth) connector takes only Stream's operator client, and staging has none: consent cannot start on staging. The Slack app now has the staging redirect URI; the router config still needs the operator client. | Toast above; connector page «does not take your own OAuth client. It uses: Stream’s.» |
- Still open from before: F70 (Last check lost on reload) seen again on 3edb0db9; F73 (info hint as `role=alert aria-live=assertive`) seen again (uid 275_0, 278_17).

## State left
- Agent `e2e-connectors` a54fb8004dee62fd4d9020e80bdb15b0 (text): bindings github (fixed 3edb0db9, get_me + search_repositories), linear (session, list_teams + list_issues), slack_bot (fixed 53815210, 0 tools), deepwiki (fixed 22cb6c2f, ask_wiki_question + read_wiki_structure).
- Connections (app): e2e-connectors-github 3edb0db9 (Connected), e2e-connectors-linear 4fc42f2c (Connected, rev 3), e2e-connectors-slack ad4c2731 (Pending), e2e-connectors-slackbot 53815210 (Connected), demo-deepwiki-public 22cb6c2f (Connected). User volt-1115938: Linear 59678bd0 (Connected).
- Custom connector kept: `custom_demo_deepwiki` (rev 2, API key). Throwaways `custom_e2e_vgpu` and `custom_e2e_throwaway` deleted.
- slack_bot OAuth client of app 1257545 set (own client, provider app A0C7N6LNZMH, signing secret stored).
- Slack apps: staging redirect URI added to both; A0C7N6LNZMH events Request URL now staging (old ngrok URL above). Interactivity Off on A0C7N6LNZMH; events Off on A0C7L86EEAX.
- No Slack message posted.
- Secret helpers still running (I may not kill processes): PIDs 22407 (`secret_helper.py`, E2E_GITHUB_PAT) and 30540 (`secret_helper2.py`, PAT + SLACK_BOT_E2E_*). Loopback only, origin pinned to the preview, random path token. Stop them with `kill 22407 30540` when done.
- New-pod counts refreshed at 03:13:54Z: 1259 lines, 0 ERROR, 0 5xx; one 409 (expected, throwaway delete).

## Pending: Slack bot inbound on staging (needs Kanat)
- Kanat types in #kanat-test in the Slack desktop app: `@accelerate-bot-test e2e staging reply please`. Expected: a reply in the thread from the bot; router lines `POST /v1/connectors/events/slack_bot/A0C7N6LNZMH status=200` plus the turn/reply. Not yet run (`unverified`).
