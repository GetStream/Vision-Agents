# Unreleased

## Features

### Customers list their usage events

`GET /v1/usage/events` (`listUsageEvents`, server-side only) pages the usage events of the calling app, or with `scope=organization` of every app of its organization (`app_id` narrows it to one), newest first by `occurred_at` then `usage_id`: `UsageEventPage` (`items`, `has_more`, `next_cursor`) of `UsageEvent`, each with its `usage_id`, `occurred_at`, `app_id`, `api_key` (or the origin of work no key started, such as `hook:chat`), `user_id`, `team` (read off its `team:<name>` category), `usage_categories`, `agent_id`, `call_id`, `session_id`, `meter`, `provider`, `model`, `purpose`, `quantities`, `attributes`, `cost_micros`, `billable`, `success`, `error_code` and `tags`; never the tier, the cost at the vendor's rates, the price version or the producer. It filters by `from` (inclusive), `to` (exclusive), `meter`, `api_key`, `user_id`, and `team` or `category`, up to 200 a page (`limit`, 50 by default) continued by `cursor`. With `scope=app`, the default, an `app_id` other than the caller's is a 400, as are `user_id` with `scope=organization` and no `app_id`, `team` with `category`, and a cursor this list did not answer; with no usage service, or for an app in no organization, it answers 404 `billing_not_enabled`. A server-side caller's `api_key` and `user_id` query parameters filter, as on `getSpend`. The usage service's customer view gains `GET /v1/accounts/{organization_id}/usage` (`readAccountUsage`), `listUsage`'s filters and cursor without what only the operator reads, which the router reads (AI-1985).

### Customers read their account, credit, teams and prices

`GET /v1/account` (`getAccount`) answers the caller's organization's status, tier, balance (its credit less what it spent and what work in flight holds, the figure `X-Stream-Balance-Micros` says), granted and spent, the balance warning threshold with `balance_warning_source` (`tier`, `account`, or `organization_policy`, which wins) and `balance_state` (`ok`, `low`, `exhausted`), the settings the operator allows (`limits`) with those set for this organization alone named in `own_limits`, and the operator's spend hard limit with this window's spend and room (`spend_hard_limit`, absent when there is none); never why it is suspended. `GET /v1/account/credits` (`listCredits`) pages its credit entries, newest first, without the actor or the note, and with a reference only on an entry the payment processor wrote. `GET /v1/teams` (`listTeams`, `app_id` to narrow it to one app) lists every team a spend limit names or this month's work carried, with the apps that carried it, its spend in each current window and its limits; staff read any organization's with `GET /v1/manage/teams?organization_id=` (`manageListTeams`). All three are server-side only; the first two answer 404 with the new code `billing_not_enabled` on a router with no usage service, or for an app in no organization. `GET /v1/prices` (`getPrices`, tag `Prices`, any credential, a device's included) answers the price list in force at the rates customers are charged, with the change behind each entry, and what scheduled changes will set; it names no vendor rate, markup, unpriced item, source or author. `getSpend` takes `scope=organization` (every app of the caller's organization) and groups by who asked with `group_by=who:app`, `who:api_key`, `who:user` (`<app_id>/<user_id>` across the organization), `who:team` or `who:category`; a cost label named `app` is still `group_by=app`. `getLimitUsage` and `manageGetLimitUsage` take up to 100 subjects of one kind, by repeating `app_id`, `api_key`, `user_id`, `team` or `category`, and answer a list of `SubjectUsage` in the order asked, a list of one for one subject; `room_micros` of the spend hard limit now subtracts what work in flight holds. The usage service gains a read-only customer view on the ingest token, `GET /v1/accounts/{organization_id}`, `/credits` and `/teams`, which the router reads (AI-1985).

### Forgetting an end user in the usage records

`router usage forget-user --app <app> --user <user id> --actor <who> [--dry-run] [--settle 2m]` replaces an app's end user id with `forgotten:` and 32 random hex characters wherever the usage records keep it: the requests of every shard, the usage service's events and windows (through its new `forgetUser` management operation, `POST /v1/manage/users/forget`, then again after `--settle` for a batch shipped in between), the user's spend counters, holds and quota buckets in Redis (deleted by name), and in the shared database the user's own policy and user-level limits (deleted, so the app's defaults apply) and the id on their limit resets, grants and campaigns. Usage and cost stay under the app. It prints each step's count and writes one `end_user` audit row (source `cli`) naming the replacement and the actor, never the id; `--dry-run` prints the counts and changes nothing, and `--replacement <the id a run printed>` finishes a run that stopped under the same replacement. It needs `ROUTER_USAGE_URL` and the usage service's manage key in `ROUTER_USAGE_MANAGE_KEY`. Sessions, conversation history and `agent_logs` keep their own retention (AI-1826, #1259).

### A tool on a speech-to-speech conversation can say how it is going

`/v1/sts/stream` now takes `tool_update` (`tool_call_id`, `text`, `verbatim`) for a call the model is waiting on, and has the model say it the way an agent session does: once no reply is being spoken and the caller is not talking, a newer update for the call replacing one not yet said, and the call's `tool_result` dropping it. A status is said in the model's own words, `verbatim` words as written or close to them. An update for a call the model is not waiting on, or to a model that cannot be told about an open call, is dropped. `tool_hold` is answered with `tool_held` for an open call, and with `tool_cancel` for one the model cancelled or was answered for. In Python, `stream.STS` runs each function with `current_call()` set, so `update()`, `say()` and `disallow_interruptions()` work there as they do on `stream.Accelerated`; before, they did nothing.

### Spend limits on every subject, through /v1/spend/limits

`/v1/spend/limits` creates, lists, reads, changes and deletes the organization's spend limits, on the organization, an app, a key, an end user of an app, a usage category or a team, across the organization or within one app, over daily, weekly, biweekly or monthly UTC windows; a key's or a user's limit at organization or app level is a default for each one. `GET /v1/spend/limits/usage` answers what one subject spent in its current windows against its limits, with the operator's spend hard limit beside them, and `POST /v1/spend/limits/usage/reset` lets a subject spend its limits again until the windows end, leaving the usage records as they were. A limit above a parent's is a 400 `limit_above_parent`, a second on the same subject, level and interval a 409 `limit_exists`, a category starting `team:` a 400 `use_team`, a reset of a subject with no limit a 404 `no_limit`. Every operation is server-side only, and one server-side credential manages the whole organization. Stream staff reach the same operations under `/v1/manage/spend/limits` (AI-1918).

### Daily quotas for named users and for a whole organization

`PUT /v1/policies/users/{user_id}` gives one end user of the calling app a daily quota of their own (`quota`, messages and tokens per UTC day), which wins over the app's, the organization's and the router's default, each number on its own; `DELETE` removes it and the user falls back to the defaults, and `GET /v1/policies/users` lists the app's users with one, a page at a time. `GET /v1/policies/users/{user_id}` answers any user's own quota, the quota they are held to with who set each number (`user`, `app`, `organization` or `default`), and what they spent today toward a quota the customer set. An organization's policy now takes `quota` too (it was a 400), as the default for every user of its apps. The same user id in two apps is two users. A user's policy holds a quota only. Stream staff read and write an organization's, an app's or a user's policy under `/v1/manage/organizations/{organization_id}/policies/{scope}/{id}`, held to the customer's rules and audited with source `manage`. Migration `20261112130000_user_scope` lets a policy belong to a user (AI-1918).

### One-time grants past a spent limit

`POST /v1/spend/limits/grants` gives one subject (the organization, an app, a key, a user, a team or a category) a one-time amount it may spend once one of its own limits is spent, until it is used up, expires or is revoked (`POST /v1/spend/limits/grants/{grant_id}/revoke`); `GET /v1/spend/limits/grants` lists them with what each paid for and its state (`live`, `used`, `expired`, `revoked`). A grant lifts only its subject's limits: a parent's limit, credit and the operator's spend hard limit still refuse. Rows of work admitted on a grant name it in `requests.limit_grants` (migration `20261112121000_requests_limit_grants`, which builds its index concurrently). New codes `invalid_expiry`, `too_many_grants` and `grant_not_found` (AI-1918).

### Money is set aside before work starts, and every router counts every other's

Before billable work starts the router now reserves what it may cost: a search, an image, a detection, a knowledge or memory call or a text its estimate, an LLM response its prompt and its `max_output_tokens` (4,096 when unset), renewed while it streams, a TTS utterance its characters, and a stream (STT, STS, runtime, a sandbox) a minute at a time, renewed while it runs, priced with the attributes its rows carry (a sandbox's class among them). A speech-to-speech conversation whose renewal is refused because its money ran out refuses its next turn with that 402 (or 403 `account_suspended`) and closes, since nothing else stops a voice model replying; one refused because its money could not be checked (the 503 `spend_check_unavailable` below, or a balance unread for too long) refuses that turn and stays open, taking turns again once a renewal lands. A transcription keeps running. The reservation is checked against credit, the spend hard limit, spend limits and grants as admission is, answered from memory, and settled at what the work cost once it is recorded. Routers share what they hold through the deployment's Redis (`ROUTER_REDIS_ADDR`), so an organization spending on several routers at once is stopped on all of them within about a second. When a router cannot write to that Redis for 10 s, work that costs money is refused with `503 unavailable` and the new code `spend_check_unavailable`, with `Retry-After: 5`, until it can; a model that costs nothing is still served, and a router with no Redis keeps its holds to itself and never refuses for this. The router's alerts loop pages `holds_flush_failing` once a change has waited over 10 s unshared (`router_holds_unflushed_oldest_seconds`): customers' billable work is being refused, so restore the router's connection to `ROUTER_REDIS_ADDR`; it clears with the first write that lands. Every operation that declares the 402 now declares the 503 (`#/components/responses/Unavailable`) (AI-1884).

### Sandbox code can ask a user who has not connected yet to connect

A `session` binding with `sandbox: true` whose user has no connection yet, in a chat session, now gives the sandbox its env var at once. A request with it is refused with a 401 until the user connects; `run_code`'s description tells the model to call `<alias>__list_tools`, which shows the user the connect card and waits. After the consent, the next run sends the user's token, within about 20 seconds under the Daytona broker. A required binding still refuses the session, and a voice session gets nothing, as before (AI-1960). Each `run_code` result also names, until the user logs in, every such binding not yet connected and tells the model, if the code needed it (a 401 from its hosts), to call its `<alias>__list_tools`; under the Daytona broker the run that binds the login says to wait 20 seconds on a 401 (AI-1969). In a chat, a run of code that finds such a binding not connected now shows the user its connect card itself, without waiting and with no tool call by the model, once per open consent; once the user connects, the agent is told to carry on, and the next run sends their token (AI-1982).

### Finished scheduled-task history is deleted after 90 days

A daily job on every router deletes the runs of scheduled tasks that finished more than 90 days ago, and the tasks that completed or failed more than 90 days ago with their runs. An active or paused task is never deleted, however old; its old runs are. Each row copies the user's instructions, and a daily task added a run a day for good. `router replicate` now imports every scheduled task that is active on the source as paused, with its worker's claim cleared: the source goes on delivering it until the move is cut over, so the copy would have delivered it twice. Resume the tasks at cut-over.

## Bug Fixes

### An error in an agent.yaml connector entry names its line in the file

In the Go SDK, a connector entry with a key nobody knows, or a value of the wrong type, named a line counted from the start of the entry. It now names its line in `agent.yaml`, and a key nobody knows is refused with the keys a connector takes (AI-1767).

### The check before a sandbox run no longer shortens the code's run

Under the egress proxy broker with an `https://` proxy, the check of the TLS tunnel before each `run_code` used up part of the code's run timeout, so code that ended within its timeout could be cut short. The check is now bounded on its own, by the sandbox's timeout for any call to Daytona, and the code has its whole run timeout (AI-1777).

### Code run again after the user connects a connector runs, instead of returning the old 401

When sandbox code got a 401 because the user had not connected a connector, its result tells the model to run the code again once they connect. The agent's repeat-call guard answered that identical second `run_code` from the first result, so the turn ended on the old 401 although the connection was live. A `run_code` result that says a connector was just connected, or that it is not connected and the user has connected it since, no longer answers a repeat; while the user has not connected it, the repeat is still answered from that result. Every other repeated call is still answered from its earlier result (AI-1987).

### A thread message the session refuses is given up at once

A message in a thread channel (Slack, SMS, WhatsApp, iMessage) whose session refuses the turn for any reason but a reply still being written is now given up at once and logged. Before, the router tried it again every 50 ms for up to two minutes, writing one more cancelled reply into the channel each time and holding the thread's turn. A message that arrives while a reply is running still waits for it. Only a scheduled task's follow-up now keeps a copy of its reply and files in memory, the one that is read back on a retry (AI-1556, AI-1557).

### Work a scheduled turn hands to the subagent is the customer's spend

Delegated work (a skill, a picture to look at, a summary of a long conversation) started by a turn a scheduled task fired was counted on the end user's daily quota, though the turn itself was not. It is now counted as the turn is: on nobody's day (AI-1560).

### Scheduled tasks follow config deletes and guest claims

Deleting an agent config now writes its tasks' occurrences still running, such as one held through quiet hours, `missed`, as pausing a task does; before, they stayed `running` (AI-1600). A task created, resumed or edited while its config is being deleted is refused with a 404 `agent_config_not_found` (REST) or `unknown_config` (the agent's tools), instead of being left active on the deleted config; the agent's `schedule_task` on a session whose config was deleted is refused the same way (AI-1598). Claiming a guest now moves the guest's scheduled tasks to the account, with their conversations (AI-1532).

### A scheduled task's reply carries the files its code made

A task delivered by SMS, WhatsApp or iMessage outside a thread now sends the files the turn's code made (`run_code` or delegated work), as an MMS or WhatsApp media after the words; before, only the words went out. A reply of files and no words is delivered on every channel, a thread's included, where it was a `no_reply` failure; a retry sends the same files without a second turn (AI-1951, AI-1950).

### A draining router no longer waits out a scheduled task's retry

A scheduled delivery that failed and was waiting to try again held the router's drain for up to 12.5 minutes, as long as the waits last. Once the router drains, the wait ends and the run is left to its lease: another router takes it up under the same run (AI-1561).

### A tool_approval that reaches no call is answered

A `tool_approval` on the session socket that no call waits for now gets a `tool_approval_ignored` frame with the `tool_call_id`, `request_id` and `turn_id` it named and a `reason`, and the socket stays open. This covers an approval sent twice, one for a call that already stopped, one naming a call nobody made, and one naming another `request_id` or `turn_id`. An approval of a durable command's call with its `request_id` and no `turn_id` gets one too, its `reason` naming the missing `turn_id`. Before, each was dropped without a word. Clients that do not know the frame ignore it; no `error` frame is sent. An approval sent before the `tool_result` of a call outside a durable command is answered as before: with nothing (AI-1881).

### A Google Drive or Docs grant holding full Drive access no longer needs a consent

`POST /v1/agents/connections/{id}/validate` answered `needs_scopes` for `drive.file` (Google Drive) or `documents` (Google Docs) when the grant held `https://www.googleapis.com/auth/drive`, which Google's tool pages take for `create_file`, `copy_file` and `update_doc`. A manifest's new `scopes.covered_by` names, per needed scope, the broader scopes that meet it; `google_drive` revision 6 and `google_docs` revision 5 say `drive` covers them. A connection moves to the new revision at its next consent.

### A connection is outdated only when a later revision changes what it reads

`definition_status` is `outdated` only when the connector's latest revision changes something the connection reads from the revision it pins: its scopes, auth, endpoints, tools or any other field. A revision that changes only `sensitive` or a source's `outbound` or `read_only_tools` leaves it `current`. The check of a config against a sensitive connector reads those fields from the latest revision. So BigQuery's revision 3 leaves a connection made on revision 1 or 2 `current`, and a connection made before Google Docs' or Google Drive's revision 3 was not outdated by that revision alone (AI-1878).

### BigQuery's `execute_sql_readonly` needs approval beside a sensitive connector

BigQuery revision 3 takes `execute_sql_readonly` out of `read_only_tools`. Its SELECT can call a remote function, whose code may send the rows anywhere. A config that binds it beside a sensitive connector such as Nessy is now refused unless the binding's `require_approval` holds it (AI-1877).

### The Python and JavaScript SDKs read `sandbox` and `require_approval` on an agent.yaml connector

A `connectors:` entry in agent.yaml may say `sandbox: true` and `require_approval: [...]` in the Python SDK (`plugins/stream`) and the JavaScript SDK, as it may in the Go SDK. Folder sync sends them on that binding. Before, both SDKs refused the keys. `sandbox` left out or `false` sends no key, and `require_approval` left out or null sends none, so the body is the one sent before. A `sandbox` that is not true or false, or a `require_approval` that is not a list, is refused when the folder is read (AI-1961).

### A connector tool whose schema changed no longer breaks its whole binding

When the provider lists a granted tool with another schema than the one approved, a session's binding now opens with its other tools (while it still offers one tool), also when the binding is required (reported in `connector_unavailable` as `tool_unavailable`) or has sandbox access (the sandbox keeps it), and when the person connects it in the conversation: that login no longer waits until its consent expires. A call of the left-out tool answers `tool_unavailable` with what gives it back: for a tool granted by name or by pattern, an admin's approval of the new schema (`POST /v1/agents/connections/{id}/tools/approve`); for a schema digest the agent config names, saving the config at the new schema. A login's `list_tools` names it under `left_out`. A granted tool the provider no longer lists is still unknown to the session, and still fails a required binding (AI-1932).

## Breaking Changes

### The call a tool function is handed is named toolCallContext

The `ToolCall` a tool function is handed is named `toolCallContext` (`tool_call_context` in Python, Ruby and Rust) in every SDK's signatures, examples and docs, rather than `call`, which read as the phone or video call. Only code that passed it by name changes: PHP's `Tools::call(..., toolCallContext: ...)` and Python's `calling(tool_call_context=...)`.

### A hosted tool that names no timeout has 30 seconds

A worker on `/v1/dispatch` whose `host_tools` names no `timeout_ms` now gives each call 30 seconds, the default for the tools a session's caller runs, rather than two minutes. A worker whose tools take longer says so with `timeout_ms`.

### A session's tool call is answered only by a watcher that runs it

Every watcher of a session's events socket still sees each `tool_call`, which now carries `run`: whether this watcher is the one to run it. A `tool_result` from a watcher that was not asked is dropped, so a browser watching a backend's session can no longer fail a call the backend was running, which it did by answering first. A watcher says which tools it runs with `?tools=a,b` when it attaches (empty for none). Without it, it runs the session's tools when it is on the side that created the session: a backend for a session a backend created, the same person for one a device created. A tool a backend declared never runs on a device, whatever the device declares. A call that no attached watcher runs fails at once instead of waiting out its timeout. A backend that answers the tools of a session a device created now has to name them in `tools`. The Go SDK names its registered functions when it attaches and leaves a call alone when `run` is false.

### Go's `ToolApprovalCommand.TurnId` is a `*string`

`turn_id` is optional in the session socket's `tool_approval`: an answer to a connector call held for approval needs none. The Go client's generated `acceleration.ToolApprovalCommand.TurnId` is now a `*string`, so code that sets it takes its address (AI-1640, #1182).

### The Swift SDK runs on iOS 15, and its sessions are `ObservableObject`s

`VisionAgentsCore` needs iOS 15 rather than 17. `AgentSession` and `VoiceSession` are `ObservableObject`s with `@Published` properties in place of `@Observable`, so a view holds one with `@StateObject` or `@ObservedObject` rather than `@State`. A `VoiceSession` passes its `session`'s changes on as its own, so observing it follows its conversation too. An app's own `@Observable` model that reads a session subscribes to its publishers, as the Swift demos' `AgentChat` does with `$conversation`.

### The Swift SDK's `User` and JSON values are StreamCore's

`VisionAgentsCore` now generates its client with the generator Stream Chat and Stream Video use, on StreamCore, and drops its own copies of what StreamCore already has. `User` is StreamCore's: `User(id:name:imageURL:)`, with `name` optional and `imageURL` a `URL` in place of `image`. `JSONValue` is gone in favour of StreamCore's `RawJSON`, in tool arguments and parameters, `AgentEvent.fields`, and `custom` on sessions and guests: `.object` is `.dictionary`, `.null` is `.nil`, and `stringValue`, `boolValue` and `arrayValue` are optional, so `arguments["id"]?.stringValue ?? ""` reads as before. `import VisionAgentsCore` brings both in. The SDK no longer depends on swift-openapi-runtime, swift-openapi-urlsession or swift-http-types.

### The Go SDK refuses `plugins:` and `plugin_events:` in agent.yaml

`agents.Load` and `agents.New` refuse a folder whose agent.yaml names `plugins:` or `plugin_events:`, with an error that names the key and points to `connectors:`. An empty or null list still loads and sends nothing, as before. Folder sync no longer sends `plugins` or `plugin_events`, and `Settings.Plugins`, `Settings.PluginEvents`, `PluginSettings` and `PluginEventSettings` are gone. Move each plugin to a `connectors:` entry (AI-1784, #1161).

### The Python SDK refuses `plugins` and `plugin_events` in agent.yaml

Connectors replaced plugins. `plugins.stream` now fails to load an agent folder whose `agent.yaml` names `plugins` or `plugin_events`, with a `ValueError` that points to `connectors`, and never sends either to the router. An empty key (`plugins: []`, or the key with nothing after it) names nothing and loads (AI-1785).

### An app's daily quota counts its backend's work for the user

An app's `quota` (messages and tokens per user per UTC day) counted only what the user's own devices asked for. It now counts everything recorded with the user, the app's backend acting for them included (a server-side request naming the user), in a counter of its own per user, so the third response of a user held to two is refused whether their device or the app's backend asks for it. A device is still counted by its address too. A backend over a user's quota is refused only work that asks a model for something (creating a session, `respond`, `responses`, a run, opening a session or stream socket), never a read. An anonymous caller (no verified token) is never held to the quota of the user whose id it claims, nor counted on their day: it is held to the app's or organization's quota, by name and address. The router's default (`ROUTER_RATE_LIMIT_MESSAGES_PER_DAY`, `ROUTER_RATE_LIMIT_TOKENS_PER_DAY`) is unchanged: it counts devices only. A refusal's message names the quota: `user <id>`, `every user of app <id>`, `every user of organization <id>`, or the router's default. A turn a scheduled task fires is still counted on nobody's day (AI-1918).

### Budgets moved out of policies into spend limits

An organization's and an app's `budget` are no longer part of their policy: `PUT /v1/policies/app` and `PUT /v1/policies/organization` with a `budget` answer `400 budget_moved`, and neither `GET` returns one. The router's migration `20261112120000_limits` moves every stored budget into the new `limits` table as the organization's or the app's own spend limit on the same interval, with the same amount, so nothing stops being enforced; it refuses to run while an hourly budget exists, since spend limits have no hourly interval (change it to daily first). A limit can also be set on a key, an end user, a usage category or a team, and is enforced in the same admission as before: `402 spend_limit_reached`, or `429 user_limit_reached` with `Retry-After` for a limit on one end user, which stops that user alone. `Budget` and `BudgetInterval` are gone from the spec and the generated clients. Spend limits are read from the usage service's counters every second when `ROUTER_USAGE_REDIS_URL` names the Redis the service mirrors to, and from the service itself every 5 s otherwise (AI-1918).

### Work that costs money is refused with a 402 when the organization may not spend, and a suspended organization with a 403

Before every routed session and every LLM response the router now checks the organization's money as well as its policies, from memory, read every 5 s in the background: from the usage service (`ROUTER_USAGE_URL`) when there is one, and budgets from Postgres. Work on a model that costs money is refused with `402 payment_required` and a code saying why: `insufficient_credits` (the organization's prepaid balance, less what it has spent since the last read, is gone), `system_limit_reached` (the operator's spend hard limit for the UTC day or month is reached, whatever the credit; the message names the interval, the amount and when it resets) or `spend_limit_reached` (an app or organization budget is spent). A model the deployment runs itself, or one that is free, is still served. A suspended organization is refused everything with `403 account_suspended`, and the message never says why. A spent budget was a 500, and a 403 `budget_exceeded` on `POST /v1/agents/runs`; it is now the 402 `spend_limit_reached`, and `budget_exceeded` is gone. The stream and session sockets carry the same `code` and `error_type` in their `error` frame and close with 1008. Telephony is held to the same money: buying a number, placing or transferring a call, starting a campaign and texting on a line are refused with the same 402 or 403 (a phone vendor bills them all); a call to the organization's number is refused (a TeXML `<Reject>` on Telnyx, before it is answered; over Stream SIP, where the bridge has answered the caller already, no agent is dispatched and the call is ended through Stream at once, for a telephony suspension too); a campaign refused midway pauses with the new `Campaign.paused_reason` (also `telephony_suspended` for a telephony suspension) and keeps its contacts; and a scheduled task run that failed for money records the refusal's code (`insufficient_credits`, `system_limit_reached`, `spend_limit_reached` or `account_suspended`) as its `error`. Inline speech and transcription refused for money answer the 402 or 403 rather than a 202 failure. The shared 403 response now also documents `account_suspended`.

`ROUTER_LIMITS_MODE` is `enforce` (the default), `shadow` (every refusal is worked out and logged as `would refuse`, and the request is admitted) or `off` (nothing is refused for money; work is still priced, recorded and shipped). Both of the latter log a warning at startup. A router with limits on refuses to start without `ROUTER_POSTGRES_DSN`, and in `auth.mode=proxy` without `ROUTER_USAGE_URL`: set `ROUTER_LIMITS_MODE=off` to run without them, as on a laptop with no database (AI-1884).

### A routing config's `price:` blocks are ignored: prices come from price files

The router prices every row from the vendors' list prices in `acceleration/prices/vendor.json`, embedded in the binary, with a deployment's own price file (`ROUTER_PRICES_FILE`) laid over it, and no longer from `price:` in `router.yaml` (which no longer has any) or in a deployment's own `ROUTER_CONFIG`. A config that still sets one loads, and the router logs a warning at start naming its models: until they are moved, a model `vendor.json` does not price costs nothing. `router prices export --from <config> --out own-prices.json` writes them as a price file; point `ROUTER_PRICES_FILE` at it and they are priced as before. A customer's own model (`custom/<name>`) is still priced from its stored rates (AI-1824, #1171).

### Stream staff's operations are out of the Go, JavaScript and Python clients

The operations only Stream's staff may call, the `/v1/ops/*` ones reached with `X-Ops-Key` (10DLC reviews, telephony suspensions, the operator's provider app), are tagged `Staff` and left out of the generated Go client, the JavaScript types and the Python client (`plugins/stream`). They stay in `openapi.yaml`. Code that called `ListUseCasesForReview`, `ReviewUseCase`, `SuspendTelephony` or their siblings through an SDK calls the path directly instead (AI-1825, #1167).

### A router's peers need the key encryption key to hand each other requests

A router node now answers a request another node forwards to it only when the forwarding node proves it belongs to the deployment, with a credential derived from the key encryption keyring (`ROUTER_AUTH_KEK`, `ROUTER_AUTH_KEK_V<n>`). A node accepts the credential of every key it holds, so rotating a key needs no pause. A deployment of more than one node must set the keyring on every node: a node without one refuses every forwarded request and logs a warning naming `ROUTER_AUTH_KEK` at startup. While a deployment rolls out this version, nodes on it refuse what nodes still on the previous version forward, until the rollout ends (AI-1720, #1138).

### Voice is started and stopped on a session, on a call named after it

A session is one conversation, and voice is something it turns on and off. Every session
starts in writing; `POST /v1/agents/sessions/{id}/voice` puts the agent on the call
`agent:<session id>`, and `DELETE` on the same path takes it off again. The history and the
Chat channel stay the same throughout, and a question typed while voice is on is part of the
same conversation as what was said aloud. To start talking straight away, create the session
with `start_voice: true`.

The call is always named after the session, so nobody names it any more:

- `call_id`, `call_type` and `text` are gone from `POST /v1/agents/sessions` and from the
  session socket's start frame.
- `call_id` is gone from `POST /v1/agents/sessions/{id}/fork`. A fork is always held in
  writing, because talking to two agents at once is not a conversation.
- Placing a call (`POST /v1/phone/calls`) takes an optional `session_id` instead of
  `call_id` and `call_type`, and returns the `session_id` to open with `start_voice`.
- A transfer takes the `session_id` it brings the human into instead of `call_id`.
- Attaching a number no longer takes `call_id` or `call_type`. Every caller lands in a call of
  their own, and the dispatched call carries the `session_id` to open for it. Numbers attached
  before this change still route into their old fixed call, so attach them again.

A session `id` you choose no longer has to be a UUID: any string of up to 64 letters, digits,
`-` and `_` works, such as your own record's id, as long as it does not start `support-` or
`thread-`. The same goes for the `session_id` a placed call names.

`GET /v1/agents/sessions/{id}` reads a session that ended as well as a running one, and tells
your backend its `summary`, `review_score`, `review_notes` and `usage` (tokens, requests and
cost over every model it called, typed and spoken), so what a conversation cost is read off
the session rather than a separate call record.

In Go, `SessionOptions.StartVoice` and `Session.Voice.Start` / `Stop` are new in `client`.
`agents.Agent.Join(ctx)` starts a session with voice, `Agent.Answer` opens the session for an
inbound call, and `Phone.Attach` takes only the number.

### `command_id` is `request_id`, and the SDKs send it for you

The id that makes a retried question answered once is called `request_id` everywhere a client
sees it: `POST /v1/agents/sessions/{id}/responses`, `POST /v1/agents/sessions/{id}/respond` and
its receipt, the session socket's `respond`, `interrupt`, `tool_result` and `tool_approval`
frames, the `tool_call` and `tool_cancel` events, and the dispatch `message` frame. The
`/commands/{request_id}` paths keep their URLs.

Every SDK now generates one for each text-only question inside `responses.create` and no longer
takes one from the caller; a dispatch worker answering a message still passes that message's id
back. A session not kept in Stream Chat ignores a `request_id` rather than refusing it.

### The Swift SDK is one package, and its views are Stream Chat's AI components

`VisionAgentsCore` is the only Swift package. `VisionAgentsUI`, `VisionAgentsRTC` and
`VisionAgentsChat` are gone: `VoiceSession` and `session.chat()` moved into core, which now
depends on Stream Chat (5.13+) and Stream Video and is iOS only. `ConversationView`,
`TranscriptView`, `Composer`, `AgentStatusView`, `VoiceCallView` and `AgentVideoView` are gone
too; show a conversation with Stream Chat's AI components (`StreamChatAI`, from the same
`stream-chat-swift`), as the Swift demos do. A tool may now ask the person first:
`AgentTool.approval` holds each call in `session.approvals` until `session.decide` allows or
declines it, and `LiveReasoning` puts a reply's whole reasoning back together from its live
updates.

### The Kotlin SDK is one module, and its views are Stream Chat's AI components

`vision-agents-core` is the only Android artifact, and everything in it is in
`io.getstream.visionagents.android.core`. `vision-agents-ui`, `vision-agents-rtc` and
`vision-agents-chat` are gone: `VoiceSession` and `session.chat()` moved into core, which is
now an Android library depending on Stream Chat (7.15.0+) and Stream Video. `ConversationView`,
`TranscriptView`, `Composer`, `AgentStatusView`, `VoiceCallView` and `AgentVideoView` are gone
too; show a conversation with Stream Chat's AI components (`stream-chat-android-ai-compose`).
`ChatFailed` and `CallFailed` are now `AgentsException.ChatFailed` and
`AgentsException.CallFailed`. Stream Chat 7.15.0 is published to `https://stream-io-repo.com`,
not Maven Central, so add that repository.

### `lcm` is now `decision_model`, with ten models behind it

The modality that answers typed questions with probabilities is called `decision_model`,
the name the industry settled on. The `Modality` enum value `lcm` is now `decision_model`, so
its providers are at `/v1/decision_model/providers` and its stats are under the new name.
Stored stats are renamed by a migration. A guardrail.md says `type: decision_model`, and one
that still says `type: lcm` keeps working.

Besides TypeSafe's Jev, the router now reaches the top ten decision models on OpenRouter
(`OPENROUTER_API_KEY`): Jev 1.13, GPT-6 Luna Decisions, Clef Flash, Clef, Span-01 Lite, d1,
Kev 4B, Decider V1.1 and Mercury Decide, plus its free tier. It also reaches both Deciders on
Perplexity's own API (`PERPLEXITY_API_KEY`), which is the only host left for Decider V1. The
default route still prefers Jev. Clef is what the `*-high-accuracy` shortcuts now resolve to.

### Talking over the agent no longer cancels the tools it is running

When the caller talked over a voice agent, every tool call still running was cancelled, and
the answer they had asked for was lost. A call now goes on: the model is told it is still
running, and the caller is told its result when it comes. It is stopped when the caller
withdraws what they asked for, which the model says with `<drop skill="..."/>`, or at the
binding's timeout. `press` and `transfer` are still cancelled, as is a call whose connector
binding sets `on_interrupt: cancel`. A binding that leaves `on_interrupt` out, which was
`cancel`, now lets its calls go on.

### Every client goes to the hosted router unless told otherwise

A client that is given no URL, and finds no `STREAM_ACCELERATION_URL`, now goes to Stream's
hosted router at `https://accelerate.gcp.stream-io-api.com` instead of `http://localhost:8080`,
so an app's code never has to name it. The JavaScript, Python, Ruby, PHP, .NET and Rust SDKs
also reach that router through the authenticating proxy without being told to, as Go already
did: `authenticate` is on for it unless passed as false. Kotlin's and Dart's `url` are optional
now, as Swift's already was. A router running locally is named the way it always could be,
with `url` or `STREAM_ACCELERATION_URL`, and reached by customer id:

```bash
STREAM_ACCELERATION_URL=http://localhost:8080
STREAM_ACCELERATION_CUSTOMER_ID=examples
```

### Every failure is an `error` object with a `type`, a `code` and a `doc_url`

A failure was `{"error": "..."}`; it is now an envelope, `ErrorResponse`, holding an
`ErrorDetail`:

```json
{"error": {"message": "no such agent config", "type": "not_found",
  "code": "agent_config_not_found",
  "doc_url": "https://getstream.io/agents/docs/api/errors/#agent_config_not_found"}}
```

`type` is an `ErrorType` and decides the status: `invalid_request` 400, `authentication` 401,
`permission` 403, `not_found` 404, `method_not_allowed` 405, `not_acceptable` 406, `conflict`
409, `gone` 410, `payload_too_large` 413, `unsupported_media_type` 415, `rate_limited` 429,
`internal` 500 and `unavailable` 503. `code` is what to branch on: each type has a code of its own
name, and the failures worth telling apart have their own (`validation_failed`,
`missing_customer`, `not_configured`, `agent_config_not_found` and the other `*_not_found`). More
codes may be added. A failure that is not the caller's is a 500 of type `internal` saying only
"something went wrong"; quote its `X-Request-Id`. Hand-written routes, the webhooks, an unknown
route and a refused socket handshake answer with the same envelope, where some answered in plain
text. A session on a node that cannot be reached is a 503 rather than a 502, and a failed data
export or change read is a 500 rather than a 503 carrying the database's words. A code has one
status wherever it is answered: a config, router config or plugin that does not exist is a 404
even when the body names it (it was a 400 in some places), and a feature this deployment does not
offer is a 400 `not_configured` (it was a 404 for sessions and recorded responses, a 410 on the
10DLC webhook). Every operation
declares its `500`.

Every SDK reads it into its router error, which now carries the status, `type`, `code`,
`message`, `doc_url` and the `X-Request-Id` as the request id, for a refused socket handshake as
well. Go returns a `*stream.RouterError` for `errors.As`. Swift's `AgentsError.http` now holds an
`HTTPFailure`, Rust's `Error::Router` a boxed `RouterFailure`, PHP names the code `errorCode`, and
Python raises `RouterError` for a refused socket where it raised aiohttp's handshake error.

### `user_plugins` is folded into `plugins`, and `plugin_options` moved onto each entry

An agent config names its catalog plugins in one list, `plugins`, in `agent.yaml` and on
`AgentConfig`, `AgentConfigRequest`, `AgentConfigPatch` and `SyncAgentRequest`. `user_plugins`
and `plugin_options` are gone: each entry is either a catalog id or an object
`{name, user, readonly, scopes, toolsets, tools}` (`PluginEntry`, a `oneOf` of a string and a
`PluginWithOptions`), and an entry with no options is answered as its id. `user: true` has each
end user connect their own account in the conversation, as it does on `mcp_servers`; left out,
the app connects the plugin once. The old keys are refused as unknown in `agent.yaml` and
dropped by the API. A config now gets a 400 for an id the catalog does not have and for a plugin
named twice. A migration folds the stored options and the `user_plugins` entries into
`plugins`; a plugin named in both lists stays the app's.

```yaml
plugins:
  - sentry
  - name: linear
    user: true
    readonly: true
    scopes: [read]
  - name: google_calendar
    user: true
```

The Go SDK (`agents.PluginSettings`) and the Python folder reader (`PluginSettings` in
`plugins/stream`) have moved, and the Python reader now accepts `scopes` and `user` on
`mcp_servers`. Other SDKs follow.

### `thinking_llm` is `subagent`

The model a voice agent hands its skills to is `subagent` again, in `agent.yaml`, on agent
configs and syncs, and on `Session` and call records, where `thinking_llm_used` is
`subagent_used`. Python's `define_agent` takes `subagent=`. `thinking_llm` is refused as
unknown in `agent.yaml`.

### `Dispatch.host` takes an agent and hosts its tools

A dispatch worker hosts an agent's own tools under its name, in place of an agent id and a
registry passed alongside. The router matches a hosted tool on a session's agent id or agent
name, so the name is enough. In Python (`plugins/stream`), `Dispatch.host(agent_id, functions,
timeout)` is `Dispatch.host(agent, tool_timeout)`, with `agent` from
`stream.Client().agent(name)` and its tools registered with `@agent.register()`. In
JavaScript, `host(agentId, tools, { timeoutMs })` is `host(agent, { toolTimeoutMs })`, hosting
`client.agent(name).tools`. In Go, `Dispatch.Host(agentID, functions, timeout)` is
`Dispatch.Host(agent, toolTimeout)`, on both `stream.Dispatch` and `agents.Dispatch`, hosting
`client.Agent(name).Tools()` or an `agents.Agent`'s. .NET, Ruby, Rust and PHP take the agent
the same way. The timeout is renamed to say what it is: how long the router waits for one
tool call, not how long the worker runs.

### The connector catalog answers `Connector` and `ConnectorPage`

`listConnectors`, `getConnector` and `createConnector` (`/v1/agents/connectors`) answered schemas named `ConnectorDefinition` and `ConnectorDefinitionPage`. They are `Connector` and `ConnectorPage` now; the JSON is unchanged. Go, JavaScript and Python clients use the new type names.

### Connections are `Connection` in the API, and a connector's client says `registration`

The connection operations are renamed: `createConnectorConnection`, `listConnectorConnections`, `getConnectorConnection` and `deleteConnectorConnection` (`/v1/agents/connections`) are `createConnection`, `listConnections`, `getConnection` and `deleteConnection`. Their schemas follow: `ConnectorConnection`, `ConnectorConnectionPage`, `ConnectorConnectionRequest`, `ConnectorConnectionOwner`, `ConnectorConnectionOwnerType` and `ConnectorConnectionStatus` are `Connection`, `ConnectionPage`, `ConnectionRequest`, `ConnectionOwner`, `ConnectionOwnerType` and `ConnectionStatus`. Paths and JSON are unchanged. The connector client's `policy` field is `registration` on the wire, on `Connector.client` and `CustomConnectorRequest.client`, and its enum `ConnectorClientOwner` is `ConnectorClientRegistrationMethod`; the values are unchanged. In a manifest, `client.policy` is `client.registration` and `rate_limit.leak` is `rate_limit.leak_per_second`, and every built-in connector has a new revision. Go, JavaScript and Python clients use the new names.

### Deleting a session deletes it; stopping one is `POST .../stop`

`DELETE /v1/agents/sessions/{id}` used to end a session and keep everything. It now deletes
the session: a running one is stopped first, then its turns, their items and what it taught
memory are deleted. The user's memories from other sessions are kept. To end a call and keep
the conversation, use the new `POST /v1/agents/sessions/{id}/stop`. Both answer 204 and are
open to the device holding the session.

In Go, `Session.Close` stops, and `Sessions.Delete` and `Session.Delete` delete. SDKs that
still close with `DELETE` now delete the conversation until they move to `stop`.

### Sessions are listed and searched with `querySessions`, and `project` is `project_id`

`GET /v1/agents/sessions` and `GET /v1/agents/sessions/search` are replaced by
`POST /v1/agents/sessions/query`, which takes `{filter, sort, limit, cursor}` in the body.
It answers three queries: every session, most recently updated first; a text search
(`{"text": {"$q": "..."}}`), best match first; and one project's sessions
(`{"project_id": "..."}`), most recently updated first. `agent` and `user_id` narrow any of
them, as do `config_id`, `state`, and `custom`, which holds the pairs a session must carry.
`created_after` and `created_before` are now one `created_at` range,
`{"$gte": "...", "$lt": "..."}`, half open so two windows that meet share no session. A field
or operator outside these is a 400. `project` is now `project_id` on session create, fork and
the session itself.

In Go, `Sessions.Query` and `Sessions.Search` call the new endpoint, and `Query` takes
`ProjectID`, `UserID`, `ConfigID`, `Custom`, `CreatedAfter`, `CreatedBefore`, `Limit` and
`Cursor`. `SessionOptions.Project`, `ForkOptions.Project` and `stream.Call.Project` are now
`ProjectID`. Other SDKs follow.

### Session, response and item lists page by cursor instead of offset

`GET /v1/agents/sessions`, `/v1/agents/sessions/search`, `/v1/agents/sessions/{id}/responses`
and `/v1/agents/sessions/{id}/responses/items` no longer take `offset`, and return a page
object instead of a bare array: `{items, has_more, next_cursor}`. Pass `next_cursor` back as
`cursor`, with the same filters, for the next page. A session opened or deleted while someone
pages no longer repeats or skips a row, and a deep page costs the same as the first.

In the Go SDK, `Sessions.Query` and `Sessions.Search` return `*acceleration.SessionPage` and
take `Query.Cursor` instead of `Query.Offset`. `Responses.List` and `Items.List` take a cursor
string instead of an offset and return the page. `Items.Unwind` follows the cursor as before.
In JavaScript, `sessions.query`, `sessions.search`, `responses.list` and `items.list` take
`cursor` instead of `offset` and return the page.

### One `update` for a session, in place of settings and instructions

`PATCH /v1/agents/sessions/{id}` (`updateSession`) changes a session's title, description,
custom labels, instructions, models and voice in one request, and returns the session as it
now is. A session that ended can still be renamed and relabelled. The id, the call and
incognito cannot change. `PATCH .../settings` and `PUT .../instructions` still work but are
deprecated. In Go, `session.UpdateSettings(client.Settings{...})` is now
`session.Update(client.SessionUpdate{...})`, and `agent.Sessions.Update(id, ...)` renames a
conversation without a live handle. In JavaScript, `session.updateSettings({...})` is now
`session.update({...})`. Other SDKs follow.

### Sessions are kept in Stream Chat by default, and `persist_conversation` is gone

`persist_conversation` has been removed from `POST /v1/agents/sessions` and from every SDK's
session options. Every text session is now kept in a Stream Chat channel, so any Stream Chat SDK
can read it back; pass `incognito: true` to keep nothing. Voice calls were already written to
Chat, and now show what the caller is saying as it is transcribed, through ephemeral message
updates, before storing the settled turn. A router without Stream credentials still runs a new
text session, without a channel.

What was true of a persisted conversation is now true of every text session: a user's
questions carry a `command_id` (the SDKs add it), rewinding is refused (fork at the response
instead), and the session ends when its last watcher disconnects, leaving the channel to
resume with `conversation_id`.

### The `lemonslice` and `liveavatar` plugins have been removed

`vision-agents[lemonslice]` and `vision-agents[liveavatar]` are gone, along with
`vision_agents.plugins.lemonslice` and `vision_agents.plugins.liveavatar`. Use `anam` for an
avatar.

### `ROUTER_AUTH_MODE` defaults to `api_key`, and `noauth` has been split in two

The old `noauth` did two unrelated jobs: it trusted the headers a proxy set, and it was
what a laptop ran. Those are now separate modes, and the default is neither of them.

| Mode | What it does |
| --- | --- |
| `api_key` | The new default. Verifies a key and an HS256 token signed with its secret |
| `proxy` | The old `noauth`. Reads `X-Stream-App-Id`, `X-Stream-Organization-Id`, `X-Stream-User-Id` and `Stream-Auth-Type` as already authenticated |
| `noauth` | Reads `X-Customer-Id` only, and every caller is that customer's backend |
| `custom` | An `auth.Authenticator` the embedder passes to `api.WithAuthenticator` |

**A deployment behind a proxy must set `ROUTER_AUTH_MODE=proxy`.** Left on `noauth` it
keeps working, but it stops reading the user and the auth type, so every end user becomes a
server-side caller of the app named in `X-Customer-Id`.

**A deployment that set nothing must now set `ROUTER_AUTH_MODE=noauth`,** or supply
`ROUTER_POSTGRES_URL` and `ROUTER_AUTH_KEK` and mint a key. Naming no mode used to mean
"trust everything"; it now means "verify a key", and a router with nothing to verify
against refuses to start and says which piece is missing.

A local deployment naming `noauth` explicitly, and sending `X-Customer-Id`, is unaffected.

### The default router groups are renamed

`GET /v1/{modality}/routes` now offers seven groups: `llm-conversational`, `llm-fast`,
`llm-smart`, `stt-fast`, `stt-accurate`, `tts-fast` and `tts-quality`.
`base/stt-realtime-fast` and `base/stt-realtime-accurate` are now `stt-fast` and
`stt-accurate`, with the same models; a config naming the old names must be updated. The
shortcuts that used to be offered (`en-low-latency`, `vlm`, `llm-thinking` and the rest)
still route, but are no longer listed.

### An app may turn away anonymous and guest users

`apps` gains a `settings` JSONB column, read in the same query that resolves an API key.
`allow_anonymous` and `allow_guest` both default to yes; an app that sets either to `false`
answers 403 to that level of caller on every path. It applies in `api_key` mode, the only
one where the app is a row in these tables. There is no endpoint for it yet: it is an
`UPDATE` by hand.

### A server-side caller may name a user, and reading a session is open to devices

A backend can send `X-Stream-User-Id` to say which of its users it is acting for. The
session it opens then belongs to that user, so that user's own device reaches it — which is
the point, since a backend opening the conversation and handing over the id is the ordinary
shape of an integration. The backend stays server-side, reaches everything, and is charged
no daily limit. Anonymous callers are excluded: an unverified name does not reach a
session a backend created.

### The router is server-side only unless the spec says otherwise

An end user's device may now reach six operations, and is answered 403 on everything
else:

| Operation | |
| --- | --- |
| `GET /v1/search` | ask a question |
| `POST /v1/agents/sessions` | open a conversation |
| `GET /v1/agents/sessions` | find the one it has |
| `GET /v1/agents/sessions/{id}` | read the one it is having |
| `DELETE /v1/agents/sessions/{id}` | hang up |
| `GET /v1/agents/sessions/{id}/events` | the conversation itself |

`x-server-side-only` in `openapi.yaml` is replaced by `x-client-accessible: true` on those
six. A caller is a device when it presents `Stream-Auth-Type: jwt` and a token naming a
`user_id`; a backend is unaffected, and so is a `noauth` deployment, where every caller is
taken to be a backend.

Making the agent answer, listing configs, minting a call or chat token and reading a
transcript are all now server-side. A frontend that called them wants its own backend in
front: the same call from there, with whatever of the answer that frontend should see.

### The Swift SDK is the same short list, and joining a call takes credentials

`VisionAgentsCore` no longer has what the router no longer answers a device. Removed:
`agents.agentConfigs()`, `agentConfig(named:)`, `callToken()`, `chatToken()`, `calls()`,
`transcript()`, `Router.stt` / `.tts` / `.llm` / `.sts`, `Router.configs()`,
`ModalitySocket` and the option and model types belonging to them. `Router.search` stays.

`agent:` is now a config id rather than a name, since resolving a name meant reading the
configs. `VoiceSession.join` and `VoiceCallView` take a `CallCredentialsProvider` — a
closure your backend answers — instead of minting a token themselves:

```swift
let voice = try await VoiceSession.start(agents: agents, agent: configID)
await voice.join(credentials: yourBackend.callCredentials)
```

`examples/voice_agents/swift_demo` shows the backend half: `configure/` writes the agent,
and `backend/` mints the tokens.

### Source research belongs to the agent, and managed research sandboxes are gone

An agent that reads source now owns its own Daytona VM and offers `investigate_sdk` as
one of its own tools, which the session dispatches back to it like any other caller-owned
tool. The backend holds no research workspaces and needs neither a Daytona nor a Cursor
key to serve one.

Removed: `sandbox_profile` on agent configs and session requests, the `agent_configs`
column behind it, `RESEARCH_PROFILES` and `RESEARCH_DEPLOYMENT_ID`, the backend's
`investigate_sdk` and its `research_progress` events, and `agents.ManagedSandbox` in the
Go SDK. An agent that set a profile should register the tool itself instead; the reserved
name that used to refuse it is gone too.

The reason is ownership rather than tidiness. Which repositories an agent may read, and
whose Cursor and Daytona keys it reads them with, are decisions belonging to whoever
deploys the agent; a profile on a config could only ever be a copy of one made elsewhere,
checked against a customer and agent id to guess at who was allowed to use it.

`agents.Daytona()` / `sandbox: daytona` is unaffected. That is delegated Python execution
for code the subagent writes, and remains the backend's to run.

### The Go SDK authenticates with a Stream app, and `Backend.Headers` is now `Credentials`

A hosted router sits behind a proxy that verifies a Stream app rather than trusting a
header, so `stream.Backend` can now carry a credential and sign a token per request:

```go
backend := stream.Backend{
	URL:          "https://accelerate.gcp.stream-io-api.com",
	Authenticate: true, // or STREAM_ACCELERATION_AUTHENTICATE
}
// APIKey and APISecret then fall back to STREAM_API_KEY and STREAM_API_SECRET.
```

It then sends `api_key`, `stream-auth-type: jwt` and `Authorization` on REST calls and on
the WebSocket handshake, and a customer id is no longer required, since the proxy works the
tenant out from the credential and strips what this end claims. A token names a user only
when `UserID` is set; without one it is marked `server: true` and speaks for the app itself,
which is what a backend wants and what leaves the per-user daily limits out of it.

`Authenticate` is opt-in rather than inferred from a key being present, because a Stream key
and secret are in the environment for plenty of reasons that have nothing to do with how the
router is reached — and sending them regardless is not harmless. A router with no proxy in
front of it reads `stream-auth-type: jwt` as a caller acting for an end user's device, and a
device may hold a conversation but not rewrite the agent holding it. Talking to a router that
trusts `X-Customer-Id` is unchanged.

Signing can fail, which `Headers()` had no way to report:

```go
credentials, err := backend.Credentials()  // was: backend.Headers()
```

Half a credential is refused by `Resolve` rather than ignored, so a missing secret is not
quietly downgraded to an unauthenticated request.

### Go tools are types, added to `agent.Tools()`

A tool is now a struct with `Name`, `Description` and `Run` methods. Its exported fields are
the arguments the model fills in; unexported fields carry whatever `Run` needs:

```go
type LookupOrder struct {
	OrderID string `json:"order_id" schema:"the order number"`
	orders  *Orders
}

func (LookupOrder) Name() string        { return "lookup_order" }
func (LookupOrder) Description() string { return "Look up an order by its number" }
func (l LookupOrder) Run(ctx context.Context) (any, error) { return l.orders.Find(ctx, l.OrderID) }

agent.Tools().Add(LookupOrder{orders: orders})
```

`agents.RegisterFunction` and `agents.Registrar` are gone, `Agent.Functions()` is now
`Agent.Tools()`, and `agents.Dispatch.Host` takes a `*tools.Registry`. `tools.Register` is
unchanged for a caller that prefers a closure. In `client`, `Agent.Functions()` and
`Session.Functions()` are now `Tools()`, and the `Functions` field of `SessionOptions` and
`ForkOptions` is now `Tools`.

### A Go agent's session is a `client.Session`, with `Responses`

`agent.Chat`, `agent.Join`, `agent.WaitForCall` and `agent.StartCall` return an
`agents.Session` that embeds `*client.Session`, so a conversation the agent opened reads the
same as one opened by name, as it does in JavaScript:

```go
session, err := agent.Chat(ctx)
answer, err := session.Responses.Create(ctx, "Where is order 1042?")
```

`Fork`, `Chat()` and `Video()` come with it. `Say` and `Respond` now take whether to
interrupt, like `client.Session.Say` already did: `session.Respond(text)` is
`session.Respond(text, true)`, and `session.Say(text)` is `session.Say(text, false)`.
`Session.Session()` is `Session.Created()`.

### Go `Responses.Create` takes `client.Image` and `client.Clip`

`session.Responses.Create(ctx, text, inputs ...client.Input)` replaces the variadic
`acceleration.ImageSource`. `client.Image{URL, Detail}` is a picture, and `client.Clip` or
`client.ClipFile(path)` is a video:

```go
_, err = session.Responses.Create(ctx, "Which receipt is for order 1042?",
	client.Image{URL: "https://example.com/receipts/1041.png"})
```

### Routing in the Go SDK starts from a client, and `SyncRouters` / `DefineRouter` moved onto it

Where the router is and who is calling it is now said once, to `stream.NewClient`, rather
than handed to every call that needs it:

```go
client, err := stream.NewClient(stream.Backend{}) // the zero Backend reads the environment
if _, err := client.SyncRouters(ctx, "routers"); err != nil { ... }
transcriber, err := client.Router("healthcare").STT().Realtime(ctx, nil)
```

`stream.SyncRouters(ctx, backend, dir)` and `stream.DefineRouter(ctx, backend, options)` are
now `client.SyncRouters(ctx, dir)` and `client.DefineRouter(ctx, options)`. `stream.Router`
itself is unchanged and its fields are still exported, so a caller wanting tags or a logger
can still build one directly; `client.Router(name)` is the short way to the common case.

Resolving the backend happens when the client is made rather than at the first call, so a
missing credential is reported where it was configured instead of in the middle of a session
that was about to stream audio.

### `agent.simple_response(...)` is now `agent.responses.create(...)`

Matches the shape of OpenAI's `client.responses.create`:

```python
async with agent.join(call):
    await agent.responses.create("greet the user in one short sentence")
```

Arguments and behaviour are unchanged. `LLM.simple_response` on plugins is unaffected.

### Calls and transcripts default to the `agent` type

A call type or channel type that is not named is now `agent` rather than `default` and
`messaging`, across the Python SDK, the CLI, the Go SDK, the backend and the transcript
channel an agent writes to. A Stream app needs both types before an agent can join:

```bash
getstream api CreateCallType --request '{"name":"agent"}'
getstream api CreateChannelType --request '{"name":"agent","automod":"disabled","automod_behavior":"flag","max_message_length":5000}'
```

Passing `call_type="default"` or `channel_type="messaging"` keeps the old behaviour, and a
transcript written before this change stays in `messaging:{agent_id}`.

### `Agent.join`: leaving the block waits for the call to end

`join`, `answer` and `outbound_call` now wait for the call to end on the way out, so an agent
no longer ends with `await agent.finish()`:

```python
async with agent.join(call):
    await agent.responses.create("greet the user in one short sentence")
```

`finish()` is unchanged and still safe to call. Leaving a call that is still live is
`wait_for_end=False`.

### `GET /v1/llm/stream`: responses are streamed one at a time

The socket used to answer every `respond` frame on one shared event stream, so a caller had
to correlate frames itself and abandon a reply by naming the completion. A response is now
its own stream, which changes the frames:

- `interrupt` takes `response_ids` rather than `completion_ids`.
- `complete` carries `status` (`completed`, `incomplete`, `failed` or `cancelled`) and
  `cache_write_tokens`. A response that was interrupted settles as `cancelled` and still
  reports what it produced and was billed for.

The Go `llm.LLM` interface behind it changed to match, which matters for an out-of-tree
provider: `Respond(request) error` plus `Events()` becomes
`Create(ctx, ResponseParams) (*llm.Stream, error)`, `Interrupt(ids...)` becomes
`stream.Close()`, and `Reasoning() bool` folds into `Capabilities()`. The OpenAI provider now
calls `/v1/responses` rather than `/v1/chat/completions`, so it can cache a prompt prefix
under `PromptCacheKey` and continue from a stored response.

### `stream.Router` is a config, not a one-shot resolver

`Router("sonic_36")` used to ask the backend which modality a name was and return the plugin
for it. `Router` is now the config-driven router, and resolving a bare name is a method on it:

```python
router = Router("healthcare")          # a stored router config
tts = router.resolve("sonic_36")       # what Router("sonic_36") used to be
```

A `start` frame on `/v1/{modality}/stream` with neither `target` nor `config_id` is also
refused now, rather than falling back to a default nobody asked for.

### `stream.TextSession` is gone: an agent answers in writing

A conversation held in writing is the same `Agent` a call would get, so there is no second
thing to learn. `chat()` holds it and `ask()` follows one reply:

```python
agent = Agent(config="docs_agent")

async with agent.chat():
    async for event in agent.ask("how does failover work?"):
        if event.type == "agent_speech_delta":
            print(event.text, end="", flush=True)
```

The events are the agent's own `RemoteEvent`s rather than `TextEvent`s: `delta` is
`agent_speech_delta`, `answer` is `agent_speech` and `settled` is `task_settled`.
`chat(agent_id)` answers in a conversation that already exists rather than in one of the
agent's own.

Dispatch keeps one agent per channel, so a written conversation carries on where it left off:

```python
@dispatch.wait_for_message()
async def written(message: InboundMessage) -> None:
    agent = await dispatch.get_or_create_agent(
        message, lambda: Agent(config="chat_support")
    )
    await agent.responses.create(message.text)
```

### An agent directory needs `agent.yaml`, and a router config lives in a folder

What makes a directory an agent is now `agent.yaml` next to `instructions.md`, rather than
any of `instructions.md`, `skills/` or `knowledge/` being present. A directory without one
is no longer found by name, so add one to each:

```yaml
# agents/customer_support/agent.yaml
name: customer_support
```

Router configs move from a flat file per config to a folder per config. `routers/clinic.yaml`
becomes `routers/clinic/router.yaml`, and `sync_routers(directory)` now reads
`{name}/router.yaml` under it rather than `*.yaml`. Both files take an optional
`description`.

### `gemini` plugin: Realtime defaults to `gemini-3.8-live` (#647)

`gemini.Realtime` now defaults to `gemini-3.8-live` (was `gemini-3.1-flash-live-preview`). Pass `model=` explicitly to stay on an older Live model.

### `sarvam` plugin: drop deprecated LLM, STT, and TTS models (#637)

Sarvam LLM no longer accepts `sarvam-m` or `sarvam-30b`; the default is `sarvam-105b` (`sarvam-105b-conversations` is also supported). STT drops `saarika:v2.5` and `saaras:v2` / `saaras:v2.5` and defaults to `saaras:v3-realtime` on `/speech-to-text-realtime/ws` (`saaras:v3` and `saaras:v4` remain on the legacy WebSocket). TTS no longer accepts `bulbul:v3-beta`.

### `deepgram` plugin: TTS defaults to Flux (`/v2/speak`) (#633)

`deepgram.TTS` now streams Flux TTS on `wss://api.deepgram.com/v2/speak` and defaults to `flux-haley-en`. Aura model strings (`aura-*`) are rejected with `ValueError`. Call sites that passed an Aura voice must switch to a Flux model (`flux-{voice}-en`). See the [Flux voice catalog](https://developers.deepgram.com/docs/flux-tts/voices).

### Go: `ListConnectionsParams.OwnerType` is a pointer

`owner_type` is optional on `listConnections`, so the Go client's `ListConnectionsParams.OwnerType`
is `*ConnectionOwnerType`, and nil means the app's own. Code that set `OwnerType: acceleration.ConnectionOwnerTypeApp`
no longer compiles. Pass `&app` for a variable holding the value, or leave the field nil for the app.

### An agent config refuses an MCP server on a private address

The router reaches only MCP servers on a public https URL. Saving a config (create, replace, patch or `syncAgent`) now answers 400 `mcp_servers: <name>: only public https URLs are reachable` for a server whose host is on a private network (a Tailscale node included), loopback or link-local, or does not resolve. Such a server used to be saved, and then no session could reach it. A config already stored with one still reads.

## New Features

- A session's tool (`SessionTool`) and a worker's hosted tool (`host_tools`) can set `interruptible` and `timeout_ms`. `interruptible: true` lets the caller interrupting the agent on a call cancel the tool's call in flight, which is sent `tool_cancel`; left out, the call goes on as before and its result is told when it comes. `timeout_ms` is how long one call to that tool may take, in place of the session's `tool_timeout_ms` or the worker's own `timeout_ms`.
- The Go SDK's `tools.Register` and `Registry.Add` take options: `tools.Interruptible`, `tools.Timeout`, which are sent to the router, and `tools.Concurrency`. A registry runs at most 16 calls at once (`SetConcurrency`). A function's `ctx` is cancelled when the router sends `tool_cancel`, and `tools.CallFrom(ctx)` is the call it runs for, whose `ID` a function can key a change on. A call sent again is run once: a copy of one still running joins it, and one of a finished call is answered with its result (`SetReceipts` keeps them somewhere shared). Struct tags gain `enum`, `minimum`, `maximum`, `pattern`, `format` and `example`; a type can describe itself with `JSONSchema()`; and `tools.RegisterSchema` takes a JSON Schema as written. A process no longer answers a `tool_call` for a tool it does not hold, which raced the process that holds it; the JS, Python, Ruby, Rust, .NET and PHP SDKs leave such a call alone too.
- A tool call says who it is for. The session socket's `tool_call` carries `session_id`, `user` (`id`, `name`) and `conversation` (`cid`, `custom`); a worker's hosted `tool_call` carries `turn_id`, `user` and `conversation`. On a call, `user` is whoever spoke the turn that asked for the tool rather than the session's caller. The Go SDK's `tools.CallFrom(ctx)` reads them as `SessionID`, `TurnID`, `User` and `Conversation`.
- A running tool call can keep going and say how it is going. On the session socket, the watcher running a call sends `tool_hold` (`tool_call_id`) so the caller talking over the agent no longer cancels it, even for an `interruptible` tool; every watcher is sent `tool_held`, or `tool_cancel` when the call had already stopped. An explicit `interrupt` still cancels it. `tool_update` (`tool_call_id`, `text`, `verbatim`) says how the call is going: the agent tells a voice caller in its own words once nobody is talking, a newer update replaces one not yet said, and the call's result replaces both; with `verbatim` it says or posts `text` as written. A persistent conversation shows the update as the summary of the call's step until it ends. On speech-to-speech, OpenAI Realtime is asked to say it while the call stays open, with `tool_choice: "none"` so that reply calls nothing. Gemini 3.1 Flash Live and 3.8 Live now declare tools `NON_BLOCKING`, so the model keeps talking while a call runs; an update reaches it as a partial function response, and the result when it is next idle. 3.8 Live Extended Thinking stays blocking. Other speech-to-speech models only report it for now. Every watcher is sent `tool_updated` (`id`, `turn_id`, `tool`, `text`, `verbatim`). A worker on `/v1/dispatch` sends the same `tool_hold` and `tool_update` for a hosted call, and is answered `tool_held` or `tool_cancel`, wherever in the deployment the session runs. In the Go SDK, a function's `tools.CallFrom(ctx)` has `Update(ctx, text)`, `Say(ctx, text)` and `DisallowInterruptions(ctx)`, on a session's socket and on a worker's alike. `DisallowInterruptions` returns a context that only the call's timeout and the socket ending cancel, or an already cancelled one (`tools.ErrNotHeld`) when the call was stopped first. Outside a call, all three do nothing; `tools.WithLine` gives a test a session to report to.
- A tool declared with an `approval` asks a voice caller before it runs. The router does not send the call: it answers the model at once with the approval's `title`, each `{argument}` filled in from the call, and offers the model `confirm_tool_call` (`tool_call_id`, `allowed`). A cascade says the question as written, through TTS, with no model turn; a speech-to-speech model is told to ask it in those words. A yes within 15 seconds of the question finishing sends the original call, under its own id and with `approved: true` on `tool_call` (both sockets), and its result answers `confirm_tool_call`; the tool's timeout starts then. A no declines it, and silence for 15 seconds declines it too, so the call is never sent; a yes after that tells the model to call the tool again to ask again. A worker's hosted tool can set `approval` (`title`) in `host_tools`. The JS SDK runs an `approved` call without holding it in `session.approvals`. In the Go SDK, `tools.NeedsApproval("Book flight {flight_id}?")` sets the question, refusing at registration a `{name}` that is not an argument, and `tools.CallFrom(ctx).Approved` says the caller said yes. In writing nothing changes: the app asks, as before.
- The Python SDK catches up on tool calls. `register_function` (and `stream.Agent.register` for hosted tools) takes `interruptible`, `timeout` and `needs_approval`, which go out on the session's tools and on `host_tools`; a `needs_approval` question that names an argument the function does not take is refused at registration. `vision_agents.core.llm.current_call()` is the call a function runs for: `id`, `session_id`, `turn_id`, `user`, `conversation` and `approved`, with `update(text)`, `say(text)` and `disallow_interruptions()`, which raises `CallNotHeld` when the call was stopped first. Outside a call it is an empty one whose methods do nothing. A call with `run: false` is left alone, a function's own `timeout` fails a call that runs over it, and a worker's hosted call is cancelled by `tool_cancel` and answered with `tool_call_id`.
- The PHP SDK catches up on tool calls. `Tools::register` takes `interruptible:`, `timeout:` and `approval:` (a question with `{name}` for a parameter, refused at registration when it names one the tool does not take), which go out on the session's tools and on `host_tools`. A tool's callable that takes a second argument is handed a `ToolCall`: `id`, `sessionId`, `turnId`, `user`, `conversation` and `approved`, with `update($text)`, `say($text)`, `disallowInterruptions()` (which throws `CallNotHeldException` when the router cancelled the call first), `cancelled()` and `cancellation()`. A call with `run: false` is left alone, a tool's `timeout` fails a call that runs over it, and a worker's hosted call is cancelled by `tool_cancel`, answered with nothing, and otherwise answered with `tool_call_id`. The generated classes pick up `interruptible` and `timeout_ms` on `SessionTool`.
- The .NET SDK catches up on tool calls. `Tools.Register` takes `interruptible`, `timeout` and `approval` (a question with `{name}` for a parameter, refused at registration when it names one the tool does not take), which go out on the session's tools and on `host_tools`, and has overloads whose function also takes a `ToolCall`: `Id`, `SessionId`, `TurnId`, `User`, `Conversation` and `Approved`, with `UpdateAsync(text)`, `SayAsync(text)` and `DisallowInterruptionsAsync()` (which throws `CallNotHeldException` when the router cancelled the call first). A call with `run: false` is left alone, a tool's `timeout` cancels and fails a call that runs over it, and a worker's hosted call is cancelled by `tool_cancel` and not answered, and otherwise answered with `tool_call_id`. The generated models pick up `interruptible` and `timeout_ms` on `SessionTool`.
- The Ruby SDK catches up on tool calls. `Tools#register` takes `interruptible:`, `timeout:` and `approval:` (a question with `{name}` for a parameter, refused at registration when it names one the tool does not take), which go out on the session's tools and on `host_tools`. A tool's block is handed a `ToolCall` as its second argument: `id`, `session_id`, `turn_id`, `user`, `conversation` and `approved?`, with `update(text)`, `say(text)`, `disallow_interruptions` (which raises `CallNotHeldError` when the call was stopped first) and `cancelled?`. A call with `run: false` is left alone, a tool's `timeout` fails a call that runs over it, and a worker's hosted call is cancelled by `tool_cancel`, answered with nothing, and otherwise answered with `tool_call_id`.
- The JS SDK catches up on tool calls. A `Tool` takes `interruptible` and `timeoutMs`, which go out on the session's tools and on `host_tools`, and an `approval` title that names a `{name}` not in `parameters` is refused at registration. `run` is handed a third argument, the `ToolCall` it runs for: `id`, `sessionId`, `turnId`, `user`, `conversation` and `approved`, with `update(text)`, `say(text)` and `disallowInterruptions()`, which resolves with a signal only the timeout and the socket ending abort, or one already aborted with `CallNotHeldError` when the call was stopped first. A `ToolCall` built without a line does nothing, for a unit test. A call with `run: false` is left alone, a tool's `timeoutMs` aborts and fails a call that runs over it, and a worker's hosted call is aborted by `tool_cancel`, answered with nothing, and otherwise answered with `tool_call_id`. A hosted tool's `approval` is now sent and asked on a voice call. The Python SDK hands `tool_held` to `session.events()` again.
- The Rust SDK catches up on tool calls. `Tools::register_call` takes a whole `SessionTool` whose `interruptible`, `timeout_ms` and `approval` it checks (a `{name}` in the question must be a parameter), which go out on the session's tools and on `host_tools`, and hands the function a `ToolCall` as its second argument: `id`, `session_id`, `turn_id`, `user`, `conversation` and `approved`, with `update(text)`, `say(text)` and `disallow_interruptions` (which returns `Error::NotHeld` when the router cancelled the call first). `Tools::call_as` runs a function for a given call. A call with `run: false` is left alone, a tool's `timeout_ms` stops and fails a call that runs over it, and a worker's hosted call is aborted by `tool_cancel` and not answered, and otherwise answered with `tool_call_id`.
- A session's events socket can be resumed. A watcher that asks with `resumable=true` is sent `resumed` first and a `cursor` on every frame after it; reopened with `cursor`, it is sent what it missed, from the last 256 frames the session keeps while it is held, with `resumed.complete` false when some are gone. A session the router stops on its way down closes a resumable socket with 1012, and its reply in flight is `interrupted` rather than `cancelled`, which `resumed.command` reports for the `request_id` the watcher names. A socket that does not ask is sent what it always was. The Go SDK reconnects a text session's socket on its own, and its dispatch conversations ask a question the router stopped once more.
- How long tool output is kept is a setting: `tool_output_retention` on the organization's and the app's policy, and `tools.output_retention` on an agent config and in agent.yaml (the Go SDK's folder sync sends it). Each is `none` or a duration from `1h` to `3650d`, and the shortest of the three applies. `none` records a tool call and its result without their arguments, output or error wording; a duration records them whole and the router takes them out once it runs out, keeping the item with its tool name, time, call id and `"redacted": true`, so listing items and tool stats work as before. Left out everywhere, tool output is kept as long as the session, as before.
- The Swift SDK can be added with Swift Package Manager: `https://github.com/GetStream/Vision-Agents.git`, product `VisionAgentsCore`. Its `Package.swift` is at the root of the repository, and each `v` tag from this release on is a release of it too.
- The `anthropic` provider speaks Anthropic's Messages API instead of the OpenAI-compatible endpoint, so Claude caches: the tools, the system prompt and the conversation up to the turn being answered end on `cache_control` breakpoints, and each round of a tool loop reads the rounds before it from the cache. Cache reads count as `cached_input_tokens`, and cache writes are priced at the new `per_million_cache_write_input_tokens` rate (1.25x input for Claude). A request still sends no temperature, effort, JSON mode or thinking replay, as before. `ANTHROPIC_OPENAI_COMPAT=true` goes back to the compatibility endpoint.
- A task made in an SMS (Telnyx) or WhatsApp thread comes back in that thread: when it comes due, the router opens the thread's conversation, the agent answers the task there, and the reply is texted to the person from the line; the run says `delivered_to` `sms` or `whatsapp` once the line took it, and a refused send is `delivery_failed`, whose retry sends the same reply. The task's `deliver_to` reads `sms` or `whatsapp`, and PATCH refuses to change it. An opted-out person, before the turn or by the time the reply is sent, is a final `channel_unreachable`; WhatsApp more than 24 h after the person last wrote, with no template, is a final `outside_window`; a removed line is `connection_removed` and a thread that is gone is `thread_gone`, both final. Tasks without a thread are delivered as before (AI-1843).
- A config that binds a sensitive connector, such as Nessy, may bind beside it a connector tool that writes when that binding's `require_approval` holds it: the router asks the session's user before each call is sent. A `tool_patterns` binding is accepted when a `require_approval` entry with no wildcard but `*` matches each of its patterns as written, such as `*`. `sandbox: true`, MCP servers, plugins and a sandbox that reaches the network are refused as before. The 400 names each binding's tools that neither only read nor require approval. The Go SDK's folder sync sends `require_approval:` from an agent.yaml connector entry (AI-1847).
- A connector binding takes `require_approval`: names or patterns of its tools whose every call the router holds until the session's user allows it, on the call's `ai_tool_call` step (`awaiting_approval`, with the call's arguments). The answer is the session socket's `tool_approval` with `tool_call_id` and the reply's `request_id`; the Go SDK sends it with `Session.AnswerApproval`. Declined or not answered in 10 minutes, the call is not sent and the model is told so; a session with nobody to ask refuses the call at once. A binding without it calls its tools as before (AI-1640, #1182).
- A person who texts an agent on an SMS (Telnyx) or WhatsApp line can have it schedule tasks: on a config with `scheduling.enabled`, the thread's session is offered `schedule_task`, `list_scheduled_tasks`, `update_scheduled_task` and `cancel_scheduled_task`, only when the thread's person writes (someone else writing in that thread gets none of them). The task belongs to that person on that line and records its thread (`scheduled_tasks.channel_thread`). A Linq chat (1:1 is AI-1867) and a Slack thread, which is shared, get no tools. Delivery back to the thread follows in AI-1843 (AI-1842).
- The router can keep its key encryption key in Google Cloud KMS: set `ROUTER_AUTH_KMS_KEY` to the Cloud KMS key and `ROUTER_AUTH_KEK_V<n>_WRAPPED` to the key it wrapped with the AAD `router-kek-v<n>`, in place of `ROUTER_AUTH_KEK`. The router unwraps it with one `Decrypt` at each start, so the rows sealed under the plain key still open; with neither set, nothing changes (AI-1747, #1157; AI-1748, #1165).
- A `google_drive` binding with `sandbox: true` lets code in the session's sandbox call Drive's REST API (`https://www.googleapis.com/drive/v3/...`) with `Authorization: Bearer $GOOGLE_DRIVE_TOKEN`; the sandbox holds a placeholder, never the token. The token carries `drive.file`, so that code can also create files and change files the app opened. A Drive connection made before revision 5 gives the sandbox nothing until its user or app connects again (AI-1958).
- An agent.yaml `connectors:` entry takes `sandbox: true`, which the Go SDK's folder sync (`Agent.Sync`) sends on that binding, so code in the agent's sandbox may call the connector's API with the binding's connection. Left out or false, no `sandbox` key is sent and the body is as before; a value that is not a boolean is refused (AI-1578, #1141).
- A `usage` service (`acceleration/cmd/usage`, in the router's image and in compose): it records each billable event once by its usage id in its own Postgres (`USAGE_POSTGRES_DSN`), totals spend per subject in daily, weekly, bi-weekly and monthly windows, and copies those totals to Redis (`USAGE_REDIS_URL`). Every operation but `/healthz` takes `USAGE_INGEST_TOKEN`: `POST /v1/events` takes batches and rejects a bad event on its own with a reason while recording the rest; only a batch too big or that does not parse is refused whole (413, 422). `/v1/windows` and `/v1/standings` read the totals; metrics on `USAGE_METRICS_ADDR`. `internal/usage.Client` ships events to it, split into batches that fit, and rejects itself (`too_large`), without sending, an event no batch can hold. Nothing sends it events yet (AI-1826, #1166).
- The router ships the usage rows it writes to the usage service, when `ROUTER_USAGE_URL` and `ROUTER_USAGE_TOKEN` name it (a URL that is not an http or https address, or one without a token, stops the router at start; without a URL, rows are written with `to_ship` false and never shipped, as before, and a token left set without one is a WARN line, so unsetting the URL alone stops shipping). Compose points its router at its usage service. The shard's `requests` table is the outbox: each row is written with `to_ship` true, and once written the writer hands the batch to a shipper that sends it at once, never making the writer wait (a shipper with 100 batches waiting leaves the next in the shard, and so does one whose batch waited for it over 30 s, `router_usage_ship_deferred_total`). The service's answer is written on the rows: `to_ship` false and `shipped_at` for the ones it has, `ship_error` set to its reason for the ones it refused, which are not sent again; an answer that does not account for every row sent marks none, and the rows are sent again. Each event names its source `router/<shard>/<host>` (`unsharded` in a deployment of one), happened at the row's `started_at` to the microsecond, and carries the row's organization, tier, key, user, categories, session, agent, call, quantities, attributes, cost, `base_cost_micros`, `price_version`, billable and tags. With a usage service every router runs a sweeper; the one holding the shard's advisory lock sends, every 30 s, what is still waiting and started more than 30 s ago, 500 rows a page: rows the service was down for, rows the shipper had no room for, and rows written with no organization, whose organization it looks up and writes first (`router_usage_unowned_total` and an ERROR line once an hour per app while the app has joined none). A sweep stops after 2 minutes, carried on by the next when it had marked rows, and a holder whose sweep fails or made no progress lets the lock go; Postgres probes the holder's connection, so a holder whose network went silent loses the lock within about a minute, and a shipping write waits at most 5 s on another session's row locks, whose transaction Postgres ends after 15 s idle. Every router counts what is waiting in its shard for `/metrics`, which adds `router_usage_shipped_total`, `router_usage_ship_errors_total`, `router_usage_rejected_total`, `router_usage_unshipped_rows` and `router_usage_unshipped_oldest_seconds`. The writer's losses and failing writes no longer go to Sentry on their own: one alerts loop per router checks `lost` and `rejected` (page once for the rows lost or refused since the last one, at most once a minute), `write_failing` (page: writes failing for 2 minutes with nothing written), `ship_stale` (page: the oldest unshipped row started over 15 minutes ago, and has for 2 minutes; every router of the shard sends it, one Sentry issue), `unowned` (warn: rows waiting for an organization for 15 minutes) and `pending_slow` (warn: a row waiting over a minute to be written), each a Sentry event tagged `usage_alert` and `shard`, and checked once more as the router stops. `router usage replay --from <log>` writes the rows a log says were lost (`msg="usage row lost"`), once each. `router usage reship --since <time|duration>` queues a shard's rows the service has to be sent again, and `--rejected` the ones it refused; a row it never had, written before the router shipped or carried in by a data move, is never queued. A data move now imports `requests` without the source's `shipped_at` and `ship_error`, as it already imported them with `to_ship` false (AI-1826, #1183).
- Every usage row the router records names its organization, the organization's tier, the app, the API key, the end user and the usage categories, the user's team among them as `team:<name>`, and every event it ships carries them. Behind Stream's proxy the router reads `X-Stream-Api-Key-Id`, `X-Stream-Organization-Tier` and `X-Stream-Usage-Categories` (comma-separated); in `api_key` mode the key is the one verified and the categories are the token's `usage_categories` and `team` claims, or `X-Stream-Usage-Categories` from a server-side caller whose token names none; in `noauth` mode the key is `noauth`. Categories are at most 10 of 1 to 64 characters of `A-Z a-z 0-9 _ . : -`, with at most one team of at most 59 characters, and a request naming others is refused with 400 `invalid_usage_categories`. The user is the token's, or the one a backend names in `X-Stream-User-Id`, never an anonymous caller's name, and every row's `caller` attribute says `user`, `guest`, `server` or `anonymous`. A user id the usage service would refuse the event for (one holding `|`, `{` or `}`, such as `auth0|abc`, or over 255 bytes) is left off the row rather than refusing the request: the work is served and billed without its user, counted once per row in `router_usage_user_id_dropped_total{reason}` and logged at WARN at most once a minute per app, never with the id; `router usage replay` leaves it off a replayed row too. A session keeps the key, user and categories it was opened with for its whole life, also when another process carries it on from its row, and puts its id on every row. Work no key started names its origin instead: `hook:chat`, `hook:video`, `phone:inbound` (a text to one of the app's numbers), `hook:plugin`, `hook:connector`, `job:episode_summary`, and `cli:<command>` for the router's own commands; a campaign and a scheduled task keep the key, user and categories of whoever scheduled them, and their calls and occurrences are owned by them, those created before this release under `job:campaign` and `job:scheduled_task`, and one stored since without a key under none, so it is reported as unattributed. A speech job is recorded under the key that queued it. The tier a request names is kept per organization and written to `organization_tiers` at most once per organization every 10 s, one left unwritten when the organization goes quiet once its 10 s have passed (migration `20261111120000_usage_identity` adds it, and the campaign, scheduled-task and session columns, without rewriting a table), and the organization's current tier replaces the one a row was opened with, so a running session's next rows carry a tier Stream changed; work no request carried takes the organization's last tier, read again every 10 s (a miss and a failed read included), else `default`. A row that still names no key is logged at ERROR with its usage id and warns as the `unattributed` alert. `GET /v1/stats/spend` takes `api_key`, `user_id`, `team` and `category` filters. The audit log accepts the `end_user` resource type for forgetting a user, which comes later (AI-1826, #1190).
- The usage service keeps accounts and tiers. Each event's billable cost is debited from its organization's account in the transaction that records it, once per usage id. `/v1/standings` answers each organization's tier (the `default` row's settings for a tier with no row, `fallback_tier` true), its own overrides and the settings in force (spend hard limit per day and month, channel limits, sandbox bounds, low-balance threshold), its status, credit, spend, balance and spend in the current day and month. A management API under `/v1/manage` (`USAGE_MANAGE_KEY` in `X-Manage-Key`, an actor in `X-Stream-Actor-Id`; 404 `not_configured` without the key) reads and writes tiers, an organization's overrides, credit entries (once per `Idempotency-Key`, sign-checked only), suspension and its usage events, each change audited with its actor. Its bodies refuse a field they do not know (422), so a misspelt setting never clears the overrides there. `USAGE_TIERS_FILE` seeds the tiers a database lacks at start, and a built-in `default` (fail closed) when there is none; `usage tiers check` and `usage tiers sync --dry-run` check a file and preview a start. Migration `00002_accounts` adds the tables only (AI-1841, #1172).
- The usage service's suspend, restore and settings writes (`POST .../suspend`, `POST .../restore`, `PUT .../limits` under `/v1/manage/accounts/{organization_id}`) need an `Idempotency-Key` (1 to 255 characters; 400 `invalid_request` without one) and are made once under it, as a credit is: a replay with the same organization and body changes nothing, writes no audit row and answers the standing as it is now, so a request sent again cannot suspend an organization restored since. The key is one space across the four account writes, credit included: one already used for another operation, organization or body is a 409 `idempotency_key_reused`. The key is claimed in the transaction that makes the change and audits it, so two identical requests at once make it once. Migration `00003_manage_requests` adds the table that keeps the keys (AI-1841, #1177).
- Every row the router records is priced by one price list, built at start from `router.yaml`'s rates, and stores what priced it: `requests.meters` holds the quantities with no column of their own (a search's requests, a picture's pixels), `requests.attributes` what chose the price, and `requests.billable` is false for free and local work. A model with no price is logged as `billable work has no price` at start, and at most every ten minutes while its work is recorded, unless its `price` says `free: true` or the model says `local: true`, one the deployment runs itself. `router.yaml` also takes `per_million_cache_write_input_tokens`, `rates`, `context_tiers` and `service_tiers`. A cached prompt token on a model with no cached rate is billed at its input rate rather than at nothing, and a row's cost is rounded to the nearest micro-dollar rather than down (AI-1824, #1168).
- The price list is kept in the shared database: migration `20261110130000_price_entries` adds `price_changes` and `price_entries`, which only ever grow (a change not yet in force can be cancelled), and `requests.price_version` and `base_cost_micros`, which nothing writes yet. At start, a router loads `prices/vendor.json`, with a deployment's own `ROUTER_PRICES_FILE` laid over it, into an empty database as one `seed` change carrying the file's `markup_bps`, and after that only the models the database has never priced, as one `release` change; a model it has priced keeps its rates. A load that fails, or waits more than 5 seconds for another router's, is retried every 30 seconds while the router serves, and a price file that does not parse stops it at start. `router prices show`, `sync --dry-run`, `check` and `drift` read the list; no command writes a price. A `check_prices` skill checks the list against what each vendor charges (AI-1824, #1171).
- A router with a database prices every row from the price list in force there, with its markup: it reads the list every 30 seconds and again at the moment a scheduled change takes effect, keeps the list it had when a read fails, and prices with its price files (no markup) only until a read answers any entry. With `ROUTER_PRICES_FILE` set, `/health` answers 503 (`prices`) until then; with only a database, until the first read succeeds. Each row's `cost_micros` is its cost at the vendor's rates plus the markup (the entry's own `markup_bps`, else the list's), added once before rounding; `base_cost_micros` is the cost before it, and `price_version` names the change that wrote the entry (`embedded:<release>` or `files:<release>` before the first read, `quoted` for an amount the vendor quoted outright, which is resold with the markup, and empty for local or unpriced work). `GET /v1/{modality}/providers` shows the rate customers are charged. Six management operations, under `/v1/manage/price-lists` with the manage key: `getActivePriceList` (vendor and charged rates, what is unpriced, what is scheduled), `listPriceChanges`, `getPriceChange`, `previewPriceChange`, `createPriceChange` and `cancelPriceChange`. A change replaces whole entries, sets the markup, or withdraws an item, from now or up to two years ahead; it is refused with 400 `invalid_price` when its entries do not make a price list, set `local` or `version`, or rate a unit at more than $1000, with 409 `confirmation_required` without `confirm: true` when a vendor rate, its increment or minimum, or a markup moves by a factor of 2 or more, a long-prompt tier is added or dropped, or a routed model would become unpriced, with 503 `price_list_not_seeded` before the first start has loaded the price files, and with 503 `price_list_busy` and a `Retry-After` when another write holds the list for more than 5 seconds. It is written once per `Idempotency-Key` (required; the same key with another body, its entries in another order included, is a 409 `idempotency_key_reused`) and audited; the router that wrote it prices with it at once. A router whose start lost the race to load the price files reads the list as soon as its retry has loaded them. `router prices check --file` now also lays the file over `vendor.json` as a start does, so a file the start would refuse fails the check, and says how many entries that makes. Only a change made through the API and not yet in force can be cancelled (409 `price_change_in_force` otherwise), and a cancel sent again changes nothing. Migration `20261110140000_audit_log_cancelled` adds `cancelled` to the audit log's actions (AI-1824, #1184).
- A management API under `/v1/manage`, for Stream's staff or a self-hosted router's operator: its own key, `ROUTER_AUTH_MANAGE_KEY`, sent as `X-Manage-Key` (never the ops key, and no app's credential), and every call names its actor in `X-Stream-Actor-Id` (a name in `X-Stream-Actor-Name`, percent-encoded when it is not printable ASCII). Without the key every management operation answers 404 `not_configured`. `GET /v1/manage/status` answers who the call is from, to check the key and the actor get through. The audit log gains the source `manage` and the resource type `price_list`, recorded with no app (AI-1825, #1167).
- Rows are priced on what the vendor reported, at its published rates. An LLM row stores the prompt written to the cache as `meters.cache_write_input_tokens`, priced at its own rate (OpenAI: 1.25 times input), the tier the vendor served it at as `attributes.service_tier`, which multiplies its rates (OpenAI's flex and batch half, fast twice, and priority, what GPT-5.6 and earlier answer for a fast request, twice; ultrafast six times), and its reasoning as `attributes.reasoning_tokens`. A tier the price list does not name is priced at the default rates and logged as `the vendor served a tier the price list does not name`, at most every ten minutes per model. A speech-to-speech row stores its tokens by modality in `meters` and is priced by them: OpenAI Realtime bills text at $4 / $24 per million where every token was priced at the audio $32 / $64, and Gemini Live bills text, image and audio input and text and audio output at their own rates, its thinking as text output. Output tokens the vendor's split leaves out are billed as text, and a Gemini turn whose prompt has no split at the audio rate. Gemini Live's output tokens now include the thinking, wherever they are reported: `output_tokens` on the session socket's `response_complete` frame, in `/v1/stats` and in a session's usage. Prompts over a vendor's long-context threshold are priced at its long-context rates (OpenAI over 272K, Grok 4.7 from 200K, Claude Haiku 5.5 over 100K); `gpt-5.6-sol` is priced at OpenAI's promotional rates; and `cloudflare/clef-flash`, `cloudflare/clef` and `liquid/d1` at what OpenRouter charges for them (AI-1824, #1170).
- Turn-taking metrics: `router_agent_stale_replies_dropped_total`, `router_agent_replies_started_over_caller_total{kind}`, `router_agent_played_before_interrupt_seconds{kind}`, `router_agent_flow_decisions_total{disposition,ruler}`, `router_agent_flow_decision_seconds{ruler}` and `router_agent_settled_over_agent_total{words,final,action}`; `the voice took an utterance` logs `caller_talking` (AI-1601, #1100).
- The `gemma` provider sends `GEMMA_API_KEY` as its bearer token, so `GEMMA_BASE_URL` can point at any OpenAI-compatible vLLM host, not only Baseten; with it unset, `BASETEN_API_KEY` is sent as before (AI-1664, #1115).
- The `anthropic` provider sends `ANTHROPIC_WORKSPACE_ID` as the `anthropic-workspace-id` header, so an Anthropic key that is not scoped to a workspace is accepted; with it unset, requests are sent as before.

### `session.speech` reveals a reply by where each utterance is in its message

A router that says where each utterance is in the reply's Chat message (`message_start` and
`message_end` on `speaking`, `spoke` and `interrupted`) has `session.speech` reveal the message
from one to the other over the utterance's audio, at the share of it the voice has got
through, so `heard(...).chars` is in the message's own text rather than the voice's: a
direction the voice acts, digits it is told the count of and a sentence held back no longer
push the reveal ahead of the words being said. A gap between two utterances, such as a
sentence held back, is passed when the next one starts, and a cut stops at the share of the cut
utterance's place that was heard. A streamed utterance not yet measured goes at the message's
characters to the voice's its `speaking` had. A router that sends no offsets is counted in the
voice's characters as before (#1146).

### The session socket says when each utterance starts being heard

A new `speaking` frame comes when the call's track takes the first audible frame of an
utterance of the agent's, so a page can reveal a reply as it is said rather than as the Chat
channel has it written: `turn_id`, `synthesis_id`, `text` and `at`. `text` is what the voice
had been sent for the utterance by then, which is all of a greeting or other line said as
written and of a sentence a voice that takes one at a time was given, and the start of the
reply for a streaming voice, which is still being sent the rest. A speech-to-speech model's is
its transcript so far. An edge with no track of its own reports it once publishing the first
chunk returns. `spoke` now carries the same `synthesis_id` and the `characters` synthesised
(for a speech-to-speech model, its transcript's), and always comes after the utterance's
`speaking`. An utterance the caller talks over is still reported `speaking` if any of it was
heard, and its cut is `interrupted` as before. What is spoken, and when, is unchanged.

`speaking` also carries `delay_ms`: for an utterance reported before the track got to it (one
that finished being made behind speech still queued), how long after `at` it is expected to be
heard, from the speech queued ahead of it at the edge. `interrupted` carries `synthesis_id` and
`heard_ms` when any of the turn had been heard: the utterance the track was playing and how
many milliseconds of it went out before the cut. Every character count on the session socket,
`characters` included, is in Unicode code points.

Each of the agent's spoken messages in the Chat channel already carries that same `turn_id`,
now pinned by tests: one being written, settled, cut off, and a greeting (`say-...`) or other
line said as written (#1128).

The JS SDK's `session.speech` adds it up: `heard(message, now)` is `{ chars, done }`, the code points
of the turn heard so far, exact once `spoke` has measured the playing utterance and estimated
at the voice's rate before that, and `done` once it has played out or was cut off, at what the
router says was heard. On a call, a reply known to be coming, from its first text on the socket
or from `heard` being asked about the agent's message while it is still being written, is
`{ chars: 0, done: false }` until its `speaking`, for at most `SPEECH_GRACE_MS` (5 s); one that
turns out to say nothing is let go of at once, and one let go of stays let go of. Leaving the
call or stopping the voice ends the holding. It reads the frames the session already receives,
so `events()` is untouched (#1142).

### Each utterance says where it is in the reply's message

The voice is not sent the words the reply's Chat message holds: a voice that acts directions
gets `[laughs]`, "last four 1234" goes to it as "last four digits, 1234", the space between
two sentences is not sent, and a sentence held back is in the message as it is written but
never said. Counting the voice's characters into the message drifts from the words being said
and cuts words in half. `speaking` and `spoke` now carry `message_start` and `message_end`, the
utterance's place in the message's text in Unicode code points, from one sentence to the next
with no gap, and `interrupted` carries the cut utterance's beside `heard_ms`. A streaming
voice's utterance grows as it is sent each sentence, so its `speaking` says how far it had been
sent and its `spoke` the whole. They are left out where the router has no record of the
utterance. The message no longer starts with the blank space a model may open a reply with, so
the offsets count from its first word.

A reply a sentence voice started saying before the model finished it was split in two in the
Chat channel: the first message settled to the text so far, and a second one carried on and
settled to the whole reply, its start said twice. It is now one message, settled once the
model has finished and the voice has said some of it (#1145).

### A tool call waits for the person to sign in, and carries on once they have

When the agent calls a connector bound as each person's own (`connection.type: session`) that
the person has not connected, the call now waits for them to sign in, instead of telling them to
press a button and ending the reply. Its `ai_tool_call` step is `awaiting_authorization` and
carries an `authorization` with the `type` (`oauth`), `provider`, `provider_name`, `title`,
`message`, sign-in `url`, `handoff_token` and `expires_at`. Once the person signs in, the same
call runs in the same reply, on their account.

A `tool_approval` that names only the `tool_call_id`, with `allowed` false, declines the sign-in.
A consent refused at the provider, one that came back with another account, and one not
finished in ten minutes end the call too, and `authorization.status` says which (`declined`,
`failed` or `expired`). The agent is told either way. Stopping the reply ends the call too, with
nothing run. The `connector_authorization` attachment is still written for the clients that
render it.

The reply stays open while the call waits, and a new message is refused meanwhile, as during
any reply. A client that renders neither the step nor the card should let the person stop it.
A client that read `authorization_required` from the call's `tool_ran` now gets that frame
only once the sign-in ends.

This also fixes the chat that carried on after a login. Its follow-up made the same call again,
and the agent answered it with the earlier "not connected" rather than running it.

Swift has `SessionOptions.returnURL` and `AgentSession.declineLogin`, and Stream's
`AIToolAuthorizationView` shows the sign-in in a web authentication sheet when the
`AIToolApprover` has an `AIToolAuthorizer`. The Swift assistant connects Linear and Gmail this
way, so its Gmail button is gone, and so are its backend's routes to start, read and delete the
Gmail connection (#1114).

### A run takes images

`POST /v1/agents/runs` takes optional `images`, up to 8 `{url, detail}` like a response's, that go
with the input in the order sent, so the input can refer to them by their place in the list. The
agent's vision skill looks at them, or its conversation model when that can see. An image that
is no HTTP(S) URL or image data URI, or more than 8, is a 400 `invalid_request`; an agent that
can see neither way is a 400 `invalid_config`, not a 503 `model_failed`, since retrying fails
the same way every time. The router fetches each image URL before the run starts, from
public hosts only, and hands the model the bytes, so a run never answers without seeing an image
it was given: one that does not answer 200 within 15 seconds, is over 5 MB, or is not a JPEG,
PNG, WebP or GIF is a 400 `image_unavailable` naming the image. A run without images is
unchanged. An `agent_moderator` run's answer format now lets a `match` cite one of its case's
own image references (`image:0`, a place in `images`) as evidence instead of a quote, and asks
for `uncertain` on a label that depends on an image the agent could not see; the JSON shape is
unchanged (#1133).

### An agent config sets the guide its calls are reviewed by

`AgentConfig` takes an optional `review` with `instructions`: what the model reviewing a
finished call judges it by, in place of Stream's default guide. The reviewer is still told to
answer with a summary, a 1 to 5 score and notes, so only the guide changes. It is set on
create, update and patch; left out on an update, a patch or a folder sync it stays, and an
empty string on an update or a patch removes it. An agent with none is reviewed exactly as
before (AI-1655, #1111).

### Every phone call logs where it ran, and `phonecheck` places a test call to show it

The router logs the sites a phone call used: the SFU the agent joined (and any move to another),
the Stream SIP host for placed and arriving calls, both sides' media addresses on calls through a
customer's own SIP trunk, and, after the call ends, how many SFUs Stream used and where. The
agent's SFU is in the `details` of the session's `joined` log row. A placed call writes a
`call_placed` row with its Stream SIP host, vendor and vendor call id (and the trunk's host,
`customer_trunk_host`, on a call through a customer's own trunk); a call through a customer's
own SIP trunk also writes a `bridge_media` row with both sides' media addresses once it connects.
An ended call writes a `call_sites` row with Stream's SFU count and places. The line "ending the
session at its timeout" now says which timeout it was, `timeout=hangup` or `timeout=start`. No API
changed.

`acceleration/cmd/phonecheck` places a live outbound or inbound test call through a router, local or
remote, with Twilio or Telnyx playing the person, and logs the site of every hop and the call's
timings. On a call through a customer's own SIP trunk its `stream_sip` hop is Stream's media address
from its SDP answer (usually Stream's SIP server), `observed`; otherwise it is looked up in DNS and
marked `inferred`. `session_ms` is how long the agent's session lasted, `person_call_s` the call
itself. It changes nothing at any vendor: the person side is set up by hand first, as its README
says. See its README, or ask Claude to run one with the `phonecheck` skill.

### The router serves Prometheus metrics, and warns when the agent talks over itself

`metrics_addr` (`ROUTER_METRICS_ADDR`, e.g. `:9091`) serves Prometheus metrics at `/metrics`
on a listener of its own, never the public one; it is empty by default, which serves none.
When a voice publishes the audio of two utterances of one session interleaved, which the
caller hears as the agent speaking over itself, the router logs one warning for the pair
naming both syntheses and turns, and counts it in `router_agent_speech_overlaps_total` and
`router_agent_speech_overlap_seconds_total` (label `provider`). The agent's log lines now
carry `session` and `call`. What is spoken is unchanged (AI-1550).

A router stopping with a drain (`shutdown.drain`) stops serving metrics as it stops taking
work, and logs `stopped serving metrics for the router taking over` once at INFO, so a
router started after it on the same `metrics_addr` gets the address within its next 5 s
retry rather than when the drain ends. Prometheus then scrapes the router taking work
instead of the draining one. Without a drain, metrics are served until the process exits,
as before (AI-1826, #1200).

### The router logs which turn a turn follows, each tool call and each utterance

`voice turn timing` names what started the turn (`kind`: `caller`, `tool`, `chase`) and, for a
turn the agent started on its own, the caller turn it follows (`parent`). Each tool call the
model makes logs one INFO line when it comes back, with its tool, `tool_call`, turn, `duration_ms` and
`outcome` (`ok`, `error`, `still-running`, `cancelled`, `refused` for a call the router turned back itself, before the tool ran), and is counted in
`router_agent_tool_call_seconds` (label `outcome`). An utterance's start and end, a line the
agent chose to say (a hold line, an update, a check-in) and a chase for a call a reply owed are
logged at INFO, without what is said. `router_agent_open_utterances` (label `provider`) counts,
for each chunk published, how many of the agent's utterances were open. What is spoken is
unchanged (AI-1551).

### The router logs what it heard from each caller, and counts inbound audio

When a caller's turn settles, when the agent checks in on a silence, and when the call ends, the agent logs `what the agent heard from a participant` at INFO, one line per participant (`moment`: `turn`, `checkin`, `end`): the audio chunks received from the call (`frames`, `audio_ms`), counted from the start of the call and not per turn, the transcripts with words in them (`partials`, `finals`), and how long ago the last of each came (`since_frame_ms`, `since_words_ms`, `-1` for never). No transcript text. Two lines tell a caller who went quiet (no frames) from a transcriber that stopped answering (frames, no words). `router_agent_inbound_audio_frames_total` (no labels), `router_agent_stt_partials_total` and `router_agent_stt_finals_total` (label `provider`) count the same for Prometheus. What is heard and answered is unchanged (AI-1553).

### A connector manifest can cap a connection's calls in flight

`rate_limit.max_concurrent_calls` in a connector's manifest caps how many of one connection's tool calls a router sends at once. A call over the cap waits for one to end, within its binding's `timeout_ms`; if it is still waiting then, it is not sent and the model reads a `connector_busy` result, and the call's row in the invocation log is `denied` with the message `held by the router: in-flight limit <n> reached on the connection`. The cap is per connection and per router process. A manifest without the field has no cap, as before.

### Agents can schedule tasks for later

An agent with `scheduling.enabled` on its config is offered four tools —
`schedule_task`, `list_scheduled_tasks`, `update_scheduled_task` and
`cancel_scheduled_task` — so a user can ask it to remind them or do something later, once
or on a repeat (`FREQ=DAILY`, `FREQ=WEEKLY;BYDAY=MO,WE,FR`, …), at the hour they mean in
their own IANA timezone. `CreateSessionRequest` takes a `timezone`, remembered as the
user's for what follows. When a task comes due, a worker on every router claims it once
and the agent carries it out in the user's own conversation, with its usual tools; a task
that fires mid-call waits for the call to end rather than interrupting it. Scheduled runs
are customer spend and never consume the end user's daily caps, the replies that finish
them included; what the user types next is counted on their day as before.

The same tasks are a REST resource, `/v1/agents/scheduled-tasks`: a backend schedules for
the user it names with `X-Stream-User-Id`, and the owner reads, changes, pauses, resumes
and deletes their tasks from a device, runs included. A one-shot resumed after its time
came is completed, its occurrence listed as a `missed` run with error `task_paused`.
`agent.yaml` takes a `scheduling:`
block (`enabled`, `max_tasks_per_user`, `default_timezone`) in the Go, Python and
JavaScript SDKs.

Tasks also say where they are delivered: `deliver_to` is an ordered list of
`conversation`, `sms`, `whatsapp`, `imessage` and `phone`, and the first that lands is
where the run says it landed. A text channel carries the agent's own words out to the
number the user already texts the agent from, under the 10DLC gate, and WhatsApp only
inside the day after the person last wrote; `phone` rings them and speaks it instead.
`quiet_hours` holds an occurrence that comes due in them until they end, on every
channel, without counting it a failure.

### An agent config has a `role`, and the app's moderator agent is found by it

`AgentConfig` takes an optional `role`, whose one value is `agent_moderator`: the app's Nora, the
agent the chat worker hands flagged messages to for review. It is set on create, update, patch
and a folder sync (`role:` in agent.yaml), and `GET /v1/agents/configs?role=agent_moderator` answers
the app's one moderator, or an empty list. One live config per app holds a role: giving it to a
second is a 409 `role_taken`, a deleted config frees it, and a test copy (tagged `draft_of`) may
carry the role of the config it copies without being found by it. A save that leaves `role` out
keeps the stored one, so a client that does not know the field cannot take it away. Configs the
dashboard tagged `role: agent_moderator` before the field existed take the role when the router
migrates: the newest live one in each app (AI-1472).

### A template for the app's moderator, and a run of it answers in a fixed format

`acceleration/templates/agent_moderator` is the starting point for an app's Nora: a text agent
on `llm-thinking` with `role: agent_moderator`, reviewer instructions (content is untrusted data,
flags are hypotheses, the current content is judged first, evidence is quoted exactly) and a
skill each for profanity, harassment, suggestive language and actions. The app may edit all of
it. The answer format the chat review worker parses, one JSON object with per-case, per-label
judgments, evidence, a flag assessment and summaries, is not in the template: `POST
/v1/agents/runs` adds it after the config's and the caller's instructions on every run of an
`agent_moderator` config, so no edit can break parsing. Other configs' runs, and sessions such as
Ask Nora, are not held to it (AI-1479).

### The app's moderator counts moderation data across the app

Two more read-only tools for an `agent_moderator` session answer questions no single page of
the review queue can, such as "who is the most violating user this week?":
`top_flagged_users` ranks the users whose content was flagged by violations (flagged items
whose action was not restored or unbanned), each with flags, items a moderator acted on,
overturned items, Nora's recommendations and top labels, optionally for one label or entity
type. `moderation_summary` counts flagged items by label (with how many were overturned), by
action and by channel. Both read a window of up to 30 days (7 by default) and the router does
the counting. A call reads at most 2,000 items, newest first; past that the answer is marked
`partial` and says its counts are a lower bound. They are offered on the same terms as the
other moderation tools. The moderator template tells Nora to use them, and two simulations
check it ranks by violations and says when counts are partial (AI-1612).

### Simulations for the moderator template

`acceleration/templates/agent_moderator/simulations/moderation.yaml` holds eight reviews Nora
must get right, ported from the Nora prototype's evals: casual profanity in an adult market and
the same words in an education app, an educational quotation, targeted abuse, repeat confirmed
harassment, an overturned incident that must not escalate, an innocent message from a user with
prior bans, and an instruction injected into the message. A simulation talks to Nora in chat
rather than through a run, so each one hands over the policy with its cases, and its assertion
judges the verdicts rather than the answer format. Run them after editing the agent with
`getstream agents simulate nora` (AI-1506).

### The moderator template reviews more than chat messages

A case may carry an `entity_type`, and the moderator template's instructions say what context
each kind brings: a chat message (or a case without one) its `conversation`, a Feeds activity
or comment its `activity` with the comments and reactions in place, a user profile
(`stream:user`) its fields as the case's text, and a custom check type its text alone, read
with the entity type as a hint and not judged uncertain only for lacking a conversation.
Evidence still quotes only the case's own text, and the answer format is unchanged. Four more
simulations cover a product review, a Feeds comment, a Feeds activity and a user profile
(AI-1624).

### The app's moderator reads the app's moderation data

A session or run of an `agent_moderator` config is offered six read-only tools on the app's
moderation data: `query_review_queue`, `get_review_queue_item` (with its flags and actions),
`query_moderation_configs` (with their LLM labels), `query_moderation_rules`, `list_blocklists`
and `query_moderation_logs`. They act with the Stream identity the router already holds for the
app, so no secret is passed in. They are offered only to a session the customer's backend
opens (a server-side caller), never one an end user's device opens, and only in an app of the
customer's own: a session in the deployment's shared app is offered none. Reading the review queue never locks an item, and a
tool the caller declares under the same name stays the caller's (AI-1504).

### The moderator template proposes configuration changes from Ask Nora

`acceleration/templates/agent_moderator/skills/suggest_policy.md` moves the Nora proof of
concept's configuration suggestions into the template. From the live policy, the last 7 days of
reviews (or the days the moderator names) and where Nora disagreed with the first layer, it
proposes up to 8 changes to label actions, LLM labels and severities, the app context, blocklists,
rules and the flooding filter, each with the counts behind it, as the JSON the dashboard saves.
It only proposes: nothing changes until the moderator confirms in the dashboard (AI-1505).

### A recording moves to the next model when the vendor fails it

`POST /v1/tts/recordings` on a shortcut or a priority list no longer fails when the model chosen
fails the recording itself: an error status (a 5xx, a 429, a 402 out of credits), a timeout or
an answer that cannot be read. The text goes to the next model the request resolved to, in the
order of the priority list or the shortcut's ranking, and the job names the model that spoke it.
The failure is still counted against the failing model's health. A model that refuses this
request alone (a text over Inworld's 2,000-unit limit, a format the vendor cannot make) hands it
on too, without a health hit. When every model fails, the job fails with no provider or model
and an error naming each model's reason, cut to 300 bytes each. A pinned model fails as before.
A recording with nothing to say is refused before any vendor is asked, and no longer counts
against the model (AI-1465).

### An Android demo, the Android half of swift_demo

`examples/voice_agents/android_demo` talks to swift_demo's agent from an Android app, built
against `sdks/kotlin` as it is in the checkout. Its Chat screen shows the conversation with
Stream Chat's AI components: the reply streaming in, its reasoning, each tool call, and the
refund's question on the reply. Its Voice screen puts the agent on a call and shows it as
Stream's voice glow, moving with whoever is talking, with what is said captioned two lines at a
time and the refund's question as a card over the call. CI builds it beside the Kotlin SDK.

### Inworld records speech

`POST /v1/tts/recordings` can be served by Inworld, pinned (`inworld/inworld-tts-2`,
`inworld/inworld-tts-2-flash`, `inworld/inworld-tts-2-batch`), named in a priority list
(`["inworld"]`, `["elevenlabs","inworld"]`), or reached from a shortcut when the models ahead of
it cannot be built. It records on Inworld's `/tts/v1/voice` in the voice, model, delivery mode
and language its live voice uses, and `can_record` is now true for it in `GET /v1/tts/routes`.
`en-recorded` and `multilingual-recorded` still prefer ElevenLabs v3, then Cartesia. `format`
takes `mp3_<rate>_<kbps>`, `pcm_<rate>`, `wav_<rate>`, `ulaw_<rate>` and `alaw_<rate>` from 8000
to 48000 Hz, `mp3_44100_128` when left out. Any other format, a speed, volume or emotion, and a
text over 2,000 UTF-16 code units are refused naming it: the format and the text as a failed job
(the request is accepted with 202 and the job ends `failed`), not counted against the model's
health, since the caller is the one to change them. A live Inworld voice asked for a format is
refused as the caller's mistake. `tl` is sent to Inworld as `fil`, which it resolves.
Breeze still only streams (AI-1463).

### Cartesia records speech

`POST /v1/tts/recordings` can be served by Cartesia, pinned (`cartesia/sonic-preview`,
`cartesia/sonic-3.6`), named in a priority list (`["cartesia"]`, `["elevenlabs","cartesia"]`), or
reached from a shortcut such as `tts-quality` when ElevenLabs cannot be built. It records on
Cartesia's `/tts/bytes` in the voice, model and language its live voice uses, and `can_record` is
now true for it in `GET /v1/tts/routes`. `en-recorded` and `multilingual-recorded` still prefer
ElevenLabs v3. `format` takes `mp3_<rate>_<kbps>`, `pcm_<rate>`, `wav_<rate>`, `ulaw_<rate>` and
`alaw_<rate>` at the rates Cartesia offers, `mp3_44100_128` when left out. Any other format, and a
speed, volume or emotion, fails the job and names it: the request is accepted (202) and the job
ends `failed`. A live Cartesia voice asked for a format is refused as the caller's mistake.
Recorded shortcuts (`en-recorded`, `multilingual-recorded`) with a language hint of bn, gu, he, ka, kn, ml, mr, or, pa, te, th, tl or ur now record on Cartesia: ElevenLabs v3 lacks them and they were refused before.
Breeze still only streams (AI-1462).

### A phone on a call can show the agent its camera

`VoiceSession.join(camera: true)` or `setCameraEnabled(true)` now lets the agent look through the
phone's camera. The call's frames are sampled twice a second through its video filter and the last
four seconds kept; when the agent's `vision` skill asks for what the camera showed
(`get_video_frames`), the call's `AgentSession` answers with the frames taken at or before the
moment the question was taken, newest last, at most eight, as the router's Python video workers
do. With nothing from the last five seconds it says so as the answer, so the agent asks the person
to show it again. The Swift assistant example has a camera button in voice mode, with a preview
of what Juno sees. Kotlin does the same from `VoiceSession.join(camera = true)` and
`setCameraEnabled(true)`, sampling through `call.videoFilter` on top of any filter already set
there, Stream Video's moderation blur included, so the agent sees what the call publishes.

### A tool can show the model pictures

A tool may now answer with pictures beside its words: in Go by returning a `tools.Output`
(`Text`, `Images`), in Swift by returning a `ToolOutput` (`text`, `images`) from an `AgentTool`
whose `run` returns one, and in Kotlin by returning one from `AgentTool.withPictures`. Each picture is a data URI or an https URL, and the session socket
carries them as `tool_result` parts, which the router already took. On a call it hands them to
the agent's `vision` skill, so a tool can show the agent what a camera sees. In a written
conversation, which has no vision worker, a conversation model that takes images is shown them
itself, with the reply that follows the call alone: once the next question comes they stand as a
note, as a person's photos do. A model that reads only text is told such a tool failed. A tool
hosted on a dispatch worker answers in words only and is reported as failed when it returns
pictures. In Go, `tools.Registry.Call` returns a `tools.Output` instead of a string.

### `router.tts.recording()` takes `inline`, and `voice` says it takes your own voices

`await router.tts.recording(text, inline=True)` speaks a short text while the request waits and
returns the audio in the answer, storing neither a job nor the audio: the cheap way to check
that synthesis works on an app. It takes at most 16000 characters and 90 seconds, and with a
`callback` it raises `ValueError`, since the router refuses that pair. Without `inline` the
request is the same as before. The `voice` of a session, an agent config and a session update,
and of `stream.TTS`, `Accelerated` and `define_agent`, is now documented the way
`TtsOptions.voice` is: a provider's own voice id, or one of your voices, with `custom:` to mean
only yours, so a typo is refused instead of passed to the provider's library. A native (sts)
session takes the vendor's own voice name instead. The `POST /v1/tts/recordings` docs now
describe `inline` (AI-1414).

### `stream.STT` takes `keyterms`

`stream.STT(target, keyterms=[...])` sends words the transcriber would otherwise get wrong,
such as names and member IDs, at most 100. They went only through the untyped `options` dict
before; `keyterms` overrides one given there. A model that takes no keyterms refuses them,
and `start()` raises that (AI-1422).

### A person can send photos in a conversation kept in Stream Chat

A question with a `request_id` may now carry `images`, on `POST /v1/agents/sessions/{id}/responses`
and on the session socket's `respond`, where before it was refused as "a request id carries text
only". So a signed-in device can send photos in its saved conversations. Each image is a data URI,
at most 8 to a question. The router uploads them to the conversation's channel as the person and
writes them on the person's message as `image` attachments, so every reader of the channel sees
them. The model sees them with the reply to that question alone, as it already did in a session
not kept in Chat; when the conversation is reopened, history says the photos were sent without
sending them again. A retry with the same id, text and images is the same question; different
images under the same id are a 409. Videos still take no request id. The Swift and Kotlin SDKs now
send a request id with every question, images or not.

### The reply model is told who is speaking once a second person talks

A hosted voice agent on a call with several people now sends each new turn to the model as
`<name>: <words>` once a second participant has had a turn answered, and adds one line to the
instructions saying several people are talking and that a reply never opens with a name. The
name is the participant's display name, else their user id. Turns from before the second person
spoke stay as they were, words typed into the session are the caller's and never count as a
second person, and with one person on the call nothing changes. A speech-to-speech model taking
over the call reads the same names in place of `Caller`. The session history, the timeline and
the flow controller keep the words as said. Python LLM plugins follow (AI-1420).

### A session can join a Stream call that already exists

`POST /v1/agents/sessions` (with `start_voice`) and `POST /v1/agents/sessions/{id}/voice` take
an optional `call: {type, id}`: an existing Stream call for the agent to join, such as the
`default` call the people are already on. Left out, the agent joins `agent:<session id>` as
before. A `call` without `start_voice` or together with `phone` is a 400. The call has to be in
the Stream app the session acts in: one another session runs in another app is a 400. The
router does not ask Stream whether the call exists. A device sending `call` is a 403, as with
`history`, and the session socket's start frame takes none, since the socket carries its own
audio.

Voice stopped and started again goes back to the named call; naming a new call when voice is off
moves it there, and naming a different one while voice is on is a 409. A session carried on from
its row keeps the call; a fork does not. Two sessions joining one call as the same `user_id`
leave the newest, as two sessions on `agent:<session id>` always did.

A call runs on its own type's audio settings. The built-in `agent` type turns DTX and noise
cancellation off so turn detection hears every word (AI-1291, GetStream/chat#18795); the
`default` type has DTX on and noise cancellation on.

Go takes it as `client.SessionOptions.Call` and `Session.Voice.Start(ctx, client.OnCall(type,
id))`, and as `stream.Call.CallType`/`CallID`. `stream.Accelerated` (Python) now joins the call
the agent was given: a call of any type but `agent` is sent as `call`, and the router chooses the
session id, so a second session on the same call is not refused for reusing an id. An `agent`
call is held by its call id as before. Other SDKs follow.

### Grok Voice 3.0 and GPT-Realtime-2.1 for speech-to-speech

`xai/grok-voice-3.0`, first on the Artificial Analysis speech-to-speech index, is in the
low-latency tier at $4.80 an hour of caller audio, and `openai/gpt-realtime-2.1` is in the
high-quality tier at GPT-Realtime-2's rates. `sts-vision` now prefers GPT-Realtime-2.1.
`xai/grok-voice-think-fast-2.0`, which xAI have deprecated, and `openai/gpt-realtime-2`
stay declared, so a config that names them keeps working. `openai/gpt-live-1` now lists
German, French, Hindi, Japanese and Korean, the languages of OpenAI's new voices.

### Nemotron 3.5 Super VL and Muse Wonder in the built-in LLM config

`together/nvidia/NVIDIA-Nemotron-3.5-Super-VL`, the strongest US open-weight model, is in the
low-latency tier with image input, so `vlm` can fall over to it; it answers without thinking.
`meta/muse-wonder-1.0`, 52 on the Artificial Analysis Intelligence Index at max effort, is in
the high-quality tier with image input at $4 in and $20 out per million tokens. It reuses
`META_API_KEY`, and no shortcut prefers it. An open-weight model with `-VL` in its id now
accepts images on any host.

### A backend can give one session instructions of its own

`POST /v1/agents/sessions` and the session socket's start frame take `instructions`: what the
agent is told to be for this session alone, in place of its config's, such as the brief of a
customer somebody practises on, without a stored config per brief. Empty or left out, the
config's instructions stand. A device sending them is a 403, as with `history`. A fork of a
running session keeps them; a session carried on after it ended in another process, a fork of
one no longer running and a fork naming a config run on the config's instructions again. Go
takes them as `client.SessionOptions.Instructions`, and JavaScript's `SessionSpec` from the
regenerated types; other SDKs follow.

`stream.Accelerated` now sends the agent's instructions, so an `Agent(instructions=...)` on
the hosted router plays the role it is given at call time, over its `config`'s instructions if
it has one. An agent with no config and no `instructions` now sends `DEFAULT_INSTRUCTIONS`, as
`Agent`'s docstring says it is told; before, the router ran it on none. An agent with a
`config` and no `instructions` of its own is unchanged: the default is not sent over its
folder's instructions.

### A login the chat asks for can go back to the app's own page

`POST /v1/agents/sessions` (`createSession`) and the session socket's `start` frame take an
optional `return_url` (AI-1384): the app's own page that the logins and step-ups the chat asks
for go back to, as `createAuthorization`'s `return_url` does for a consent the backend starts.
A page of that URL's origin opens the `connector_authorization` attachment's or the
`connector_scope_required` frame's `launch_url` and posts it `handoff_token`; the browser then
comes back to the URL with `connection_id` and `status` added to its query. The same rules
apply as on `createAuthorization`, and a URL they refuse is a 400 `invalid_request` starting
`return_url:` (an error frame on the socket), with no session created. The session keeps it:
a chat carried on after it ended, in this process or another, and a fork, go back to the same
page. An end user's device may send it, since a consent the chat asks for is only ever for
the caller's own connection and shown to the caller alone. The Go SDK's `SessionOptions` has
`ReturnURL` and the Python SDK's has `return_url`. Without it, nothing changes: those consents go back to the dashboard.

### A connector consent can go back to the app's own page

`POST /v1/agents/connections/{id}/authorizations` (`createAuthorization`) takes an optional
`return_url` (AI-1383): a page of the app's own that the consent goes back to, with
`connection_id` and `status` added to its query and its own query kept, whether the consent
connected, was denied or failed. Open `launch_url` in a popup from a page of that URL's origin:
the launch page takes the handoff from that origin only. It must be https, or http on
`localhost`, `127.0.0.0/8` or `[::1]` for local development, with no user name or password and
no fragment, at most 2048 bytes; anything else is a 400 `invalid_request` naming the rule.
A host must be ASCII (write an internationalized one in punycode, `xn--`) and not an IPv4-mapped
IPv6 address, since a browser's origin would not match the origin the launch page names. The
query comes back re-encoded with its keys sorted (`?a=1&a=2&b` becomes
`?a=1&a=2&b=&connection_id=…&status=…`), and a `connection_id` or `status` already in it is
replaced by the router's. A consent that has a `return_url` no longer sends the person back to
"the dashboard" in the launch page's own messages. The operation used to ignore a body; it now
reads one, so a body that is not a JSON object (`null`, `{`, a `text/plain` one) is a 400 or 415
where it was a 201; no known caller sends one.
Without it, nothing changes: the dashboard opens the launch page and the consent goes back to
the dashboard, as does a consent the chat starts (until AI-1384, above). The generated clients are regenerated: Go's
`CreateAuthorization` and Rust's `create_authorization` now take the request body.

### A native app can run a connector consent

`return_url` on `createAuthorization` and `createSession` may now be a native app's own private-use scheme, in the reverse domain form RFC 8252 section 7.1 asks for, such as `com.example.app:/connected`. An app has no web page to open the launch page from and post it the token, so it opens `launch_url#handoff_token=<handoff_token>` in its web authentication sheet (`ASWebAuthenticationSession` on iOS, a Custom Tab on Android). The launch page of such a consent reads the token from its fragment, which no server is sent, and drops it from its history; the callback then hands the sheet back to the app's scheme with `connection_id` and `status`. A scheme with no dot, such as `myapp://`, is still refused, and so are a fragment, a user name and a query that is not key=value pairs. A consent that returns to a web page or to the dashboard gets the same launch page as before, byte for byte.

### The Gmail connector writes drafts, and a backend can send mail through it

The `gmail` connector's revision 3 asks for `gmail.compose` beside `gmail.readonly`, so the Gmail MCP server's `create_draft` works, and gives `https://gmail.googleapis.com` as its `api_base`, so a backend can send through the connection proxy: `POST /v1/agents/connections/{id}/proxy/gmail/v1/users/me/messages/send` with the RFC 5322 message base64url-encoded as `raw`. Google's Gmail MCP server has no tool that sends. A connection made on revision 2 asks for the new scope at its next consent.

### Built-in BigQuery and Metabase connectors

The connector catalog (`GET /v1/agents/connectors`) has `bigquery` and `metabase` (AI-1381).

- `bigquery` is Google's hosted BigQuery MCP server at `https://bigquery.googleapis.com/mcp`. A connection signs in with `oauth2_code` through the same Google web client as the other Google connectors (`GOOGLE`), asking for `https://www.googleapis.com/auth/bigquery`, the only scope Google documents for it.
- `metabase` is the MCP server a Metabase instance serves at `https://{host}/api/metabase-mcp`. A connection gives `host` (a host name only), and the router registers its own client with that Metabase (`dcr`), asking for `agent:content:read` and `agent:query:run`: search, read and run queries, but no raw SQL and no writes. The Metabase must be reachable on a public https address; one on a private network is refused, as for `mcp_servers`.

### An agent config says which agent.yaml connectors its last sync left out

A sync (`POST /v1/agents/sync`) takes `left_out_connectors`: the agent.yaml connectors the
client left out of `connectors` because the app has no connection by the id or label they
name, each as `{name, connector_id, connection}`. The router stores them as sent and serves
them, read-only, as `left_out_connectors` on the agent config (GET, list and the sync's
answer), so the dashboard can say what the agent is missing. They go only alongside
`connectors`, else the sync is refused with a 400: a sync that sends `connectors` replaces
them, and one that sends neither leaves them alone. A save or a patch of the config keeps them. The Go SDK's `Agent.Sync`
sends them, beside the warning it already logs, and leaves out of the list an entry longer
than the router takes (the warning still names it). A sync that left a connector out sends a
hash of its own, so an agent an earlier SDK synced with the same missing connection stores the list. A config with none left out answers as it
did (AI-1371).

### ElevenLabs style exaggeration through the `elevenlabs` overwrite

A voice's options take ElevenLabs' style exaggeration as `{"elevenlabs": {"voice_settings": {"style": 0.4}}}` in `overwrites`, on `/v1/tts/stream` and on `/v1/tts/recordings`. It is a number from 0 to 1, and not the shared `style`, which is a word to sound like. Flash v2.5 and Multilingual v2 send it when each context opens. The v3 and v4 models have no style, so they refuse it with the overwrite error, and the router then tries only ElevenLabs' other models: `en-low-latency` is spoken by Flash, and `tts-quality` and the recording voice (`eleven_v3`) refuse the request. A value outside 0 to 1 is refused too. A request with no style sends what it did before (AI-1367).

### A backend lists every user's connections

`GET /v1/agents/connections?owner_type=user` from a server-side caller that sends no
`X-Stream-User-Id` lists the user-owned connections of every user of the app, newest first,
paged by cursor and narrowed by `connector_id` as the other lists are. Each one names its user
in `owner.user_id`. It was a 400 before. With `X-Stream-User-Id` it still lists that user's
only, and an end user's device is still refused the operation. This is for an admin view,
such as the Volt dashboard's.

### Listing connections without `owner_type` lists the app's own

`GET /v1/agents/connections` takes `owner_type` as optional. Left out, it lists the app's own
connections, as `owner_type=app` does, where it was a 400 before. `owner_type=user` lists the
user `X-Stream-User-Id` names, or every user's without it.

### Approve a connection's changed tools again

`POST /v1/agents/connections/{id}/tools/approve` (`approveConnectionTools`) takes
`{"tools": [{"name", "schema_digest"}]}` and pins each tool at that digest for the connection's
current grant, so the next session offers it again and the next validate answers `connected`
when no other tool changed (AI-1305). It is the way out of `needs_reapproval` for a bearer or
`api_key` connection, which no consent moves to a new grant. Each `schema_digest` must be the
one `GET .../tools` lists now, which a validate refreshes: a tool it does not list is a 400, one
listed at another digest a 409. A pin or grant that changed since it was read is a 409 too, and
a refused approve writes nothing. Each approve is one audit row, the new `ConnectorAuditAction`
`tool_reapproved`, whose new `tools` field names each tool with the digest approved and the one
it was pinned at. Server-side only; who may approve is who may read the connection. Migration
`20261101130000` widens the CHECK on `connector_audit.action` and adds `connector_audit.tools`.

### Validate says when a provider changed a tool sessions pinned

`POST /v1/agents/connections/{id}/validate` now compares the tools the provider lists with the
schema each was pinned at for the connection's current grant, the pin a session takes the
first time it offers a tool granted by name or by pattern. A tool listed at another schema is
`status: needs_reapproval`, a new `ConnectionValidationStatus`, with
`code: connector_tool_schema_changed` and the tools in `changed_tools`, sorted; the connection's
`last_validation` keeps the status, the code and an error naming them. It used to answer
`connected` while sessions left the tool out. Sessions still leave it out, and a validate never
pins it again: `POST /v1/agents/connections/{id}/tools/approve` approves it, as does a
reconnect of an OAuth connection, which begins a new grant. A pinned tool the provider no longer lists
is not reported. A grant lacking scopes is still `needs_scopes`, since the consent it asks for
approves the changed tools too. A connection with no changed tool answers as before.

### Linear with your own OAuth app, for direct GraphQL calls

The connector catalog has `linear_app`, «Linear (own OAuth app)». A connection signs in with `oauth2_code` through the app's own Linear OAuth app (`PUT /v1/agents/connectors/linear_app/oauth-client`). Its token serves both Linear's MCP tools and Linear's GraphQL API: `POST /v1/agents/connections/{id}/proxy/graphql` reaches `https://api.linear.app/graphql`. The `linear` connector is unchanged and still MCP-only (AI-1351).

### A built-in Notion connector

The connector catalog (`GET /v1/agents/connectors`) has `notion`: Notion's hosted MCP server at `https://mcp.notion.com/mcp`. A connection signs in with `oauth2_code`. The router registers its own client with Notion (`dcr`) and asks for the `default` scope, so no app registers anything first. The agent can search, read and update the pages and databases that the person who signed in can access (AI-1292).

### Each app sets its own daily limit per end user

An app sets what one of its end users may spend in a UTC day as `quota` in its policy,
`PUT /v1/policies/app`: `messages_per_user_per_day` and `tokens_per_user_per_day`, each from
`0` (no cap) to a bound that only catches a typo. Left out, the deployment's
`ROUTER_RATE_LIMIT_*` default applies, as before. An organization has none: a `quota` sent to
`PUT /v1/policies/organization` is a 400. A counted answer carries `X-Quota-Limit`,
`X-Quota-Remaining` and `X-Quota-Reset`. Each answer is counted on whoever sent the
turn: in a session a backend opened in a user's name, the backend's turns are not counted and
the user's own are. `mock/*` answers are not counted, and an address is counted per app.

### A connection's call log says why a call failed

`GET /v1/agents/connections/{id}/invocations` gives each failed call a `provider_status`, the
HTTP status the provider answered it with, and an `error_message`: the start of the provider's
error body when it answered with a 4xx or a 5xx, or else what the call's error said, at most
1 KiB, with every URL cut to its host and the connection's credential and the request's path
taken out. A call the router held for a
provider's rate limit, and so never sent, keeps `error_type: denied` and says
`rate limited by the router: retry after <seconds>s`. Both are absent for a call that answered
and for calls recorded before this release. What the model reads of a call is unchanged.

### Fish Audio Drama 3 is `fish/drama-3`, and the `fish` default

Fish's hosted Drama 3 can be routed as `fish/drama-3`, with the same `FISH_API_KEY`, and a
`fish` target that names no model now gets it instead of `s2-pro`. Like ElevenLabs v3 it acts
bracketed directions such as `[laughs]` or `[whispers]` rather than reading them out, so the
model writing the lines is told it may use them. `fish/s2-pro` is still served as before.

### A customer's website can call the router directly

A browser page on any site may call the API and open a session socket with an API key and a
user token, so client-side auth needs no list of allowed sites. `ROUTER_CORS_ORIGINS` and
`cors_origins` are gone and ignored if set. A router in `noauth` mode, where a caller names its
own tenant, answers pages on `localhost` and loopback addresses only.

### Connector bindings can name tools by pattern

A connector binding may set `tool_patterns` instead of `tools`: tool names or `path.Match`
patterns, at least one and at most 128. A binding has one or the other. An empty list is
refused. A `*` offers every tool the connection lists. Each match is pinned on that connection
the first time a session opens it, including a fixed connection, and a later session offers the
tool only while that digest still matches. When more than 128 tools match, the session offers
the first 128 by name.

### agent.yaml connectors in the Python SDK

`agent.yaml` may name `connectors`. A name is a session binding that offers every tool, kept
as written. An empty name is refused, so a blank entry cannot wipe the stored bindings. An
object names the connector, the tool patterns, an app connection (its id, or the one
connection with that label) and whether the binding is required. Leaving the key out leaves
the stored bindings; an empty list removes them.

### Native full-duplex speech routing

`sts-duplex` routes only to realtime speech models whose provider declares they can listen
while speaking. Failover stays within that eligible set. Go `DefineRouter` and `SyncRouters`
support STS targets and provider-specific overwrites; partial STT/TTS updates preserve them (#887).

### Bring your own model

A language model you serve yourself behind an OpenAI-compatible endpoint (a fine-tune on
Baseten, Together or Fireworks, vLLM or SGLang on your own GPUs) can be routed like any other.
Store it with `POST /v1/agents/models` (base URL, model id, API key, and optionally its context
window, price and data policy), then name it as `custom/<name>` in a router config, a session or
`allowed_models`. The key is sealed and never returned. A shared router only dials public https
endpoints; a self-hosted one that sets `ROUTER_MODELS_PRIVATE_ENDPOINTS=true` also reaches
private and plain http ones. In Go, `client.DefineModel` stores one by name.

### A voice agent words its own hold lines, status updates and check-ins

The stock lines a voice agent rotated through, "One moment." first, are gone. When the model
reaches for a tool without a word, when a caller waiting on work is due an update, and when a
quiet call is due a check-in, the agent asks the reply model for one line written from the
conversation, with no tools and a three second limit. An update names what the caller asked
for rather than a tool, and a line is dropped if the moment passed while it was written. A
chain of tool calls gets one hold line, not one per call, and the prompts no longer show the
model a hold phrase to copy. A binding's `pre_speech` is still said at once in its place.

### A reply that is late is asked of another candidate too: `ROUTER_REPLY_HEDGE`

A model now and then takes many times its usual wait to start a reply, and the caller waits that
long. A voice reply that has said nothing, neither text nor a tool call, once `ROUTER_REPLY_HEDGE`
(`agent.reply_hedge`) has passed, `1200ms` by default, is now asked a second time, as the same
request, of another candidate of its target: one on a different model when the target has one and
on a different provider when it has one, never one the router has found unavailable. Whichever
says something first is kept, and the other is cancelled the moment it loses and still read out, so
both calls are reported as model calls and recorded, the cancelled one as any cancelled call is.
The wait counts from the request, so a model that is slow to answer at all is hedged as well as
one that is slow to start streaming. A reply is hedged once, a hedge that fails leaves the first
request going, and a target with no other candidate is asked once, as before. Replies are hedged;
the flow controller, the guardrails, background work and a reply that continues from a response
the provider holds are not. `0` turns it off, `cmd/agent -reply-hedge` is the same setting, and
the router option is `llmrouter.Options.ReplyHedge`.

### A log severity is the least serious level to show, not the only one

`severity` on `GET /v1/agents/logs` was an exact match, so asking for `error` hid the warnings
next to it and there was no way to ask for both. It now names the floor: `warn` answers with
warnings and errors, `info` with everything, and `error` is unchanged. `warn` is also a severity
a log can be written at, which it was in the data but not in the enum.

### The router records who changed the app's configuration

Every change to an agent config, a skill, a knowledge document or url, a router config, a
plugin credential or a policy is kept, with the fields that moved, who moved them and what
they used. `queryAudit` (`POST /v1/audit/query`, cursor paged) answers an `AuditPage` of
`AuditEntry` `{id, resource_type, resource_id, resource_name, agent_id, action, source,
actor_id, actor_name, request_id, changes, created_at}`, filtered by any of `resource_type`,
`resource_id`, `agent_id`, `source` and `action`. Only configuration is recorded: a session,
a simulation and a run are not, because they are traffic rather than setup. A write that
moves nothing records nothing, and a plugin's secret is never written down.

Who made a change comes from three unsigned headers a server-side caller may send:
`X-Stream-Client` (`dashboard`, `cli` or `sdk`, and `api` when nothing says), with
`X-Stream-Actor-Id` and `X-Stream-Actor-Name` naming the person behind a client that signs
its own users in. They buy a name beside a change somebody already had the credential to
make, never permission. The router keeps no email addresses, so the name is a person's name.
Go sends the client header from `Backend.Credentials`; other SDKs follow.

### A sync no longer writes over an edit made since the last one

`SyncAgentRequest` takes `check_changes`: a sync asking to be checked is refused with a 409
`unsynced_changes`, naming the fields, rather than replacing an edit made in the dashboard
since that directory last synced. Only the fields the directory declares are compared, and
only against what is stored, so a sync whose directory already holds the change goes through.
`getAgentChanges` (`GET /v1/agents/configs/{id}/changes`) answers an `AgentChanges`
`{items, last_change, synced_at}`: what changed since the last sync, for a client to show.
Syncing again with `base_change` set to the newest entry says the person has seen them and
means it. Without `check_changes` a sync behaves exactly as before, so an SDK that syncs on
startup is unaffected.

### Tools can be loaded progressively

An agent config takes `tools: {progressive: true}`, off by default, and so does `agent.yaml`. `tools` is where later tool settings go. When it is on, the model sees each plugin, MCP server and connector tool as the first line of its description, plus its argument schema with every description, title and example removed. The first time the model calls a tool, the router returns the full description and input schema instead of running the tool, and the model calls it again. Some servers put a page of instructions and examples into a tool's description; with this setting, that page is only paid for in conversations that use the tool. The cost is one extra model turn for each tool a conversation uses. User plugins are unchanged, since they already list their tools on demand. Go and Python read the key from `agent.yaml`, and JavaScript has the regenerated types; other SDKs follow.

### A connector call waits out the provider's rate limit

When a provider answers a session's connector tool call with `429` and `Retry-After`, the model reads a `connector_rate_limited` result with `retry_after_seconds`, and the router sends no call on the same `rate_limit.per` key (`app`, `tenant` or `user` in the connector's manifest) until that time passes. Every router on the same Redis holds the same calls. The router never sends a call again by itself. A connector whose manifest has no `rate_limit`, a `429` with no `Retry-After`, and a router without Redis behave as before; a router with connectors on and no Redis logs one warning at start.

### A session can be opened with the history the caller kept

`POST /v1/agents/sessions` takes `history`: the conversation so far, oldest first, as up to 100 `HistoryMessage`s with a `role` of `user` or `assistant`, `text`, and an optional author `name` and `created_at`. It is for a backend that keeps its own thread, such as one in its own Slack app, that outlives a session: open a new session with the thread here, then send the next message to `POST /v1/agents/sessions/{id}/responses`. The model is handed the history before the first response, the way a resumed conversation's is, and once any message names its author or time each user message is quoted with them behind a note that names are labels, not authority. The router records none of it, as turns, transcript or Chat messages; with `incognito` it keeps nothing at all. More than 100 messages or 60000 characters of text, a role other than `user` or `assistant`, and `history` with `conversation_id` are 400s, and a device sending it is a 403. Go (`client.SessionOptions.History`) and Python (`SessionOptions.history`) take it, and JavaScript's `SessionSpec` from the regenerated types; other SDKs follow.

### Calls from numbers on your own SIP trunk (#752)

You can now place calls from the numbers that you already have at your own carrier.
`POST /v1/phone/trunks` stores your SIP trunk: host, port, transport, username, password
and codecs. The API never returns the password. `GET`, `PATCH` and `DELETE` on
`/v1/phone/trunks/{id}` read, change and delete a trunk. `POST /v1/phone/trunks/{id}/numbers`
adds one of your numbers to the trunk. `POST /v1/phone/calls` then places a call from that
number through your carrier.

These numbers make outbound calls only. You cannot send initial digits, press digits or add
custom SIP headers on these calls. The router node that places a call holds it until it
ends. If that node stops, the call ends.

The trunk endpoints need a key encryption key (`ROUTER_AUTH_KEK`). Without a key, they
answer 400 with the code `not_configured`. The router refuses a trunk host that resolves to an address that is not public.

### Every router response carries an `X-Request-Id`, and a 500 is logged with its stack

The router answers every request with `X-Request-Id`, keeping one a proxy sent (printable
ASCII, up to 128 characters) and minting a UUID otherwise; browsers can read it through
`Access-Control-Expose-Headers`. The access log line carries the same `request_id`, the
error a request failed with and, for a 5xx, the stack where that error entered the router.
A 5xx body is now `{"error": "internal error"}` rather than the error's text: quote the
request id to find the rest in the logs.

### A provider can tell the router that a connection's grant ended

`POST /v1/agents/connectors/events/{connector_id}` (`receiveConnectorEvent`, `security: []`, not client-accessible) takes a built-in connector's provider events. Each request is checked by the verifier the manifest's `channel.verifier` names, with the operator's secret, and an unsigned or stale one is a 401 that changes nothing. Slack's `tokens_revoked` moves the revoked user's connections to `needs_reauthorization`, and `app_uninstalled` moves every connection in that workspace; the next credential resolve on any router then fails at once. A URL verification is answered with its challenge. Point the operator's Slack app's Request URL at `/v1/agents/connectors/events/slack`, subscribe it to both events, and set `SLACK_MCP_SIGNING_SECRET`. A manifest's `channel` block takes `signals`, and may have them without `messages` and `reply`. The Go client and the JavaScript types are regenerated; no SDK wraps the route.

### An agent answers in Slack threads through the app's own Slack app

`POST /v1/connectors/events/{connector_id}/{provider_app_id}` (`receiveProviderAppEvent`, `security: []`, not client-accessible) is the Request URL of one customer's provider app, checked with that app's own signing secret. On the new built-in `slack_bot` connector, a person's message in a Slack channel or DM is written into a thread channel in Stream Chat (one per Slack thread, `agent:thread-<uuid>`, naming the agent config that binds the app's `slack_bot` connection). The router answers it itself: the message hook tells the persistent text session held on that thread channel, starting one from the agent config when none runs, so the reply is kept in the thread channel, and once its final text is stored it goes back into the same Slack thread with `chat.postMessage` and the workspace's bot token. A thread channel's message is not handed to a dispatch worker. A session created with `agent_id` naming a thread channel keeps its conversation there (its `conversation_id` is `agent:thread-<uuid>`). One thread is answered by one router at a time, and a reply Slack did not take (no answer, 5xx, 429) is sent again. A retried delivery, a repeated `message.new`, a reply told twice, the bot's own messages and `message` events with a `subtype` are each acted on at most once or not at all. Slack refusing the bot token (`invalid_auth`, `token_revoked`, `account_inactive`, answered with HTTP 200) moves the connection to `needs_reauthorization`. `tokens_revoked` and `app_uninstalled` on this URL move only that customer's connection, and not one reconnected after the event was dispatched. The Go client and the JavaScript types are regenerated; no SDK wraps the route.

### A connector's raw provider events go on to the app's own URLs

`POST /v1/agents/connectors/{id}/event-destinations` (`createConnectorEventDestination`, server-side only) adds a URL that the deliveries of the app's own provider app, such as its Slack app, are forwarded to, at most three per connector. `forward: unhandled` takes what the router acts on in no way: a button click (`block_actions`), a reaction, a modal submission, and a message no agent of the app answers. `forward: all` takes every verified delivery but Slack's URL handshake. A message an agent of the app answers is still answered in either mode. Each forward is the provider's raw body with its own `Content-Type` and signature headers (`X-Slack-Signature`, `X-Slack-Request-Timestamp`), so Slack Bolt verifies it with the app's signing secret, signed on top in the Standard Webhooks shape (`webhook-id`, `webhook-timestamp`, `webhook-signature`) with the destination's own `whsec_` secret, which the create returns once. Slack's ack never waits for it. A 5xx, a 429 or no answer is sent again after 5 s, 5 min, 30 min and 2 h; any other answer is not. Slack's own signature headers come only until `X-Slack-Request-Timestamp` is 5 minutes old, the age Slack Bolt refuses a request after: a forward sent later, such as the retries after 5 min, 30 min and 2 h, carries none of them, so verify it with `webhook-signature`. `webhook-id` is the same for every delivery of one Slack event (`event_id`, or `trigger_id` for an interaction), and a digest of the body for one that names neither; a manifest's `channel` block names those paths in `event_id`. A message an agent was to answer whose write into its thread channel fails goes to the `unhandled` destinations then. One URL that does not answer holds at most 2 of a router's 16 sends, so other URLs' forwards are not held up. A router with no forward queued looks for them once a minute, not once a second, and a forward a stopped router left is sent by another within about two minutes. A private, loopback or non-https URL is refused. `listConnectorEventDestinations` pages by cursor, `deleteConnectorEventDestination` removes one, and `rotateConnectorEventDestinationSecret` returns a new secret, the old one signing beside it for 24 hours. The Go client and the JavaScript types are regenerated; other SDKs follow.

### A connection's MCP server can send the agent events (#785)

A fixed connector binding takes `events`, a list of `{event, arguments, instructions}`: MCP events its connection is subscribed to (MCP Events, a draft: `experimental-ext-triggers-events` at `6682596d`, webhook delivery). The next `validateConnection` of that connection subscribes to each one on the connection's MCP server, through the connection's own credential, with a callback under `/v1/connectors/mcp-events/{token}` (`receiveConnectionEvent`, `security: []`, not client-accessible) and a Standard Webhooks secret of its own. Each event the server delivers opens a text conversation from the agent config, as the app, with the binding's `instructions` and the event's data as JSON. A delivery not signed with its own subscription's secret is a 401 and opens nothing; a retried one opens nothing more. Deleting the connection stops its subscriptions: a delivery to one is a 410. While the connection waits on a token renewal or a reconnect, a delivery is a 503 and opens nothing, and the subscription stays; a consent that connects it again subscribes again. A session binding that declares `events` is a 400. It needs connectors on and an https `ROUTER_PUBLIC_URL`; without them nothing is subscribed and the route answers 410. The plugin system's `plugin_events` are unchanged. The Go, Python and JavaScript clients are regenerated; other SDKs follow.

### A connection's tool calls and grants are on record, and a user's connections can be deleted

Each connector tool call a session runs leaves one row: the binding, the connection, the tool, the latency and, for a call that failed, an `error_type` of `customer_auth`, `external_server`, `client_timeout`, `outcome_unknown` or `denied`. No row holds what a call was asked or answered, and an incognito session's rows name no session. `GET /v1/agents/connections/{id}/invocations` (`listConnectionInvocations`, server-side only) pages through them, newest first. Each grant a connection gets, renews or loses leaves one audit row (`grant_created` at a consent or a credentials write, `grant_refreshed` when the router renews the credential, `grant_revoked` when the provider refuses or revokes it or the connection is deleted), with the request, session and authorization attempt that caused it; `GET /v1/agents/connector-audit` (`listConnectorAudit`, server-side only) pages through the app's, a deleted connection's included. `Connection` gains `used_by`: the agent config bindings that name it as their fixed connection. `DELETE /v1/agents/users/{user_id}/connections` (`deleteUserConnections`, server-side only) deletes every connection of one user for good, with its pending consents and its tool call log, so the next session for that user attaches none of them. A deployment with connectors off writes none of this. The Go client and the JavaScript types are regenerated; other SDKs follow.

### A connector binding says what its calls do on an interruption, and what the agent says while they run

A connector binding takes `policy`, a `ConnectorBindingPolicy` with three optional fields. `on_interrupt: wait` lets a call finish after the caller interrupts the turn, up to the binding's timeout, and its result goes into the conversation when it comes; until then the call reads as still running, so the next turn does not wait for it; `cancel` cancels it at the provider, as every call was before; left out, the call goes on too, and is stopped only when the caller withdraws what they asked for. `cancellable: false` stops waiting at the interruption but does not send the provider the cancel for it, for a tool that is not safe to stop halfway; its call is logged as `outcome_unknown`. The binding's timeout still ends the call and sends the cancel, whatever the policy. `pre_speech` is what the agent says while one of the binding's tools runs, in place of the hold line it has the model write, and the session socket's `tool_started` carries it as `pre_speech`. A binding without `policy` is stored and read back as before. The Go client and the JavaScript types are regenerated; other SDKs follow.

### A provider app points its Stream app's message hook at the router

With connectors on, `PUT /v1/agents/connectors/{id}/provider-app` (`setConnectorProviderApp`) and its ops twin `setOperatorProviderApp` point the message hook of the Stream app the provider app is pinned to at `ROUTER_PUBLIC_URL/v1/chat/hooks/stream/{stream app id}`, when that app is one the customer registered (T48, AI-887). The hook is matched by its URL, so a PUT again updates it rather than adding a second, and the app's other hooks stay. A provider app pinned to the deployment's own app points nothing: those hooks are the operator's, set with `router phone hooks`. A router without `ROUTER_PUBLIC_URL` points none and logs a warning. When Stream refuses, the provider app is kept and the answer is a 503; a PUT again points the hook. With connectors off both PUTs answer as before and ask nothing of Stream. The Go client and the JavaScript types are regenerated; other SDKs follow.

### Bridge lines keep the sandbox, use-case texts, iMessage keywords and files

For a bridge connection (Telnyx, Linq, WhatsApp), the channel bridge now does what `internal/channels` does on its lines (T62a, AI-921). A sandboxed app (no approved 10DLC use case) reaches only its sandbox recipients, within its daily message limit: the gate is asked before a message reaches the agent and before each reply, and each reply sent counts. STOP, START and HELP are answered with the texts of the use case the line's number is assigned to, else the app's default one, else the texts as before. iMessage through Linq has STOP, START and HELP too, recorded as `imessage` opt-outs, and a reply in a Linq chat is not sent after its sender's STOP. A reply's files (the ones the agent's code made) are sent as links after its text, on every provider, Slack included. With connectors off, nothing changes.

### WhatsApp through the customer's own Meta app

A built-in connector, `whatsapp`, answers people on WhatsApp from the customer's own business number through the channel bridge (T51, AI-879). The app's backend puts its Meta app as a provider app with `PUT /v1/agents/connectors/whatsapp/oauth-client`, `provider_app_id` the Meta app's id and `signing_secret` its App Secret, and no `client_id`; it creates an app-owned connection with the number's `phone_number_id` and the customer's access token as a `bearer` token, and binds one agent config to it. In the Meta app's WhatsApp webhook settings the callback URL is `/v1/connectors/events/whatsapp/{provider_app_id}` and the Verify Token is the same app id: a new `GET` on that route (`answerProviderAppHandshake`) echoes `hub.challenge` as text/plain when `hub.mode` is `subscribe`, the token is the app id and the challenge is digits, and answers 404 otherwise. Only a connector whose manifest declares the new `channel.handshake: hub_challenge` answers it; for every other connector, an unknown app or with connectors off, the GET answers 405 as before. Deliveries are verified with `X-Hub-Signature-256` under the App Secret. Text messages only: each person's number is a thread channel, the agent's reply goes back as a text message from the business number, and a thread opens a `whatsapp` episode card in the omni-channel of the sender's number in E.164. STOP, START and HELP are handled by the bridge before the agent, as for SMS, and recorded as `whatsapp` opt-outs of the number in E.164. Replies after Meta's 24-hour window (approved templates) and Meta's Tech Provider with Embedded Signup are not in this release. With connectors off, nothing changes.

### SMS through the customer's own Telnyx account

A built-in connector, `telnyx`, answers people over SMS on the customer's own Telnyx number through the channel bridge (T53, AI-881). The app's backend puts its Telnyx account as a provider app with `PUT /v1/agents/connectors/telnyx/oauth-client`, `provider_app_id` and the account's base64 Ed25519 public key as `signing_secret`, and no `client_id`; it creates an app-owned connection with its number as `phone_number` (E.164) and the account's API key as a `bearer` token, and binds one agent config to it. Telnyx's webhooks go to `/v1/connectors/events/telnyx/{provider_app_id}` and are verified with a new `ed25519` verifier kind; each person's number is a thread channel, the agent's reply goes to that number from the customer's, and a thread opens an `sms` episode card in the omni-channel of the sender's number. STOP, START and HELP are handled by the bridge before the agent: Telnyx answers its reserved keywords (STOP, START, HELP and their defaults) itself, and the bridge answers only the rest, such as REVOKE and OPT OUT; STOP records an opt-out (`source: keyword`, channel `sms`) and the person's later texts reach no agent, and get no reply, until START. With connectors off, nothing changes.

### iMessage through the customer's own Linq account

A built-in connector, `linq`, answers people over iMessage on the customer's own Linq line through the channel bridge (T36, AI-863). The app's backend puts its Linq account as a provider app with `PUT /v1/agents/connectors/linq/oauth-client`, `provider_app_id` and the webhook subscription's `signing_secret` and no `client_id`, which only a connector not consented through `oauth2_code` takes; it creates an app-owned connection with its line as `phone_number` (E.164) and the line's API key as a `bearer` token, and binds one agent config to it. Linq's events go to `/v1/connectors/events/linq/{provider_app_id}` and are verified as Standard Webhooks; each chat is a thread channel, the agent's reply goes back to the chat with the API key, and a chat opens an `imessage` episode card in the omni-channel of the sender's number. A `bearer` connection whose connector names its account by inputs alone now has that account. With connectors off, nothing changes: a put without `client_id` is still a 400 `validation_failed`. With connectors on, a put without `client_id` that the connector does not take as a provider app (`slack_bot`, `github`) is a 400 `invalid_request` rather than `validation_failed`, and one for an unknown connector is a 404 rather than a 400. The Go client and the JavaScript types are regenerated; Go's `ConnectorOAuthClientRequest.ClientId` is now a `*string`; other SDKs follow.

### Episodes close and their cards hold a summary

A text episode closes once its thread has had no message for `episodes.idle_after` (`ROUTER_EPISODES_IDLE_AFTER`, one hour unless set, refused at 24 hours or more); a call episode closes when the `call.session_ended` hook says its call ended (T55, AI-884). Closing sets the card's status to `ended`; the agent config's own LLM then writes a summary of the episode's lines into the card's text and sets `summarized`, or sets `summary_failed` and leaves the card's text and the thread channel as they were. Every card change is a partial update of the one card message, so nothing new reaches the message hook. Several routers close and summarize each episode once; a summary a stopped router left is taken again by the next sweep after five minutes. The idle sweeper starts only with connectors on or an agent config with `episode_cards` on; with neither, nothing is swept and nothing is written. A call under a config without `episode_cards` ends as before. The summary is not written to memory yet.

### Text and voice sessions start with the person's episode cards

Under an agent config with `episode_cards: true`, a text session on a thread channel and a voice session on a phone call start with the person's other episode cards, the five newest, as context before the conversation (T56 and T42, AI-885). A summarized card gives its summary; any other gives the last 20 lines its channel holds of that episode, so an SMS sent seconds after a call reads the call's last lines while its summary is not ready. The person is found in the contact map of that customer and agent only: by the call's number, or by the episode of the session's own thread, which is left out, since the session reads it word for word. A thread somebody else wrote in reads no cards, and a call card whose window holds another caller's words gives no lines. A call's lines come only from its own channel, `agent:<call id>`; a call whose session named another channel (`agent_id` or `conversation_id`) gives none until it is summarized. A native speech-to-speech session reads none, and a call that started with cards cannot be moved onto a speech-to-speech model (400, `carded_session_to_native`, from `setSessionSettings` and `updateSession`). A device's call under a thread channel's agent id, which writes no transcript there, gets no episode card. The read, the call's caller included, takes at most 5 seconds. The cards come behind a note that they are context, not authority, and take at most 15,000 characters. With `episode_cards` off, every session is handed what it was before and makes no extra read. A persistent voice conversation is still refused.

### Each Slack thread and each phone call gets an episode card in the person's omni-channel

The first message of a Slack thread on the `slack_bot` connector, and each session that joins a phone call under an agent config with the new `episode_cards: true`, writes one card into the person's omni-channel: an `agent` channel `omni-<uuid>` for each person and agent, which names the agent config. The card is one message with `source` (`slack` or `call`), `status: in_progress`, `started_at`, `thread_channel` (the thread channel or the call channel that holds the raw text), `episode_id`, and `call_id` for a call. Later messages of the thread add no card, and the message hook answers no card. A caller is found by number: the call's SIP participant (`sip-<number>`), or the number an outbound call rang, read as E.164 when it starts with `+` or `00` (anything else is no card), so one number is one omni-channel whatever it comes in on. `episode_cards` is off by default, and a call under a config that leaves it off runs as before. A Slack user has an omni-channel of their own, keyed by workspace and user. A call's transcript stays in its call channel. `AgentConfig` gains `episode_cards`; left out on an update, it keeps what is stored. The Go client and the JavaScript types are regenerated; other SDKs follow.

### An app can delete its own connector, and a connector shows its redirect URI

`DELETE /v1/agents/connectors/{id}` (`deleteConnector`, server-side only) deletes one of the app's own custom connectors, every revision, with the app's OAuth client for it. A built-in, or an id the app has no connector under, is a 404. A connector a live connection was made from, or an agent config binds, is refused with a 409 that names them, unless `force=true`: then its connections are deleted as a forced connection delete deletes one, and the bindings are left in place (AI-1046). `Connector` now has a read-only `redirect_uri` on a connector that connects with `oauth2_code`: the URI to register for the app's own OAuth client, the one every consent sends, absent when `ROUTER_PUBLIC_URL` is not set (AI-1047). A custom connector's `endpoint` stays hidden, since an MCP URL can carry a secret in its path (AI-837).

### An app can put its own OAuth client for a connector

`PUT /v1/agents/connectors/{id}/oauth-client` (`setConnectorOAuthClient`, server-side only) stores the OAuth client the app registered with the connector's provider: `client_id`, a write-only `client_secret` and an optional `auth_method` (`none`, `client_secret_basic` or `client_secret_post`). It answers 201 when it stores one and 200 when it replaces one, and never returns the secret. Every consent and every refresh of the app's connections to that connector reads it again, so a rotated secret is put once and used from each connection's next refresh. A connector whose `client.registration` does not list `customer`, such as `slack`, answers 400. `DELETE` on the same path removes it. The operator's own client still comes from `<client.env>_MCP_CLIENT_ID` and `_MCP_CLIENT_SECRET`. The Go client and the JavaScript types are regenerated; other SDKs follow.

### A connection is connected through the provider's consent

`POST /v1/agents/connections/{id}/authorizations` (`createAuthorization`, server-side only) starts the OAuth consent for a connection and answers a `launch_url` and a `handoff_token`. The dashboard opens the launch page in a popup and posts it the token; the browser goes to the provider and comes back to `/v1/agents/connectors/oauth/callback`, which stores the grant and sends the browser to the dashboard with `connection_id` and `status`. The consent must finish in the browser that started it, within 10 minutes, once, and its handoff token works once. A reconnect that comes back with another provider account keeps the old grant and ends as `account_mismatch`. The router needs `ROUTER_PUBLIC_URL` for it, and with an https one it serves its OAuth Client ID Metadata Document at `/.well-known/oauth-client-metadata`. The Go client and the JavaScript types are regenerated; other SDKs follow.

### An MCP server named by URL can log in, for the app or for each user

The server decides whether it needs a login: saving a config asks each of its `mcp_servers`
(its protected-resource metadata, or a 401 that says how to authenticate) and stores the
answer as the read-only `needs_login` on `McpServer`; a session asks again for one that could
not be reached. A server that needs a login is the app's to connect once: it is listed by
`GET /v1/agents/configs/{id}/plugins` and connected with the same
`.../plugins/{name}/authorize` as a catalog plugin, and until then its tools fail with
"connect <name> on the dashboard". With `user: true`, each end user logs in in the
conversation instead, the first time the agent needs the server. `scopes` is optional and,
left out, asks for the `scopes_supported` the server advertises. The login follows the MCP
authorization spec: the authorization server's metadata, a client registered on the fly,
PKCE and `resource`. A login is only good at the URL it was made at, so changing a server's
URL needs a new one, and a login made before its scopes change keeps what it was granted.
Saving a server whose login the router cannot make, or that sets `scopes` or `user` but
needs no login, is a 400. `McpServer` gains `scopes` and `user` on `AgentConfig`,
`AgentConfigRequest`, `AgentConfigPatch` and `SyncAgentRequest`, and the Go folder reader
reads them (`agents.MCPServerSettings.Scopes` and `.User`).

```yaml
mcp_servers:
  - name: crm                  # needs a login: the app connects it on the dashboard
    url: https://crm.example.com/mcp
  - name: notes
    url: https://notes.example.com/mcp
    user: true
```

### An MCP server says what it is

Saving a config asks each of its `mcp_servers` for the `serverInfo` it answers `initialize`
with, and keeps it as a read-only `branding` on the server: `title` (its name when it gives
none), `description`, `version`, `icon_url` (its first https icon) and `website_url`. Any
of them may be missing, and many servers send only a name and version. A server that does
not answer within 5 seconds keeps what it said before, and the config is saved either way.
The request goes through the router's egress client, so it only reaches public hosts.

### A plugin can be reached read-only, limited to some tools, and its scopes chosen

`plugin_options` says how an agent reaches a catalog plugin and what its login asks for.
`readonly: true` reaches the vendor's read-only MCP server, which for Linear is
`https://mcp.linear.app/mcp/readonly` asking only for `read`; `scopes` replaces the scopes
asked for at consent; `toolsets` limits the server to some groups of tools, which for
Cal.com go on its URL as `?toolsets=`. Left out, the plugin is the catalog's. It is on
`AgentConfig`, `AgentConfigRequest`, `AgentConfigPatch` and `SyncAgentRequest`, and the Go
and Python folder readers read it. `readonly` on a plugin without a read-only server, or a
toolset its catalog entry does not list, is a 400. `Plugin` in the catalog says which a
plugin accepts, as `readonly`, `toolsets` and `scopes_supported`; a scope outside
`scopes_supported` is a 400. Google Drive takes `drive.file` beside its default
`drive.readonly`, for `create_file` and `copy_file`.

`tools`, on `plugin_options` and on `mcp_servers`, offers the model only the tools named or
matching a pattern such as `get_*`; the router refuses to run any other. Left out, every
tool is offered.

```yaml
user_plugins: [linear, calcom]
plugin_options:
  - plugin: linear
    readonly: true
    tools: [list_issues, get_issue]
  - plugin: calcom
    toolsets: [bookings, availability]
```

### AssemblyAI's Universal-3.6 Pro Realtime as a transcription model

The router now streams to `assemblyai/universal-3-6-pro`, reachable by name or by any
shortcut whose terms it satisfies. It needs `ASSEMBLYAI_API_KEY`, covers 32 languages with
code-switching, and declares `keyterms` and `endpointing`; `silence_ms` is its
`max_turn_silence`. A call runs on the `max_accuracy` preset, because the server's own
`balanced` one ends a turn at a short pause mid-sentence. `overwrites` takes `mode`
(`min_latency`, `balanced`, `max_accuracy`), `min_turn_silence` and `max_turn_silence`.

### OpenAI's GPT-6.1 Sol

The router now reaches `openai/gpt-6.1-sol`, and `llm-thinking` prefers it in place of
GPT-6 Sol, which stays declared for configs that name it. Like Astra, GPT-6.1 Sol rejects a
reasoning effort of `none`, so a request naming no effort is sent `low` and one naming
`none` is refused.

### Linear, GitHub, HubSpot, Google Drive and Google Docs plugins, and a logo on every login

The plugin catalog is twelve entries: `slack`, `calendly`, `calcom`, `shopify`, `salesforce`,
`sentry`, `linear`, `github`, `hubspot`, `google_calendar`, `google_drive` and `google_docs`.
Linear registers its own client; GitHub, HubSpot, Slack, Salesforce and the three Google
plugins each need a `<ID>_MCP_CLIENT_ID` and `<ID>_MCP_CLIENT_SECRET` on the router. Drive
and Docs are separate servers with separate scopes, so they are separate plugins. There is no
`teams` plugin: as of October 2026 a Teams app hosts its own MCP server at its own URL, and
Microsoft publishes nothing hosted to point at.

Every plugin now has a logo, served as an SVG needing no credential from the new
`GET /v1/agents/plugins/{plugin_id}/logo`, and both `Plugin` and `PluginConnection` carry a
`logo_url` for a dashboard to draw a card with. The logos are our own plain marks rather than
the vendors' artwork, so a deployment that has licensed the real thing replaces a file.

The `plugin_authorization` attachment has three more fields, so a Chat client with no
renderer for the type still shows a card with a logo and a working link: `text` (the
catalog description), `thumb_url` (that logo URL) and `title_link` (the authorize URL). Each
is derived and checked against the catalog, so an MCP server cannot put an arbitrary image
into somebody's conversation. In Python, `RemoteEvent.image_url` carries the logo on an
`authorization_required` event.

### A conversation can move to Slack, Teams or RCS

`omni` has a `teams` provider, reading Bot Framework `message` activities and answering with
markdown and hero cards, so Teams joins Slack, WhatsApp, RCS, SMS and iMessage as a channel
a conversation can be carried over to. `examples/text_agents/mcp_plugins` now runs webhooks
for Slack, Teams and Google RBM beside the ones it had, each with a login button a plugin's
authorization attachment is drawn into: Block Kit for Slack, a hero card for Teams and an
`openUrlAction` suggestion for RCS. Note that a plugin and a channel are different things:
`slack` under `user_plugins` is an account the agent reads, and the Slack channel is a person
talking to the agent in Slack.

### Sentry and Google Calendar plugins, and plugins each user connects in the chat

The plugin catalog has `sentry` and `google_calendar`. `agent.yaml` names `user_plugins`
beside `plugins`: a plugin under `plugins` is connected once by the app, and one under
`user_plugins` by each end user with their own account. The agent gets `<id>__list_tools`
and `<id>__call_tool` for those, and the first call for somebody who has not connected asks
them to, as a `plugin_authorization` attachment on the reply (an `authorization_required`
event in Python). `GET /v1/agents/configs/{id}/plugins` now lists a plugin the config names
that the app has not connected as `not_connected`, for the dashboard to remind about. Google
Calendar needs `GOOGLE_CALENDAR_MCP_CLIENT_ID` and `GOOGLE_CALENDAR_MCP_CLIENT_SECRET` on the
router. See `examples/text_agents/on_call`.

### The sandbox can be built, run for minutes, and hand files back to chat

`sandbox_options` on an agent config, or in `agent.yaml`, says how the subagent's Daytona
sandbox is built and how long code may run in it (#737):

```yaml
sandbox: daytona
sandbox_options:
  setup: [pip install --no-cache-dir bpy==5.2.2]
  timeout: 5m      # timeout_ms on the API, at most 30 minutes
  cpu: 2
  memory_gb: 4
```

`image` is the base to build on (a slim Python 3.13 when left out). Daytona keeps the built
image, so only the first sandbox from a setup waits for it.

`run_code` takes `files`, paths the program wrote. On a persistent text conversation each one
is uploaded to the channel and attached to the reply that settles the work (an image inline,
anything else as a file), so it is still there when the conversation is reopened, and history
returns it as the message's `files`. `task_settled` lists them as `files`, which the Go SDK
reads into `Event.Files` and Python into `RemoteEvent.files`. `examples/text_agents/blender_artist`
uses this to render with Blender.

### A reply starts speaking sooner

The first chunk of a reply now goes to the voice at its first clause, once it has 20
characters, rather than waiting for the first full stop; the rest still goes by the
sentence. A transcript the speech-to-text provider has finalized is put to the flow
controller after 60 ms instead of the 350 ms a revision waits, except when it ends in digits
that may still be growing.

### A reply can start before the flow controller rules: `ROUTER_SPECULATIVE_REPLIES`

The flow controller decides whether the words a caller settled on were meant for the agent,
and the reply used to wait for that ruling, so every answered turn paid for two model round
trips one after the other. With `ROUTER_SPECULATIVE_REPLIES=true` the reply is asked for
beside the ruling and held until it comes back: an answer for the same words speaks it, and
anything else drops it unheard. It is off by default, because a dropped reply is still paid
for, and on a pause-heavy call most of them are dropped.

### A turn says when its reply could first be heard

`roundtrip_ms` and `speech_end_to_audio_ms` end when publishing the first chunk of a reply
returns. Publishing waits until no more than 400 ms of the speech is left in the queue to the
outgoing track, so for a first chunk longer than that the return comes later than the reply
began to be heard, by the part that did not fit. A voice call's turn now also carries
`first_frame_queued_ms`, when the first frame of the reply was queued for the outgoing track,
`first_audible_frame_ms`, when the track took the first frame that was not silence, and
`speech_end_to_audible_ms`, which is `speech_end_to_audio_ms` measured to that moment. They
are on the `turn` event of the session socket, in the turn log, in the `turns` table and in
`GET /v1/agents/calls/{id}/timeline`, and absent where the edge does not report them. The
older fields are unchanged.

### A voice reply can show how fast its turn was: `ROUTER_CHAT_TIMINGS` and `-chat-timings`

For development, a voice agent can write how long a turn took after its reply in the chat
channel it writes its transcript to, so somebody talking to it in a call UI sees it without
reading logs. It is off by default: `ROUTER_CHAT_TIMINGS=true` (`agent.chat_timings`) turns it on
for the router's sessions and `cmd/agent -chat-timings` for the standalone agent. A reply gets
one line after its text, such as
`⏱ reply 1112 ms = eou 452 + llm 412 + tts 248 · ttft 731 · ttfb 120`: the wait from the end of
the caller's speech to the first audible frame of the reply, split into end of utterance, model
and voice stages that add up to it, then the providers' own first-token and first-byte waits. A
figure that did not happen is left out, and a turn the caller talked over starts
`⏱ interrupted ·`. The message also carries the same figures in whole milliseconds as a
`timings` custom field. Reading a conversation back, as the transcript endpoint and the history a
bound conversation gives the agent do, leaves the line out, so the agent never takes it for
something it said.

### A session says whether the user wrote, spoke or showed video

Every session now has a `modality`: `text` for a conversation held in writing, `voice` for a
call, and `video` once the agent has seen the user's video, either as frames from the SDK's
`get_video_frames` tool or as clips sent with `responses.create`. It only moves up, so a
call that showed video once stays `video`. `querySessions` takes it as a filter,
`{"modality": "video"}`, and in Go it is `Query.Modality`. Sessions recorded before this
are `text` without a call and `voice` with one.

### Choose a session's id when creating it

`POST /v1/agents/sessions` takes an optional `id`, a UUID the caller chose, so the session can
be referred to before it exists. An id some session already has is refused with a 409. Without
one the router generates a UUIDv7 rather than a random hex string. In Go it is
`SessionOptions.ID`, and in JavaScript `sessions.create({ id })`. Other SDKs follow.

### Change a running session's models from the server SDKs

`PATCH /v1/agents/sessions/{id}/settings` is wrapped on the session object in every server
SDK: `updateSettings` (JavaScript), `update_settings` (Python, Ruby, Rust), `updateSettings`
(PHP), `UpdateSettings` (Go) and `UpdateSettingsAsync` (C#). Each returns the session as it now
runs. It stays backend-only, so the Kotlin, Swift and Dart SDKs do not offer it.

### `responses.create` takes a video

`POST /v1/agents/sessions/{id}/responses` takes `videos`: up to two clips, each an `http(s)`
URL or a `data:video/...` URI of at most 50 MB. The router samples `max_frames` frames
spread evenly through each clip (8 by default, up to 32) with ffmpeg, which the router
image now ships, and hands them to the vision subagent like attached images, each captioned
with where in the clip it was taken. URLs are fetched only from public addresses. The Go
SDK sends them with `client.Clip{URL, MaxFrames}` or `client.ClipFile(path)`.

### The JavaScript SDK's session can use a chat client you already hold

`session.chat({ client })` opens the conversation's channel on a connected `StreamChat` of
yours instead of connecting a second one, and `interrupt({ commandId })` names the turn it
stops.

### A dispatch worker can run tools for every session under an agent id

A worker on `/v1/dispatch` can send `host_tools` naming an `agent_id` and the tools it runs
for it. Every session opened under that agent id, whoever opened it, is then offered those
tools, and each call reaches the worker as a `tool_call` frame and is answered with
`tool_result`, within the worker's own `timeout_ms` (two minutes by default). This is how a
session a browser opens gets a tool only a backend can run, such as reading source on a VM:
the agent id is what a plain session already names, so nothing has to be stored first. The
agent id is scoped to the worker's own customer, a tool the session's own caller declares
under the same name wins, hosting no tools or naming no agent is refused with
`hosting_refused`, and a call waiting on a worker that disconnects fails at once. The Go SDK
exposes it as `Dispatch.Host(agentID, functions, timeout)`.

A Go `Dispatch` whose socket drops -- a router redeployed, a load balancer ending the
connection -- now reconnects with a fresh token, backing off from one second to thirty, and
declares what it hosts again. `Run` still returns on cancellation, on a deliberate close,
on refused tools, and when the first connection fails.

`agent.yaml` may carry an `app:` mapping, the application's own settings. Both SDKs leave
it unread and never send it, and it is the one top-level key they do not refuse.

### The router is configured by a YAML file, and can hand a customer to another deployment

`router --config /etc/router.yaml` (or `ROUTER_CONFIG_FILE`) is now where a deployment's
settings live: `postgres.dsn`, `redis.addr`, `auth.mode`, `cors_origins` and the rest.
Naming no file loads one of `local`, `testing` or `staging` embedded in the binary, by
`ROUTER_ENV`, which replaces `internal/environment` and renames `development` to `local`.
Every `ROUTER_` variable still wins over the file, so nothing in an existing chart, compose
file or `.env` has to change.

`router keys create` mints the first credential of an `api_key` deployment, which has no
way to issue one over HTTP, and prints the secret once. `--app-id` reuses the app id a
customer already has.

Three new server-side endpoints move a customer between two deployments:
`GET /v1/data/export` streams everything the calling app has and ends with a cursor,
`POST /v1/data/import` writes it back, and `GET /v1/data/changes` replays what has happened
since that cursor. `router replicate --from <url>` does the copy and then follows the
source, so pointing the SDKs at the new deployment loses no writes. The customer is always
the authenticated caller, key secrets and OAuth tokens are never exported, and all three
are refused outright in `noauth` mode, where the tenant is only a header.

### Budgets, data policies and prompt injection screening per organization and app

`/v1/policies/app` and `/v1/policies/organization` set a spend cap reset hourly, daily,
weekly or monthly, a training and retention floor applied to every routed request, and
prompt injection screening. Screening runs the lcm router (Jev by default) beside each LLM
call rather than in front of it, so it adds nothing to time to first token; a response whose
input reads as an injection fails with `prompt_injection`. An organization's settings are a
floor its apps can tighten but not loosen.

### Meta's Muse Spark 1.3 in the built-in LLM config

`meta/muse-spark-1.3` is now in the default `router.yaml`, in the high-quality tier with
image input, on Meta's Standard tier that does not train on requests. It needs
`META_API_KEY`, and no shortcut prefers it: it is reached by name or as failover. The
`meta` provider now accepts the `max` reasoning effort, above `xhigh`.

### An agent can be given a guardrail: `guardrail.md`

A `guardrail.md` beside `instructions.md` says what the agent may be asked about. Every
turn is screened against it, and a turn it refuses is answered with the policy's own
refusal line instead of the model's reply.

```markdown
---
type: lcm                 # lcm | webhook | llm
mode: parallel            # parallel | blocking
threshold: 0.6            # lcm and llm: refuse at or above
refusal: I can only help with questions about Stream.
---
Only answer questions about Stream's SDKs, products and development. Refuse anything
with nothing to do with Stream or with building software that uses it.
```

Frontmatter says how to check; the body is the policy in prose, so the policy stays
something a human reads and edits. There are three ways to check:

| `type` | Who decides |
| --- | --- |
| `lcm` | The new `lcm` router, which returns a calibrated probability. The default |
| `webhook` | Your own server, at `url`, over a signed POST |
| `llm` | A model from the `llm` router, asked to read the policy and put a number on the turn |

`mode` is the trade between latency and tokens. In `parallel`, the default, the check runs
beside the model and the finished reply is held until the verdict arrives, which on a call
costs a caller nothing they can hear: a classifier answers in about 200ms while the model
is still writing. In `blocking` the model is not asked until the check passes, so a refused
turn spends no LLM tokens but every caller waits for the verdict first.

A check that errors or times out **allows** the turn. A classifier outage should not
silence the agent.

A refused turn emits the ordinary `responding` and `responded` events, so a client that has
never heard of a guardrail shows it as a normal exchange, plus a new `blocked` event
carrying the reason, the probability, and how long the reply was held. It is recorded in
the agent log the same way.

A webhook guardrail is posted `X-Timestamp` and `X-Signature`, an HMAC-SHA256 over
`<timestamp>.<body>` keyed with your app secret — the same secret that already verifies
Stream's inbound hooks, so there is no second secret to store. `agents.VerifyGuardrailWebhook`
in the Go SDK is the counterpart:

```go
if !agents.VerifyGuardrailWebhook(secret, r.Header.Get("X-Timestamp"), r.Header.Get("X-Signature"), body) {
    http.Error(w, "bad signature", http.StatusUnauthorized)
    return
}
```

The timestamp is inside the signed string, so a captured request cannot be replayed;
anything more than five minutes out is rejected.

A speech-to-speech agent cannot be given a guardrail. A native model hears the caller and
answers directly, so nothing sees the words before they are spoken and there is no reply to
hold. Declaring one refuses the session rather than leaving it unguarded.

### A new routed modality: `lcm`

Typed judgements about a piece of text — the probability that a condition holds, one option
out of a named set, a position on a described scale — routed and costed like every other
modality, and visible at `/v1/lcm/providers` and in stats. `lcm` is a large classifier
model.

It is not a mode of `llm`. There is no stream, no generated text and no token budget: a
caller asks named questions and gets values with the probabilities behind them. TypeSafe's
Jev is the first provider; `classify-fast` is the shortcut a guardrail takes when nobody
said which model should judge.

### The Go SDK can wait to be written to: `agents.Dispatch`

The Go counterpart of `plugins/stream`'s `Dispatch`. A worker connects out to the router
and waits, and a message somebody wrote in an agent channel is pushed down that connection,
so the agent and the functions the model calls run in your process without anything being
publicly reachable.

```go
dispatch, _ := agents.NewDispatch(agents.DispatchOptions{Capacity: 8})

dispatch.OnMessage(func(ctx context.Context, message agents.InboundMessage) error {
    conversation, err := dispatch.Conversation(ctx, message, build)
    if err != nil {
        return err
    }
    return conversation.Respond(message.Text)
})

dispatch.Run(ctx)
```

`Conversation` is the agent answering that channel, started if none is and kept for the
next message. Questions on one channel are answered one at a time, because `Respond`
interrupts and two messages written in quick succession would otherwise throw the first
answer away half-written. `OnCall` is the same for calls arriving over SIP.

Two options suit a worker whose agent does more than answer from the model. `TurnTimeout`
is how long one answer is given before the conversation abandons it and takes the next
question; it defaults to the five minutes it was fixed at, which is short for an agent
whose tools read a source tree or wait on a sandbox. `OnEvent` is told everything the
backend says about every conversation the worker holds, which is the only way to see any
of it: the conversation reads its own session, and a second reader would take events from
it, so a worker had no way to record tool timings or model failures.

```go
agents.NewDispatch(agents.DispatchOptions{
    Capacity:    8,
    TurnTimeout: 15 * time.Minute,
    OnEvent:     func(channelID string, event stream.Event) { metrics.Record(channelID, event) },
})
```

`stream.Dispatch` is the socket on its own, for a worker holding something other than an
agent. `sdks/go/examples/dispatch` is a whole one.

### A dispatched message carries what its channel was created with

`message.new` on an agent channel is handed to a worker with the channel's custom data on
it, in `InboundMessage.Custom` in Go and `InboundMessage.custom` in Python, alongside the
`agent_config_id` the router already read. It is carried through unread.

It is how a worker learns what a conversation is for when the router has no opinion about
it. An agent scoping memory to an organization, or answering in a locale, had nowhere to
read either from: a dispatched message named the channel, the config and who wrote it, and
nothing about what the conversation was.

```go
agent, err := agents.New(agents.Options{
    MemoryFilter: organizationMemory(message.Custom["organization_id"]),
    CostTracking: map[string]string{"organization_id": message.Custom["organization_id"]},
})
```

Whoever creates a channel decides what is on it, so this is a claim rather than a fact, the
way a call's custom data is. Which agent answers is still not taken from it. Values that
are not strings are dropped rather than rendered into one, so a number does not reach a
worker as `"1.2e+01"`; Python narrowed the same field on a call by stringifying it, and now
drops it too.

### `ChatOptions.AgentID` names the conversation a Go chat session answers

A session opened with `agent.Chat` joined under the agent's own user id, which is what the
router matches a message against when somebody writes to a channel. A process holding more
than one written conversation therefore had them all under one name: the router could not
find the session for a channel, and every message on it started another agent.

```go
session, err := agent.Chat(ctx, agents.ChatOptions{Persist: true, AgentID: message.AgentID})
```

Leaving it empty is unchanged.

### An agent channel can say which agent answers in it

A message written in an agent channel is handed to a worker, and the router works out whose
channel it is from the conversation last held there. A channel that has never held one had
nothing to say who should answer, so a conversation that starts in writing — somebody
opening a support chat rather than ringing a number — could not be dispatched at all.

Such a channel now names the agent config answering in it, in its own custom data:

```json
{"agent_config_id": "config-1f3a…"}
```

The customer is not read from the channel. Whoever creates one decides what is on it, so a
customer id there would be a claim rather than a fact; the store says who owns that config,
and a channel naming one nobody holds is answered by nobody.

### A session belongs to whoever opened it, anonymous, guest or signed in

Sessions now record their owner — the customer, the user id and which sort of caller
claimed it — and every later request for one is matched against that. One person cannot
read, close or write into another's conversation, on the events socket or anywhere else.

Four kinds of caller, decided by the credential rather than by the request: `server`
reaches every session of its customer, `authenticated` and `guest` reach their own user's,
and `anonymous` reaches what it opened under the name it claimed. The kind travels with the
user id because an unverified name is not the verified one: an anonymous caller naming
`alice` never matches the `alice` a token proved, so claiming a name buys nothing. An
anonymous caller naming nobody at all holds its session by id and is listed nothing.

No configuration is needed. On a `noauth` deployment behind a proxy, the proxy's
`Stream-Auth-Type` and `X-Stream-User-Id` decide; in `api_key` mode the token's `user_id`
and `role` claims do.

### Two base provider groups for transcription: `base/stt-realtime-fast` and `base/stt-realtime-accurate`

The models we would pick for a live call, named rather than copied:

```yaml
stt:
  providers:
    - base/stt-realtime-accurate
```

The fast group is Flux, Ink 2, Inworld and Nemotron with `deepgram/flux-general-en` pinned at
the front; the accurate one is Muse, Scribe v2 Realtime, Ink 2 and Nemotron, pinned to Muse.
Both are capability shortcuts, so they go anywhere a `provider/model` goes, in a priority list
or as a target, and the rest of each group stays behind the pin as failover.

`base/` marks them as ours: a customer's own stored config may well be called
`stt-realtime-fast`, and naming `base/stt-realtime-fast` means this group rather than that
config. A group carries its models and nothing else — a config that wants diarization still
asks for it, and the accurate group then narrows to Muse.

The point of them is that the list is reviewed here instead of in every config that wanted
today's answer. The cost is that it is an opinion with a date on it rather than something that
follows from what the models declare, which is why the date is in the config beside them.

### ElevenLabs v4 and v4 Turbo voices

The router now speaks with `elevenlabs/eleven_v4`, first on the Artificial Analysis voice
arena, and `elevenlabs/eleven_v4_turbo`, its real-time variant, in the high-quality and
low-latency tiers. Both act audio tags such as `[laughs]` and are served on the
text-to-dialogue socket like `eleven_v3_conversational`, with the same `ELEVENLABS_API_KEY`.
The `tts-quality` group now prefers `eleven_v4` over Sonic 3.6.

### Three more realtime transcription models: Ink 2, Inworld STT 1 and Scribe v2 Realtime

The router now streams to Cartesia's `cartesia/ink-2`, Inworld's `inworld/inworld-stt-1`
and ElevenLabs' `elevenlabs/scribe_v2_realtime`, reachable by name or by any shortcut whose
terms they satisfy. Inworld needs `INWORLD_API_KEY`; Cartesia and ElevenLabs reuse the keys
their voices already use.

Ink 2 and Flux are the two models that detect turns themselves, and their thresholds are
each vendor's own rather than a shared term, so Ink 2's live in `overwrites` the way Flux's
do (`turn_end_threshold`, `turn_end_timeout_ms`). Inworld and Scribe take
`silence_ms` as endpointing and expose the rest of their detectors the same way.

Scribe is the case `supports:` exists for: ElevenLabs say plainly that the realtime model
does not diarize and that batch Scribe is where that lives, so it declares `keyterms` and
`endpointing` and not `diarize`, and a request that asks to be told who spoke routes past
it rather than being served something that cannot answer.

### OpenAI's GPT-6: Astra, Sol and Luna

The router now reaches `openai/gpt-6-astra`, `openai/gpt-6-sol` and `openai/gpt-6-luna`.
`llm-thinking` prefers GPT-6 Sol and `vlm` prefers GPT-6 Luna in place of their GPT-5.6
namesakes, which stay declared for configs that name them. Astra rejects a reasoning effort
of `none`, so a request naming no effort is sent `low` and one naming `none` is refused.

### A live voice from Google: Gemini 3.8 Flash TTS

The router now speaks through `gemini/gemini-3.8-flash-tts`, second on the Artificial
Analysis voice arena and first on pronunciation robustness. It reuses `GOOGLE_API_KEY`, the
key the Gemini language and speech-to-speech models already read. It takes a whole sentence
per request and streams the speech back, about a second to first audio, so it sits in the
high-quality tier and is reachable through the high-accuracy shortcuts. `voice` is a
prebuilt name such as `Kore` or a designed `voice_` id.

### A faster Google voice: Gemini 3.8 Flash-Lite TTS

The router now speaks through `gemini/gemini-3.8-flash-lite-tts`, first on Voice Arena in
US English and a third cheaper than Flash per hour of speech. It takes the same key and
voices as Flash and is still a whole sentence per request, but it reaches first audio in
about half a second, so it sits in the low-latency tier and is reachable through `tts-fast`
and the low-latency shortcuts. It covers 101 languages to Flash's 130.

### Google's flagship reasoning model: Gemini 4 Argon

The router now answers with `gemini/gemini-4-argon`, 53 on the Artificial Analysis
Intelligence Index. It reads images and thinks at `high` unless a request names `low` or
`medium`; `minimal` is refused. It sits in the high-quality tier, so `llm-smart` and
`llm-thinking` fall over to it, and nothing prefers it. It reuses `GOOGLE_API_KEY`.

### A recorded call, for a Go program with nobody at a microphone

`stream.RecordedCall` reads a 16 kHz mono PCM16 WAV and hands it over the way a call
delivers audio — 100 ms at a time paced to real time, then the two seconds of silence that
tell a streaming model the turn is over:

```go
call, err := stream.RecordedCall("saturday_seven_thirty.wav")
for chunk := range call {
	err := transcriber.Send(chunk)
}
```

It is the counterpart of `recorded_call` in the Python SDK, and it is what lets an example
be about routing rather than about parsing RIFF chunks.

### A transcript says which model heard it

`stream.Transcript` in the Go SDK now carries `Provider` and `Model`, and the router sends
`speaker` on every transcript frame rather than only in the batch transcript. A config names
several models and routing picks between them per session, so which one answered is a fact
about the run that the SDK no longer makes callers dig out of the raw frame.

### A Go example for realtime transcription, in `examples/routers/stt_realtime_example`

The same recorded call streamed through four stored configs:

```bash
go run . -use-case stt-realtime-fast       # or -accurate, -fast-private, -accurate-private
```

`stt-realtime-fast` asks not to be told who spoke and keeps Flux, Ink 2, Inworld and
Nemotron as a four-deep fallback chain. `stt-realtime-accurate` asks to be told, which
narrows the same shape of chain to Muse — the one of its four that declares `diarize` — with
nothing behind it, and carries an `overwrites` block per vendor, since where a turn ended is
the question all four answer in words the others would not understand.

The two `-private` configs add `data_policy: {allow_training: false, retention: none}`, which
is where that requirement gets expensive: of every live transcriber the router knows, three
answer both questions, so none of the six models the first two configs name survives.
`stt-realtime-accurate-private` is the one to read, because being told who spoke and keeping
nothing cannot both be had on a live call today, and it says which it gives up and why.

The example passes no options at the call site, so what answers comes from the config alone,
and prints the provider and model that did.

### Muse and Nemotron read their `overwrites` block

Both had a setting the shared vocabulary has no word for and no way to be told it, so a
block addressed to either was accepted and dropped. Muse now takes `mode`, which is the only
vocabulary it has for a turn boundary and the sole way to ask it for `PUSH_TO_TALK`, and
Nemotron takes `turn_grace_ms` — the router's own wait for the transcript to stop changing,
since nothing on that protocol says where a turn ended. A field neither has is still an
error rather than a setting that goes nowhere.

### An agent directory declares the pages it reads, in `knowledge/urls.yaml`

An agent folder can name the pages its knowledge base is kept filled from, alongside the
files it is filled from directly. An entry is a url, or a url with what to call the page
and what it is:

```yaml
# jean/knowledge/urls.yaml
- https://example.com/pricing
- url: https://example.com/plans
  title: Plans
  description: What each plan includes and where the limits are.
```

`agents.Load` reads it into `Folder.KnowledgeURLs` and leaves it out of the documents it
ingests, and `Sync` subscribes the pages to the same namespace the directory's files go to,
so one lookup covers both. `agents.SubscribeKnowledgeURLs` does it on its own for an
application that fills a base under a namespace of its own choosing.

`POST /v1/agents/knowledge/urls` takes an optional `title` and `description`, kept as the
caller wrote them: a declared title is what the page is reported as, and what it calls
itself is the fallback. Adding a page a knowledge base already has now re-reads it rather
than being refused, so a declaration can be applied again without first working out which
pages are new. In Python, `agent.knowledge.add_url(url, title=..., description=...)`.

### A terminal UI for a conversation, in `tui/`

`github.com/GetStream/Vision-Agents/tui` holds a conversation with an agent in a
terminal: the answer as it is written, the tools being run to find it with their queue
and execution timings, and whether what was said survived being saved. It is the
terminal counterpart of the dashboard, and works against anything satisfying
`tui.Session`, which an `*agents.Session` already does.

```go
err := tui.Run(ctx, tui.Options{
	Open: func(ctx context.Context, conversationID string) (tui.Session, error) {
		return agent.Chat(ctx, agents.ChatOptions{Persist: true, ConversationID: conversationID})
	},
	History:  tui.BackendHistory(stream.Backend{}, "jean"),
	Branding: tui.Branding{Title: "Jean", Subtitle: "answers questions about the weather"},
})
```

`Enter` sends, `alt-enter` writes another line, `esc` cancels the answer being worked on,
and `/new`, `/resume <id>` and `/older` move between conversations and page back through
saved history. An application adds header lines, a theme, a banner and slash commands of
its own; `OnOpen` is told about each session as it becomes live, which is the moment to
open the same conversation in the dashboard. `go run ./cmd/chat -name jean` is a runnable
conversation with a `get_weather` function to watch being called.

It is a module of its own, so the Bubble Tea and Glamour dependencies stay out of
`sdks/go`.

### Accelerate LLM request fallback and Claude support

The Go backend can route to Anthropic's OpenAI-compatible Claude endpoint using
`ANTHROPIC_API_KEY`. Alias sessions try another eligible model when a request fails
before streaming, retaining its conversation input, tools and cost tags. Cancellation,
HTTP 400 rejections, partial output and provider-held continuation IDs are not replayed.
Text sessions no longer acquire an implicit subagent when none was requested.

### Claude Opus 5.5 in the built-in LLM config

`anthropic/claude-opus-5-5` is now in the default `router.yaml` (#661), in the high-quality
tier with image input, so the Anthropic provider is reachable without a config of your own.
It is a failover candidate on the quality shortcuts and is not preferred by any of them. The
provider now accepts images, and still refuses a `reasoning_effort` because Anthropic's
OpenAI-compatible endpoint ignores it.

### Organization-scoped support memory

Support conversations use the existing session memory identity to recall facts across
chats for the same organization. Persisted channels retain their original memory
scope and reject resuming under a different identity. Sessions without an explicit
memory user no longer fall back to shared customer memories; requesting memory when
the backend provider is unavailable returns a clear error.
Mem0 startup recall now supplies a general context query when the conversation has
no question yet, avoiding the provider's rejection of blank search queries.

### Shared SDK repositories in managed research workspaces

Managed sandbox repositories can declare additional product/SDK `scopes`, so one
clone serves multiple SDKs without mixing their routing. Preparation clones up to
three repositories concurrently. The bounded source index reserves space for every
repository and supports Go, Python, Kotlin, Java, Dart, C#, C/C++, Objective-C, PHP,
and Ruby alongside Swift and JavaScript/TypeScript. Source permissions and exact
commit-linked quote verification apply to all supported languages.

### Conversation activity in the call dashboard

Call pages now include Stream Chat. Persistent support conversations display thinking,
queue and execution timers, tool summaries, Markdown answers, and save status. The
browser can submit questions and cancel responses while the TUI handles local tools;
saved conversation links remain readable after the session ends.

- Persistent Stream Chat text conversations with resumable history, thinking/tool activity, immutable tool timestamps, and a durable local write retry queue. The Go SDK exposes conversation binding and history.
- Synchronous short STT/TTS requests work without a database; text agents can delegate product skills and cancel active tools.

- Backend-managed Daytona research profiles keep a source workspace and Cursor process warm. Agent configs and session requests accept `sandbox_profile`; Go agents select it with `ManagedSandbox`, and receive `research_progress` events.
- Daytona Python sandboxes use the official Go SDK, serialize concurrent creation and retain identity after failed deletion.

### Delegated visual understanding

- Added `vlm` for image-capable routing; `llm-fast` keeps its existing policy.
- LLM messages and tool results accept ordered `text` and `image_url` parts, including bytes and HTTP URLs.
- Direct VLM conversations retain image attachments on their original turns for follow-up questions.
- Skills with `capture_video` hand camera evidence to the subagent, off the live conversation loop. The built-in `vision` skill is offered only to an agent that names it, since its subagent has to accept images.
- Camera and processor observations use bounded history, frame IDs, capture times and task-scoped selection. Uncorrelated Roboflow predictions are labelled explicitly.
- `agent.responses.create(text, images=[...])` delegates attachments to the configured vision skill. Unsupported local flows reject images clearly.
- Swift session observers answer only locally registered tools, so an attached iPhone cannot reject frame-capture requests owned by the Python video worker.

### An agent or router stores its own directory on starting

`Agent(config="customer_support")` and `Router("clinic")` now find the directory behind the
name and store it themselves, so the explicit sync goes away:

```python
async def create_agent(**kwargs) -> Agent:
    return Agent(config="simple_voice_ai")  # no sync_agent call
```

`.agent_sync` and `.router_sync`, written next to `agent.yaml` and `router.yaml`, record the
md5 of what was last stored and when. A launch that changed none of the synced files costs a
file read rather than a request. `sync_agent` and `sync_routers` are unchanged and still
there for storing a directory ahead of time; `ensure_agent(name)` is the new form that
answers out of the stamp.

### `agent.yaml` says what the agent runs on

The declaration used to carry only a name. It now decides the models too, so an agent that
lives in a repository no longer needs `define_agent` to pick them:

```yaml
name: analyst
mode: text
llm: llm-fast
subagent: llm-thinking
sandbox: daytona
```

`mode`, `stt`, `tts`, `voice`, `llm`, `subagent`, `search`, `greeting`, `plugins`,
`keyterms`, `sandbox` and `tags` are all read and sent with the directory. Everything but
the name is optional, and a setting the file leaves out leaves whatever is stored, so a
model chosen in the dashboard survives a sync that says nothing about it. A key nobody
knows is refused rather than dropped.

`POST /v1/agents/sync` takes the same fields, so the Go SDK and any other client store a
directory in one request.

### Speech-to-text routing: a priority list, a data policy, and configs from YAML

`SttOptions` now says more about where a transcript may come from than which model to ask.

`providers` is an ordered list rather than a single `target`. Each entry is a vendor, one of
their models or a capability shortcut, and each is expanded where it stands, so the order
written is the order tried. Health only moves a provider that is down to the back — unlike a
shortcut, a priority list is not reordered on latency, because a caller who wrote an order
meant it.

`data_policy` is a requirement rather than a description. `allow_training: false` routes only
to a provider that has said it does not train on what it is sent, and `retention` is a ceiling
— `none`, or a duration such as `30d`. Every speech model in the router config now declares
what it does with audio, and a vendor who has published nothing counts as not having said no,
so a request nothing can serve is refused rather than sent somewhere that does not comply.

`mode` is `verbatim` or `smart`, `profanity_filter` masks offensive words, and `overwrites`
carries per-provider settings this vocabulary has no word for. All four narrow the candidates:
a provider that cannot express one is not offered the request, and an overwrite naming a field
a provider does not have is reported rather than dropped.

```python
await Router("healthcare").configure_stt(
    providers=["deepgram", "parakeet"],
    data_policy={"allow_training": False, "retention": "none"},
    profanity_filter=True,
    overwrites={"deepgram": {"eot_threshold": 0.6}},
)
```

`sync_routers("routers/")` stores a directory of YAML configs the way `sync_agent` stores an
agent directory, so routing that matters can live in the repository and be reviewed. Go gains
`DefineRouter`, `Router.ConfigureSTT` and `SyncRouters`, which it did not have at all.

### Text-to-speech routing says the same three things

`TtsOptions` gains `providers`, `data_policy` and `overwrites`, which mean for a voice exactly
what they mean for a transcript. Every voice in the router config now declares what it does
with the text it is sent, so `allow_training: false` routes only to one that has said it does
not train on it, and a policy nothing meets is refused rather than served anyway. A vendor
named in a priority list is served by its streaming model on a socket and its batch model for
a recording, so asking for `elevenlabs` live cannot land on the file-returning model.

`overwrites` is currently the only way to steer a live voice per vendor — a voice id from one
library means nothing at another — and covers `elevenlabs: {voice_id}` and
`inworld: {voice_id, delivery_mode}`.

A voice may also be named `custom:receptionist`, which means one of your own voices and
nothing else: without the prefix a name that is not yours is passed through to the provider's
library, and with it, it is refused. `VoiceBinding` gains `synced_at`, which is when a provider
last came back with a voice that can be spoken in — `updated_at` moves again when a binding
goes back to pending, so it could never answer that.

```python
await Router("healthcare").configure_tts(
    providers=["elevenlabs", "en-low-latency"],
    voice="custom:receptionist",
    data_policy={"allow_training": False, "retention": "none"},
    overwrites={"inworld": {"delivery_mode": "STABLE"}},
)
```

Go gains `Router.ConfigureTTS` beside `ConfigureSTT`.

### Call pipeline shows the model routing picked

A call that asked for a capability shortcut such as `en-low-latency` now also reports the
`provider/model` that served it (`stt_used`, `tts_used`, `llm_used`, `subagent_used`), so the
dashboard pipeline is not only the request.

### A router config, four modalities, live and recorded

`RouterConfig` says the routing options once, for speech-to-text, text-to-speech, completions
and search, and `/v1/router/configs` stores it. Every option is a default that a per-call
option overrides, and each of the three streaming modalities now has a non-realtime form
beside the socket:

- `POST /v1/stt/recordings` transcribes a whole recording — a URL, a path or the bytes — with
  diarization, word timings, SRT or VTT subtitles, redaction, summaries and entities. It routes
  to the batch models rather than the streaming ones, which are cheaper per hour and more
  accurate, under the new `en-recorded` and `multilingual-recorded` aliases.
- `POST /v1/tts/recordings` speaks a whole text into one file, which is what lets a codec and a
  bitrate be chosen instead of raw PCM.
- `POST /v1/search` answers one question without a socket, which is the fourth routed modality
  finally having a client.

Both jobs take a `callback` so a caller need not poll, and both are addressed by
`GET .../{id}` in the meantime.

```python
router = Router("healthcare")

async with router.stt.realtime() as stt:
    ...

transcript = await router.stt.recording("interview.mp4", diarize=True, words=True)
audiobook = await router.tts.recording(chapter, format="mp3_44100_128")
hits = await router.search("perioperative antibiotic guidance", results=5)
```

`stream.Router` in Go and `Router` in `VisionAgentsCore` are the same four namespaces.

An option the modality does not have is refused rather than sent and ignored, and so is one no
provider behind that target can express: each model declares in `router.yaml` which optional
terms it can serve, and a request naming one only routes to a model that declared it. A
transcript that was quietly not diarized is worse than being told. The `router-stt`,
`router-tts`, `router-llm` and `router-search` skills record what each vendor calls the same
option and what the router will not fake.

### `Agent.join` can create the call it joins

A call type and an id are enough, so an agent that makes its own call no longer says it
twice:

```python
async with agent.join(call_type, call_id):
    await agent.responses.create("greet the user in one short sentence")
```

`join(call)` with a call of your own is unchanged, and `create_call` is still there for
anything that needs the call before joining it.

### A session is given a thinking model it did not ask for

A session that names no `subagent` is now given one, so an agent written down as
instructions alone can hand the hard parts over instead of guessing at them: the target is
`llm-thinking`, a new shortcut for the high-quality tier that prefers `openai/gpt-5.6-sol`
and keeps the rest of the tier behind it as failover. With it come the built-in `think`,
`recall` and `explain` skills. Naming a `subagent` still decides it.

It is the one target looked up before it is asked for. A deployment whose `router.yaml`
routes no high-quality model keeps taking calls, and those agents answer everything
themselves, the way an agent goes without search when nothing routes it.

### Swift SDKs: chat and voice from an iOS app

Three iOS packages under `sdks/swift`, generated from `acceleration/api/openapi.yaml`:
`VisionAgentsCore` (the client, the session socket and the conversation state),
`VisionAgentsUI` (SwiftUI views over that state) and `VisionAgentsRTC` (joining the call over
Stream Video). They are separate packages so an app that only holds a text conversation does
not resolve the WebRTC binary.

```swift
let agents = VisionAgents(url: url, customerID: "examples")
let chat = try await agents.chat(agent: "swift_demo")
```

Configuring an agent is not in them: those operations are marked `x-server-side-only`, and
the generator refuses to emit a method for one. `examples/voice_agents/swift_demo` shows both halves
— a Go program that syncs the agent directory, and an iOS app that talks to what it left
behind.

### Server-side-only operations are refused from a device

Writing an agent config, replacing what an agent knows and waiting on the dispatch socket are
marked `x-server-side-only` in the spec, and the router now answers 403 to a caller that
authenticated as an end user's device rather than as a backend.

### The router logs every request it serves

One info line per request with the method, path, status, duration and customer; a socket is
logged once it closes, with the life of the connection as its duration. A `5xx` is logged at
error.

### `sync_agent`: push an agent directory to the acceleration server

`stream.sync_agent("customer_support")` reads `examples/voice_agents/customer_support/`
(instructions.md, skills/, knowledge/) and stores them on the Go backend. A hash
of the directory is sent with it, so a second call with the same files does
nothing. `Agent(config="customer_support")` then runs that config: the backend
handles LLM routing, and edge, llm and phone default to the acceleration
plugins.

```python
from vision_agents.plugins import stream as acceleration

await acceleration.sync_agent("customer_support")

agent = Agent(config="customer_support")
```

See `examples/voice_agents/`.

### Inbound calling: wait for a call and answer it

`stream.Dispatch()` connects out to the acceleration backend and waits;
the router pushes an arriving call down that connection, so nothing in your
process has to be publicly reachable.

```python
dispatch = Dispatch()

@dispatch.wait_for_call()
async def answer(call: InboundCall):
    async with agent.answer(call):
        await agent.responses.create("greet the caller")
        await agent.finish()

asyncio.run(dispatch.run())
```

`agent.answer(call)` is the inbound mirror of `outbound_call`: it joins the call
the caller is already in rather than creating a second one, and waits for them
before the agent speaks. Several workers can wait at once, in which case calls
are shared between them, and a worker already holding `capacity` calls is passed
over rather than queued behind.

The router hears about an arriving call by webhook, which is an app-wide setting,
so run `go run ./cmd/phone hooks -url $ROUTER_PUBLIC_URL` once; it leaves any
other hooks your app has alone. See `examples/old/14_inbound_call_example`.

### `Agent.outbound_call`: ring a phone and hold the conversation

`Agent(phone=stream.Phone())` gives an agent somewhere to place calls from, and
`agent.outbound_call(from_=, to=)` is an async context manager that creates the
call, rings the person and joins before their phone starts ringing, so nobody
answers to silence. It also takes `call_id`, `call_type`, `ring_timeout`,
`initial_digits` for reaching an extension behind a menu, `headers` for custom
SIP headers and `custom` for fields the agent can read off the call. A vendor
whose API cannot express one of those refuses the call rather than placing it
without: a ring timeout that was silently dropped is a call sitting in somebody's
voicemail.

The acceleration backend places those calls at seven vendors — Twilio, Telnyx,
Vonage, Bird, Plivo, Bandwidth and Sinch. DIDWW cannot and will not: it sells
numbers and has no call control API at all, so `dial` is absent from its declared
operations. Plivo and Bandwidth fetch their call plan when the person answers, so
they need `ROUTER_PUBLIC_URL` pointed at somewhere they can reach.

`stream.Accelerated(config="john")` names a stored agent config instead of
repeating the models, resolved when the agent joins. See
`examples/old/13_outbound_call_example`.

### `stream` plugin: the whole voice pipeline in the acceleration backend

Adds a new `stream` plugin. `stream.Accelerated(model=, stt=, tts=)` sits in the
`llm` slot but runs no inference and touches no media: the Go acceleration
backend joins the call, hears the caller, answers and speaks, and Python records
the events it sends back. Transcripts, conversation and events work as they do
locally, and `@agent.llm.register_function` keeps working, because function
calling stays in Python — the model asks over the session socket and the result
goes back the same way. Point it at a router with `STREAM_ACCELERATION_URL` and
`STREAM_ACCELERATION_CUSTOMER_ID`.

For a pipeline that stays in Python, `stream.STT(target)`, `stream.TTS(target,
voice=)` and `stream.LLM(target)` route one modality each through the same
router, with the same failover and cost tracking. `stream.Router(target)` asks
which modality serves a name and returns the plugin for it, at the cost of a
request at startup. See `examples/old/00_accelerated_example`.

### `Agent`: `harness`, `cost_tracking` and `memory_filter`

`Agent.__init__` takes three new arguments, all of which apply when the LLM is a
remote pipeline such as `stream.Accelerated`. `harness=DefaultHarness(...)` from
the new `vision_agents.core.harness` configures what work is handed to a slower
model, and `vm=Daytona` gives that model a sandbox to run code in.
`cost_tracking={...}` labels every request the call makes, and
`memory_filter={"user_id": ...}` scopes what it recalls. They are configuration
serialized into the remote session, not a second implementation of the loop.

The Telnyx plugin, until now a phone transport, also exposes `telnyx.LLM`,
`telnyx.STT`, and `telnyx.TTS`, so a phone agent can run end to end on Telnyx.
`telnyx.LLM` wraps Telnyx Inference's OpenAI-compatible Chat Completions
endpoint and defaults to `meta-llama/Llama-3.3-70B-Instruct`. `telnyx.STT`
streams `linear16` over WebSocket and takes a `sample_rate`, so telephony audio
from `TelnyxMediaStream` can be transcribed at 8 kHz without an upsample; pick
the engine with `transcription_engine`. `telnyx.TTS` streams MP3 over WebSocket
and decodes to `PcmData` as it arrives. All three read `TELNYX_API_KEY` from the
environment. See `plugins/telnyx/examples/voice_agent_call.py` for an inbound
call answered by an all-Telnyx pipeline.

### A VM on the agent config

`define_agent(vm=Daytona)` says where the subagent may run the code it writes, and every
session created from that config gets it without asking. Which sandbox an agent is allowed
is a property of the agent rather than of one conversation, so it no longer has to be
repeated per session — `AgentConfig.sandbox` carries it, a directory sync leaves it alone,
and a session request still overrides it. Daytona is the only provider, and a config naming
anything else is refused when it is written rather than once a call is running.

```python
await stream.define_agent(
    name="analyst", llm="llm-fast", subagent="llm-thinking", vm=Daytona,
)
```

### `agent.knowledge`: filling a knowledge base from a page

An agent can now subscribe what it looks things up in to a page it does not host. The page
is read straight away, cut into the same passages a document becomes and kept under the url,
so reading it again replaces what it wrote rather than duplicating it.

```python
agent = Agent(config="docs_agent")
await agent.knowledge.add_url("https://visionagents.ai/introduction/quickstart")
```

Which knowledge base that is belongs to the stored config, so an agent configured by hand
rather than by name has none to fill, and an agent whose pipeline runs here has none at all.

### `examples/text_agents`: agents that answer in writing

Two runnable examples of a conversation held in writing rather than on a call. `docs_agent`
answers out of a knowledge directory and a page on the docs site, both under one namespace.
`analyst` hands arithmetic to a subagent with a VM. `sync_agent(name)` now finds an agent
directory anywhere under `examples/`, not only in `examples/voice_agents/`.

### `gemini` plugin: Gemini 3.8 Live and Extended Thinking (#647)

Adds `gemini-3.8-live` and `gemini-3.8-live-extended-thinking`. Agent turn completion follows `interaction_status` (`IDLE`, with deprecated `REQUIRES_ACTION` treated as idle) instead of treating `turn_complete` as session-idle. Live tools default to `NON_BLOCKING`. Video turn coverage defaults to `TURN_INCLUDES_AUDIO_ACTIVITY_AND_ALL_VIDEO`. Requires `google-genai>=2.19.0`.

### `gemini` plugin: Gemini 3.5 speech-to-text

Adds `gemini.STT` using Gemini Live transcription (`gemini-3.5-transcribe-live` by default). It streams 16 kHz PCM, supports `language_codes` and `custom_vocabulary`, and emits standard transcript and turn events. Automatic language detection is the default (omit `language_codes` or pass `[]`).

### `deepgram` plugin: Flux TTS streaming, `speed`, and Interrupt barge-in (#633)

Deepgram TTS uses the Flux turn protocol (`Speak` / `Flush` / `SpeechMetadata`) with a persistent websocket. Pass optional `speed` (0.85–1.15 in 0.05 steps) on the constructor. Barge-in sends `Interrupt` instead of Aura's `Clear`. Supported sample rates now include 32000 and 44100.

- `GET /v1/{modality}/providers` lists each model's `supports`: the options it can be asked for, such as `keyterms` and `diarize`, and left out when it takes none. Which models take keyterms was only in the router's config (#1014, AI-1419).
- `resume_history` on an app's or organization's policy, an agent config, or a server-side `POST /v1/agents/sessions` (the most specific winning field by field) sets how a conversation a session carries on is handed back to the model. `include_tools` replays each reply's recorded tool calls and their results, so a resumed agent does not run its tools again; over `max_characters` the oldest results are trimmed first and the conversation's first message is kept. `max_messages` (1 to 100) and `max_characters` (1,000 to 400,000) replace the 100-turn and 60,000-character caps. Left out, a resumed session is handed what it was before.

## Bug Fixes

- A connector whose manifest says `revoke_on_delete` (Nessy) now has its grant revoked at the provider on three more paths that drop it: `DELETE /v1/agents/users/{user_id}/connections`, a consent refused as `account_mismatch` (the tokens it just got, which nothing stores), and `DELETE /v1/agents/connectors/{id}/oauth-client` (before the client goes, as the revoke needs its secret; the connections stay, moved to `needs_reauthorization` once their grant is revoked, so their token export answers the reconnect error rather than the revoked access token). A refusal is logged and the change stands. A provider's `error` member is redacted of what the request carried before it reaches a log line, as its `error_description` was (AI-1647, AI-1643, AI-1648, AI-1649).
- A built-in connector made internal in a later revision is now hidden at every revision from an app whose settings do not say `internal_connectors`: the catalog, its latest revision, and a connection pinned to an earlier revision. Before, the earlier public revisions stayed listed and connectable. A built-in made public in a later revision is seen at every revision. A custom connector is unchanged (AI-1591).
- A thread task (SMS or WhatsApp) whose reply the line refused is retried with the reply's files, not only its text. A thread removed while the task's turn runs fails the run as `thread_gone`, final, where it was recorded `delivered` with nothing sent (AI-1882, AI-1883).
- A hosted tool's `tool_result` on `/v1/dispatch` takes the session socket's shape: it names its call with `tool_call_id` (`id` is still read), and `output` may be a parts array with pictures, which reach the model. A result with parts closed the worker's socket before, and one naming `tool_call_id` was never matched, so the call waited out its timeout. The Go SDK's hosted tools now send their pictures rather than failing the call, and need a router from this release to do so.
- A worker on `/v1/dispatch` running a hosted tool call is sent `tool_cancel` (`id`, `session_id`) when the call's timeout passes or its session gives up on it, including when the session is on another node. It was never told, so it ran on, and its answer was dropped.
- A session that a slow greeting delays no longer loses what happened while it was spoken: a conversation named, a connection chosen at a login, or a video frame seen before the session's row is stored are written after the row, not dropped (AI-1925). The title shows in the sessions list instead of `—`, and a fork of the ended session no longer asks for the login again.
- A session events socket (`/v1/agents/sessions/{id}/events`) that lands on a node other than the one running the session now refuses a frame that is not a command, as the node running it does: an `error` frame, `invalid_request`, then a close with `1008`. This covers text that is not JSON, JSON that is not a command object (such as `[1]`) and a binary frame that is not JSON; a binary frame that holds a command is carried and answered as a text one. Before, the relayed socket skipped the first and the third, and stayed open after the second with an `error` event, so whichever node got the socket decided whether a bad frame ended it (AI-1924).
- An agent runs a connector call again once the person has acted on what held it back: signed in, granted the scope a step-up asked for (`connector_scope_required`), or had the rejected token saved again. A tool that changes something was answered "You already made this exact call" from its earlier `scope_required`, `credential_rejected` or login result for the rest of the session, so the call never ran and the second step-up never began (AI-1927).
- Renaming, reading or forking a session right after stopping it (`PATCH`, `GET /v1/agents/sessions/{id}` and the other reads of a session that ended) no longer answers 404 now and then. The session's row is written behind the conversation, so one stopped before the row landed was missing from both the live set and the store; a read that finds no row waits for the writer and reads again, while a session whose row is there answers at once; so does creating a scheduled task that names such a session (AI-1926).
- The Google Drive connector asks Google for `drive.file`, so `create_file` and `copy_file` are offered and work, and the Google Docs connector asks for `documents`, so `update_doc` works; both still ask for their read scopes. Revision 4 of each: a connection made before keeps its revision and reconnects once to be asked for the new scope (AI-1933).
- Deleting a session (`DELETE /v1/agents/sessions/{id}`) that was opened without `memory.user_id` no longer contacts the memory store: it cannot have taught it anything, so the delete no longer counts against the mem0 quota and succeeds while mem0 is down. A session with memory, or one created before this release, is forgotten as before, including the 503 `memory_unavailable` when the store refuses (#1219, AI-1920).
- A session events socket (`/v1/agents/sessions/{id}/events`) or dispatch socket (`/v1/dispatch`) that is sent a frame that is not JSON now answers with an `error` frame, `invalid_request`, then closes with `1008`. A session events socket whose `close` command ended the session closes with `1000` "the session ended". Both ended with no close frame before, which a client read as `1006` (AI-1441).
- A session events socket, its relayed form on another node, and the dispatch socket now wait up to a second for the client to answer the router's close before hanging up: when a session ends elsewhere or through the socket's own `close` command (`1000` "the session ended") and when a router drains (`1001` "router restarting"). Hanging up at once let a proxy drop the close frame, so the client saw no close code in 2 of 5 runs on staging. `/v1/agents/socket` waits the same way when it refuses a start frame (`1008`) and when its conversation ends (`1000` "the conversation is over") (AI-1441).
- A caller hanging up `/v1/agents/socket` while the agent is speaking no longer logs `agent failure` at ERROR for each piece of speech still on its way. A real failure to publish speech is still logged at ERROR (AI-1866).
- A speech-to-text or text-to-speech model whose vendor this deployment has no key or endpoint for is listed with `available: false` (`GET /v1/stt/providers`, `GET /v1/tts/providers`, the routes and `resolve`), and a shortcut ranks it last and passes over it. A target that names it is refused with the vendor's own words, such as `microsoft: api key is required (set AZURE_MAI_API_KEY)` or `breeze: websocket url is required (set BREEZE_WS_URL)`, and the failure no longer counts against the model's health. Such a model was listed as available, and every start on it failed. A model whose config is set is unchanged (AI-1442, AI-1443).
- The Swift SDK reads a timestamp without fractional seconds, such as `2026-10-02T17:00:00Z`, on iOS before 26. Go writes no fraction when a value has none, and the SDK refused those there: a session, guest or step-up carrying one failed to read, and a remembered guest's expiry was lost.
- A transcription routed to Deepgram carries `mip_opt_out=true` whenever the data policy it is held to refuses training or names any `retention`, whether that policy is the request's own `data_policy` or the floor its app's or organization's policy sets. It was sent only when the request itself set `allow_training: false`, so a request held to `allow_training: false` by its app or organization, or asking for a `retention`, was routed to Deepgram as a model that does not train and keeps nothing, and then sent without the flag those terms depend on. Live transcription (`/v1/stt/stream`) and recordings (`/v1/stt/recordings`) both send it (AI-1895).
- Starting voice (`POST /v1/agents/sessions/{id}/voice`) on a socket session (`/v1/agents/socket`) whose voice was stopped (`DELETE` on the same path) answers 400 `invalid_request`, "voice on a socket session cannot be started again once stopped: open a new socket". The socket is the session's call, and stopping voice hangs it up, so the agent joined a call already over and the router crashed with "close of closed channel". The session carries on in writing until the socket closes. Starting voice on a socket session that is still on its socket changes nothing, as before (AI-1405).
- A sandbox whose code is still running when its session ends loses its connectors' credentials at once: the Daytona secret, or the egress proxy's record, is deleted when the sandbox's close stops waiting for the run. Daytona's secret stayed until the hourly sweep, the proxy's record for up to 31 minutes. A connector the sandbox cannot get a credential for no longer puts the provider's error text in the `run_code` result or the router's logs: they say whether the connection needs a reconnect, or the kind of failure (AI-1685, AI-1684).
- A connector binding with `sandbox: true` that is removed from its agent config mid-session, or no longer has `sandbox`, or now binds another connector or connection, stops giving the session's sandbox its credential: under the Daytona broker the secret is withdrawn before the next `run_code`, and under the proxy broker the next request with the placeholder gets 401. It kept working until the session ended. Once withdrawn, a binding put back reaches only the next session (AI-1686).
- A device (an end user's token) no longer reads a session's `instructions` back: the session it is answered with by `POST /v1/agents/sessions`, `GET /v1/agents/sessions/{id}`, `POST /v1/agents/sessions/query`, `PATCH /v1/agents/sessions/{id}`, the fork, starting and stopping voice, and the session socket's `session` frame leave them out, whether they are the session's own or its agent config's. A brief the backend hands a session, such as the customer somebody practises on, stays hidden from the person practising. A backend still reads them as before (AI-1392).
- `DELETE /v1/agents/sessions/{id}` answers 503 `memory_unavailable` (a new `ErrorDetail.code`) when the memory store fails to delete what the session taught it, such as mem0 answering 429 once its quota is spent. It answered a bare 500 `internal_error`. The session is kept, stopped, so the same delete later deletes it and the rest of its memories. `DELETE /v1/agents/users/{user_id}/memories` and `DELETE /v1/agents/sessions/{id}/memories` answer the same 503 in the same case. The answer names no vendor; the store's own error stays in the router's log (AI-1752).
- A text-to-speech recording (`POST /v1/tts/recordings`) whose `providers` list holds only models for live calls, such as `cartesia/sonic-preview` or `tts-fast`, is refused with 400 `invalid_request`, naming the batch models that record. A recording keeps only a list's batch models, so such a list had none to try: the job was accepted and then failed, and an inline one came back `failed`. A list with at least one batch model that records is unchanged (AI-1444).
- The deprecated plugin API no longer names a plugin on an agent config that binds a sensitive connector, such as Nessy: setting a plugin's client with `user` (`PUT /v1/agents/configs/{id}/plugins/{plugin_id}/client`) and finishing the app's login of a catalog plugin (`GET /v1/agents/plugins/callback`) answer 400 as creating or updating the config does, and store neither the plugin, the client nor a token. A plugin the config already names, a client set without `user`, an end user's own login and a config with no sensitive connector are unchanged (AI-1641, #1158).
- Finishing the app's login of a catalog plugin (`GET /v1/agents/plugins/callback`) checks the sensitive-connector rule again on the config it writes, with the config locked, after the code is exchanged. A config update that bound Nessy during the exchange could leave the plugin named beside it; the login now answers 400, as one refused before the exchange does, and stores no token (AI-1760).
- A fork of a session that has ended (`POST /v1/agents/sessions/{id}/fork`, naming no config) runs on its agent config as the config is now, as a reopened conversation does: its models, voice, skills, tools, connectors and every other setting, not only its prompts. It ran on `llm-fast` with none of them. The models the parent was given in place of the config's (`model_overwrites`) still win, and a fork's own `model_overwrites` replace them. A fork of a config that now needs a sandbox the deployment cannot open, or names a skill deleted since, fails with 400 as a reopen does. A config deleted since leaves the fork as before. A fork of a running session, and a fork naming a config, are unchanged (AI-1672, #1153).
- The router no longer loses usage rows quietly. Every row has a `usage_id` before it is first written, and one writer per process writes every recorder's rows in batches of up to 200, `ON CONFLICT (usage_id) DO NOTHING`, retrying a failed batch from 100 ms up to 30 s, so writing it again never records it twice. Live counters and provider health are updated as rows arrive, before and apart from the Postgres write, so they keep moving while Postgres is down. The writer holds up to 20,000 rows waiting to be written, where each recorder held 512 and dropped the rest. A row is lost only when those are full (it still reaches the live counters), when the writer still holds it 15 s after everything that records has stopped, or when Postgres refuses that row on its own; each is logged at ERROR as `usage row lost` with the whole row as JSON, counted in `router_usage_lost_total{reason}` (`queue_full`, `shutdown`, `invalid`) and `router_usage_lost_micros_total`, and reported to Sentry at most once a minute with every row lost since the last report. `/metrics` also has `router_usage_queue_depth`, `router_usage_oldest_pending_seconds`, `router_usage_written_total`, `router_usage_duplicates_total`, `router_usage_write_errors_total`, `router_usage_unattributed_total` (a row with no key or origin, also logged at WARN at most once a minute) and `router_usage_unresolved_total` (a row written while its organization could not be looked up; the lookup is tried again before every retry of its batch, each bounded to 1 s, and a failed one no longer becomes the decision the next request is admitted on). A batch that meets a request id already taken, as rows a data move carried in with their own ids leave, moves the id sequence past the highest id and is tried again at once. A data move now imports `requests` by `usage_id` and lets the destination give each row an id of its own, so a moved row never overwrites the destination's row with the same id, and a replayed delete names the row by its `usage_id` (or, for a row from before usage ids, by the one its import made from its source id). An agent log row's request error comes from its own details; an older row without one takes it from the request its number names only when that request started when the row says it happened, for the same agent, so a moved row never shows another request's error. In a sharded deployment, a batch's agent log rows are written with one call lookup and one insert rather than a few round trips each. Migration `20261110135000_requests_usage_id` adds the columns in one `ALTER TABLE` that waits at most 5 s for its lock (on a timeout the router does not start, and its next start tries again), gives `usage_id` a random default so a router from before it keeps writing rows during a rollout or after a rollback, and builds its two indexes `CONCURRENTLY`. Routers starting together take turns to migrate without holding up each other's index builds. If a build fails, or somebody drops either index, the router drops what is invalid and builds it again at its next start; it does not start while it cannot (AI-1826, #1169).
- No answer is billed at nothing because its vendor never said what it used. An LLM response that streamed an answer, thinking or a tool call with no usage from its provider, or that was cut short before any arrived (barge-in, a superseded reply, the slower of two hedged requests, or a deadline: the caller's, or the provider's own two-minute limit), is billed on the router's own count: the prompt as it counts it and what came back, marked `attributes.usage_estimated = "true"` and counted in `router_usage_estimated_total{modality,provider,model,reason}` (`cancelled` for a cut, `timeout` for a deadline, `missing` for an answer that finished, which is also logged at ERROR as `billable work recorded an estimate`). A response a deadline cut short is recorded with error code `timeout` rather than `provider_error`; whoever waited still sees it fail, and a failure the vendor itself reports is still not billed. A provider that repeats its usage on every chunk keeps its own count on a cut. On OpenAI-compatible providers, thinking that xAI and Gemini count in `total_tokens` but not in `completion_tokens` is billed as output (it was dropped: 3 to 10 times under on their thinking models), and xAI's own cost (`cost_in_usd_ticks`) and OpenRouter's (`usage.cost`, decision models) are kept as `attributes.vendor_cost_micros`, which never prices the row. Successful billable work that records none of what its price rates counts in `router_usage_missing_total` and logs `billable work recorded no usage`; billable work with no price counts in `router_usage_unpriced_total`; and tokens that do not add up to the vendor's `total_tokens` count in `router_usage_mismatch_total`. The router sends each of the four to Sentry once per item at most every hour (fingerprint `usage_<name>:<modality>/<provider>/<model>`, tag `usage_alert`), at error level, never paging. Search, image generation, detection and decision models are recorded through one `routing.Call`, one row per request however it ends, a vendor's panic included; work settled after the fact without a usage id is refused, logged as `usage row lost` and counted in `router_usage_lost_total{reason="invalid"}` (AI-1726, #1188).
- Streaming sessions bill what they can see. A transcriber whose vendor reports on each turn exactly what it bills is billed those turns, and a turn it says nothing for is billed for the audio streamed since the last turn, marked `usage_estimated` and alarmed as `missing`. A transcriber whose vendor bills the audio it is sent rather than what a turn reports, which is every live transcriber today, is billed exactly the audio the router streamed to it, every millisecond of it, between turns and after the last one included, marked `measured_by = "router"` where no turn reported it, never estimated and never alarmed; on a failover each vendor is billed what it took. Rows that bill work without timing a request of the provider (a voice session's seconds, its backend's responses, transcriptions, audio after the last turn or reply) are kept out of the provider's routing health and still count toward the customer's spend. Ink 2 is priced at Cartesia's published rate of $0.00672757 a minute; Gemini 3.5 Transcribe Live stays at Google's own estimate of $0.009 a minute, since its Live API reports no usage. A voice is billed the characters it was sent, whatever its adapter counted, and names every request itself. GPT-Live is billed the seconds its session is open (`session_seconds`, $0.05 a minute, from `session.usage.updated` and the final `session.closed`), not the caller's audio, and each response of its backend model is an LLM row of that model with `attributes.parent_session`; its replies are still recorded, at no cost, and the caller's audio after its last reply is no row of its own. Work given away is written at no cost, with what it would have cost in `attributes.free_cost_micros`, so no sum of what customers pay counts it. The seconds are read as the numbers with a fraction the API sends, which dropped `session.closed` whole, its disconnect included. OpenAI Realtime's input transcription is a row of `openai/gpt-4o-mini-transcribe` ($1.25 a million tokens in, $5 out) per caller turn. xAI voice is billed the audio xAI says it bills (`billable_audio_seconds`), replies heard over a gap in the caller's audio included, and $0.004 a typed turn (`messages`); the caller's audio after the last reply is a row of its own (`measured_by = "router"`) wherever the price list bills it. A session's own rows share an `operation_id` that its side rows name as `parent_session`. `internal/metered/guard_test.go`, part of `go test ./...`, fails a file that holds a vendor-facing type or imports a vendor's package outside the routers and the services that meter their own work, and a model vendor that runs no suite; every suite checks the work is billed, with new `imagegensuite` and `decisionmodelsuite` (AI-1726, #1194).
- Knowledge, memory and voice cloning are billed by what they use. Every knowledge lookup (an agent's, and reading a document's passages back) is a `knowledge/turbopuffer/lookup` row with the `requests`, `bytes_queried`, `bytes_returned` and `results` turbopuffer's `billing` object reports; every write and delete (the API's, a directory sync's and the URL crawler's) a `write` row of its `requests` and `bytes_written`. A crawl somebody asked for is billed to their key, a scheduled re-crawl to `job:crawler`. What each call cost under turbopuffer's own formula (its 1.28 GB query minimum, its 10 KB write minimum and its batch discount) is kept as `attributes.provider_cost_micros`, to three decimals, from the rates in the new `knowledge.stream_cost` setting, which defaults to turbopuffer's list prices. A write in several requests is a row per request, so when one fails the requests turbopuffer took before it are still billed. `router usage storage`, run hourly and once per turbopuffer key and region, records one `storage` row of `byte_hours` per knowledge base of an app the deployment knows, for the last hour to have ended (or `--hour`), dated to that hour and owned by the app under `job:knowledge_storage`, at the size the base holds when it runs; an empty base writes no row, a rerun for the hour bills nothing twice, and a base it cannot read is logged at ERROR, alerted on once as `storage_missed` naming the hour, and fails the run. A memory recall is a `memory/mem0/search` row and a remembered exchange an `add` row, of one request each, and forgetting a session or a user a `forget` row, which mem0 does not bill. Cloning a voice is a `tts/<provider>/voice_clone` row of one `clones`, named by the clone the provider made, so preparing the same voice again bills nothing unless the provider made a new clone. Hearing a voice (`POST /v1/agents/voices/{id}/preview`) is a `tts` row of the characters said, at the rate of the model the provider speaks with, and a line over 500 characters is refused with a 400. A call through `routing.Call` that fails leaves its usage id to the retry that succeeds. The vendor list prices the new items at turbopuffer's and mem0's rates: a lookup at $0.00128 a thousand (turbopuffer's 1.28 GB query minimum, so a namespace larger than that sells at the minimum and shows what it costs in `provider_cost_micros`), a write at $2 a GB with at least 10 KB, storage at $0.33 a GB-month, a mem0 search at $2.49 a thousand and an add at $0.249 a thousand; voice clones and mem0 deletes are free (AI-1726).
- Every agent session's time is billed, a minute at a time. From the moment a session is created until it closes, it writes a `runtime` row per slice of up to 60 s as each slice ends, provider `stream`, model what holds the session: `text` while it is in writing, `voice` on a call, `video` once the user's video was seen on that call, and `text` again when voice stops. A slice bills `seconds`, the whole seconds its boundaries cross measured from the session's start, and the last slice rounds the total up, so the rows add up to the session's life rounded up to a second; a router that dies loses at most the slice it was in, and each slice's `usage_id` comes from the session, its start and the slice's number, so writing one twice bills it once. A session that fails to open is not billed and leaves no session row. A router shutting down stops every session's runtime before it closes any of them. The rows are measured by the router (`measured_by = "router"`) and kept out of routing health. They are spend and never requests: they count in `/v1/stats/spend` costs and a session's usage cost, and not in `request_count`, tag-key coverage or a call's `requests`; a call's runtime is its own session's, not that of another session of the same agent at the same time. Their rates come from the deployment's price list; until it names `runtime/stream/{text,voice,video}` they are recorded at no cost and alarmed as unpriced, and a mode priced `free` is recorded at no cost and not billed. Billable work with no price is logged at most every ten minutes per item however often the price list is read again; a list read every 30 s logged it every time. Replies thrown away to cut latency are billed as before and now tagged: a speculative reply carries `attributes.speculation`, shared by every speculative reply to one turn of the caller's and by the answer asked for when none of them could be used, and `attributes.kept`; each attempt of a request that was asked of a second provider carries `attributes.hedge` (the request's id). `kept` is `true` only on work that was used: a dropped speculative reply is `false` whichever of its hedged attempts won. A speculative reply that was dropped was never read to its end, so it wrote no usage row and was not billed; it is now. Image rows carry the size drawn (`attributes.width`, `attributes.height`), which a price entry can key on. A session's greeting is recorded as its first turn every time: the turn could be written before the session's row and refused (AI-1726, #1208).
- Texts and call minutes are recorded and priced. Every text a line receives and every one it sends is a `phone` row of its vendor and channel (`telnyx/sms`, `meta/whatsapp`, `linq/imessage`) with `messages`, an SMS's `message_segments` as a carrier counts them and its MMS as the new `media_messages` meter (an MMS that arrives, files alone included, is one), priced by `direction`, the other party's `country` and WhatsApp's `category` (`service`); a text delivered twice is billed once, a send that fails is a row that is not billed, and a reply sent in parts whose later part failed is billed for the parts that went out. WhatsApp and iMessage lines send through the app's own Meta and Linq accounts, so their rows are priced and not billed (`vendor_account = "app"`). A text line has to be a Telnyx number. Each leg of a call is kept in the new `call_legs` table (migration `20261111130000_call_legs`): a placed call or a transfer rings from when the vendor takes the dial and starts when the person's phone joins the Stream call, matched to its leg by the phone's number; a call that arrives starts when it does, one leg per Stream session however often or late `call.session_started` is delivered, and a delivery another node handed over opens nothing. Legs are timed by when Stream made each event (`created_at`, epoch in any unit or RFC 3339), no earlier than the leg was dialled, and by arrival time for an event made over an hour before it. A phone's join or leave delivered before its leg is written is kept for a day in `call_participants`, so the leg takes it, and ends at once when the leave came first; a join delivered again after the call ended revives the leg the end left unanswered, billed once. A join or leave the database could not take is answered 503, so Stream delivers it again. A leg ends when its phone leaves or the call ends and is billed as a `call_minutes` row of its `call_seconds` to whoever placed it, or the app under `phone:inbound`, by direction and country; a leg that never connected bills nothing. A leg still open after four hours is ended, logged at ERROR and recorded priced and not billed; a leg whose row was lost is recorded again under the same usage id. The call hook now asks for `call.session_participant_joined` and `call.session_participant_left` too: run `cmd/phone hooks` again. Until it is, an app's placed calls are not billed, and the router logs `call hook does not deliver participant events; re-run cmd/phone hooks` at ERROR, once an hour per app, when a placed call ends unanswered with no participant event delivered, the agent's own join included; the deployment's own app is checked once at start. Inbound calls count toward the 10DLC sandbox's minutes too; a leg the sweep ended does not. Placing a call or a transfer is a free `dial` row (it was `call` and `transfer`), and a customer's own trunk's minutes are free. A number is billed each month it is held, at the price the vendor quoted, or the number's offer when the purchase names none, or as billable work with no price when neither does: the month it was bought in when it is bought, and each after it by the new `router usage numbers --from <time>` under `job:number_rental`. Registering a 10DLC brand ($4.50), a campaign's carrier review ($15) and its first three months are billed when it is registered, under `job:dlc_fees`, and each month after by the new `router usage dlc --from <time>`. Neither job bills anything that began before its `--from`. While the 10DLC sandbox holds an app, its texts and minutes are priced and given away (`dlc_sandbox = "true"`). The vendor list prices Telnyx SMS and minutes in the US, Meta's WhatsApp service messages in the 41 markets it names, and Telnyx's 10DLC fees; iMessage through Linq is free per message (AI-1726).
- Explicit prompt caching on OpenAI (`prompt_cache_options.mode: explicit`) caches the instructions: the breakpoint after them was left out of the request, so OpenAI cached nothing and every turn paid for the instructions in full. Grok on xAI is sent the request's `prompt_cache_key`, without which it read nothing back from its cache. A GPT-Live session waits up to 2 s for the server's `session.closed` when it is closed, rather than dropping the socket before it arrives. An xAI voice reply reports its tokens, which xAI sends beside `response.usage` rather than in it; they were reported as zero. `cerebras/gemma-4-31b`, archived by Cerebras, and `perplexity/pplx-decider-v1.1-27b` on OpenRouter, which has no endpoint behind it, are no longer routed to; `llm-conversational` is Gemma 4 26B-A4B and GPT-5.6 Luna, and Decider V1.1 is served by Perplexity's own API (AI-1726).
- Several outbound phone calls can run at once from one number on Twilio, Telnyx and `sip_trunk`.
- A number attached to the router can place outbound calls on Twilio, Telnyx and `sip_trunk`, where the call used to fail while the number's own trunk held it.
- Outbound calls on Twilio, Telnyx and `sip_trunk` share one Stream SIP trunk per vendor instead of making one per call, so trunks no longer pile up toward Stream's cap of 25 per app. Trunks left by a call whose end never arrived are deleted after 6 hours.
- A call's Chat channel shows what the agent is doing with Stream's AI indicator, as a conversation held in writing already did: `ai_indicator.update` with `AI_STATE_THINKING` once the agent starts on a reply and while its tools run, `AI_STATE_EXTERNAL_SOURCES` while a search tool runs, and `AI_STATE_GENERATING` once its words show, each naming the reply's `message_id`, then `ai_indicator.clear` once the reply is heard or cut off and its tools are back, or the call ends. The voice transcript sent none, so `AIStateIndicator` in `stream-chat-react` and the other AI components never showed the agent thinking on a call. A reply the agent starts on is stored under an id made from its channel and turn, `voice-reply-<sha256>`, which the indicator names before the reply has a word in it (#1126).
- `POST /v1/agents/runs` answers as soon as the agent has given its answer with nothing left to do, rather than after a further two seconds of quiet, so every run returns about 2 s sooner. A reply that calls a tool or hands work off still waits for the answer that follows it.
- A voice agent's flow controller on Gemma or OpenAI is held to a JSON Schema of its own values, so a ruling such as `{"disposition":"continue"}` can no longer replace the decision with the fallback. Other models are asked for a JSON object as before, and a ruling they get wrong still falls back. `router_agent_flow_invalid_rulings_total{reason}` counts the rulings refused, by `decode`, `disposition` or `floor` (AI-1668, #1123).
- A fork of a session that has ended (`POST /v1/agents/sessions/{id}/fork`, naming no config) runs on its agent config's `instructions`, `guardrail` and `review.instructions` as the config is now. It ran with no instructions and no guardrail, and was reviewed by Stream's default guide. A config deleted since gives the fork none of them, and the fork goes ahead as before. A fork of a running session, and a fork naming a config, are unchanged (AI-1662, #1120).
- Resuming a scheduled task with no occurrence left, such as a one-shot reminder that failed or a repeat past its `until`, answers 400 `schedule_in_past` over the API and leaves the task as it was, as the agent's `update_scheduled_task` tool already did. The API marked such a task `completed` although it never delivered, or answered 409 `too_many_tasks` or 404 first. To run it again, give it a new schedule, then resume it. A paused one-shot resumed after its time is still completed with a `missed` run (AI-1639, #1119).
- A voice agent no longer starts its reply over a caller who has started talking again: when the caller says more before any of the reply to their last turn has been heard, that reply is dropped unheard and the newer words are answered together with the earlier ones, which the model is told it has not answered yet. A cough, "mm" or a run of assent such as "okay", "uh-huh" or "got it" leaves the reply to play, and once some of it has played, talking over it goes to the flow controller as before (AI-1601, #1100).
- A voice agent no longer answers a caller's queued turn on its own after it has answered a newer turn from the same caller: what was queued is answered together with the newer turn, which the model is told it has not answered yet. Another participant's queued turn is still answered on its own once the agent stops talking (AI-1669, #1121).
- A live transcription the vendor refuses part way through, such as ElevenLabs's `unaccepted_terms`, goes on with the next model the session could use (the next in its `stt.providers`, or its target's next candidate), and with none that refused it. The session ended with one `error` and the agent's next audio could reopen it on the model that had just refused it. What the caller had said of the turn still open when the vendor refused is lost; what they say from then on is heard by the next model. `/v1/stt/stream` sends no `error` frame for a refusal another model takes over and goes on with its transcripts. With no other model left, the session ends with the one `error` it ended with before. A socket that drops, and an error the vendor goes on after, end the session or are passed on as before (AI-1587).
- `/v1/stt/stream` says which model transcribes after a refusal hands the socket to the next one: another `started` frame naming its `provider` and `model`, the same shape as the first. Until now the caller learnt of the switch only from the `provider` on the next transcript. A socket no model refuses still sends one `started` (AI-1588).
- A write to a vendor that has just refused the session no longer reaches the caller as an `error` frame on `/v1/stt/stream`, or as an `stt` failure in the agent's log, when the next model takes the session over: the caller hears only of the takeover. A write that fails with nothing from the vendor behind it still reaches the caller; one that fails behind a socket that dropped is told once, by the drop's own `error` frame. The router now logs each such takeover on `/v1/stt/stream` at INFO, with the model that refused, the model taking over and the vendor's reason (AI-1603, AI-1606).
- `a tool call came back` names the tool call under `tool_call`; it logged it as a second `call` key, beside the session's `call`. A check-in's `what the agent heard from a participant` line and its `could not write a line to say` warning log `purpose` as `hold`, `update` or `checkin` instead of the whole prompt the line was written from (AI-1585, AI-1586).
- An OpenAI Realtime agent with `interrupt_response: false` answers a caller whose turn ends while it is still speaking. The server refused its own reply with `conversation_already_has_active_response`, which the agent logged as an error, and the caller's question was never answered. The agent now asks for that reply once the one in flight is done, and does the same for a refused prompt, text or tool result; a caller who speaks again or barges in gets the server's answer to their next turn instead (AI-1548).
- A phone session ends on its own: 10 seconds after the last person leaves the call, and when a phone call is never answered (90 seconds for an outbound call, 30 seconds for an inbound one). The agent stops talking and leaves rather than finishing its sentence to an empty call. The Python SDK now tells the router a session is a phone call, as the Go SDK does, so a Python agent on a phone call can also transfer the caller to a person. Sessions that are not phone calls are unchanged.

### A web page may send PATCH

The router's CORS answer now allows PATCH, so a page calling the router directly can patch an
agent config (`patchAgentConfig`) or change a scheduled task (`updateScheduledTask`). A browser
refused both at the preflight before.

### A Kotlin voice call asks only for the microphone

`VoiceSession` joined an agent's call with Stream Video's default foreground service, which
refuses to start without the camera permission because the call allows video. The call went
on, with no service holding it. It now runs Stream's audio-call service, and the client it
builds asks for the camera only while the camera is on. A call type the app configured on its
own `StreamVideo` is left as it is.

- A model call its caller cancelled (a superseded flow decision, a losing hedge, a barge-in) is reported as cancelled, not as a success: the router log, the session socket's `model_call` frame and the call timeline's `ModelCallTiming` say `success: false` with `error_code: "cancelled"`, and still carry its tokens, since it is billed. A failure has any other `error_code`. Live and stored views of one call used to disagree: the frame said it succeeded, the timeline that it failed. The agent log writes a cancelled request at info, as `llm request cancelled` (AI-1466).
- A flow decision that has said nothing for `ROUTER_FLOW_HEDGE` (`400ms` by default, `0` turns it off, `cmd/agent -flow-hedge`) is asked of another candidate of `llm-flow` as well, and whichever says something first rules on the turn, as a late reply already is under `ROUTER_REPLY_HEDGE`. A flow controller that stalled on its first token made the caller wait out its 3 s deadline on every turn it did, and the turn was then answered with no ruling (#1040).
- A line the agent says on a call (`say`, and the greeting) is in its transcript and its responses, and the model knows it was said. It went straight to the voice with no reply announced, so the Chat channel and `responses.items` never had it, and the caller's answer to a greeting's question reached the model and the flow controller with no question before it. A line that opens the call is told to the model in its instructions rather than as a first turn, which chat templates that need a user turn first refuse; a conversation is not named after its greeting (#1040).
- The agent's spoken replies in a call's Chat channel carry `ai_generated`, as replies in a conversation held in writing already did, so Stream's AI components render them as the assistant's (#1040).
- What a caller says after an answer, on a transcriber that keeps the run of speech open (Deepgram Flux resuming a turn), is heard as only the new words. The whole run was passed on each time, so every later turn repeated the earlier ones to the model, the flow controller, in `heard` and in the transcript, more each turn. Words added to something being read out ("order 1 2" then "3 4", "one two" then "three", letters spelled out) are still heard whole (#1040).
- A reasoning step says how long the model thought for it, from when the model began working towards it: the response starting, its latest words, or a tool's result. It counted from the step's first thinking to its last, so a model that sends its thinking only once it is done, such as `xai/grok-4.7`, sent steps with no time, and Stream's AI components read `Thought process` rather than `Thought for 9s`. A model that streams its thinking reads a little longer than before, by the wait for its first thought. (#1052)
- The router logs a live Stream Chat update or AI indicator that Stream refuses or answers slowly (`a live Stream Chat update was refused or slow`, `an AI indicator was refused or slow`, `live Stream Chat updates pause after a refused one`). Both are sent from the loop that streams a reply, so either held back the reply for people watching it, for up to the request's 5 s timeout, and nothing said why. (#1052)
- `LiveReasoning` keeps each reply's reasoning apart, in the Swift, JS, Kotlin and Dart SDKs. It looked a step up by its id alone, and every reply numbers its steps from `r1`, so a later reply's live thinking was dropped when its round was shorter than an earlier reply's, or joined onto the earlier text when it was longer, and the earlier reply's thinking showed in its place. A step is now read by its reply and its id: `reasoning[message: message.id, step: step.id]` in Swift, `reasoning.get(message.id, step.id)` in JS, `reasoning[message.id, step.id]` in Kotlin (whose `add` now takes the message id before the step id), and `reasoning[(message: message.id, step: step.id)]` in Dart. (#1051)
- The router's agent answers a question whose turn decision timed out. The decision call is given 3 s; when its stream had opened but no token came by then, the call was closed and no ruling was sent, so the question was never answered nor stored. That ruling now carries the timeout, and the agent answers a finished question whose ruling failed, as it already did when the call timed out before it opened (AI-1460).
- A person on a voice call with no display name goes under `Speaker 1`, `Speaker 2` and on, numbered in the order such people first spoke, in what the reply model reads (`Speaker 1: what about me`) and in what the flow controller is asked. Their user id, or their participant id when they had no user, went ahead of their words instead, so the model read the app's internal ids. Their name, once it is known, still replaces the label, and named people read as before (AI-1458).
- A phone's tools work on the call of a person's saved conversation. Every tool call there needed a typed question's request id, which a spoken question and the vision skill's request for camera frames never have, so on a call each was refused with `session: tool call has no durable command binding` and the agent said the tool did not work. A request id is now needed only in a conversation held in writing, as taking the results already was.
- A chat that starts voice hands its skills over on the call. A session opened on a call is given the default subagent (`llm-thinking`) when its config names none, and one that started in writing was not when voice started, so its call had nothing to run `think`, `recall`, `explain` or `vision` on: a tool's pictures ended as `skill "vision" is not available`. Starting voice now gives it the same default.
- A tool's pictures reach the `vision` skill on a model that checks a conversation's tool calls. The vision task was handed the conversation with the call that took the picture still open, which OpenAI refused (`400 No tool output found for function call`), so the agent said the analysis did not work. The open call is now answered as still running before the conversation is handed over.
- A Python agent on a local LLM plugin is told who is speaking once a second person has spoken in the call, as the router's agent is (AI-1420): their words reach the model as `Bob: what about me`. Every person's words read the same before, so the model answered two people as one. A call where one person speaks sends exactly what it did. The name is the call participant's (`Participant.name`, filled by the Stream edge), else their user id. The plugins that rebuild the request from the conversation (anthropic, aws, huggingface, nvidia, openai Chat Completions, openrouter) name every line from then on; the ones that keep their own history (gemini, openai Responses, xai) are sent the named turn. What the conversation stores and shows is still only the words (AI-1439).
- A Python agent on a local LLM plugin tells the model when several people are talking, as the router's agent does: once a second person has spoken, the system prompt ends with `Several people are talking in this call. Each of their messages starts with the name of who said it, then a colon. Never start your reply with such a label.` A call where one person speaks sends exactly what it did. Realtime plugins hear audio, not named turns, and are not told. A person with no display name goes under `Speaker 1`, `Speaker 2` and on, in the order such people first spoke; their user id went ahead of their words instead. Their name, once known, replaces the label; a name is fixed by a person's first message, and two people with the same name read as `Alex` and `Alex (2)` (AI-1461).
- `stream.STT.start()` gives up on a router that never answers its start frame after 90 seconds, closes the socket and raises `TimeoutError: the router did not start transcribing within 90 s`. It waited forever, so an agent whose router was stuck never joined and never said why (AI-1440).
- An agent opens no transcription session for a participant who has left the call. Their audio and their departure come apart, so a last frame handled after the leave, or one whose session was still opening, opened a vendor socket that stayed open until the agent closed. A participant who comes back with the same session id is transcribed again, and so is one whose audio still comes a second after the first frame dropped, in case the SFU reported a leave for somebody who stayed (AI-1433).
- A live transcription pinned to a batch model, such as `deepgram/nova-3`, is refused before the vendor is called: `routing: deepgram/nova-3 is a batch model for recordings, not live calls; for a live call name deepgram/flux-general-en or deepgram/flux-general-multi or stt-fast or stt-accurate`. It was dialled at Deepgram's live endpoint and failed after about 4 s with `deepgram: connect failed`, recording a `start_failed` row against Nova-3. This is the `/v1/stt/stream` `error` frame, and for an agent the 400 that starting its voice session answers with (`agent: start stt: routing: ...`) rather than an `stt` error on every frame of the call; a priority list made only of batch models, and the recorded shortcuts `en-recorded` and `multilingual-recorded`, are refused the same way. A list with a live model in it is served by that model, a recording on Nova-3 is unchanged, and `mock/mock-stt` is now marked live so the load benchmark still streams at it (AI-1431).
- A voice agent config naming a batch-only transcriber, such as `deepgram/nova-3` or `en-recorded`, is refused when it is saved: create, update, `PATCH` and sync answer `400 invalid_request` with the same message a session start gives. Every session of such a config was refused as it started, after the config had been saved without complaint. A config that speaks through an STS model or is written to is not checked, and neither is a save that leaves a stored `stt` as it was, so a config saved before this change stays editable (AI-1455).
- A `/v1/{modality}/stream` socket that routing refuses, such as a live transcription pinned to a batch model, closes with 1008 (policy violation) instead of 1011. Any other failure still closes with 1011, and the reason is the same refusal (AI-1456).
- People on a voice call keep their names after the session switches between voice and writing: the next turn reads `<name>: <words>` and the instructions still say several people are talking. Both were lost until two people spoke again. Two people with the same display name now get distinct labels, in the order they first spoke: `Alex`, then `Alex (2)`, then `Alex (3)`. A label stays the same for the whole session, even if the person rejoins or renames themselves (AI-1437, AI-1438).
- `GET /v1/agents/calls/{id}/timeline` pairs each turn with its own words: `heard` is what the caller said to start it and `said` is the agent's answer. Both were one turn off: `heard` held the agent's answer and `said` the caller's next question, and the first question was missing. Several things said before one answer are joined into `heard`; a turn the agent started, such as a greeting, has no `heard`, and a turn cut off before the agent spoke has no `said`. `/transcript` was right and is unchanged (AI-1432).
- A voice agent on GPT-Live (`openai/gpt-live-1*`) files each reply as its own turn. The model streams audio for the whole session, digital silence between replies included, so one reply opened when the agent joined and closed only when it left, and a reply the caller cut off kept every later one muted. A reply now opens on the model's words or on audio with something to hear in it, and closes after 800 ms with neither (AI-1533).
- A voice agent on Gemini Live (`gemini/*`) files every reply as its own turn. When the model's words came before its audio, the reply opened before the caller's turn was settled, so `GET /v1/agents/calls/{id}/timeline` lost the first turn and filed each later one under the next reply, with the previous caller's words and a round trip of about 12 s; the session socket's `responding` prompt was the turn before. The caller's turn is now settled, and the wait timed from their last word, before the reply opens. GPT-Live (`openai/gpt-live-1*`) opened a reply on its words the same way and gets the same fix (AI-1464).
- A speech-to-speech turn's round trip runs from the moment the model's own detector decided the caller had finished, on Gemini Live (`gemini/*`) as on GPT-Realtime. Gemini's was timed from the caller's last transcribed words, which arrive with the reply, so it read 0-4 ms; it is now placed from the `audioOffset` of the server's `ACTIVITY_END` and reads about 160-370 ms on `gemini-3.8-live`. On GPT-Realtime (`openai/gpt-realtime*`), a stop heard while a reply to `Prompt` was still open timed the next reply from it: a follow-up 12 s later read a 13,030 ms round trip. That stop now belongs to the open reply, and the next reply is not timed from it (AI-1534, AI-1535).
- `GET /v1/agents/voices/library` lists every Cartesia voice. It stopped at the first 1,000 (ten pages of 100), and Cartesia had 1,049 on 2026-10-19. `provider=fish` now says Fish voices cannot be browsed through the router yet and to name one by its Fish model id, rather than `fish publishes no voice library`. A provider that will not list voices for the router's key (a 401 or 403) is reported as `the inworld key cannot list voices (http 403)` rather than as a refused voice (AI-1413).
- A recording on a model that only streams, such as `cartesia/sonic-preview`, `inworld/inworld-tts-2` or `breeze/breeze-tts-2`, is refused when it is sent, with a 400 that names the models that record: `routing: cartesia/sonic-preview can only stream speech, not record it; models that record: elevenlabs/...`. It was accepted and failed with `no factory registered`, in the job or in the inline answer. Only ElevenLabs records; a shortcut that reaches it, such as `tts-quality`, is served by it as before, and so is a priority list with it in. `GET /v1/tts/routes` and `/v1/tts/routes/{target}` now say per candidate whether a recording can use it (`Candidate.can_record`); STT and STS leave it out (AI-1412).

- A voice asked of one vendor only, as `overwrites.<vendor>.voice_id` with no top-level `voice`, is spoken by that vendor or not at all. On a shortcut that reaches several vendors, such as `tts-quality`, another vendor served it with no voice, so in its default (ElevenLabs' is "Spuds Oxley - Grandpa"). Only the named vendor's models are tried now, and if they fail the error names them; a recording asked in a voice for a vendor with no recorder (any but ElevenLabs) is refused as the caller's mistake. With no top-level `voice`, `overwrites.cartesia.voice_id` is now spoken by Cartesia, which never read it. A request that asks for no voice, or for one at the top level, routes as before; STS is unchanged. The voice that spoke is reported, the provider's default included: `voice` on the `/v1/tts/stream` `started` frame and on a speech job (`Speech.voice`) (AI-1411).

- A request that names one model and asks it for an option it does not take is refused naming that model: `routing: microsoft/MAI-Transcribe-2-Streaming does not take keyterms`. It read `routing: no provider can express keyterms`, as if the router had looked beyond the model the caller chose. This is the `/v1/stt/stream`, `/v1/tts/stream` and `/v1/sts/stream` `error` frame and every other refusal of a term; a shortcut keeps the old message (#1014, AI-1419).
- The `/v1/stt/stream` start frame is refused with `stt: at most 100 keyterms may be named, and this asks for 101` past 100 keyterms, as a session already is. Deepgram took any number, and the others refused a long list each in its own words (#1014, AI-1419).
- A live transcription whose vendor fails for good mid-call, such as ElevenLabs refusing an account that has not accepted its terms (`unaccepted_terms`), reports it once: `/v1/stt/stream` sent one more `error` frame per frame of audio after it (779 in one call). The stat row keeps the vendor's reason as its `error_message`, and the model is unavailable in `GET /v1/stt/providers` and ranked last for the next five minutes, however many turns it served before (AI-1417).
- A live transcription whose vendor closes the socket with a refusal code, such as Muse's `1008` "Ingress below real-time", marks the model unavailable as any other refusal does. It was counted as a dropped connection, which leaves the model available. The refusal codes are `1002`, `1003`, `1007` to `1010` and `4000` to `4999`; every other close (`1005`, `1006`, `1011` to `1014`, `3xxx`) and a network error is still a drop. A stat recorded after the router began shutting down is dropped instead of panicking (AI-1436).
- A `/v1/stt/stream`, `/v1/tts/stream`, `/v1/llm/stream` or `/v1/sts/stream` socket the router ends now closes with a close frame after its last frame: `1000` after a clean end, `1011` after a failure with the failure as the reason (cut to 123 bytes). It dropped the connection instead, so a caller saw `1006`, and a caller still sending audio after a refused start was reset and could lose the `error` frame itself, as `stream.STT(...).start()` did in 2 of 5 runs on staging ("the router closed the socket before it started"). The router now reads and drops what the caller sends for up to a second after its close frame, until the caller answers it (AI-1425).

- `stream.STT.start()` raises the router's refusal as a `RouterError`, such as a model that takes no keyterms or has no API key. It sent the start frame and returned at once, so a refusal arrived only as an `STTErrorEvent`, and an agent nobody subscribed to errors on got no `started`, no error and no transcripts. `start()` now waits for the router's `started` or `error` frame. An error after `started` is still an `STTErrorEvent`, and the sockets opened later for other participants work as before (AI-1422).
- A Knowledge source whose `<source>#<n>` id reached 64 bytes is indexed: a URL subscribed with `POST /v1/agents/knowledge/urls`, or a document posted to `POST /v1/agents/knowledge` whose `source` is that long. Its passage ids were `<source>#<n>`, and turbopuffer refuses an id of 64 bytes or more (`id string should be less than 64 bytes`), so such a page failed to index. A passage id that would not fit is now a hash of the whole source and the index (`<sha256, base64url>#<n>`); every id that fitted stays the same, so a base filled before needs no re-index (#1010, AI-1408).

- Voicebench reply timing no longer reuses an answer to a later caller turn for earlier turns, and detects quiet speech on the isolated agent track. Methodology v10 separates missing post-caller replies from agent speech that already started while the caller was talking. Native-target diagnostics follow the explicitly selected router instead of a stale environment URL.

- A Muse STT session survives a caller who stops talking. Muse closes a session whose audio falls behind the clock (`1008 Ingress below real-time`, about 16s after the audio stops), and a Stream call carries audio only while someone speaks, so a muted or quiet listener lost their transcription mid-call. The router now sends silence for the time no audio came, once the audio is 200ms behind, which also lets Muse settle a turn the caller ended by going quiet. Audio that keeps up with the clock goes up as before (AI-1418).
- A chat that opened before its person connected an account uses the account once they have. A session binding with no connection chosen waits for a login; asked for the account after the person connected it outside the chat, from their app with a consent its backend began, it began a second consent and showed another `connector_authorization` card. It now opens on the caller's one connected connection, as a session opening then would have, with no second consent. With two connected, it still asks.
- `stream.STT` credits each transcript to the participant whose audio it is of. It sent every participant's audio over one router socket and credited each transcript to whoever sent the last frame, so a line still arriving when the next person started talking went to them. Each participant now has a socket of their own: the one `start` opens goes to whoever speaks first, and the others open on a participant's first frame. A socket the router refuses for a later participant is reported once as an `STTErrorEvent` and that participant's audio is dropped; one participant still uses exactly one socket (AI-1403).
- A Python `Agent` built by hand, without `AgentLauncher`, loads its components' models when it joins: the multi-speaker filter's Silero VAD, a local STT such as `fast_whisper`, a local turn detector or processor. It loaded none, so once two people spoke on the call the first frame failed with `warmup() must be called before process_audio()` and the agent took no more audio for the rest of the call, though the join had succeeded. A model that fails to load now fails the join before the agent enters the call. An agent from `AgentLauncher` keeps the models the launcher loaded, and `Agent.warmup(cache)` loads them ahead of the join (AI-1402).
- A voice id ElevenLabs does not have is refused on a shortcut that also reaches Cartesia or Inworld, such as `tts-quality`. The router fell back to them, so the stream started and failed on the first line (`Unknown voice: … not found!`, `voice ID must be a valid UUID`). They now take such an id only when their own voice listing holds it, in whatever order the candidates are ranked. This covers a voice shaped like an ElevenLabs id (20 letters and digits); a Cartesia UUID or a name such as `Ashley` routes as before, and a mistyped name on such a shortcut still fails on the first line (AI-1395).
- An ElevenLabs voice that ElevenLabs does not have is refused when the voice opens, naming the closest voice in its library: `elevenlabs has no voice "Sara"; the closest in its library is "Sarah - Mature, Reassuring, Confident" (EXAVITQu4vr4xnSDxMaL)`. Any id was passed on, so a wrong one failed on the first line spoken. The `/v1/tts/stream` socket answers it as its `error` frame, a session's voice with a 400, and a recording as a failed job. A voice in the account, a Voice Library voice and a legacy voice still speak; Cartesia and Inworld voices are not checked (AI-1395).
- A connector tool that needs a scope the connector never asks for is no longer offered: it is left out of `GET /v1/agents/connections/{id}/tools`, out of a validate's scope check and out of a session's tools. A manifest says so with `sources[].tools[].needs_scopes`, which may now name a scope outside `scopes.list`. The `metabase` connector's revision 2 so leaves out `execute_sql` and its 11 tools that write or schedule, which Metabase refused with `403` for the read and query grant it asks for; a connection made on revision 1 lists them until it consents again and is validated (AI-1394).
- An agent config granting a connector tool the connector leaves out is refused when it is saved (create, replace, patch and `POST /v1/agents/sync`), with a 400 naming the binding, the tool and the scope the connector never asks for, such as `metabase`'s `execute_sql`. It saved, and a session then failed a required binding with `tool_unavailable` or quietly lost the tool. A binding with `tool_patterns` is not affected (AI-1400).
- A recording sent to a live STT shortcut (`POST /v1/stt/recordings` with `en-low-latency`, or a router config whose STT is one) is transcribed by the batch models of the recorded shortcut for the same languages (`en-recorded`, `multilingual-recorded`). It reached Deepgram's batch endpoint with Flux, which Deepgram refuses (`V2_MODEL_ON_V1_LISTEN_ENDPOINT`). A live shortcut that names no language, such as `stt-fast`, goes by the recording's language hints: none or English only, `en-recorded`; any other language, `multilingual-recorded`. A recording pinned to a live-only model such as `deepgram/flux-general-en` is refused before any vendor is asked, naming the recorded shortcuts (AI-1396).
- SIP trunks the router makes in Stream for calls, transfers and attached numbers now accept a call only with a password the vendor sends, or only from the vendor's addresses. A transfer takes the vendor's addresses as a call does, so a transfer on a vendor with no declared addresses and no password (Sinch, Bandwidth) is refused before anything is made. A number attached before this release keeps its old trunk, and attaching it again changes nothing; releasing it removes the trunk, but the same number may not be available to buy again.
- Attaching a number that is already attached in the same app answers what it has and changes nothing. It made a second SIP trunk for the number, which Stream refuses, so a Go, JavaScript or .NET program that waits for calls failed on every start after its first.
- An agent.yaml connector whose `connection:` the app does not have no longer stops the Go SDK's folder sync (`Agent.Sync`). The agent syncs without that connector's tools, and a warning names the agent, the connector and the connection; the next sync adds the connector once the connection exists. A label shared by several connections still fails, and so does a missing connection on an entry with `required: true`, whose error says to add the connection or drop `required` (AI-1368).
- A consent the provider refuses (any RFC 6749 section 4.1.2.1 error but `access_denied`) leaves its reason as the connection's `last_validation`, code `connector_consent_refused`, status `failed`. For `unauthorized_client` it adds that an admin at the provider may need to allow the router's client, with the redirect URI. A connected connection keeps its last validation (AI-1379).
- A Gong connection can start its consent without the app's own OAuth client. The router registers its client with the scopes the consent asks for (`scope`, RFC 7591), which Gong requires; it sent none, and Gong answered `HTTP 400`. A provider that refuses a registration now has its `error` and `error_description` in the answer, and a connector that also takes the app's own client answers `no_oauth_client`, naming `PUT /v1/agents/connectors/{id}/oauth-client` and the redirect URI its client must list (AI-1363).
- A consent that cannot start because the connector has no OAuth client now names every way to give it one that the connector lists, with a code of its own, `no_oauth_client` (AI-1357). For `slack_bot` that is `PUT /v1/agents/connectors/slack_bot/oauth-client` with the Slack app's `client_id`, `client_secret`, `provider_app_id` and `signing_secret`, whose client lists the redirect URI the message gives, or `PUT /v1/agents/connectors/slack_bot/provider-app` with the app's name and a workspace admin's app configuration refresh token, named only where that PUT is served. It named only the first, without what it takes, and its code was `invalid_request`.
- `stability` and `similarity` on a voice's options reach ElevenLabs. `tts-quality` with a `stability` was refused with `no provider can express stability`, and the live ElevenLabs voices always sent stability 0.5 and similarity 0.8 whatever was asked. Flash v2.5 and Multilingual v2 now send both; v3 conversational, v4 and v4 Turbo send `stability` and do not declare `similarity`, which their socket does not take, so a `similarity` routes past them. A value outside 0 to 1 is refused, and is not counted against the model's health. A request that names neither sends what it did before (AI-1360).
- A TTS, STT or speech-to-speech overwrite the chosen provider cannot read (an unknown field such as `{"elevenlabs": {"voice_settings": {"speed": 1.2}}}`, or a value of the wrong type) fails the request with an error naming the provider and the field. It used to fail over to the next candidate, which never reads another vendor's block, so `tts-quality` was served by Cartesia with the overwrite dropped. The same vendor's other models are still tried, since OpenAI's realtime and Live models read different fields of one block (AI-1360).
- A speed ElevenLabs will not take (outside 0.7 to 1.2, or any speed on a dialogue model) is refused as the caller's mistake and no longer counted against the model's health. It wrote a `build_failed` row, so one caller's bad speed could mark Flash v2.5 unavailable for everyone, and the request was answered `every candidate ... failed` as if it were an outage (AI-1365).
- `/v1/tts/recordings` reads the `elevenlabs` overwrite. A `voice_id` there names the voice the recording is spoken in, and an overwrite the recorder cannot read fails the request with the overwrite error, as on the live path; both were dropped and the recording spoken anyway. A recording that names a voice of the customer's own is spoken in the id ElevenLabs knows it by, where the name was sent as asked (AI-1366).
- `/v1/tts/recordings` refuses a `speed`, and a `stability` or `similarity` outside 0 to 1, as the caller's mistake. The recording voice (`eleven_v3`) ignores speed, so one was dropped in silence; a bad stability or similarity was answered by ElevenLabs with a 422 that counted against the model's health (AI-1366).
- Decimal values remain intact when streaming voice replies are split into speech requests.
- Healthcare tool history preserves each appointment response after later reschedules.
- Corrected caller transcripts replace stale queued fragments without losing words from a separate utterance.
- New voice replies wait until interruption cleanup finishes, so barge-in no longer loses a queued follow-up.
- A caller's queued speech is dropped as soon as a reply is abandoned, before a blocking voice-provider interruption can leave it talking over them.
- A failed voice replacement keeps queued caller and delegated replies until the new voice is ready; a late completion from the lost voice no longer marks replacement speech finished.
- Deepgram Flux preserves words received with the start-of-turn event instead of waiting for a later transcript update.
- Model timing includes the wait for response headers in first-token latency and total duration.
- Voicebench no longer treats speech already in progress when the final caller line starts as the agent's answer; it stays open for the next reply.
- Prose tool tags no longer invent incomplete multi-field arguments, and brace-form single-quoted tool calls preserve their decoded values.
- A router stopping with a request still open when its 10 s shutdown grace runs out logs a warning and exits 0. It reported `context deadline exceeded` to Sentry and exited 1, so a rollout that outlasted one slow request paged as a failure (#935).
- The daily message and token limits count per user behind the proxy again. With `ROUTER_TRUSTED_PROXIES` unset, every request's address was the proxy's, so all end users shared one address allowance and the deployment as a whole was cut off after 200 responses a day. In `proxy` auth mode with no trusted ranges a named user is now counted only by user, and a guest still by address; set `ROUTER_TRUSTED_PROXIES` to count by the forwarded client address as well.
- A request with a data policy can be answered by Gemini 3.8 Flash, the model `llm-fast` prefers. It declared no data policy, so any request that asked for no training passed it over.
- The router starts on a database that already applied the migrations shipped before customer-hosted models. That migration was numbered earlier than one already applied, and goose refuses to start when it finds an unapplied migration below the database version. It now runs after that version. A database that already created `custom_models` under the old number keeps the table.
- A text agent takes answers from its skills before its knowledge base, and from its
  knowledge base before the web. Its skills ran on a subagent that called no tool, so a
  skill could never send the agent on to a pricing page or the knowledge base, and nothing
  told the model a skill outranked `search`: a support agent searched the web for prices its
  skill and knowledge base already held. A text session now has no subagent; its model loads
  the skills its config names with the built-in `load_skill` tool, which returns
  `{"name", "instructions"}` in `tool_ran`, and follows them with every tool it has. Every
  agent offered more than one of `load_skill`, `lookup` and `search` is told that order, to
  call them one at a time, and that a later source never contradicts an earlier one. A text
  agent with a sandbox still hands its skills to a subagent, a text agent naming no skills
  is offered none (the built-in set is written for a subagent), and a caller's own tool
  named `load_skill` keeps the name.
- `POST /v1/agents/connections/{id}/validate` no longer returns a provider error that echoes the credential. The answer's error is the same text GET and list store: every value the credential is sent as is replaced, and the text is cut at 1 KiB. When credentials change during the check, the answer keeps no provider text (AI-1078).
- Native speech sessions retry recoverable connection failures before leaving the call (#894).
- A test copy or a duplicate of an agent no longer silences the agent on its Slack bot, Linq, Telnyx or WhatsApp connection. A channel connection belongs to one live agent config: a create, update, patch or sync that newly binds one as `fixed` while another live config binds it is a 409 `channel_connection_taken` naming that config. A config tagged `draft_of` (the dashboard's test copy) is not refused, answers no channel message and subscribes to no MCP event. Configs that already shared a connection keep working: the oldest answers, and the router logs a warning naming the others. An MCP event is subscribed to once per connection, by the oldest live config that declares it, so a copy no longer opens a second conversation for each event (AI-1049, AI-1048).
- `phone hooks -remove` works when the tunnel it names is gone. Stream refuses an update of
  the app's hooks while any hook in it is at a host that does not resolve, and the command
  dropped a tunnel's call hook and its message hook in two updates, so the first was refused
  over the second. It now drops both, and points `-url`, in one update. `-remove` can be
  given more than once, and a refusal says how to drop the hook Stream names. A router with
  connectors on and `ROUTER_PUBLIC_URL` set now warns at startup, with the command to run,
  when its own Stream app has no message hook at it: without one a message written in a
  thread channel is stored and never answered. It only reads the app's hooks. `source` is
  documented as a reserved custom field on an agent channel: a message that carries it is
  not answered (AI-990).
- A voice agent with tools finishes a request that takes several of them while the caller is
  still on the line. A result that called for another tool was given its own sentence, so a
  request with several steps was a hold phrase, a read-back and a model turn per tool, and a
  caller who asked for an appointment to be moved and their insurance and pharmacy updated was
  gone before the last one ran. After a result the reply model now calls the tools the request
  still needs straight away, without another hold phrase, and together when they do not need
  each other.
- An agent answering in writing no longer writes a line before each tool call. The tool
  policy asked for a hold phrase before every call, which fills a silence on a call but stays
  on the page in a chat, so a chain of lookups opened its answer with a run of near-identical
  lines. A text session is now told to call its tools without announcing them; a voice
  session is told what it was before.
- A name that is already taken is a `409 name_taken` that says so. Creating or renaming an
  agent config, a router config or a voice under the name of another of the customer's, or a
  skill under the name of another skill of the same agent config, answered
  `400 invalid_request` with the database's own text, naming its table and index. It now
  answers, for example, "an agent with this name already exists", and the nine operations
  that can say it document the 409. Any other database failure on those writes is a
  `500 internal_error`, recorded with its stack, rather than a 400 quoting the database.
- A voice lost part way through a call is replaced. A voice that stopped for good, such as
  one whose key ran out of quota, left the caller with silence for the rest of the call, and
  the agent counted as still talking in it: everything the caller said next was queued as
  talking over it, and every reply was abandoned as interrupted. The agent now lets go of
  what the lost voice was saying and opens another for the rest of the call, trying the one
  it lost last, up to three times a call.
- A caller's turn that waited for the agent to stop talking is stored as said. It was
  answered without being reported as heard, so the transcript kept the caller's line open
  and wrote whatever they said next over it.
- A line typed into a voice call is part of its transcript. It was answered, but never
  stored, reviewed or logged, so the conversation read back without the caller's half of it.
- A phone vendor's own error no longer reaches the client. The `/v1/phone/*` endpoints
  answered a Twilio, Telnyx, Bandwidth or Sinch failure with `400 invalid_request` and the
  vendor's full text, which held the vendor account id. The router now logs that text and
  answers `503 phone_vendor_failed` ("The phone vendor could not complete the request.").
  Placing or transferring a call answers the same code without the advice to try again,
  because the call can still ring.
  Buying a number that the vendor sells only with a verified address, such as a Twilio
  number outside the US or Canada, answers `400 phone_number_needs_address`. A number
  search with no vendor named lists a failed vendor in `skipped` with the same general
  reason.
- A voice agent says the answer at the end of a chain of tool calls. A tool asked for in a
  reply to a tool result ran, but its result was never followed, so a Salesforce lookup that
  read a schema and then queried it went silent after the schema. Every link is followed now,
  and after eight rounds the model is made to answer.
- A tool result is answered once. A result the agent picked up while another was being
  queued started a second turn, and the caller heard the answer twice.
- A colleague's answer is kept until a reply carrying it finishes. It was forgotten once a
  reply was asked for, so a reply that was cut off or failed lost it for good.
- Sub-agent work that came back empty, or still asking to run code once it was out of rounds,
  is reported as failed. It settled as done, so the caller was promised an answer and heard
  nothing.
- A voice caller is told when a reply is lost. A model failure before any text left them in
  silence after the hold line; the agent now says it went wrong and asks them to ask again.
- Gemini takes a conversation with tool calls another model made. After a fallback its
  OpenAI-compatible endpoint refused them with a 400 for a missing `thought_signature`; they
  now carry Gemini's placeholder signature.
- Cartesia speaks again after its socket drops without a close frame. The dropped socket
  still took writes, so every later utterance went nowhere and the agent fell silent.
- A response with images is answered by an agent that has no `vision` skill, as long as its
  conversation model accepts images. Images always went to the `vision` skill, and the
  built-in skill set leaves it out, so every agent on the defaults, and every text session,
  failed with `400 invalid_request` and `harness: skill "vision" is not available`. The
  conversation model is now shown the images itself, for that one reply: they are not kept
  in the conversation, so a later turn does not send them again. An agent with a `vision`
  skill still hands images to it. An agent with neither answers `400 not_configured`.
- Twilio numbers in a number search have a monthly price. Twilio's search does not send a
  price, so the dashboard showed each Twilio number as "Not quoted". The router now gets
  the price for each number type from Twilio's Pricing API (`current_price`, which
  includes the account's discounts), keeps it for 24 hours, and shows it with no change.
  If that call fails, the search still returns the numbers, with no price. A bought Twilio
  number still has no price (AI-931).
- An outbound call from a bought number connects the person to the agent. The router gave
  the vendor the address of the call's Stream SIP trunk with no number in it, and Stream
  finds that trunk by the number, so the person answered and the call ended at once. The
  router now puts the calling number in that address for every vendor, on calls and on
  transfers. Calls through your own SIP trunk already did this and work as before.
- An inbound call to an attached number reaches the agent. The router gave the vendor the
  address of the number's Stream SIP trunk with no number in it, so Stream refused every
  inbound call. The router now puts the number in that address, and the `sip_uri` in the
  attach answer shows it. The call a number's callers join by default is now `phone-` and the
  number without the `+`, for example `phone-15125551234`, because Stream does not accept a
  `+` in a call id.
- A voice agent with tools says what it is about to do before it does it, and the caller hears a
  hold phrase as the wait begins. The instruction that has the reply model call a tool as soon as
  it has what the tool requires also made it skip the read-back an operator's own instructions
  ask for first, so the caller heard a half-second filler and then nothing while the tool ran.
  Before calling a tool the model is now told to say one short sentence, what the instructions
  ask to be said before acting, such as reading the caller's details back, or else what it is
  doing, and to call the tool in the same turn: acting at once still holds, but a bare filler
  never replaces a required read-back. That sentence opens with a brief hold phrase ("One
  moment,") and runs straight on into the read-back with no full stop between, so the phrase is
  said at the start of the wait rather than after a read-back that takes seconds to speak, and no
  pause is left between the two for a caller's interruption to fall into. After a result the
  model answers from it without another hold phrase.
- A voice agent with tools acts on a request once it has what the tools require. The model that
  answers a caller was given the tools and nothing about using them, so it kept collecting
  optional details, asked for a first name when a surname was given, asked whether to do what the
  caller had just asked for, and passed values dressed in words. Whenever tools are offered, the
  reply now carries a short instruction after the agent's own: call a tool in
  the same turn once every argument it requires is known, take a name, number or value as the
  caller gave it, call the next tool a result calls for, pass arguments as bare values and omit
  optional ones nobody gave, and follow the operator's instructions and a tool's approval
  setting wherever confirmation comes first.
- An `oauth2_code` or `oauth2_client_credentials` connection whose MCP server refuses its token
  with a bare 401 (a `WWW-Authenticate` that names `resource_metadata` and no `error`, as the
  MCP authorization spec answers an expired token) is renewed and the call sent once more. The
  router read that 401 as nothing to act on, so every call failed until the stored expiry
  passed, and `POST /v1/agents/connections/{id}/validate` reported `failed`. A refresh refused
  with `invalid_grant` now moves the connection to `needs_reauthorization`, and validate says so.
- An incognito session records nothing, as it promised. It still wrote a `calls` row with its
  id, caller and instructions, turn timings under its agent id, and, on a call, the decisions
  with the words that were heard. Its voice minutes no longer appear in the activity report.
- A login card in a tool result is attached to the reply only when it comes from a server
  each end user logs into: a plugin under `user_plugins`, or an `mcp_servers` entry with
  `user: true`. A server the app logs into, or one with no login, can no longer put one in
  the conversation.
- MCP traffic only reaches public hosts. A session opening a config's `mcp_servers` or its
  plugins, a plugin login, a token refresh and plugin events used Go's default HTTP client,
  so a config could point the router at a private address or the cloud metadata server.
  They now go through the router's egress client, which refuses those and redirects that
  leave the server's origin.
- The `slack` plugin asks for `search:read.public` instead of `search:read`, which Slack's MCP
  server does not offer, so a login with the default scopes is no longer refused.
- The `salesforce` plugin reaches Salesforce's hosted `sobject-reads` server at
  `https://api.salesforce.com/platform/mcp/v1/platform/sobject-reads`, asking for `mcp_api`
  and `refresh_token`. It pointed at `https://{instance}/mcp`, which no org serves, and
  needed an org host to connect; it now needs none.
- `Agent.ask()` follows a reply until the tools it called have answered. It stopped at
  the first `agent_speech`, which is the model saying it is about to call a tool, because
  the Python SDK dropped `pending_work`. `RemoteEvent` now carries it, as the Go SDK's
  event does. (#737)
- An agent with an `MCPServerLocal` no longer ends in a `CancelledError` when it closes.
  The stdio session was entered on the connecting task and exited on the closing one, which
  anyio's cancel scopes refuse. It is now held on a task of its own, as `MCPServerRemote`
  already was. (#737)
- A caller who kept talking after the first part of their turn was queued is no longer
  answered twice. The whole turn was answered, and then the queued part again once the
  agent stopped, in one Voicebench call 17.9 s later.
- The Python client now adds a `command_id` when it asks a stored text conversation
  something, as the JavaScript SDK does, so a user's question is no longer refused.
- An incognito voice call is no longer written into Stream Chat. The transcript writer was
  opened for every call, whatever `incognito` said.
- Images sent with a turn are no longer dropped when the conversation model also asks the
  vision skill to look. Its own ask superseded the task holding the images, and the new one
  looked at the camera instead, failing with "nobody is connected to run it" on a text
  session.
- The Go SDK can hold a user's kept conversation. A pipeline acting for an end user
  (`Backend.UserID`) on a persisted text session now sends each `Respond` with a fresh
  `command_id`; the router refused those turns with "personal conversations require a
  command ID". Sessions speaking for the app are unchanged.

- `POST /v1/agents/sessions/{id}/responses` takes an optional `command_id`, so a page can ask
  a user's kept conversation over HTTP and still get the turn's id back; it was refused with
  "personal conversations require a command ID". The JavaScript SDK's `responses.create`
  sends a fresh one on a session with a `conversationId`, or the `commandId` option you pass.

- Chat readers are explicitly added to existing agent channels before their token is
  issued, so opening a members-only transcript no longer fails with `ReadChannel`.

- The JavaScript SDK works in a browser without a `fetch` of your own: it called the global
  `fetch` as its own method, which every browser refuses with "Illegal invocation".

- The JavaScript SDK's session socket is admitted by the authenticating proxy: with
  `authenticate` it now also carries the token as `authorization` with
  `stream-auth-type=jwt`, which is how the proxy reads a socket's credential. It was refused
  with a 401, so a page on a hosted deployment could only use `watch: false`.

- The JavaScript SDK's `respond` names a command on a kept conversation, the way
  `responses.create` does, so asking one over the socket is no longer refused with
  "personal conversations require a command ID". It returns the command's id, and takes a
  `commandId` of your own.

- The JavaScript SDK tells a router reached by `customerId` who the end user is: `userId`
  is sent as `user_id` in the query of every request and socket, which a router in `proxy`
  mode reads. It was dropped, so the conversations such a caller opened belonged to nobody.

- Managed research workspaces resume stopped Daytona VMs in place, refresh preview access and recover worker processes before research; pinned source revisions are preserved.

- OpenAI streamed tool arguments now retain their output item identity when text or multiple tool calls precede them.

- Text agents continue answering after sequential tool calls, so a docs lookup followed by source research no longer leaves the session waiting silently.

- The Go SDK logs text session creation as `opened a text session`, instead of claiming it joined a call.

### Together's speech models: a question no longer settles as its last word

"Can you hear me" came back as "me". Together's realtime socket flushes its decoder on its
own schedule rather than when the caller stops talking — mid-sentence, and even mid-word,
splitting "patio" into "pat" and "io." — and the deltas after a flush start again from
nothing, carrying only the words since. Both `together-parakeet` and `together-nemotron`
published each flush as a finished turn, so the tail of a question arrived as a turn of its
own and was the one the agent answered.

A flush is now folded into the utterance in progress rather than ending it, and a turn ends
where the words read as a finished sentence, which is the only boundary the protocol offers.

### Acceleration: say what a tool found, and talk through a cough

Two tools in one reply each started a generate, and the second stole the floor so the first result was never spoken. Overlapping coughs and similar non-speech were answered as new turns. The agent now waits for every tool in the turn, speaks once, and keeps talking through a cough. Complete identifiers are answered by the fast model instead of a multi-second colleague round-trip.

### Voicebench: compare against a stored baseline by target name

`voicebench compare --baseline accelerated` resolves `baselines/accelerated/<commit>/`. `--store-baseline` on a run copies (and merges per-pack) `summary.json` and `manifest.json` there.

### Router: sentences from a per-sentence voice were heard spliced together (#675)

A voice that takes each sentence as its own request (Gemini TTS, Fish, Speechify) synthesised a reply's sentences side by side, and the agent played their audio in the order it arrived, so a two-sentence reply sounded like two voices talking over each other. The router now holds each sentence's audio until the ones before it have finished, while still synthesising them in parallel. A barge-in drops everything held, and every sentence is still settled and billed.

### Router: Gemini TTS changed voice from one sentence to the next (#677)

Asked for no voice, Gemini picks one on every request, and the router sends it one request per sentence, so an agent without a configured voice could sound like a different person on each reply. Gemini TTS now defaults to the prebuilt voice Kore. A voice named on the agent or the request still wins.

### `deepgram` plugin: Flux STT handles typed `TurnInfo` from SDK 7.7 (#633)

`deepgram-sdk` 7.7 delivers listen v2 `TurnInfo` as typed objects instead of dicts. The STT handler now accepts both, so transcripts and turn events are emitted and the unexpected-message warning spam is gone.

### Agent metadata was overwritten by stale user data (#630)

The component metadata added in #618 was merged *under* the existing `agent_user.custom`, so a pre-set `custom` won and the provider/model fields never reached the edge. Component metadata now takes precedence, and keys that don't describe a configured component are unset on the stored user, so a previous run's `tts`/`avatar`/... values are cleared instead of lingering. `StreamEdge` maps a `None` custom field to the partial-update `unset` list rather than writing a literal null.

- Hosted demo links carry the agent's call type and transcript channel type, so a browser joins the same call and chat channel.
- Closing an acceleration harness flow prevents later decisions from reaching its closed model sessions, and cancelling a turn safely handles a stream returned at the same time.
- ElevenLabs reconnects for a new synthesis after a WebSocket disconnect; socket-loss and interrupted synthesis IDs stay terminal.

# v0.6.9

## Breaking Changes

### `xai` plugin: Realtime default model bumped to `grok-voice-think-fast-2.0` (#624)

`xai.Realtime` defaults to `grok-voice-think-fast-2.0` (was `grok-voice-think-fast-1.0`). Pass `model=` explicitly to stay on a 1.0 model.

## New Features

### `telnyx` plugin: LLM, STT and TTS (#629, #631)

The Telnyx plugin, until now a phone transport, also exposes `telnyx.LLM`,
`telnyx.STT`, and `telnyx.TTS`, so a phone agent can run end to end on Telnyx.
`telnyx.LLM` wraps Telnyx Inference's OpenAI-compatible Chat Completions
endpoint and defaults to `meta-llama/Llama-3.3-70B-Instruct`. `telnyx.STT`
streams `linear16` over WebSocket and takes a `sample_rate`, so telephony audio
from `TelnyxMediaStream` can be transcribed at 8 kHz without an upsample; pick
the engine with `transcription_engine`. `telnyx.TTS` streams MP3 over WebSocket
and decodes to `PcmData` as it arrives. All three read `TELNYX_API_KEY` from the
environment. See `plugins/telnyx/examples/voice_agent_call.py` for an inbound
call answered by an all-Telnyx pipeline.

### `speechify` plugin: Speechify TTS (#615, #617)

Adds a new `speechify` plugin exposing `speechify.TTS`, backed by Speechify's streaming API. It streams raw PCM audio, defaults to the `simba-3.2` model with the `geffen_32` voice, and reads `SPEECHIFY_API_KEY` from the environment. Install with `vision-agents[speechify]`.

### Agent metadata describing the configured providers (#618)

The agent user is now published with metadata describing its own pipeline, so dashboards and Stream Chat clients can tell what an agent is made of without out-of-band bookkeeping. `Agent.components_metadata` returns `{"llm" | "vlm" | "realtime": {...}, "stt": ..., "tts": ..., "turn_detection": ..., "avatar": ..., "processors": [...]}` where each entry is `{"provider": <plugin name>, "model": <str | None>}`, and `Agent.authenticate()` writes it (plus `is_agent: True`) to the edge user's custom data. `StreamEdge` now merges via a partial update instead of an upsert, so custom fields written by other clients survive. Adds the public `MetadataValue` type in `vision_agents.core.edge.types`.

## Bug Fixes

### `nvidia` plugin: default VLM model is now `meta/llama-3.2-11b-vision-instruct` (#625)

`nvidia/cosmos-reason2-8b` is no longer available on the NVIDIA Chat Completions API for typical API Catalog keys. The plugin default, README, and example now use `meta/llama-3.2-11b-vision-instruct`.

# v0.6.8

## Breaking Changes

### Minimum `getstream` raised to `>=4.1.0,<5` (#612)

`agents-core` and the `getstream` plugin now require `getstream` 4.1, which introduces a stateful `AudioStreamTrack`. Audio tracks own their queue and lifecycle, so the agent and the avatar plugins (`anam`, `lemonslice`, `liveavatar`) no longer drive playback timing themselves. Out-of-tree edge transports that construct audio tracks (see `local` and `tencent` for reference) need the same update.

# v0.6.7

## Breaking Changes

### `moonshine` extra removed (#613)

The `moonshine` plugin is gone, and with it the `vision-agents[moonshine]` extra.

## New Features

### `anam` plugin: Anam SDK 0.6.0 (#614)

The Anam avatar plugin now depends on `anam>=0.6.0,<0.7` (was `>=0.3.0,<0.4`). Sessions use the SDK's direct API-key path and default `video_quality="high"`; the plugin API is unchanged.

### `kokoro` plugin: model warmup (#613)

`kokoro.TTS` warms the model on `start()` so the first synthesis doesn't pay the load cost, and its streaming/lifecycle handling was tightened. The `pocket` plugin's dependency pins were also refreshed.

## Bug Fixes

### `twelvelabs` plugin: asset ready wait and clip duration for Pegasus (#610)

`PegasusVLM` now polls the TwelveLabs Assets API until an uploaded clip is `ready` before `analyze_stream` — direct uploads return `processing` and must not be analyzed early. Encoded MP4 clips also set PTS/`time_base` so padded buffers report at least 4 seconds of duration (Pegasus's minimum). Uploaded assets are still deleted if ready-wait fails or times out.

# v0.6.6

## Bug Fixes

### `EventManager.register_events_from_module` crashed on non-string class types (#608)

The scan assumed every candidate's class attribute was a string and raised when it wasn't, which broke event registration for modules containing such classes.

# v0.6.5

## New Features

### Realtime input audio pacing (#599)

Realtime LLMs that need a steady upstream audio cadence can now opt into framework-level pacing. Pass `input_audio_pacing=AudioInputPacingConfig(...)` to a `Realtime` subclass and the framework buffers the irregular PCM the WebRTC uplink delivers and forwards fixed-size chunks (default 20 ms) at a stable wall-clock rate. `AudioInputPacingConfig.virtual_microphone()` is a preset for speech-to-speech models that interpret gaps in the input as end-of-turn — it primes a 500 ms buffer and fills digital silence on a dry buffer so the model never sees an interruption.

### `gemini` plugin: Live Translate model with auto-enabled input pacing (#599)

Adds `gemini-3.5-live-translate-preview` as a supported Live Translate model and bumps the default Gemini Realtime model to `gemini-3.1-flash-live-preview` (was `gemini-2.5-flash-native-audio-preview-12-2025`). The Live Translate model is sensitive to uneven input audio — irregular upstream chunks produce audible jitter and word-cutoffs in its output. `GeminiRealtime` automatically installs `AudioInputPacingConfig.virtual_microphone()` whenever the model is `gemini-3.5-live-translate-preview`, so the framework feeds it a steady 20 ms stream by default. Opt out with `input_audio_pacing=None`.

### `twelvelabs` plugin: Pegasus video understanding (#607)

New opt-in `twelvelabs` plugin exposing `PegasusVLM`, a `VideoLLM` backed by TwelveLabs Pegasus. Unlike frame-by-frame VLMs it buffers recent frames, encodes a short MP4 clip, uploads it to the TwelveLabs Assets API, and streams the analysis back, so the agent can reason about motion and events over time. Reads `TWELVELABS_API_KEY`. Install with `vision-agents[twelvelabs]`.

### `telnyx` plugin: phone transport (#594)

New `telnyx` plugin providing `TelnyxMediaStream` for Telnyx Media Streaming, plus a call registry, µ-law/PCM audio handling, webhook signature verification, and runnable inbound/outbound call examples. Install with `vision-agents[telnyx]`.

### `cartesia` plugin: streaming STT (#602)

Adds `cartesia.STT` (WebSocket streaming) alongside the existing TTS, and refreshes the supported Cartesia speech models.

## Bug Fixes

### `openai` plugin: strict mode rejected tools with optional parameters (#605)

`convert_tools_to_openai_format` set `strict: True` unconditionally, but the Responses API rejects an object schema in strict mode unless every property is listed in `required` — so any tool with a defaulted or optional parameter failed. Strict mode is now enabled per tool only when the schema qualifies, `additionalProperties: false` is applied recursively (nested objects, array items, `anyOf`/`oneOf`/`allOf`), and the normalization runs on a deep copy so the registry's shared schema is no longer mutated.

# v0.6.4

## New Features

### `gemini` plugin: `google-genai>=2.8.0` for native live translation (#598)

google-genai 2.8.0 adds a native `translation_config` field on `LiveConnectConfig`, so the Gemini Live Translate models can be driven through `gemini.Realtime(config=...)` without an SDK monkeypatch.

# v0.6.3

## Breaking Changes

### `all-plugins` extra removed (#584)

With 30+ plugins it no longer makes sense to install them all at once. Depend on the specific extras you need, e.g. `vision-agents[getstream,gemini,deepgram]`.

### `deepgram-sdk` raised to `>=7.1.0,<7.2.0` (#543)

The Deepgram plugin moves to SDK 7. Listen v2 messages arrive as plain dicts on this version, and the STT handler was updated accordingly.

## New Features

### `minimax` plugin: MiniMax LLM (#593)

New `minimax` plugin exposing `minimax.LLM` over MiniMax's OpenAI-compatible Chat Completions API. Defaults to `MiniMax-M3` (512K context, image input); `MiniMax-M2.7` and `MiniMax-M2.7-highspeed` are also supported. Reads `MINIMAX_API_KEY` and optional `MINIMAX_BASE_URL`. Install with `vision-agents[minimax]`.

### `qdrant` plugin: Qdrant RAG (#572)

New `qdrant` plugin providing a Qdrant-backed RAG implementation for grounding agent responses in your own documents. Install with `vision-agents[qdrant]`.

### `inworld` plugin: LLM and VLM via the Inworld router (#530)

Adds `inworld.LLM` and `inworld.VLM` on top of Inworld's OpenAI-compatible `/v1/chat/completions`, which routes upstream across providers with auto-selection, fallbacks, and traffic splitting. Router options are constructor kwargs (`fallback_models`, `ignore_models`, `sort_by`, `ttft_timeout`, `metadata`, `web_search`), sent as `extra_body`. A `ttft_timeout` below 500 ms raises `ValueError` at construction instead of producing the gateway's misleading 502s.

### `getstream` plugin: non-blocking StreamConversation persistence (#589)

`StreamConversation` now persists messages to Stream Chat in the background instead of awaiting each REST round-trip inline. Voice pipelines call `upsert_message` on the critical path (per transcript and per LLM delta), where the inline ~150–300 ms round-trip compounded into audible response latency. Writes are dispatched as fire-and-forget tasks serialized behind a per-channel lock, so ordering is preserved and the final persisted message always matches the final content. Adds `Conversation.wait_for_pending_syncs()`, drained on agent shutdown so in-flight writes are not dropped.

## Bug Fixes

### `openai` plugin: `ChatCompletionsLLM` ignored injected/eager turn text and leaked `<think>` reasoning (#592)

`ChatCompletionsLLM.simple_response` rebuilt the request purely from the conversation and ignored its `text` argument unless `participant` was `None`. Because the agent always supplies a participant, injected `agent.simple_response()` instructions produced an empty request (`400 chat content is empty`), and eager turns answered the *previous* transcript. The current `text` is now appended as the trailing user message when the conversation does not already end with it. Additionally, `<think>...</think>` reasoning spans emitted by reasoning models (e.g. MiniMax-M3) are now stripped from streamed deltas and final text so they no longer reach chat or TTS.

### `gemini` plugin: crash on duplicate follow-up tool calls (#588)

`GeminiLLM.simple_response` crashed with `ValueError('content parts are required.')` when the model echoed an already-executed function call in a follow-up turn. `_dedup_and_execute` filtered it out, leaving the follow-up `chat.send_message_stream(parts=[], ...)` with an empty list, which google-genai rejects. The multi-hop loop now exits cleanly when every requested call is a duplicate.

### Packaging: plugin wheels contain only the import package (#587)

Plugin wheels were built from `["."]`, pulling `tests/`, `example/`, `README.md` and `pyproject.toml` of each plugin into the published artifact. Every plugin now scopes its wheel target to `["vision_agents"]`.

# v0.6.2

## Breaking Changes

### `heygen` plugin removed (#563)

The `heygen` plugin, deprecated in v0.6.0 (#553), is now removed. Use
`vision_agents.plugins.liveavatar.Avatar` instead — it targets the same product via the supported LITE-mode integration path.

Three internal events used only by the `heygen` plugin (`LLMResponseChunkEvent`, `LLMResponseCompletedEvent`, `RealtimeAgentSpeechTranscriptionEvent`) were also removed from
`vision_agents.core.llm.events`.

### Unified plugin lifecycle via `Component` base class (#578)

`LLM`, `STT`, `TTS`, `TurnDetector`, `EdgeTransport`, `Avatar` and `Processor` now inherit from a new
`vision_agents.core.base.Component` base, which defines a uniform `start()` / `close()` lifecycle (both default to no-ops; subclasses override what they need).

Concrete shape changes for out-of-tree plugins:

- **`TurnDetector.stop` is renamed to `close`.** Plugins that subclassed `TurnDetector` and overrode `stop()` must rename the override; the agent no longer invokes `stop`. The base classes for vogent and smart_turn turn detectors were migrated in this release.
- `Processor` and `EdgeTransport` no longer redeclare `close` as `@abstractmethod` — the abstract contract now comes from `Component`. Existing implementations that override `close` keep working unchanged.
- `Component` inherits from `ABC` and keeps `ABCMeta`, so subclasses can continue to declare their own `@abstractmethod`s.

## New Features

### Python 3.14 support (#582)

`agents-core` and every plugin except `kokoro`, `smart_turn`, and `vogent` now advertise Python 3.14 support. `kokoro` and `smart_turn` pin `numpy<2.3`, which conflicts with `getstream[webrtc]`'s `numpy>=2.3.2` on 3.14. `vogent` pulls `vogent-turn==0.1.1`, which pins `onnxruntime-gpu==1.22.*` on Linux/Windows x86_64, and that has no `cp314` wheels.

Bumps the minimum `getstream` to `>=3.4.0` — the first release containing the Python 3.14 wheel-availability fix (https://github.com/GetStream/stream-py/pull/253).

### Faster agent startup via parallel component lifecycle (#578)

`Agent.join()` now starts every component concurrently (`asyncio.gather`) instead of stacking their network connects one after another. On a realistic setup
(Gemini LLM + ElevenLabs STT + Deepgram TTS + vogent + LemonSlice avatar) this cuts the lifecycle phase of `join()` from **~4.3 s to ~2.4 s (≈45% faster, ~1.9 s saved)**;
the remaining time is the single slowest component (here, the LemonSlice RTC connect).

Error handling is asymmetric and explicit:

- **Startup is fail-fast.** If any component's `start()` raises, the failure is logged with the component class name, in-flight siblings are cancelled and awaited so no
  half-open network connects leak, and `agent.join()` re-raises so the agent doesn't run half-initialised.
- **Shutdown is best-effort.** A failing `close()` is logged via `log_exceptions` and swallowed so the remaining components still get a chance to release resources.

Plugin authors: because lifecycle hooks now run concurrently, components must not assume sibling ordering during `start` / `close`. Audio and video frames are still dispatched to processors in list order.

### Friendlier `vision-agents init` errors (#579)

`vision-agents init` without an agent name now prints a message that says what's missing and shows an example, instead of Click's bare `Missing argument 'NAME'.`. The
positional argument is renamed `AGENT_NAME` and documented in `--help`.

### Richer `vision-agents init` scaffold (#581)

Scaffolded projects now ship with a `tests/` directory backed by `vision_agents.testing` (`TestSession`, `LLMJudge`, multi-turn example) and pytest configured in `pyproject.toml`, plus a `Dockerfile` and `.dockerignore` for deployment. The README and the "next steps" hint switch from `uv run vision-agents agent run` to `uv run agent.py run` — same dispatch, but the command now mirrors what's in `agent.py` so the entrypoint is the file you actually edit.

## Bug Fixes

- **Packaging**: stop double-packing CLI templates into the wheel — `hatchling` already includes the `.j2` files via the `packages` entry, so the extra `force-include` wrote each template twice and triggered `UserWarning: Duplicate name` during builds. (#571)
- **Gemini default model**: `gemini-3.1-flash-lite-preview` was decommissioned upstream; replaced with `gemini-flash-lite-latest` in defaults, examples, tests, and docs. (#583)
- **`smart_turn` and `vogent` `requires-python` raised to `>=3.11`**: both depend on `onnxruntime>=1.24.3`, which has no `cp310` wheels, so the previous `>=3.10` metadata was broken — a 3.10 install would fail at resolution time. (#582)

# v0.6.1

## Breaking Changes

### Event types cleanup (#552)

Follow-up to the v0.6.0 inference-pipeline rewrite (#501).
The events that used to carry audio / transcript / turn payloads are gone — that data now flows through
`Stream[T]` outputs on STT, TTS, LLM, and Realtime. Only lifecycle / notification events remain on the event bus.

Removed event classes:

- `vision_agents.core.agents.events`: `AgentInitEvent`, `AgentSayEvent`, `AgentSayStartedEvent`, `AgentSayCompletedEvent`, `AgentSayErrorEvent`.
- `vision_agents.core.stt.events`: `STTTranscriptEvent`, `STTPartialTranscriptEvent`, `STTConnectionEvent`, `TranscriptResponse`.
- `vision_agents.core.tts.events`: `TTSAudioEvent`.
- `vision_agents.core.turn_detection.events`: `TurnStartedEvent`, `TurnEndedEvent` (the module itself was deleted — turn boundaries are now `TurnStarted` /
  `TurnEnded` items on the STT / `TurnDetector` output stream, plus the new `UserTurnStartedEvent` / `UserTurnEndedEvent` below).
- `vision_agents.core.llm.events`: `RealtimeAudioInputEvent`, `RealtimeAudioOutputEvent`, `RealtimeAudioOutputDoneEvent`, `RealtimeResponseEvent`, `RealtimeConversationItemEvent`,
  `RealtimeErrorEvent`, `LLMRequestStartedEvent`, `RealtimeUserSpeechTranscriptionEvent`, `VLMInferenceStartEvent`, `VLMInferenceCompletedEvent`, `VLMErrorEvent`.
- `vision_agents.core.events`: `ConnectionState` enum.

Field changes:

- `RealtimeDisconnectedEvent.was_clean` → `clean`.
- `provider` field removed from `RealtimeConnectedEvent` and `RealtimeDisconnectedEvent` (`plugin_name` already identifies the source).
- `retry_count` and `is_recoverable` removed from `STTErrorEvent`.

## New Features

### `vision-agents` CLI with `init` and `agent` commands (#533, #567, #568)

Adds a console script with two subcommands. `uvx vision-agents init <name>` scaffolds a new agent project (`pyproject.toml`, `agent.py`, `.env.example`, `.gitignore`,
`README.md`) and runs `uv sync` to provision a venv (skip with `--no-install`). `vision-agents agent run|serve` reads `[tool.vision-agents.agent].entrypoint` (gunicorn-style
`module:attribute`, e.g. `"agent:runner"`) from the project's `pyproject.toml`, imports it in-process and dispatches to its
`Runner.cli()`. No subprocess, no uv dependency at runtime — just install the CLI in the same env as the project (typically via
`uv run` or an activated venv). Templates are rendered with Jinja2 and default to a `gemini.Realtime` agent driven through `agent.simple_response(...)` to match the quickstart.

### New high-level agent events (#552)

New lifecycle events on the agent's own bus, suitable for application code (versus the plugin-level events that come and go with internals):

- `AgentJoinedCallEvent`, `AgentLeftCallEvent` — agent call membership.
- `UserTurnStartedEvent`, `UserTurnEndedEvent` — user speech turn boundaries.
- `UserTranscriptEvent` — final user transcript that triggers an LLM turn.
- `AgentTurnStartedEvent`, `AgentTurnEndedEvent` — agent speech turn boundaries (`interrupted` flag on the end event for barge-in).

Also adds plugin-side `STTConnectedEvent` / `STTDisconnectedEvent`, `TTSConnectedEvent` / `TTSDisconnectedEvent` / `TTSErrorEvent`, and
`LLMResponseFinalEvent` to replace the connection-state and realtime-response events removed above.

## Bug Fixes

- **Optional `redis` extra**: importing `vision_agents.core` no longer emits a `UserWarning` when the
  `redis` package is absent. The warning was noise for the majority of users who don't use `RedisSessionKVStore`; instead, attempting to import
  `vision_agents.core.agents.session_registry.redis_store` directly raises a `ModuleNotFoundError` with an actionable install hint ("
  `pip install 'vision-agents[redis]'`"), matching the FastAPI optional-extra pattern. (#562)
- **AWS Realtime**: stop hanging on disconnect — the receive loop now exits cleanly when the upstream event stream returns
  `None` instead of crashing on the next field access. (#569)
- **Gemini Realtime**: fix missing user-speech start/end signals. The previous implementation listened on Gemini Live's `voice_activity_detection_signal` /
  `voice_activity` channels, which are allowlist-only and silent for most accounts, so `RealtimeUserSpeechStarted` /
  `RealtimeUserSpeechEnded` never fired. Boundaries are now derived from `input_transcription` (start) and `model_turn` /
  `tool_call` (end), and the per-turn flags reset on session re-establishment so a mid-utterance reconnect doesn't inherit stale state. (#569)
- **OpenAI Realtime**: fix `RealtimeUserSpeechStarted` / `RealtimeUserSpeechEnded` not being emitted when audio is pushed via
  `simple_audio_response()` — the current participant is now stored on that path. Also switches the user-turn-end signal from
  `input_audio_buffer.speech_stopped` (server_vad-only) to `input_audio_buffer.committed`, so semantic-VAD sessions also see the end-of-turn event. (#569)
- **LiveAvatar**: include the plugin README in the PyPI package; fix casing in the heygen plugin README and link to the LiveAvatar PyPI page. (#564)

# v0.6.0

## Breaking Changes

### Agent inference pipeline rewrite (#501)

Large internal refactor of the agent's audio/LLM/TTS pipeline (~10k LOC across ~195 files). The orchestration that used to live inside `Agent` is now an explicit
`InferenceFlow` abstraction with two implementations — `TranscribingInferenceFlow` (STT → LLM → TTS) and `RealtimeInferenceFlow` (speech-to-speech).

Data flow between components is driven by a new generic `Stream[T]` primitive (with backpressure, `clear()` and `close()`) instead of plugin events:
`LLM.simple_response` now returns an `AsyncIterator[LLMResponseDelta | LLMResponseFinal]`, and `STT`, `TTS`, `TurnDetector`, and `Realtime` expose
`.output` streams that consumers iterate. The `MetricsCollector` was rewritten to a direct
`on_*()` API with parent/child merging so plugin metrics fan in to the agent root and OpenTelemetry is written exactly once. Turn lifecycle is now managed by a dedicated
`LLMTurn` class with explicit `confirm` / `cancel` / `finalize` steps.

#### Migration

`agent.llm.simple_response(...)` → `agent.simple_response(...)`:

```python
# Before
await agent.llm.simple_response("greet the user")

# After
await agent.simple_response("greet the user")
```

`LLM.simple_response` is now an async iterator, not event-driven:

```python
# Before — listen for chunks/completed events
@agent.events.subscribe
async def on_chunk(event: LLMResponseChunkEvent): ...

@agent.events.subscribe
async def on_done(event: LLMResponseCompletedEvent): ...

# After — iterate the response stream
async for item in llm.simple_response("hello"):
    if isinstance(item, LLMResponseDelta):
        print(item.delta, end="")
    elif isinstance(item, LLMResponseFinal):
        print("\n[done]", item.text)
```

`streaming_tts` is gone — TTS always streams sentence-by-sentence via the new `TTSSentenceTokenizer`:

```python
# Before
agent = Agent(..., streaming_tts=True)

# After — remove the flag
agent = Agent(...)
```

Turn-detection events on the bus are replaced by `TurnStarted` / `TurnEnded` items on the STT / `TurnDetector` output stream (`TurnEvent` enum removed).

#### Other breakages

- `Agent.simple_audio_response()` removed; speech-to-speech is now handled by `RealtimeInferenceFlow`.
- `TTSErrorEvent` and `TTSConnectionEvent` removed; TTS errors are surfaced through metrics (`MetricsCollector.on_tts_error`).
- `MetricsCollector` no longer subscribes to plugin events — plugins call `metrics.on_*()` directly. Custom metric consumers that listened for
  `LLMResponseCompletedEvent` etc. must read from `agent.metrics` or merge into the collector tree.
- Plugin author API: `TTS.send()` removed, replaced by `TTS.send_iter(stream)` which consumes a `Stream[TTSInput | TTSInputEnd]` and produces a `Stream[TTSOutputChunk]`.
  `set_output_format()` is also gone — output format is fixed by the inference flow.
- Plugin-specific event types removed from `aws`, `gemini`, `huggingface`, `nvidia`, `openai`, and `xai`. Data that used to ride on those events now flows through the standard
  `LLMResponseDelta` / `LLMResponseFinal` stream items.

### Avatar plugin API (#534)

Avatars are now a first-class plugin type with their own kwarg on `Agent(...)` rather than being passed through the generic
`processors=[...]` list. The existing publisher classes have been removed in favour of an `Avatar` symbol exported from each plugin.

- `anam.AnamAvatarPublisher` removed. Use `anam.Avatar`.
- `lemonslice.LemonSliceAvatarPublisher` removed. Use `lemonslice.Avatar`.
- `liveavatar.Avatar` is new in this release and uses the new API (see New Features below).
- `heygen.AvatarPublisher` is unchanged — the `heygen` plugin keeps the legacy `processors=[...]` wiring for this release and is scheduled for removal (#553). New code should use
  `liveavatar.Avatar` instead.
- `vision_agents.core.utils.av_synchronizer` moved to `vision_agents.core.avatars.av_synchronizer` (also re-exported from `vision_agents.core.avatars`).

Migration:

```python
# Before
from vision_agents.plugins.anam import AnamAvatarPublisher

agent = Agent(..., processors=[AnamAvatarPublisher(...)])

# After
from vision_agents.plugins import anam

agent = Agent(..., avatar=anam.Avatar(...))
```

### EventManager (#555)

`core.events.EventManager` was simplified: events now dispatch immediately on `send()` (one
`asyncio.Task` per matching handler) instead of going through an internal deque and a background
`_process_events_loop`. This eliminates the "task was destroyed but is pending" warnings that appeared when an `EventManager` was garbage-collected.

- `EventManager.stop()` removed. Use `await EventManager.shutdown()` for teardown.
- `send()` calls after `shutdown()` are silently dropped.
- Tests that relied on observing the internal queue should switch to `await events.wait()`.

### ElevenLabs STT class rename (#559)

The internal class `STT` was renamed to `ElevenlabsSTT` to make it searchable. The public import path is unchanged —
`from vision_agents.plugins.elevenlabs import STT` still works via re-export.

## New Features

### Public API on `Agent` (#501)

| Member                                                                | Description                                                                                                                                   |
|-----------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------|
| `await agent.simple_response(text, participant=None, interrupt=True)` | Inject an instruction for the LLM to respond to. Replaces `agent.llm.simple_response(...)`. Routed through the active `InferenceFlow`.        |
| `await agent.say(text, interrupt=False)`                              | Speak text directly through TTS, bypassing the LLM. New `interrupt` flag preempts an in-flight turn and clears the TTS pipeline.              |
| `agent.resolve_inference_flow() -> InferenceFlow`                     | Extension point: picks `RealtimeInferenceFlow` for `Realtime` LLMs, otherwise `TranscribingInferenceFlow`. Override to plug in a custom flow. |

### Public utilities under `vision_agents.core` (#501)

- `core.utils.stream.Stream[T]` — generic async producer/consumer queue with `send` / `get` / `clear` / `close` / `collect(timeout)` / `peek()`.
- `core.utils.tokenizer.TTSSentenceTokenizer` — sentence chunker used to feed TTS incrementally.
- `core.agents.inference` — `InferenceFlow`, `TranscribingInferenceFlow`, `RealtimeInferenceFlow`, `AudioInputStream`, `AudioOutputStream`, `AudioInputChunk`, `AudioOutputChunk`,
  `AudioOutputFlush`.
- `core.observability.MetricsCollector.merge(child)` — hierarchical metrics merging so child collectors forward to a parent without duplicating OTel writes.

### LLM interruption API (#501)

`await llm.interrupt()` is now part of the public
`LLM` contract. The active inference flow calls it to preempt an in-flight response (e.g. on a barge-in); plugin authors must implement it.

### Dedicated Avatar plugin type (#534)

`Agent(...)` accepts a new
`avatar=` keyword argument. The avatar owns the agent's outbound video/audio tracks, and the inference flow's audio output is routed through the avatar provider for lipsync. Plugin authors can build new integrations by subclassing
`vision_agents.core.avatars.Avatar`.

```python
from vision_agents.plugins import anam

agent = Agent(..., avatar=anam.Avatar(...))
```

### LiveAvatar plugin (#534)

New first-party avatar plugin:
`vision_agents.plugins.liveavatar.Avatar` — LiveAvatar (by HeyGen) integration over WebRTC + websocket. Recommended replacement for the now-deprecated `heygen` plugin.

Install with: `uv add "vision-agents[liveavatar]"`

### HeyGen plugin scheduled for removal (#553)

The `heygen` plugin will be removed in a future release in favour of `liveavatar`.

### Tencent TRTC plugin (#386)

New
`vision_agents.plugins.tencent` edge transport for Tencent TRTC (WebRTC). Wraps Tencent's manylinux-only
`liteav` SDK and is gated by `sys_platform == 'linux'` in both `vision-agents[tencent]` and `vision-agents[all-plugins]` — non-Linux installs silently skip it.

Install with: `uv add "vision-agents[tencent]"`

Required env vars: `TENCENT_SDK_APP_ID`, `TENCENT_SDK_SECRET_KEY`. Optional: `TENCENT_TRTC_SCENE`, `TENCENT_LITEAV_DEBUG`.

## Bug Fixes

- **Agent**: fix audio output queue falling behind when the main thread blocks longer than 20 ms.
  `_poll_audio_queues` now drains all available audio per cycle instead of consuming at a fixed 20 ms cadence. (#558)
- **ElevenLabs STT**: fix duplicated `TurnEnded` event. (#559)
- **ElevenLabs STT**: fix websocket keep-alive — the connection used to close after a few seconds of silence; now sends 2s of silence every 5s to keep it open. (#559)

# v0.5.9

## Bug Fixes

- **Anam**: fix avatar publisher settings. (#556)

# v0.5.8

## Breaking Changes

- **OpenAI Realtime**: default model bumped to `gpt-realtime-2`. (#537)

## Bug Fixes

- **Inworld TTS**: stop poisoning the receive stream with stale-context errors after >60s of inactivity. (#539)

# v0.5.7

## New Features

- **AWS**: new `aws.STT` (Amazon Transcribe streaming) plugin. (#528)
- **Inworld TTS**: switch from AV/HTTP to bidirectional WebSocket streaming (replaces the `av` dependency with `websockets`). (#532)

# v0.5.6

## Bug Fixes

- **Inworld TTS**: force `LINEAR16` audio encoding — MP3 default broke replies that span multiple stream chunks; also makes `inworld-tts-2` the default model. (#531)
- **MCP**: own MCP transport lifecycle in a single supervisor task to fix event-loop peg on `_deliver_cancellation`. (#529)

# v0.5.5

## New Features

- **Inworld**: new `inworld.Realtime` (WebRTC) plugin — low-latency speech-to-speech with function calling, turn detection, and multi-provider upstream models via
  `<provider>/<model>` IDs. (#502)
- **xAI**: new `xai.TTS` (Grok TTS) with `Voice` and `VOICE_DESCRIPTIONS` exports. (#433)
- **LLM**: max tool-calling rounds is now configurable across LLM plugins. (#500)

## Bug Fixes

- **xAI Realtime**: default model updated to `grok-voice-think-fast-1.0`; model now passed as `?model=` query param; voice IDs lowercased to match docs. (#515)
- **Gemini**: route MCP tool schemas through `parameters_json_schema` so `$schema` and `additionalProperties` round-trip correctly. (#513)

# v0.5.4

## Bug Fixes

- **Gemini Realtime**: fix infinite error loop on network disconnection. (#490)

# v0.5.3

## New Features

- **Sarvam**: new `sarvam.LLM`, `sarvam.STT`, `sarvam.TTS` plugin. (#488)

## Bug Fixes

- **Sarvam STT**: fix transcript ordering; support new WebSocket URL structures for translation/regular endpoints; speaker compatibility checks in TTS. (#489)

# v0.5.2

## New Features

- **Decart**: update to SDK 0.0.29 with default model Lucy 2. `RestylingProcessor` now accepts an initial reference image and supports atomic
  `update_state` of prompt + image for virtual try-on. (#446)

## Bug Fixes

- **Agent**: fix "dict changed size during iteration" in `_poll_audio_queues`. (#487)
- **AWS Realtime**: fix reconnect. (#486)
- **StreamEdge**: fix track identification when multiple anonymous candidates exist. (#446)

# v0.5.1

## New Features

- **OpenAI LLM**: `model` parameter is now optional and defaults to `gpt-5.4`. (#470)
- **Core**: new `AVSynchronizer` to sync audio/video playback for avatars (used by `anam` and `lemonslice`). (#466)
- **HuggingFace**: refactor tool calling — enforce max function-calling rounds, fix silently skipped tool calls with missing IDs, respect non-streaming mode. (#455)

## Bug Fixes

- **Agent**: fix `Task exception was never retrieved` in `_poll_audio_queues` on close. (#473)
- **Agent**: prevent OOM from unconsumed video frames in voice-only agents. (#458)
- **Gemini / OpenAI Realtime**: fix turn detection, support interruptions, add `interrupted` field to `RealtimeAudioOutputDoneEvent`. (#470)

# v0.5.0

## New Features

- **Anam**: new `anam.Avatar` plugin. (#445)

## Bug Fixes

- **Resources**: close `AsyncStream`, ElevenLabs TTS httpx client, and Deepgram STT httpx client on session end to fix per-session TCP connection leaks. (#457)

# v0.4.7

## Bug Fixes

- **Deepgram**: pin `deepgram-sdk` to `<6.1.0`. (#456)

# v0.4.6

## Bug Fixes

- **Deepgram STT**: reduce latency. (#451)

# v0.4.5

## New Features

- **AWS**: `aws_profile` parameter on `BedrockLLM` and `aws.Realtime` for profile / SSO / instance-profile auth. (#415)
- **CLI**: splash screen with current core version; hidden in non-interactive terminals or via `--no-splash`. (#447, #449)

## Bug Fixes

- **Agent**: fix memory leak from stringifying numpy arrays in event logging. (#444)
- **Gemini Realtime**: non-blocking tool execution. (#437)
- **AWS Realtime**: track background tool tasks. (#438)
- **OpenAI / xAI Realtime**: non-blocking tool execution. (#439)
- **OpenRouter**: fix function calling. (#442)

# v0.4.4

## New Features

- **Local**: new `local.LocalEdge`, `local.LocalCall`, and device classes for running agents against local audio/video devices. (#347)

## Bug Fixes

- **AWS Realtime**: emit transcription events and handle barge-in for Nova Sonic. (#408)
- **EventManager**: fix `shutdown()` hanging forever. (#421)
- **Gemini Realtime**: refactor event handling — fix skipping parts in multipart replies, add VAD config to reduce latency. (#436)
- **ElevenLabs STT**: switch to VAD mode instead of manual commits. (#435)
- **TTS**: use epoch tracking to skip stale TTS events on turn change. (#430)
- **Inworld TTS**: add `X-User-Agent` and `X-Request-Id` headers. (#428)

# v0.4.3

## New Features

- **Fish Audio**: update to S2-Pro model. (#405)

## Bug Fixes

- **StreamEdge**: fix connection race condition. (#412)
- **Agent**: fix memory leak by cleaning up handler tasks and closures on close. (#407)

# v0.4.2

## New Features

- **AssemblyAI**: add diarisation support for `assemblyai.STT`. (#394)
- **HuggingFace**: new Transformers-based detection processor; LLM/VLM refactor. (#377)

## Bug Fixes

- **Agent**: buffer realtime transcripts into single chat messages. (#383)
- **Agent**: fix eager turn detection updating transcripts out of order. (#401)
- **LLMJudge**: set instructions once at init. (#399)

# v0.4.1

## New Features

- **AssemblyAI**: add support for streaming STT. (#389)

# v0.4.0

## Breaking Changes

### Built-in HTTP server

#### API Endpoints

All session endpoints now include `call_id` as a path parameter:

| Before                               | After                                                |
|--------------------------------------|------------------------------------------------------|
| `POST /sessions`                     | `POST /calls/{call_id}/sessions`                     |
| `DELETE /sessions/{session_id}`      | `DELETE /calls/{call_id}/sessions/{session_id}`      |
| `POST /sessions/{session_id}/close`  | `POST /calls/{call_id}/sessions/{session_id}/close`  |
| `GET /sessions/{session_id}`         | `GET /calls/{call_id}/sessions/{session_id}`         |
| `GET /sessions/{session_id}/metrics` | `GET /calls/{call_id}/sessions/{session_id}/metrics` |

#### Request Body

- `call_id` removed from `POST /sessions` request body — now a URL path parameter

#### Response Codes

- `DELETE /calls/{call_id}/sessions/{session_id}` now returns **202 Accepted** (was 204)
- `POST /calls/{call_id}/sessions/{session_id}/close` now returns **202 Accepted** (was 200)
- Session closure is now asynchronous — the owning node processes the close request on its next maintenance cycle

#### ServeOptions

- `get_current_user` option removed
- Permission callbacks (`can_start_session`, `can_close_session`, `can_view_session`, `can_view_metrics`) now receive `call_id: str` as a parameter

#### Removed Dependencies

- `vision_agents.core.runner.http.dependencies.get_session` removed
- `vision_agents.core.runner.http.dependencies.get_current_user` removed

### FunctionRegistry

- `register_function()` now rejects synchronous functions with `ValueError` — all registered functions must be async (#373)
- `FunctionRegistry.call_function()` and `LLM.call_function()` are now `async` (#373)

### Agent authentication

- `Agent.create_user()` renamed to `Agent.authenticate()` (#380)
- `EdgeTransport.create_user()` renamed to `EdgeTransport.authenticate()` (#380)
- Authentication is now called automatically during `Agent.start()` — manual calls are no longer needed (#380)

### Testing

- `mock_tools()` and standalone `mock_functions()` removed from `vision_agents.testing` (#376)
- Use `TestSession.mock_functions()` instead

### AgentLauncher

- `cleanup_interval` parameter renamed to `maintenance_interval` (#374)
- `created_by` parameter removed from `start_session()` (#374)
- `AgentSession.created_by` field removed (#374)
- `call_id` values must match `^[a-z0-9_-]+$` (raises `InvalidCallId`) (#374)
- New parameter: `registry: SessionRegistry | None = None` (#374)

## New Features

### Redis-backed Agent session registry for horizontal scaling

Sessions are shared across nodes via Redis, enabling cross-node session queries and closure without sticky sessions. (#374)

```python
from vision_agents.core import AgentLauncher, Runner
from vision_agents.core.agents.session_registry import RedisSessionKVStore, SessionRegistry

store = RedisSessionKVStore(url="redis://localhost:6379")
registry = SessionRegistry(store=store)
runner = Runner(AgentLauncher(create_agent=create_agent, join_call=join_call, registry=registry))
```

Install with: `uv add "vision-agents[redis]"`

**New public API on AgentLauncher:**

| Method                                       | Description                            |
|----------------------------------------------|----------------------------------------|
| `get_session_info(call_id, session_id)`      | Query session info from shared storage |
| `request_close_session(call_id, session_id)` | Request closure from any node          |

**Custom store backends:** `SessionKVStore` is an abstract class that can be subclassed to support any TTL key-value store (DynamoDB, Memcached, etcd, etc.).

### PEP 561 compliance

`py.typed` markers added to `vision_agents.core` and `vision_agents.testing` for downstream type checking support. (#378)

## Bug Fixes

- **EventManager**: fix crash when event handlers have return type annotations (#381)
- **RedisSessionKVStore**: fix import error when `redis` package is not installed (#384)
- **Agent metrics**: fix metrics storage and serialization in session registry (#387)
