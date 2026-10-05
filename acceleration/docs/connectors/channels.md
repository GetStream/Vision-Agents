# Connectors: inbound channels, tools and the omni-channel conversation

Oct 2, 2026 · @Kanat Kiialbaev

Exported from Claude Docs on 2026-10-06 (https://claude.ai/code/artifact/b2be31d9-a1ea-44f4-a413-b8c65adf46f6). The Claude Doc is the source of truth; this copy is a snapshot.

One agent answers a person on SMS, WhatsApp, Slack, iMessage and phone calls. The agent keeps one history for each person. Part 1 shows the design and the end-to-end flows. Part 2 holds the other parts of the design and the details: step tables, code references, rules and open questions.

The section [Terms](#m729fz3d1s3.24616) defines each term and its name in the code.

# Part 1 · Overview

## Summary

1. **What we build.** One agent answers a person on SMS, WhatsApp, Slack, iMessage and phone calls. Inbound channels bring messages to the agent. Tools let the agent act in other services.
2. **Channel bridge in the Router.** It receives the provider webhook. It writes each external thread word for word into its own thread channel. The message hook that exists today wakes the agent. The bridge sends the reply back to the same thread.
3. **Omni-channel with episode cards.** The omni-channel is the person's agent channel. It keeps one episode card for each call and each text thread. When the episode ends, the Router writes a summary into the card.
4. **Connectors serve both sides.** One connector layer serves tools and inbound channels: `store.ConnectorConnection`, `core.Resolver`, `core.Scheme`, `core.Verifier`, the proxy and token export. Tools use the `sources` block of `core.Manifest`. Inbound channels use the new `channel` block. An inbound channel is never a kind of tool.
5. **One provider unit for each customer.** Each customer has its own Slack app, WhatsApp business account, Telegram bot or SMS account.
6. **Flexible integration.** A customer uses the whole platform or only some parts: tokens, events, the proxy or our agent. This is the same [flexibility](#m729fz3d1s3.112218) as Vercel Connect.
7. **Open questions.** Thierry has not agreed to the design yet. How to join one person across channels is not decided. See Why this design.

## Flexibility: the customer chooses how much of the platform to use

The design gives the same flexibility as Vercel Connect. Vercel Connect gives customer code a token and forwards provider events ([tokens](https://vercel.com/docs/connect/concepts/tokens), [triggers](https://vercel.com/docs/connect/concepts/triggers)). The Router gives both. The Router can also run the agent, keep the history and send the replies.

&#91;embedded content: integration modes A, B, C and Vercel Connect · who runs each layer\]

The proxy implements no Slack method. It sends the request on unchanged, so a new Slack method needs no change in the Router.

Details: [Integration modes](#m729fz3d1s3.78034) · [Direct calls: the proxy](#m729fz3d1s3.84393) · [Direct calls: token export](#m729fz3d1s3.85357).

## The whole picture

The agent has two sides. On the left, inbound channels bring messages to the agent. On the right, tools let the agent act in other services. Under both sides, `store.ConnectorConnection` keeps the tokens.

&#91;embedded content: the agent · inbound channels, agent channel, tools, connections\]

An inbound channel is not a tool. An inbound channel starts a conversation. The LLM calls a tool inside a conversation.

## End to end: SMS, WhatsApp and a phone call

The diagram shows one person who sends an SMS, writes in WhatsApp and then calls. SMS and WhatsApp enter through the channel bridge in the Router. The phone call enters through the call hook: the number vendor bridges the call into Stream Video. One contact map in the Router turns the phone number into one omni-channel for the person. Each SMS thread, WhatsApp thread and call keeps its text in its own thread channel or call channel. The omni-channel gets one episode card for each of them.

&#91;embedded content: one person on SMS, WhatsApp and a phone call · one omni-channel\]

Details: [End to end: where each step runs](#m729fz3d1s3.109197) · [How the agent knows it is the same person](#m729fz3d1s3.51172).

### Step by step: one SMS

The diagram shows every call between services for one SMS and its reply. Read it from top to bottom. The Stream Chat API has two columns. The omni-channel, the person's agent channel, gets the episode card. The thread channel of this SMS thread gets the messages word for word. Steps 18–21 close the episode after an idle period.

&#91;embedded content: one SMS · 21 steps, 3 webhooks\]

Details: [One SMS: every call](#m729fz3d1s3.109230).

### Step by step: how the Router gets a token

The channel bridge (step 15 of the SMS diagram) and every tool get a token the same way: through `core.Resolver`. The token stays in the Router.

&#91;embedded content: how the Router gets a token · 11 steps\]

Details: [How the Router gets a token: what exists](#m729fz3d1s3.109277).

### Step by step: one phone call

A phone call does not use the channel bridge, a connector or a tool. The person and the agent talk in a Stream Video call, and the agent speaks its replies. The omni-channel, the person's agent channel, gets one episode card for the call and later the summary. A call is one episode. The full transcript goes to the call channel: it is the thread channel of a call.

&#91;embedded content: one phone call · steps 1–9, then the episode card in the omni-channel (10–18)\]

Details: [One phone call: every call](#m729fz3d1s3.109250).

### Why the omni-channel gets summaries, not the raw text

**Decision:** the omni-channel gets one episode card with the summary for each episode. An episode is one call, or one run of messages on one external thread. The raw text stays in the call channel or the thread channel. The model gets a smaller context: the episode cards, the last messages of the current thread word for word, and memory facts. This is our proposal.

**Example.** On Monday the person has an 8-minute call and books a cleaning for Thursday at 15:00. On Tuesday the person sends an SMS: «can I move it to 4?». The model context has one line, «Call on Monday, 8 min: booked a cleaning, Thursday 15:00», and the SMS. The full transcript stays in the call channel for an operator or a dispute.

Details: [Episodes: the options and the episode card](#m729fz3d1s3.109318).

## Why this design, and an example

**This design is our proposal.** Thierry has not agreed to it yet.

Reasons:

- All inbound channels write episode cards to one omni-channel. Thus the agent has one shared history across channels.
- The Router keeps the bot token. The Router refreshes and revokes it the same way as tool tokens.
- The message hook exists today. The channel bridge needs no new entry into `session.Session`.

Botpress, Twilio Conversation Orchestrator, Intercom Fin and Chatwoot use the same pattern. A channel adapter in the platform writes each message into one shared conversation. The agent reads that conversation.

**Example: Athena in the Slack channel #support.** The `store.ConnectorConnection` owner is `app`, because several people are in the session (architecture doc, one-way door 7).

1. An admin clicks “Connect Slack” one time. The Router creates a `store.ConnectorConnection` row with the bot token (owner `app`, account = Slack team). If Slack MCP does not accept the bot token, the tool needs a second row with the user token ([`providers/slack.yaml`](https://github.com/GetStream/Vision-Agents/blob/ead4a273f4d3623fff2a2286d5422725aa0af2e2/acceleration/internal/connectors/providers/slack.yaml)).
2. A user writes in #support: «@athena what did we decide yesterday about the release?».
3. Slack sends an event. The channel bridge checks the signature. It writes the message to the thread channel of this Slack thread. The omni-channel gets an episode card.
4. The message hook receives the message and starts `session.Session`.
5. The agent calls the tool `slack.read_thread` with the user token from its own row. If Slack MCP accepts the bot token, the second row is not necessary (`unverified`).
6. `chatlog.Log` writes the reply to the thread channel. The channel bridge sends the reply to the Slack thread with the bot token.
7. The next day, the user opens Athena in the browser and speaks. The voice session reads the omni-channel of the user. The agent sees the episode card of the Slack thread from yesterday. This is omni-channel.

**To decide with Thierry:**

- [ ] Does he agree to the channel bridge in the Router? He asked «how do we want to approach this?» on October 1.
- [ ] Is the omni-channel with episode cards the required history for all inbound channels?
- [ ] How do we join one person from different inbound channels into one omni-channel?

Details: [Architecture changes](#m729fz3d1s3.42018) · [Cost of this design](#m729fz3d1s3.109598) · [How other companies do it](#m729fz3d1s3.72499).

# Part 2 · Details

This part holds the other parts of the design and the details: step tables, code references, rules and open questions. Each section of Part 1 links to its details here.

## End to end: where each step runs

| Step | Where |
| --- | --- |
| SMS and WhatsApp enter through | the channel bridge in the Router (new) |
| The phone call enters through | the call hook in the Router (exists) |
| The contact map lives in | the Router, one copy (new) |
| SMS and WhatsApp messages go to | the thread channel of that thread. One episode card goes to the omni-channel |
| The phone call goes to | an episode card in the person's omni-channel. The transcript stays in the call channel `agent:<call id>` |
| The agent sees SMS, WhatsApp and the call together | yes, as episode cards in the omni-channel |

## Where each part runs

The Router and the Stream Chat API are two services. The thread channel and the omni-channel live in the Stream Chat API. The channel bridge, the message hook and `session.Session` run in the Router. The numbers follow one Slack message and its reply.

&#91;embedded content: where each part runs · Router, Stream Chat API, Slack · 8 calls\]

The Router calls the Stream Chat API to write the episode card and the messages (steps 2, 3 and 6). The Stream Chat API calls the Router only through the message hook webhook (step 4). Slack calls the Router only through its Events API webhook (step 1).

## One SMS: every call

**How Stream Chat gives the Router a new message.** The Router subscribes to a Stream Chat webhook. At start, the Router adds its own URL to the event hooks of the Stream app, for the event `message.new` only (`PointMessageHook` → `UpdateApp`, [`internal/chat/hooks.go:26-32,80-116`](https://github.com/GetStream/Vision-Agents/blob/ead4a273f4d3623fff2a2286d5422725aa0af2e2/acceleration/internal/chat/hooks.go#L26-L32)). Stream Chat then sends every new message of the app to `/v1/chat/hooks/stream`. The Router checks the signature and keeps only messages in an agent channel, with text and without `source` ([`internal/api/messagehooks.go:61-128`](https://github.com/GetStream/Vision-Agents/blob/ead4a273f4d3623fff2a2286d5422725aa0af2e2/acceleration/internal/api/messagehooks.go#L61-L128)).

**The reply to the SMS is not a tool call.** The LLM does not decide to send it. The channel bridge sends it, and it uses the connector layer (`core.Resolver`, `core.Scheme`) for the vendor credential. A tool is only for an action that the LLM chooses, for example «read a Slack thread».

| # | Who calls whom | How | In code? | Source |
| --- | --- | --- | --- | --- |
| 1 | Person → SMS vendor | SMS | — | — |
| 2 | SMS vendor → channel bridge | webhook, signed by the vendor | no | design |
| 3 | channel bridge | `core.Verifier` checks the signature. The contact map gives the omni-channel. The thread link gives the thread channel | no | proposal |
| 4 | channel bridge → Stream Chat API | first message of an episode only: `SendMessage` of an episode card to the omni-channel, `source: sms`, `status: in_progress`. The message hook ignores its `message.new`, because the card has `source` | no | proposal |
| 5 | channel bridge → Stream Chat API | `SendMessage` to the thread channel as the person, without `source`. `chatlog.Log` uses the same API | no | [`internal/chatlog/chatlog.go:621-650`](https://github.com/GetStream/Vision-Agents/blob/ead4a273f4d3623fff2a2286d5422725aa0af2e2/acceleration/internal/chatlog/chatlog.go#L621-L650) |
| 6 | Stream Chat → message hook | webhook `message.new` | yes | [`internal/api/messagehooks.go:61`](https://github.com/GetStream/Vision-Agents/blob/ead4a273f4d3623fff2a2286d5422725aa0af2e2/acceleration/internal/api/messagehooks.go#L61) |
| 7 | message hook | `addressed()`: channel type `agent`, text, no `source`. A thread channel is of type `agent` | yes | [`internal/api/messagehooks.go:113-128`](https://github.com/GetStream/Vision-Agents/blob/ead4a273f4d3623fff2a2286d5422725aa0af2e2/acceleration/internal/api/messagehooks.go#L113-L128) |
| 8 | message hook → `session.Session` | a running session: `Session.Ask`. No running session: `dispatch.AssignMessage` gives the message to a worker, and the worker starts a session. The thread channel is the conversation of the session | yes | [`internal/api/messagehooks.go:131-190,292-300`](https://github.com/GetStream/Vision-Agents/blob/ead4a273f4d3623fff2a2286d5422725aa0af2e2/acceleration/internal/api/messagehooks.go#L131-L190) |
| 9 | `session.Session` → Stream Chat API | reads the episode cards of the person in the omni-channel as extra context | no | proposal |
| 10 | `session.Session` | the LLM writes the answer. It can call tools | yes | — |
| 11 | `session.Session` → Stream Chat API | `Log.Reply` sends the answer to the thread channel as a new message with `source: agent` | yes | [`internal/session/session.go:496-497`](https://github.com/GetStream/Vision-Agents/blob/ead4a273f4d3623fff2a2286d5422725aa0af2e2/acceleration/internal/session/session.go#L496-L497), [`internal/chatlog/chatlog.go:298-303`](https://github.com/GetStream/Vision-Agents/blob/ead4a273f4d3623fff2a2286d5422725aa0af2e2/acceleration/internal/chatlog/chatlog.go#L298-L303) |
| 12 | Stream Chat → message hook | webhook `message.new` for the reply | yes | same as step 6 |
| 13 | message hook | the reply has `source`, so it is not a question. The Router does not answer it. This stops the loop | yes | [`internal/api/messagehooks.go:113-128`](https://github.com/GetStream/Vision-Agents/blob/ead4a273f4d3623fff2a2286d5422725aa0af2e2/acceleration/internal/api/messagehooks.go#L113-L128) |
| 14 | message hook → channel bridge | the thread channel is linked to an SMS thread, so the hook gives the reply to the outbound half | no | proposal |
| 15 | channel bridge | `core.Resolver` gives the vendor credential. `core.Scheme` adds it to the request | interfaces only | [`internal/connectors/core/resolver.go:17`](https://github.com/GetStream/Vision-Agents/blob/ead4a273f4d3623fff2a2286d5422725aa0af2e2/acceleration/internal/connectors/core/resolver.go#L17) |
| 16 | channel bridge → SMS vendor | the vendor's send API | no | vendor docs, `unverified` |
| 17 | SMS vendor → person | SMS | — | — |
| 18 | Router | no message on the thread for the idle period: the episode closes. The idle period is a setting | no | proposal |
| 19 | Router → Stream Chat API | `UpdateMessagePartial` on the card: `status: ended`. It sends no webhook | no | [`internal/chatlog/chatlog.go:551-556`](https://github.com/GetStream/Vision-Agents/blob/ead4a273f4d3623fff2a2286d5422725aa0af2e2/acceleration/internal/chatlog/chatlog.go#L551-L556) uses the same API |
| 20 | Router | the LLM writes the episode summary | no | proposal |
| 21 | Router → Stream Chat API | `UpdateMessagePartial`: `text` = summary, `status: summarized`. Then the facts go to memory | no | proposal |

**No dependency cycle.** Webhooks only come into the Router (steps 2, 6, 12). The Router calls Stream Chat (steps 4, 5, 9, 11, 19, 21) and the vendor (step 16). Stream Chat and the vendor never call each other. The runtime loop at step 12 stops at step 13.

## One phone call: every call

| # | Who calls whom | How | In code? | Source |
| --- | --- | --- | --- | --- |
| 1–2 | Person → vendor → Stream Video | the vendor bridges the call over SIP into a Stream inbound trunk. The caller becomes the participant `sip-{{caller_number}}` | yes | [`internal/phone/phone.go:14-17`](https://github.com/GetStream/Vision-Agents/blob/ead4a273f4d3623fff2a2286d5422725aa0af2e2/acceleration/internal/phone/phone.go#L14-L17), [`internal/phone/stream.go:21-25`](https://github.com/GetStream/Vision-Agents/blob/ead4a273f4d3623fff2a2286d5422725aa0af2e2/acceleration/internal/phone/stream.go#L21-L25) |
| 3 | Stream Video → call hook | webhook `call.session_started`. The Router asks for two events only: `call.session_started` and `call.session_ended` | yes | [`internal/phone/hooks.go:17-22`](https://github.com/GetStream/Vision-Agents/blob/ead4a273f4d3623fff2a2286d5422725aa0af2e2/acceleration/internal/phone/hooks.go#L17-L22), [`internal/api/callhooks.go:58,113`](https://github.com/GetStream/Vision-Agents/blob/ead4a273f4d3623fff2a2286d5422725aa0af2e2/acceleration/internal/api/callhooks.go#L58) |
| 4 | call hook → worker | dispatch sends the call to a worker connected to `/v1/dispatch`, with `caller_number` | yes | [`internal/api/dispatchws.go:376-384`](https://github.com/GetStream/Vision-Agents/blob/ead4a273f4d3623fff2a2286d5422725aa0af2e2/acceleration/internal/api/dispatchws.go#L376-L384) |
| 5 | worker | the contact map turns the number into a conversation id | no | proposal |
| 6 | worker → `session.Session` | `POST /v1/agents/sessions` with the call id and the conversation id. The API accepts `conversation_id` | the API yes, this use no | [`internal/api/generated.go:9384`](https://github.com/GetStream/Vision-Agents/blob/ead4a273f4d3623fff2a2286d5422725aa0af2e2/acceleration/internal/api/generated.go#L9384), [`internal/api/sessions.go:513`](https://github.com/GetStream/Vision-Agents/blob/ead4a273f4d3623fff2a2286d5422725aa0af2e2/acceleration/internal/api/sessions.go#L513) |
| 7 | `session.Session` → Stream Video | the session joins the call through `streamedge` | yes | [`cmd/router/main.go:910-916`](https://github.com/GetStream/Vision-Agents/blob/ead4a273f4d3623fff2a2286d5422725aa0af2e2/acceleration/cmd/router/main.go#L910-L916) |
| 8 | `session.Session` → omni-channel | read the omni-channel: the episode cards of the person | no for voice | [`internal/session/manager.go:229-231`](https://github.com/GetStream/Vision-Agents/blob/ead4a273f4d3623fff2a2286d5422725aa0af2e2/acceleration/internal/session/manager.go#L229-L231) |
| 9 | person ↔ `session.Session` | audio both ways through SIP and Stream Video | yes | — |

**How the call is stored.** When the session starts, the Router writes one episode card with `source: call` into the omni-channel, the person's agent channel (step 10). `chatlog.Log` writes the phrases into the call channel `agent:<call id>`, as it does today (step 12). When the call ends, the Router updates the card: first the status, then the summary (steps 15–17). The card has a link to the call channel. Then the Router writes the facts into memory (step 18). Steps 5, 8, 10 and 15–18 are new. The section «The episode card» below defines the card.

**Where the call transcript lives today.** In two places:

- **Stream Chat:** one channel of type `agent` for each call, id = the call id, written live by `chatlog.Log`. If the worker sets `agent_id` or `conversation_id`, the channel has that id instead ([`cmd/router/main.go:918-926`](https://github.com/GetStream/Vision-Agents/blob/ead4a273f4d3623fff2a2286d5422725aa0af2e2/acceleration/cmd/router/main.go#L918-L926), [`internal/session/spec.go:87-88,343-344`](https://github.com/GetStream/Vision-Agents/blob/ead4a273f4d3623fff2a2286d5422725aa0af2e2/acceleration/internal/session/spec.go#L87-L88)). What the workers pass today: `unverified`.
- **Postgres:** the Router records sessions, turns (`agent_responses`) and their items (`agent_response_items`, with the text of what the person said and what the agent answered) in the background ([`internal/session/records.go:79-88`](https://github.com/GetStream/Vision-Agents/blob/ead4a273f4d3623fff2a2286d5422725aa0af2e2/acceleration/internal/session/records.go#L79-L88), [`internal/store/models.go:1235-1295`](https://github.com/GetStream/Vision-Agents/blob/ead4a273f4d3623fff2a2286d5422725aa0af2e2/acceleration/internal/store/models.go#L1235-L1295)). An incognito session is not recorded.

Thus the call channel already exists today. What is missing is the omni-channel, the episode card and the link between them.

The reply does not go back through a connector or a tool. The agent speaks it in the call.

## How the Router gets a token: what exists

What exists at [`ead4a273`](https://github.com/GetStream/Vision-Agents/commit/ead4a273f4d3623fff2a2286d5422725aa0af2e2): `store.ConnectorConnection` with a sealed grant, and `core.Scheme` with `oauth2code` (`Retrieve`, `Wrap`). `core.Resolver` and its `CredentialStore` are interfaces only ([`internal/connectors/core/resolver.go:15-41`](https://github.com/GetStream/Vision-Agents/blob/ead4a273f4d3623fff2a2286d5422725aa0af2e2/acceleration/internal/connectors/core/resolver.go#L15-L41)). The resolver cache in step 2 is design: `Invalidate` «drops anything cached», but no cache exists yet.

## Episodes: the options and the episode card

**What the code does today.**

- `chatlog.Log` writes every phrase of a call into the agent channel with `source: speech` ([`internal/chatlog/chatlog.go:239-270`](https://github.com/GetStream/Vision-Agents/blob/ead4a273f4d3623fff2a2286d5422725aa0af2e2/acceleration/internal/chatlog/chatlog.go#L239-L270)). Thus SMS and WhatsApp in the same agent channel mix with the raw call transcript.
- The Router reads history back 200 messages at a time ([`internal/chatlog/reader.go:13-16`](https://github.com/GetStream/Vision-Agents/blob/ead4a273f4d3623fff2a2286d5422725aa0af2e2/acceleration/internal/chatlog/reader.go#L13-L16)). A session can start with `ContextTruncated` ([`internal/session/manager.go:246`](https://github.com/GetStream/Vision-Agents/blob/ead4a273f4d3623fff2a2286d5422725aa0af2e2/acceleration/internal/session/manager.go#L246)). One long call, or many short messages on several threads, can fill this window and push older episodes out. The number of lines in a typical call is `unverified`.
- A memory package keeps facts between conversations: «A call ends and its history goes with it. A memory store keeps the facts worth carrying forward» ([`internal/memory/memory.go:1-5`](https://github.com/GetStream/Vision-Agents/blob/ead4a273f4d3623fff2a2286d5422725aa0af2e2/acceleration/internal/memory/memory.go#L1-L5)). It is scoped by `memory.Scope.UserID`, so it needs the same person key as the contact map.

|  | Raw only: all text in one timeline | Summary only | Episode cards in the omni-channel, raw text in thread and call channels (chosen) |
| --- | --- | --- | --- |
| What the model gets | every line | 3–5 lines for each episode | the current thread word for word, episode summaries, memory facts |
| Exact quotes («you said 3 pm») | yes | no | yes, from the raw record |
| Context size and cost | large: speech-to-text errors, fillers, interruptions | small | small |
| Handoff to a human and audit | full | lost, unless stored elsewhere | full |
| Wrong detail in the summary | no risk | risk | risk, but the raw record is next to it |
| Timing | instant | needs an LLM pass after each episode. An SMS 10 seconds after a call can arrive before the summary | the same. Until the summary is ready, use the last raw lines |
| New code | none | summarizer and an episode-end trigger | summarizer, episode-end trigger, context builder |

#### The episode card

Each episode has one card in the omni-channel, the person's agent channel. An episode is one call, or one run of messages on one external thread. A text episode ends after an idle period. The length of this period is a setting. The Router creates the card at the start of the episode and updates it two times. The card stays one message, so an episode takes one place in the 200-message history window. This is our proposal. It is not in the code.

**Fields of the card.**

| Field | Value | Example |
| --- | --- | --- |
| `text` | empty while the episode goes on, then the summary | «Booked a cleaning, Thursday 15:00» |
| `source` | a new value for each kind: `call`, `sms`, `whatsapp`, `slack`, `imessage`. The message hook ignores every message with `source` ([`internal/api/messagehooks.go:113-128`](https://github.com/GetStream/Vision-Agents/blob/ead4a273f4d3623fff2a2286d5422725aa0af2e2/acceleration/internal/api/messagehooks.go#L113-L128)) | `call` |
| `call_id` | the Stream Video call id. Only for a call | `abc123` |
| `status` | `in_progress` → `ended` → `summarized`, or `summary_failed` | `summarized` |
| `started_at`, `ended_at` | the first and the last activity. For a call: `call.session_started` and `call.session_ended` | `2026-10-05T10:02:00Z` |
| `thread_channel` | the channel with the raw text: the call channel or the thread channel | `agent:abc123` |

**Who writes the card and when.**

| When | Who | Stream Chat API call | Webhook to the Router? |
| --- | --- | --- | --- |
| The episode starts: a call session starts, or the first message on a thread arrives | Router | `SendMessage`: a new card, `status: in_progress` | yes, `message.new`. The hook ignores it because it has `source` |
| `call.session_ended` arrives, or the session closes. For text: the idle period ends | Router | `UpdateMessagePartial`: `status: ended`, `ended_at` | no. The Router asks only for `message.new` ([`internal/chat/hooks.go:32`](https://github.com/GetStream/Vision-Agents/blob/ead4a273f4d3623fff2a2286d5422725aa0af2e2/acceleration/internal/chat/hooks.go#L32)) |
| The LLM finishes the summary | Router | `UpdateMessagePartial`: `text` = summary, `status: summarized` | no |
| The summary fails | Router | `UpdateMessagePartial`: `status: summary_failed` | no |

`UpdateMessagePartial` exists in the code today: `chatlog.Log` uses it for the final text of a spoken reply ([`internal/chatlog/chatlog.go:551-556`](https://github.com/GetStream/Vision-Agents/blob/ead4a273f4d3623fff2a2286d5422725aa0af2e2/acceleration/internal/chatlog/chatlog.go#L551-L556)).

**How the agent reads the card.**

- `status: summarized`: the agent reads the summary.
- `status: in_progress` or `ended`: the summary is not ready. The agent reads the last lines of `thread_channel`. This solves the race when an SMS arrives right after the call.
- `status: summary_failed`: the agent reads the last lines of `thread_channel`.

Inside the same thread, the session uses the thread channel as its conversation. It reads the last messages of that thread word for word. The episode cards give the other episodes of the person. The in-app Stream Chat channel is a special case: it is itself the thread channel of in-app chat.

**Other work for the episode card**:

1. Write the facts of the episode into memory, scoped by the person from the contact map.
2. When the Router builds the context, it reads the card. It reads raw text only from the current thread, and from other episodes only in the cases above.

**Open question:** does the person in Stream Chat see the raw call transcript or only the summary card? This is a product decision. It does not change storage.

A voice session must learn to read the episode cards in the omni-channel. Today only a text session reads history ([`internal/session/manager.go:228-231`](https://github.com/GetStream/Vision-Agents/blob/ead4a273f4d3623fff2a2286d5422725aa0af2e2/acceleration/internal/session/manager.go#L228-L231)). The next section explains the contact map.

## How the agent knows it is the same person

**The key is the phone number.** SMS, WhatsApp and a phone call all come from a phone number. A new contact map turns this number into one conversation id. The conversation id names the omni-channel of the person. Each external thread also gets its own thread channel. This is our proposal. It is not in the code.

The steps for each inbound message or call:

1. Get the sender number from the inbound event.
2. Normalize the number to E.164 format, for example `+15550100`.
3. Look up the contact map: (customer, agent, number) → conversation id. If there is no row, create a new omni-channel and a new row.
4. Start or wake `session.Session` with this conversation id.
5. `session.Session` reads the last messages of the thread channel and the episode cards in the omni-channel.

**Where the number comes from.**

| Inbound channel | Where the Router gets the number | In code? | Source |
| --- | --- | --- | --- |
| Phone call | The SIP caller becomes the participant `sip-{{caller_number}}`. Dispatch gives `caller_number` to the worker. | yes | [`internal/phone/stream.go:21-25`](https://github.com/GetStream/Vision-Agents/blob/ead4a273f4d3623fff2a2286d5422725aa0af2e2/acceleration/internal/phone/stream.go#L21-L25), [`internal/dispatch/dispatch.go:59-60`](https://github.com/GetStream/Vision-Agents/blob/ead4a273f4d3623fff2a2286d5422725aa0af2e2/acceleration/internal/dispatch/dispatch.go#L59-L60), [`internal/api/dispatchws.go:381`](https://github.com/GetStream/Vision-Agents/blob/ead4a273f4d3623fff2a2286d5422725aa0af2e2/acceleration/internal/api/dispatchws.go#L381) |
| SMS | The sender field of the vendor webhook. | no | vendor docs, `unverified` |
| WhatsApp | The sender number in the provider webhook. Twilio adds the prefix `whatsapp:`. | no | vendor docs, `unverified` |

**What exists and what is new.**

| Part | In code today | Needed |
| --- | --- | --- |
| The conversation id selects the agent channel | yes. `chatlog.Log` writes to the channel from `ConversationID` ([`cmd/router/main.go:918-926`](https://github.com/GetStream/Vision-Agents/blob/ead4a273f4d3623fff2a2286d5422725aa0af2e2/acceleration/cmd/router/main.go#L918-L926)). Without it, the channel id is `spec.AgentID`, and for a call `AgentID` is the call id unless the caller sets `agent_id` ([`internal/session/spec.go:87-88,343-344`](https://github.com/GetStream/Vision-Agents/blob/ead4a273f4d3623fff2a2286d5422725aa0af2e2/acceleration/internal/session/spec.go#L87-L88)). Thus today each call has its own channel ([`internal/chatlog/chatlog.go:193-196`](https://github.com/GetStream/Vision-Agents/blob/ead4a273f4d3623fff2a2286d5422725aa0af2e2/acceleration/internal/chatlog/chatlog.go#L193-L196)) | set `ConversationID` from the contact map for every inbound channel |
| A text session reads the history | yes. A persistent text conversation loads the earlier messages ([`internal/session/manager.go:228-237`](https://github.com/GetStream/Vision-Agents/blob/ead4a273f4d3623fff2a2286d5422725aa0af2e2/acceleration/internal/session/manager.go#L228-L237)) | nothing |
| A voice session reads the history | no. The code says «persistent conversations require text mode» ([`internal/session/manager.go:229-231`](https://github.com/GetStream/Vision-Agents/blob/ead4a273f4d3623fff2a2286d5422725aa0af2e2/acceleration/internal/session/manager.go#L229-L231)) | new: a phone call must read the episode cards in the person's omni-channel |
| The call path sets `ConversationID` | no. The call hook, dispatch and phone code do not set it: `grep -rln ConversationID internal/api/callhooks.go internal/dispatch internal/phone` finds nothing, October 2 | new: the call path or the worker sets it from the contact map. The session API accepts `conversation_id` ([`internal/api/sessions.go:513`](https://github.com/GetStream/Vision-Agents/blob/ead4a273f4d3623fff2a2286d5422725aa0af2e2/acceleration/internal/api/sessions.go#L513)) |
| The contact map | no | new table: (customer, agent, E.164 number) → conversation id |

**Where the contact map lives.** One copy is in the Router. The channel bridge and the call path read the same map.

**Risks.**

- A caller can spoof the caller number. Do not use the number alone for sensitive actions. The competitor doc names STIR/SHAKEN attestation (`StirVerstat` at Twilio) as one check («Personal token in a voice channel»).
- One number can belong to several people, for example a family phone.
- A person can use WhatsApp on a different number. Then the map needs a manual link, for example a one-time code.
- The omni-channel id and the thread channel id must not contain the raw phone number. Use a hash or a random id.

**Open question:** Slack and the Stream Chat app give no phone number. To join them with the phone, the person must link the accounts, for example with a login or a one-time code.

## How each inbound channel reaches the Router

Three inbound channels work today through Stream products. The other inbound channels need the channel bridge. Each one has its own webhook and its own signature check. Messages go into one thread channel for each thread. One episode card for each episode goes into the omni-channel.

&#91;embedded content: how each inbound channel reaches the router · 3 exist, 5 new\]

| Inbound channel | How it reaches the Router | Status | Source |
| --- | --- | --- | --- |
| Phone call | The number vendor bridges the call over SIP into a Stream inbound trunk. Stream Video sends a call event to the call hook (`/v1/phone/hooks/stream`). The Router asks a worker to answer. `chatlog.Log` writes the transcript. | exists | [`internal/phone/phone.go:14-17`](https://github.com/GetStream/Vision-Agents/blob/ead4a273f4d3623fff2a2286d5422725aa0af2e2/acceleration/internal/phone/phone.go#L14-L17), [`internal/phone/hooks.go:15`](https://github.com/GetStream/Vision-Agents/blob/ead4a273f4d3623fff2a2286d5422725aa0af2e2/acceleration/internal/phone/hooks.go#L15), [`internal/api/callhooks.go:58`](https://github.com/GetStream/Vision-Agents/blob/ead4a273f4d3623fff2a2286d5422725aa0af2e2/acceleration/internal/api/callhooks.go#L58) |
| Voice in an app (Athena web, iOS) | The app joins a Stream Video call through the SDK. Athena gives its chat id, so voice uses the same agent channel. | exists | [`internal/chatlog/chatlog.go:109-111`](https://github.com/GetStream/Vision-Agents/blob/ead4a273f4d3623fff2a2286d5422725aa0af2e2/acceleration/internal/chatlog/chatlog.go#L109-L111) |
| Stream Chat | The client writes to the agent channel. Stream Chat sends `message.new` to the message hook (`/v1/chat/hooks/stream`). | exists | [`internal/chat/hooks.go:26-32`](https://github.com/GetStream/Vision-Agents/blob/ead4a273f4d3623fff2a2286d5422725aa0af2e2/acceleration/internal/chat/hooks.go#L26-L32), [`internal/api/messagehooks.go:61`](https://github.com/GetStream/Vision-Agents/blob/ead4a273f4d3623fff2a2286d5422725aa0af2e2/acceleration/internal/api/messagehooks.go#L61) |
| Slack | Slack Events API → channel bridge. The `hmac_header` verifier checks the Slack signing secret. | new | architecture doc, stress test «Slack, Enterprise Grid» |
| WhatsApp | Provider webhook → channel bridge. Through Twilio, the `twilio_signature` verifier checks `X-Twilio-Signature`. Meta Cloud API directly: `unverified`. | new | architecture doc, stress test «Twilio» |
| iMessage | Linq or Sendblue: HMAC-signed webhook → channel bridge. The customer brings their own account. The official path is Apple Messages for Business through an MSP. | new | competitor doc, «iMessage: a channel, not a connector» → «Two paths» |
| SMS | Inbound SMS from the number vendor → channel bridge. Numbers have the `sms` capability. The Router has no inbound SMS handling. | new | [`internal/phone/phone.go:47`](https://github.com/GetStream/Vision-Agents/blob/ead4a273f4d3623fff2a2286d5422725aa0af2e2/acceleration/internal/phone/phone.go#L47); `grep -rni "sms_url\|inbound.*sms" internal` returns nothing, October 2 |
| RCS, Telegram | The same pattern: a provider webhook and its own verifier. | new | Thierry, October 1: «slack, whatsapp, rcs, texting, imessage, maybe telegram» (architecture doc, «Risks») |

A new inbound channel needs three things: a provider manifest, a verifier for its signature and a way to send replies. Agent channels, `session.Session` and tools already exist.

Tools work the same way for every inbound channel. An inbound channel is not a tool (architecture doc, one-way door 9). iMessage uses the same kind of channel bridge. The reason: the Linq MCP server runs only over stdio, and the Router does not run stdio servers (competitor doc, «Voice agents and Router»).

## Other channels: WhatsApp, SMS, iMessage, Telegram

The design works for every messaging channel. The episode, the thread channel, the episode card, the proxy and token export do not depend on the channel. Six things depend on the channel. The `channel` block of `core.Manifest` describes them, so a new channel needs a manifest, not new Router code. This is a proposal. Rows marked `unverified` need a check.

| Channel | The customer's own unit at the provider | How it connects | Token | Inbound check | Thread key |
| --- | --- | --- | --- | --- | --- |
| Slack | one Slack app for each customer | the Router creates it with `apps.manifest.create` | OAuth, bot and user tokens, 12 hours with rotation | HMAC with the signing secret of the app | channel + `thread_ts` |
| WhatsApp (Meta Cloud API) | the customer's WABA and phone number, under one Stream Meta app (Tech Provider) | the customer completes Embedded Signup; the Router exchanges the code for a business token ([Meta](https://developers.facebook.com/documentation/business-messaging/whatsapp/embedded-signup/onboarding-customers-as-a-tech-provider)) | business token of the customer | `X-Hub-Signature-256`: HMAC-SHA256 of the raw body with the app secret of our Meta app ([details](https://hookdeck.com/webhooks/skills/whatsapp-webhooks)) | person's number ↔ business number |
| SMS (Twilio, Telnyx) | the customer's number or account at the vendor | API key of the vendor | static key | vendor signature | number ↔ number |
| iMessage | the customer's account and number at Linq or Sendblue. Apple Messages for Business is open only through an Apple-approved MSP ([Bird](https://bird.com/explained/apple-messages/what-is-a-messaging-service-provider-and-how-do-i-choose-one)) | API key of the provider | static key | Linq: HMAC-SHA256 of `{timestamp}.{rawBody}` ([Linq](https://docs.linqapp.com/guides/webhooks/index.md)). Sendblue: a secret in the `sb-signing-secret` header ([Sendblue](https://sendblue.co/docs/webhooks/)) | number ↔ number |
| Telegram | the customer's bot | the customer creates the bot in BotFather; no API creates a bot | bot token; expiry not documented (`unverified`) | the `X-Telegram-Bot-Api-Secret-Token` header set by `setWebhook` ([Bot API](https://core.telegram.org/bots/api#setwebhook)) | `chat_id` |

Microsoft Teams, Discord and RCS are not checked. They use other signature methods, for example JWT for the Bot Framework and Ed25519 for Discord (`unverified`).

**What the `channel` block describes.**

1. **Scheme.** `oauth2code` serves Slack and the code exchange of WhatsApp Embedded Signup. Telegram, Linq, Sendblue, Twilio and Telnyx need `api_key`. `core.Scheme` already plans it ([`core/scheme.go:20`](https://github.com/GetStream/Vision-Agents/blob/ead4a273f4d3623fff2a2286d5422725aa0af2e2/acceleration/internal/connectors/core/scheme.go#L20)).
2. **Verifier.** One HMAC verifier with parameters (header, algorithm, signed bytes: body, or timestamp and body) and one shared-secret header compare. These two cover every row of the table.
3. **Event routing.** Slack and Telegram: each app or bot has its own webhook URL, so the URL names the customer. WhatsApp: one Tech Provider app has one webhook for all customers, so the Router finds the customer by `phone_number_id` in the event. WhatsApp needs a global index on this field.
4. **Outbound policy.** WhatsApp allows free-form replies for 24 hours after the person's last message. After that, only approved templates ([Bird](https://bird.com/explained/whatsapp/what-is-the-24-hour-customer-service-window)). Thus the channel bridge needs a template path, and the idle period that closes a WhatsApp episode is shorter than 24 hours.
5. **Token export.** `api_key` keys and bot tokens are static: they do not rotate and do not expire after 12 hours. An exported key is a long-lived secret. For these channels the proxy is the default. Token export needs an explicit opt-in.
6. **Identity.** SMS, WhatsApp and iMessage join the contact map by the E.164 number. Telegram and Slack give no number, so the person links the accounts (see «Open question» in «How the agent knows it is the same person»).

**The rule «one app for each customer» becomes «one provider unit for each customer».** For Slack the unit is an app. For WhatsApp, Meta sets a different model: one Tech Provider app for all customers and a WABA with a number for each customer. How Meta isolates quality and bans between WABAs is `unverified`. For Telegram the unit is a bot. For SMS and iMessage the unit is an account or a number.

## Who moves messages: the channel bridge

The channel bridge moves messages between an external inbound channel and its thread channel. It also writes the episode card to the omni-channel. It is new code in the Router. It does not exist today. It has two halves:

- The inbound half receives the provider webhook and writes the message to the thread channel. For the first message of an episode, it also writes the episode card to the omni-channel.
- The outbound half takes the agent reply and sends it to the same external thread.

Phone and the Stream Chat app do not need the channel bridge. A call writes to its call channel. The in-app Stream Chat channel is itself the thread channel of in-app chat.

&#91;embedded content: channel bridge · 10 steps, in and out\]

The inbound half uses the entry that exists today. The channel bridge writes to the thread channel, which is an agent channel too. Stream Chat sends `message.new`. The Router wakes `session.Session`.

The outbound half also uses the message hook. For a written reply, `session.Session` calls `Log.Reply`, and `chatlog.Log` sends the reply as a new message with `source: agent` ([`internal/session/session.go:496-497`](https://github.com/GetStream/Vision-Agents/blob/ead4a273f4d3623fff2a2286d5422725aa0af2e2/acceleration/internal/session/session.go#L496-L497), [`internal/chatlog/chatlog.go:298-303`](https://github.com/GetStream/Vision-Agents/blob/ead4a273f4d3623fff2a2286d5422725aa0af2e2/acceleration/internal/chatlog/chatlog.go#L298-L303)). Stream Chat delivers it as `message.new`. The hook does not answer it, but it gives it to the outbound half when the channel is linked to an external thread. A spoken reply is different: `chatlog.Log` streams it and saves the final text with `UpdateMessagePartial` ([`internal/chatlog/chatlog.go:469,551-556`](https://github.com/GetStream/Vision-Agents/blob/ead4a273f4d3623fff2a2286d5422725aa0af2e2/acceleration/internal/chatlog/chatlog.go#L469)). SMS, WhatsApp and Slack are written channels, so this difference does not apply to them. The section «Step by step: one SMS» shows the full order.

| Step | What it does | In code? | Source |
| --- | --- | --- | --- |
| 1 Event endpoint | Receives the provider webhook. | no. The design has `POST /v1/agents/connectors/events/{connector_id}` for token signals. The channel bridge uses the same endpoint. | architecture doc, «AI-816: keep, change, add» → Add |
| 2 `core.Verifier` | Checks the signature: `hmac_header` for Slack and Linq, `twilio_signature` for WhatsApp through Twilio. Drops retries and the bot's own messages. | interface only ([`internal/connectors/core/signal.go:7`](https://github.com/GetStream/Vision-Agents/blob/ead4a273f4d3623fff2a2286d5422725aa0af2e2/acceleration/internal/connectors/core/signal.go#L7)) | architecture doc, «Layers and interfaces». How Slack marks bot messages: `unverified` |
| 3 Find `store.ConnectorConnection` | Finds the row by the account in the event (Slack `team_id`). | the table exists; lookup by account does not | architecture doc, stress test Slack: identity `$.team.id + $.authed_user.id` |
| 4 Thread → thread channel | Keeps the link between an external thread and its thread channel. A new thread gets a new thread channel. A new episode gets a new episode card in the omni-channel. | no. New table | our proposal |
| 5 Author → Stream Chat user | Maps the external id (Slack `U…`, phone number) to a Stream Chat user. | no | our proposal. How to join one person across inbound channels is an open question |
| 6 Write | Writes the message to the thread channel as the person, without the `source` field. | `chatlog.Log` already writes through the Chat API | `addressed()` accepts only such messages: [`internal/api/messagehooks.go:113-128`](https://github.com/GetStream/Vision-Agents/blob/ead4a273f4d3623fff2a2286d5422725aa0af2e2/acceleration/internal/api/messagehooks.go#L113-L128), [`internal/chatlog/chatlog.go:43,62`](https://github.com/GetStream/Vision-Agents/blob/ead4a273f4d3623fff2a2286d5422725aa0af2e2/acceleration/internal/chatlog/chatlog.go#L43) |
| → message hook → `session.Session` | Stream Chat sends `message.new`. The Router wakes or starts `session.Session`. | yes | [`internal/chat/hooks.go:26-32,80`](https://github.com/GetStream/Vision-Agents/blob/ead4a273f4d3623fff2a2286d5422725aa0af2e2/acceleration/internal/chat/hooks.go#L26-L32), [`internal/api/messagehooks.go:61,131`](https://github.com/GetStream/Vision-Agents/blob/ead4a273f4d3623fff2a2286d5422725aa0af2e2/acceleration/internal/api/messagehooks.go#L61) |
| 7 Agent reply | Receives the agent reply: `message.new` with `source: agent` in a linked thread channel. The message hook gives it to the outbound half. | the webhook exists; the hand-off to the bridge does not | [`internal/chatlog/chatlog.go:239,262-270,298`](https://github.com/GetStream/Vision-Agents/blob/ead4a273f4d3623fff2a2286d5422725aa0af2e2/acceleration/internal/chatlog/chatlog.go#L239) |
| 8 External thread? | Looks up the link from step 4. If there is no link, the conversation is a plain Stream Chat conversation. Nothing is sent. | no | our proposal |
| 9 `core.Resolver` | Gives the bot token from the inbound channel's `store.ConnectorConnection`. Tools use the same mechanism. | interface only ([`internal/connectors/core/resolver.go:17`](https://github.com/GetStream/Vision-Agents/blob/ead4a273f4d3623fff2a2286d5422725aa0af2e2/acceleration/internal/connectors/core/resolver.go#L17)) | architecture doc, one-way door 4 |
| 10 `core.Scheme` and send | Adds the token. Sends the reply to the provider API (Slack: `chat.postMessage`, `unverified`). | [`schemes/oauth2code`](https://github.com/GetStream/Vision-Agents/tree/ead4a273f4d3623fff2a2286d5422725aa0af2e2/acceleration/internal/connectors/schemes/oauth2code) exists; sending does not | [`internal/connectors/schemes/oauth2code/`](https://github.com/GetStream/Vision-Agents/tree/ead4a273f4d3623fff2a2286d5422725aa0af2e2/acceleration/internal/connectors/schemes/oauth2code) |

## Where connectors are in this design

Connectors are a layer under both sides. A connector is not a tool and not an inbound channel. A connector is a provider manifest, an account (`store.ConnectorConnection`) and a way to get and add a token. Tools and the channel bridge both use this layer.

&#91;embedded content: connector layer · what tools and the channel bridge use\]

**What changes in the layer for inbound channels.** `core.Manifest` gets a second block next to the tools block. The architecture doc calls it «a manifest block for events and a reply API» (two-way door «Where a channel's transport lives»). Our proposal for its content:

- which `core.Verifier` checks the inbound webhook;
- where the event keeps the account, thread, author and text (JSON paths, like the capture rules in the design);
- where and how to send a reply (endpoint and body template).

Inbound channels do not need the tool parts: `core.Binding`, `core.ToolGrant`, `core.ToolSource` and the Dispatcher. An inbound channel does not go through the LLM as a tool call. It starts a conversation and delivers the reply. One Slack app can be an inbound channel and a tool at the same time. Vercel does this: one connector gives channel credentials and MCP tools (architecture doc, one-way door 9, «eve with Vercel Connect»).

**What exists in code at `ead4a273`:** the manifests [`providers/slack.yaml`](https://github.com/GetStream/Vision-Agents/blob/ead4a273f4d3623fff2a2286d5422725aa0af2e2/acceleration/internal/connectors/providers/slack.yaml) and [`linear.yaml`](https://github.com/GetStream/Vision-Agents/blob/ead4a273f4d3623fff2a2286d5422725aa0af2e2/acceleration/internal/connectors/providers/linear.yaml) (tools only), the table `connector_connections` and [`schemes/oauth2code`](https://github.com/GetStream/Vision-Agents/tree/ead4a273f4d3623fff2a2286d5422725aa0af2e2/acceleration/internal/connectors/schemes/oauth2code). `core.Resolver` and `core.Verifier` are interfaces only ([`internal/connectors/core/resolver.go:17`](https://github.com/GetStream/Vision-Agents/blob/ead4a273f4d3623fff2a2286d5422725aa0af2e2/acceleration/internal/connectors/core/resolver.go#L17), [`signal.go:7`](https://github.com/GetStream/Vision-Agents/blob/ead4a273f4d3623fff2a2286d5422725aa0af2e2/acceleration/internal/connectors/core/signal.go#L7)). There are no implementations: `grep -rn ") Verify(r\|core.Resolver" internal` finds nothing outside tests, October 2. The Dispatcher is in the architecture doc only.

Slack shows why both sides use the same layer. The Slack inbound channel needs a bot token: the Slack app receives events and the bot posts replies. The Slack tool uses a user token today ([`internal/connectors/providers/slack.yaml:1-4`](https://github.com/GetStream/Vision-Agents/blob/ead4a273f4d3623fff2a2286d5422725aa0af2e2/acceleration/internal/connectors/providers/slack.yaml#L1-L4)). Thus one Slack app has two `store.ConnectorConnection` rows. The Router keeps both rows. In the design, one `core.Resolver` gives both tokens.

## Integration modes: full platform, customizations, pass-through

The platform has three layers. Each layer has a public API. A customer uses all three layers or only some of them. Our own upper layers call the lower layers through the same API. In every mode, the customer has its own Slack app. This is the target design. It is not in the code.

**Three layers.**

| Layer | What it does | Public API |
| --- | --- | --- |
| 1 · Connector layer | registers the customer's Slack app, runs OAuth, keeps and refreshes tokens, receives and forwards provider events, proxies calls | connections, proxy, token, event destinations |
| 2 · Conversation layer | episode cards in the omni-channel, thread channels, contact map, channel bridge | Stream Chat API |
| 3 · Agent layer | `session.Session`: LLM, MCP tools, hosted tools | session API (`POST /v1/agents/sessions`) |

**Three modes.**

- **A · Full platform.** Layers 1, 2 and 3. A Slack event goes through the channel bridge to its thread channel and the agent. The omni-channel gets one episode card. Tools go through the Slack MCP server. The customer writes the agent config only.
- **B · Platform with customizations.** Mode A, plus customer code for what the Slack MCP server does not offer. A hosted tool calls any Slack Web API method through the proxy. Raw events, for example buttons, reactions and modals, go to a customer URL or worker.
- **C · Pass-through.** Layers 1 and 3 only. We register the customer's Slack app and keep its tokens. We forward Slack events to the customer. The customer calls our agent through the Router API, keeps the history itself and replies through the proxy or with an exported token. Mode C is the loosest coupling: the customer can use the proxy, token export and event forwarding together. Today the customer opens a text session with `incognito: true`, so that we record nothing, and sends each message with `POST /v1/agents/sessions/{id}/responses`. The session keeps the history while it runs. The session API has no field for history from the caller ([`CreateSessionRequest`](https://github.com/GetStream/Vision-Agents/blob/ead4a273f4d3623fff2a2286d5422725aa0af2e2/acceleration/internal/api/generated.go#L2040), [`CreateResponseRequest`](https://github.com/GetStream/Vision-Agents/blob/ead4a273f4d3623fff2a2286d5422725aa0af2e2/acceleration/internal/api/generated.go#L2027)). Thus a thread that outlives a session needs a new history field.

The diagram in [Flexibility](#m729fz3d1s3.112218) shows which layers Stream runs in each mode.

### Direct calls: the proxy

The customer calls any Slack Web API method through the Router. The Router adds the token and sends the request on. The Router knows no Slack method. Thus we keep no layer over the Slack API. The proxy is the default way for direct calls.

&#91;embedded content: the proxy · 8 steps, the token stays in the Router\]

### Direct calls: token export

Some customer code must call Slack itself, for example with Slack Bolt. For this code the Router gives a short-lived token. The diagram shows the flow.

&#91;embedded content: the customer calls Slack directly · 13 steps, the Router gives only the token\]

|  | Proxy | Token export |
| --- | --- | --- |
| Where the token is | stays in the Router | in customer code |
| Audit | every call | every token request |
| Revoke | at once | when the token expires |
| Cost | one more hop. The Router is on the path of each call | none |
| Use | default | opt-in, for each connector |

**Trade-off.** The proxy keeps the strongest guarantee: neither the LLM nor customer code sees the token. It costs one hop, and the Router is on the path of each call. Token export removes the hop, but the token leaves the Router. A Slack app for each customer keeps both risks inside one customer.

## Integration modes: one Slack app for each customer

**Each need, compared with Vercel Connect.**

| What the customer wants | Our design | Vercel Connect |
| --- | --- | --- |
| We run everything: agent, history, replies | Mode A · full platform | Connect gives tokens and events only. The agent is customer code |
| Its own code where the Slack MCP server is not enough | Mode B · a hosted tool calls any Slack Web API method through the proxy | Customer code calls Slack with a token |
| Its own agent and its own history | Mode C · pass-through: we keep the Slack app and the tokens, the customer calls our agent through the Router API | Yes: customer code runs the agent, for example with the Vercel Chat SDK and the Vercel AI SDK |
| Call Slack with the official Slack SDK | The proxy (default): the SDK base URL points to the Router, and the token stays in the Router. Token export: opt-in | `getToken`: the token goes to customer code |
| Raw Slack events: buttons, reactions, modals | Event forwarding to a customer URL, signed with a key of that customer | Triggers: verified, then forwarded to at most three destinations and signed with a key of the connector |
| Its own Slack app | Yes, in every mode | Yes |

**How tools reach Slack today.** Tools go through the hosted Slack MCP server, not through the Slack Web API: `mcp: https://mcp.slack.com/mcp` and the only source `kind: mcp` ([`providers/slack.yaml:28,109-111`](https://github.com/GetStream/Vision-Agents/blob/ead4a273f4d3623fff2a2286d5422725aa0af2e2/acceleration/internal/connectors/providers/slack.yaml#L28)). Slack defines the tool list. The Router discovers it and pins each schema digest in `core.ToolGrant`. We implement no Slack API method for tools. The limit: the agent can use only what the Slack MCP server offers. The channel bridge is the one exception. It receives the Events API webhook and sends `chat.postMessage`. This surface is small and fixed, and the manifest `channel` block describes it.

**Each customer has its own Slack app, in every mode.** The Router creates it with Slack [`apps.manifest.create`](https://docs.slack.dev/reference/methods/apps.manifest.create). The response has `client_id`, `client_secret` and `signing_secret`. The app has the customer's name and icon. A customer can also bring its own app (`client.registration: customer`). The shared Stream app (`client.registration: operator`) serves only Stream's own agents, for example Athena. Reasons:

- A Slack ban or rate limit touches one customer only. Slack evaluates Web API rate limits for each app «per method, per workspace» ([rate limits](https://docs.slack.dev/apis/web-api/rate-limits)).
- Each app has its own Request URL, for example `/v1/connectors/events/{provider_app_id}`. The URL names the tenant and the `signing_secret`. Thus the Router needs no global lookup by `team_id`.
- The app is created in the customer's workspace and is not distributed. Slack limits distributed apps outside the Marketplace: since May 29, 2025, `conversations.history` and `conversations.replies` allow 1 request per minute ([changelog](https://docs.slack.dev/changelog/2025/05/29/rate-limit-changes-for-non-marketplace-apps)). Internal apps are outside this limit: «internal customer-built applications are not impacted». Thus the app stays in one workspace, without public distribution.
- To create the app, the Router needs an app configuration token. It belongs to one user and one workspace. It expires after 12 hours, and `tooling.tokens.rotate` renews it ([app manifests](https://docs.slack.dev/app-manifests/configuring-apps-with-app-manifests)). Thus a workspace admin gives it to Stream once, at connect time.

## Direct calls: rules, the Slack SDK and code changes

**The proxy.**

- Endpoint, as a proposal: `ANY /v1/agents/connections/{id}/proxy/{path}`. It is server-side only.
- The base URL comes from the manifest. Only provider hosts are allowed, so a token never goes to another host.
- On a 401 the Router calls `Invalidate`, refreshes and retries once.
- Each call writes an audit row. The Router limits the rate for each customer. It returns the Slack `429` and `Retry-After` as they are.
- The Router calls Slack from fixed egress IP addresses. Slack can restrict the tokens of an app to a list of IP ranges ([security](https://docs.slack.dev/authentication/best-practices-for-security)). For apps we create, we set the Router ranges. Then a leaked token does not work outside the Router. The manifest sets this list in `settings.allowed_ip_address_ranges`, at most 10 items ([app manifest](https://docs.slack.dev/reference/app-manifest)).

Nango offers both: a proxy where «tokens never touch your code» ([proxy](https://nango.dev/platform/request-proxy)) and credential retrieval ([get connection](https://nango.dev/docs/reference/backend/http-api/connection/get.md)). Pipedream Connect gives credentials only for the customer's own OAuth client: «the connected account must be using your own OAuth client» ([retrieve account](https://pipedream.com/docs/connect/api-reference/retrieve-account.md)).

**The customer uses the provider's own SDK.** We write and maintain no SDK for Slack. The customer uses the official Slack SDK in its language, for example `slack_sdk` for Python or `@slack/web-api` for Node.

- With token export, the SDK gets our short-lived token: `WebClient(token=...)`. The SDK calls Slack directly. The Router is not on the path.
- With the proxy, the SDK gets one setting: the base URL points to the proxy instead of `https://slack.com/api/`. `slack_sdk` has `base_url` ([WebClient](https://docs.slack.dev/tools/python-slack-sdk/reference/web/client.html)). `@slack/web-api` has `slackApiUrl` ([WebClientOptions](https://docs.slack.dev/tools/node-slack-sdk/reference/web-api/interfaces/WebClientOptions)).
- The proxy checks the caller, replaces `Authorization` with the token of the customer's Slack app, sends the request on unchanged and returns the response unchanged.
- The proxy implements no Slack method. For the proxy, `chat.postMessage` is only a path to forward to `slack.com`. A new Slack method needs no change in the Router.

```python
# The official Slack SDK, pointed at the proxy. The header name is a proposal.
client = WebClient(
    base_url="https://<router>/v1/agents/connections/<id>/proxy/api/",
    headers={"<Stream auth header>": "<server credential>"},
)
client.chat_postMessage(channel="C123", text="hi")
```

With the proxy, the SDK gets no Slack token: `WebClient` has no `token`. The customer sends its Stream server credential instead, so that the proxy knows the caller. The proxy adds the Slack token. The connection `<id>` in the path selects the grant, and thus the kind of token: bot or user. Each `store.ConnectorConnection` holds one grant.

**Rules for token export.**

- The endpoint is server-side only. The worker authenticates as on `/v1/dispatch`, which refuses client-side credentials ([`api/dispatchws.go:31-38`](https://github.com/GetStream/Vision-Agents/blob/ead4a273f4d3623fff2a2286d5422725aa0af2e2/acceleration/internal/api/dispatchws.go#L31-L38)).
- The Router gives a token only for a connection in the agent's `core.Binding`, or for the caller's own user connection.
- Only the customer's own Slack app can export. The shared Stream app never exports.
- The token carries all scopes of the grant: Slack issues no narrower token from a refresh token. [`oauth.v2.access`](https://docs.slack.dev/reference/methods/oauth.v2.access) has no scope argument. Least privilege comes from separate connections with narrow grants.
- Without token rotation, a Slack access token «never expires». With rotation, it expires after 12 hours, and rotation «may not be turned off once it's turned on» ([token rotation](https://docs.slack.dev/authentication/using-token-rotation)). Each app we create has `token_rotation_enabled: true` in its manifest.
- If the app restricts tokens to the Router IP addresses, an exported token does not work. Thus each connector chooses: IP restriction with the proxy only, or token export.
- The refresh token never leaves the Router. Each token request writes an audit row.
- Token export is off by default.

**Bot and user tokens.** The Router gives both kinds, for token export and for the proxy. The connection selects the kind.

|  | Bot token (`xoxb-…`) | User token (`xoxp-…`) |
| --- | --- | --- |
| Subject | `app` | `user` |
| Connection | `owner_type: app`, one for each workspace | `owner_type: user`, one for each person |
| Scopes in the Slack manifest | `oauth_config.scopes.bot` | `oauth_config.scopes.user` |
| Acts as | the bot of the customer's Slack app | one person |
| Who gets it | agent code, when the connection is in the agent's `core.Binding` (`Selection: fixed`) | only a session where this person is the verified caller (`Selection: session`) |

The rule for user tokens is stricter. A customer worker is trusted server-side code. But it must not get the user token of any person from a `user_id` alone. Else customer code can act as any employee who once connected Slack. `core.Binding` already has this rule: «`session`, the verified caller's own connection chosen when the session starts» ([`core/source.go:47-56`](https://github.com/GetStream/Vision-Agents/blob/ead4a273f4d3623fff2a2286d5422725aa0af2e2/acceleration/internal/connectors/core/source.go#L47-L56)). The same rule applies to the proxy: a path with a user connection is accepted only in the session of that user.

Vercel Connect is less strict. Code with a project OIDC token can request a user token for any `id`. A personal access token can request only its own user ([authentication](https://vercel.com/docs/connect/concepts/authentication)). Token rotation applies to both kinds: bot and user tokens expire after 12 hours ([token rotation](https://docs.slack.dev/authentication/using-token-rotation)).

**What changes in the code.**

1. A provider app record for each (app\_pk, connector): Slack `app_id`, `client_id`, sealed `client_secret` and `signing_secret`, owner Stream or customer. The Router creates it with `apps.manifest.create`.
2. An event endpoint for each provider app. `core.Verifier` checks the event with the `signing_secret` of that app.
3. The proxy operation, server-side only.
4. Token export. `core.AccessCredential` keeps its secret private: «Nothing else reads it» ([`core/scheme.go:115-130`](https://github.com/GetStream/Vision-Agents/blob/ead4a273f4d3623fff2a2286d5422725aa0af2e2/acceleration/internal/connectors/core/scheme.go#L115-L130)). Thus `core.Scheme` gets an optional `Export`. Only bearer schemes implement it: `oauth2code`, later `api_key`.
5. The token operation, server-side only.
6. An audit table for proxy calls and token requests.
7. Event destinations for raw event forwarding (modes B and C).
8. A session call that takes the history from the caller (mode C). Today there is none.
9. SDK methods in the Go SDK first, then in the other SDKs (AGENTS.md, «SDK changes»).
10. The schemes `api_key` and `client_credentials`. `core.Scheme` already plans them ([`core/scheme.go:20`](https://github.com/GetStream/Vision-Agents/blob/ead4a273f4d3623fff2a2286d5422725aa0af2e2/acceleration/internal/connectors/core/scheme.go#L20)).

## Dependency direction: why there is no cycle

There is no dependency cycle. Stream Chat does not use connectors. Stream Chat does not know the Router. The agent and the channel bridge use connectors, and both run in the Router.

&#91;embedded content: dependency direction · today and where a cycle could be\]

- **Router → Stream Chat is a code dependency.** The Router writes the agent channel with `GetOrCreateChannel` ([`internal/chatlog/chatlog.go:227`](https://github.com/GetStream/Vision-Agents/blob/ead4a273f4d3623fff2a2286d5422725aa0af2e2/acceleration/internal/chatlog/chatlog.go#L227)). The Router sets its own webhook URL in the Stream app settings: `PointMessageHook` → `UpdateApp` ([`internal/chat/hooks.go:80-116`](https://github.com/GetStream/Vision-Agents/blob/ead4a273f4d3623fff2a2286d5422725aa0af2e2/acceleration/internal/chat/hooks.go#L80-L116)).
- **Stream Chat → Router is only a webhook.** Stream Chat sends the generic event `message.new` to the URL from the settings. The Router checks the signature ([`internal/api/messagehooks.go:61`](https://github.com/GetStream/Vision-Agents/blob/ead4a273f4d3623fff2a2286d5422725aa0af2e2/acceleration/internal/api/messagehooks.go#L61)). This is dependency inversion. Calls from Stream Video work the same way ([`internal/phone/hooks.go:15`](https://github.com/GetStream/Vision-Agents/blob/ead4a273f4d3623fff2a2286d5422725aa0af2e2/acceleration/internal/phone/hooks.go#L15)).
- **There is a loop at runtime, not in code.** The Router writes to Stream Chat, and Stream Chat wakes the Router. This loop exists today for agent replies. `addressed()` prevents an echo: the Router ignores messages with the `source` field ([`internal/api/messagehooks.go:113-128`](https://github.com/GetStream/Vision-Agents/blob/ead4a273f4d3623fff2a2286d5422725aa0af2e2/acceleration/internal/api/messagehooks.go#L113-L128)). The channel bridge writes the person's message without `source` (step 6). The channel bridge drops its own message when Slack sends it back (step 2).

## Architecture changes

**The design needs no redesign of the connector architecture.** The model stays: `store.ConnectorConnection` with an owner, `core.Resolver`, `core.Scheme` and `core.Manifest` with revisions. The architecture doc planned for inbound channels: «Connection, manifest, Resolver, the inbound endpoint and Verifier registry, all unchanged» (two-way door «Where a channel's transport lives»). Checked on [`connectors/planning`](https://github.com/GetStream/Vision-Agents/tree/connectors/planning) @ [`ead4a273`](https://github.com/GetStream/Vision-Agents/commit/ead4a273f4d3623fff2a2286d5422725aa0af2e2), October 2.

In the connector layer:

1. Add a `channel` block to `core.Manifest`.
2. Change the `core.Verifier` result so that it can carry a message, not only a `core.Signal`.
3. Add the event endpoint. It must separate token signals from messages.
4. Add an index to find `store.ConnectorConnection` by the account in the event.
5. Add a built-in Slack manifest for the bot token.

In the Router, outside the connector layer:

6. Add the channel bridge with an inbound half and an outbound half.
7. Add a table that links an external thread to its thread channel.
8. Add a map from an external author to a Stream Chat user.
9. In the message hook, give agent replies in a linked thread channel to the outbound half of the channel bridge.

For episodes, in the Router (proposal):

- Create one thread channel for each external thread. A call keeps its call channel.
- Write one episode card into the omni-channel when an episode starts. Update it with `UpdateMessagePartial` when the episode ends and when the summary is ready.
- Close a text episode after an idle period. The length of the period is a setting.
- Summarize each closed episode and write its facts into memory.
- Let `session.Session` read the episode cards of the person as extra context. A voice session must also learn this ([`internal/session/manager.go:229-231`](https://github.com/GetStream/Vision-Agents/blob/ead4a273f4d3623fff2a2286d5422725aa0af2e2/acceleration/internal/session/manager.go#L229-L231)).

The design needs no new entry into `session.Session`. The message hook delivers the message.

### Details of the connector-layer additions

| Addition | Code today | Size |
| --- | --- | --- |
| `core.Manifest`: a `channel` block (how to read an event, where to reply) | `core.Manifest` has `Sources` for tools and no block for inbound channels ([`internal/connectors/core/manifest.go:28-66`](https://github.com/GetStream/Vision-Agents/blob/ead4a273f4d3623fff2a2286d5422725aa0af2e2/acceleration/internal/connectors/core/manifest.go#L28-L66)) | a new field and a new manifest revision. Existing rows do not change |
| `core.Verifier` returns more than token signals | `Verify` returns `[]core.Signal`: revoked, uninstalled, rotated only ([`internal/connectors/core/signal.go:5-27`](https://github.com/GetStream/Vision-Agents/blob/ead4a273f4d3623fff2a2286d5422725aa0af2e2/acceleration/internal/connectors/core/signal.go#L5-L27)). No implementations | a small interface change. The cost is lowest now, because no implementation exists |
| The event endpoint separates signals and messages | design only (`POST /v1/agents/connectors/events/{connector_id}`) | the endpoint is planned for signals anyway. Messages add one branch |
| Find `store.ConnectorConnection` by the account in the event | the column `account_id` exists. The only index is `connector_connections_owner_idx` ([`migrations/20261002193000_connector_connections.sql:39,68-70`](https://github.com/GetStream/Vision-Agents/blob/ead4a273f4d3623fff2a2286d5422725aa0af2e2/acceleration/migrations/20261002193000_connector_connections.sql#L39)). The Slack account id is `team_id:user_id` ([`providers/slack.yaml:98`](https://github.com/GetStream/Vision-Agents/blob/ead4a273f4d3623fff2a2286d5422725aa0af2e2/acceleration/internal/connectors/providers/slack.yaml#L98)), but the channel bridge must find the row by team | an index and an identity rule for the bot token |
| A Slack manifest for the bot token | the built-in [`providers/slack.yaml`](https://github.com/GetStream/Vision-Agents/blob/ead4a273f4d3623fff2a2286d5422725aa0af2e2/acceleration/internal/connectors/providers/slack.yaml) uses the user-token flow for Slack MCP ([`slack.yaml:1-4`](https://github.com/GetStream/Vision-Agents/blob/ead4a273f4d3623fff2a2286d5422725aa0af2e2/acceleration/internal/connectors/providers/slack.yaml#L1-L4)). Only the test fixture [`core/testdata/manifests/slack.yaml`](https://github.com/GetStream/Vision-Agents/blob/ead4a273f4d3623fff2a2286d5422725aa0af2e2/acceleration/internal/connectors/core/testdata/manifests/slack.yaml) uses the bot token | a second built-in manifest. The manifest format already supports it |

Two items do not depend on inbound channels. Tools need them too:

- `core.Resolver` has no implementation ([`internal/connectors/core/resolver.go:17`](https://github.com/GetStream/Vision-Agents/blob/ead4a273f4d3623fff2a2286d5422725aa0af2e2/acceleration/internal/connectors/core/resolver.go#L17)). Tools need it.
- `store.ConnectorConnection` has tool columns `cached_tools` and `tools_digest` ([`connector_connections.sql:53-55`](https://github.com/GetStream/Vision-Agents/blob/ead4a273f4d3623fff2a2286d5422725aa0af2e2/acceleration/migrations/20261002193000_connector_connections.sql#L53-L55)). No change is necessary: a row for an inbound channel keeps them empty.

**Bot and user tokens in Slack.** The Slack tool uses a user token today ([`slack.yaml:1-4`](https://github.com/GetStream/Vision-Agents/blob/ead4a273f4d3623fff2a2286d5422725aa0af2e2/acceleration/internal/connectors/providers/slack.yaml#L1-L4)). The inbound channel needs a bot token: the Slack app receives events and the bot replies. Does the hosted Slack MCP accept a bot token? `unverified`. If not, one Slack app has two `store.ConnectorConnection` rows: bot for the inbound channel, user for the tool. The Router keeps both rows, with one refresh and one revoke path.

**Do now:** change the `core.Verifier` result so that it can carry a message. This is the only change where waiting adds rework. The other additions can wait for the first inbound channel.

### Tenancy: one Stream app per customer

The tenant of every connector record is the customer's Stream app. In the Router tables the column is `customer_id`: behind the Stream gateway (auth mode proxy) it holds the Stream app id as a decimal string, the same value as `app_pk` in the `chat` database. This matches the branch `nash/project-tenancy`: a customer is its Stream app, `CustomerOf(app int64)` ([`internal/streamapp/streamapp.go:97`](https://github.com/GetStream/Vision-Agents/blob/8436b0b36c6848fdd539044a991dbba7dae524bc/acceleration/internal/streamapp/streamapp.go#L97)). On `accelerate`, `withCustomer` also takes `principal.AppID`. The branch adds five rules for this design. They are proposals.

1. **Channels live in the customer's app.** The omni-channel and the thread channels are Stream Chat channels of the customer's app, not of the deployment app. `chatlog.Log` and the channel bridge write through the client of the customer's app. In the branch, a session that a hook started already acts in the app of that hook ([`internal/session/manager.go:1131-1137`](https://github.com/GetStream/Vision-Agents/blob/8436b0b36c6848fdd539044a991dbba7dae524bc/acceleration/internal/session/manager.go#L1131-L1137)).
2. **Each episode keeps its app.** An episode closes later, after the idle period. Then the Router must know which app gets the summary. Thus the episode, the thread link and the provider app keep a `stream_app_pk` pin, as the branch does for sessions and calls ([`migrations/20261002120000_stream_app_pins.sql`](https://github.com/GetStream/Vision-Agents/blob/8436b0b36c6848fdd539044a991dbba7dae524bc/acceleration/migrations/20261002120000_stream_app_pins.sql)).
3. **A message hook in each customer app.** The branch checks an inbound hook with the secret of its own app (`internal/streamapp/hooks.go`). The branch does not set the hook in a customer app yet: registration does not call `PointMessageHook`. Only the operator command `phone hooks -app <id>` sets it, by hand for each app ([`cmd/phone/main.go:568`](https://github.com/GetStream/Vision-Agents/blob/8436b0b36c6848fdd539044a991dbba7dae524bc/acceleration/cmd/phone/main.go#L568)). The Router must set its hook in each customer app; otherwise `message.new` from a thread channel does not reach the Router. This comes later.
4. **One sealing scheme, two tables.** The key of the customer's Stream app stays in the `stream_apps` table of the branch. The secrets of the customer's Slack app stay in `store.ConnectorConnection` with owner `app`, read through `core.Resolver`. Both tables seal with the same `auth.Sealer` and keyring. The branch builds one sealer for both in `newSecretSealer` ([`cmd/router/main.go:212`](https://github.com/GetStream/Vision-Agents/blob/8436b0b36c6848fdd539044a991dbba7dae524bc/acceleration/cmd/router/main.go#L212)). Its Stream keys use their own AAD purpose ([`internal/streamapp/sealed.go:14`](https://github.com/GetStream/Vision-Agents/blob/8436b0b36c6848fdd539044a991dbba7dae524bc/acceleration/internal/streamapp/sealed.go#L14)).
5. **Forwarded events use the customer's signing key.** Raw event forwarding signs each request with a key of that customer or destination, not with the deployment secret. This key is a separate signing secret, not the API secret of the customer's Stream app: a receiver that checks a signature must not get full API access. In the branch, the guardrail webhook signs with the API secret of the customer's Stream app ([`internal/session/manager.go:1117`](https://github.com/GetStream/Vision-Agents/blob/8436b0b36c6848fdd539044a991dbba7dae524bc/acceleration/internal/session/manager.go#L1117)). On `accelerate`, `plugins.NewWebhookSecret` already makes a separate `whsec_` secret ([`internal/plugins/events.go:163`](https://github.com/GetStream/Vision-Agents/blob/84a48818e1ea5b51ff402c4fc0102514dbe42529/acceleration/internal/plugins/events.go#L163)).

**No org level.** Org links stay out of this design. In the Stream platform, `app_pk` separates all data, and `org_id` is only an attribute of the app. Checked on the live `chat` database on October 5: 124 of 131 tables have `app_pk` or `app_id`; only `moderation_stats_daily` has `org_id`, as «a non-key column». In Stream's chat backend, `ApplicationConfig` keeps `OrganizationID` as a field of the app, and org is used for billing, features, logs and dashboard access. If a customer later wants one Slack workspace for several of its apps (for example staging and production), a separate org-level link can come on top, like project links in Vercel Connect.

## Cost of this design

**Cost:** the Router gets channel bridge code. The channel bridge does four tasks:

1. Receives events from Slack.
2. Links a Slack thread to its thread channel and writes the episode card.
3. Drops repeated events.
4. Sends the agent reply to Slack.

The first channel bridge is for Slack. This is the first Athena scenario: «slack connection and multiplayer» (Nash, October 1).

**Cost.**

- Build: the channel bridge, the contact map and the five connector-layer additions. Agent channels, the message hook and the token store exist today.
- Run: one service. There is no second deployment and no second on-call rotation.
- Each SMS: two Stream Chat writes (the message and the reply) and two `message.new` webhooks. The price per message for our Stream app: `unverified`.
- Latency: one more round trip, Router → Stream Chat → message hook → Router. It is not measured: `unverified`. Measure it on staging. If it is too slow, the channel bridge can write the message and also call `Session.Ask` directly. Then the session has two entries.
- Load: vendor webhooks come into the Router. If they need their own scaling, move the channel bridge to a separate service.

## How other companies do it

Platforms that run the agent inside the platform use the same pattern as this design. A channel adapter runs in the platform. It changes the provider message to one common format. It writes the message to one shared conversation. The agent reads that conversation. The reply goes back through the same adapter. The table uses public docs, read on October 2.

| Company | How it works | Same as this design? |
| --- | --- | --- |
| [Botpress](https://botpress.com/docs/integrations/sdk/integration/messaging/) | One integration defines `channels` and `actions` ([definitions](https://botpress.com/docs/definitions)). A message arrives at `webhook.botpress.cloud`. The integration calls `getOrCreateConversation`, `getOrCreateUser` and `createMessage`. Tags keep the external ids. The handler in `channels` sends the bot reply. | Yes. One connector serves the inbound channel and the tools. The message goes into the conversation before the bot reads it. |
| [Twilio Conversation Orchestrator](https://www.twilio.com/en-us/changelog/conversation-orchestrator-is-now-generally-available) | GA on May 5, 2026. «One unified conversation model works across Voice, SMS, WhatsApp, RCS, and Chat». A setting groups conversations: by customer profile (the default), by participant address (SMS and voice on one number) or by address and channel. | Yes. Grouping by participant address is our contact map by phone number. |
| [Intercom Fin](https://www.intercom.com/help/en/articles/7120684-fin-ai-agent-explained) | Fin works in Messenger, WhatsApp, Facebook, Instagram and SMS. Fin and human agents share one customer record. Fin Voice brings the same agent to phone calls. | Yes, for the shared history. The docs do not say where the adapters run. |
| [Chatwoot](https://chatwoot.com/docs/user-guide/add-inbox-settings) (open source) | One Rails app. Each connected channel is an inbox. Messages from WhatsApp, Twilio SMS and other channels get one format, go into one queue and become a conversation in an inbox ([third-party overview](https://pyshine.com/Chatwoot-Open-Source-Omni-Channel-Customer-Support/)). | Yes. The adapters run in the main service. |
| [Azure Bot Service](https://learn.microsoft.com/en-us/azure/bot-service/bot-service-manage-channels) | The Bot Connector Service changes the channel schema to the Activity schema and back. It runs apart from the bot code. | Partly. The adapter is a separate service. But there the bot is customer code and the connector is the platform. Here the agent is part of the platform. |
| [Rasa](https://rasa.com/docs/rasa/connectors/custom-connectors) | `InputChannel` receives the message. `OutputChannel` sends the reply. Both run in the Rasa server and talk to the bot directly. | No. The connector docs show no conversation that channels share. |

**Other designs, and why this design does not use them.**

- The channel bridge gives the message directly to `session.Session`, as in Rasa. Then the channels have no shared history. Omni-channel needs a second design later.
- The channel bridge runs in a separate service, as the Azure Bot Connector Service does. Then the bot token is outside the Router. That service needs its own encryption, refresh and revoke. It also needs its own Stream Chat webhook for `message.new` and a second copy of the contact map.
- A later move is still possible. The channel bridge talks to the Router only through Stream Chat channels. Thus a move to a separate service changes the deployment, not the design.

## What a connector is in the code today

The merged AI-816 work (#727, #729, #730) covers tools only. It stores accounts at providers and gives their tokens to the agent. Checked on [`connectors/planning`](https://github.com/GetStream/Vision-Agents/tree/connectors/planning) @ [`ead4a273`](https://github.com/GetStream/Vision-Agents/commit/ead4a273f4d3623fff2a2286d5422725aa0af2e2), October 2.

In the code today, «connector» means tools only. The Router has no Slack, WhatsApp or iMessage inbound channel.

| Code name | What it does | Example | Where in code |
| --- | --- | --- | --- |
| `store.ConnectorDefinition`, `core.Manifest` | Describes one provider: OAuth endpoints, scopes, schemes. | Slack, Linear | [`internal/connectors/providers/slack.yaml`](https://github.com/GetStream/Vision-Agents/blob/ead4a273f4d3623fff2a2286d5422725aa0af2e2/acceleration/internal/connectors/providers/slack.yaml), [`linear.yaml`](https://github.com/GetStream/Vision-Agents/blob/ead4a273f4d3623fff2a2286d5422725aa0af2e2/acceleration/internal/connectors/providers/linear.yaml); table `connector_definitions` ([`migrations/20261002190000_connector_definitions.sql`](https://github.com/GetStream/Vision-Agents/blob/ead4a273f4d3623fff2a2286d5422725aa0af2e2/acceleration/migrations/20261002190000_connector_definitions.sql)); API `/v1/agents/connectors` ([`internal/api/connectors.go:169`](https://github.com/GetStream/Vision-Agents/blob/ead4a273f4d3623fff2a2286d5422725aa0af2e2/acceleration/internal/api/connectors.go#L169)) |
| `store.ConnectorConnection` | Keeps one account and its sealed token. The owner is `app` or `user`. | the Stream Slack workspace | [`internal/store/connections.go:72`](https://github.com/GetStream/Vision-Agents/blob/ead4a273f4d3623fff2a2286d5422725aa0af2e2/acceleration/internal/store/connections.go#L72); `owner_type` in [`migrations/20261002193000_connector_connections.sql:31,63`](https://github.com/GetStream/Vision-Agents/blob/ead4a273f4d3623fff2a2286d5422725aa0af2e2/acceleration/migrations/20261002193000_connector_connections.sql#L31) |
| `store.ConnectorAuthorizationAttempt` | Keeps one OAuth attempt. An attempt is used one time only. | “Connect Slack” button → popup → callback | [`internal/store/connections.go:137`](https://github.com/GetStream/Vision-Agents/blob/ead4a273f4d3623fff2a2286d5422725aa0af2e2/acceleration/internal/store/connections.go#L137); [`migrations/20261002193100_connector_authorization_attempts.sql`](https://github.com/GetStream/Vision-Agents/blob/ead4a273f4d3623fff2a2286d5422725aa0af2e2/acceleration/migrations/20261002193100_connector_authorization_attempts.sql) |
| `core.Scheme` | Gets a token and adds it to a request. | OAuth 2.0 authorization code | [`internal/connectors/core/scheme.go:17`](https://github.com/GetStream/Vision-Agents/blob/ead4a273f4d3623fff2a2286d5422725aa0af2e2/acceleration/internal/connectors/core/scheme.go#L17), [`internal/connectors/schemes/oauth2code/`](https://github.com/GetStream/Vision-Agents/tree/ead4a273f4d3623fff2a2286d5422725aa0af2e2/acceleration/internal/connectors/schemes/oauth2code) |
| `core.Binding`, `core.ToolGrant` | Give one agent config the tools of one connector. A grant names one exact tool and its schema digest. | Athena can call `slack.read_thread` | [`internal/connectors/core/source.go:47,62`](https://github.com/GetStream/Vision-Agents/blob/ead4a273f4d3623fff2a2286d5422725aa0af2e2/acceleration/internal/connectors/core/source.go#L47) |

`store.ConnectorConnection` does not know if its token is for a tool or for an inbound channel. Thus both sides can use it. The architecture doc has one rule: an inbound channel is never a kind of tool (one-way door 9).

## Where the omni-channel conversation lives

The omni-channel is the person's agent channel. It keeps one episode card for each episode: each call and each run of messages on one external thread. The raw text stays in the call channel or the thread channel. The omni-channel is a history. It does not transport messages from Slack or WhatsApp.

**What omni-channel means.** It is a contact-center term. One agent is available in all inbound channels. For the agent, it is one conversation. Example: a patient writes in WhatsApp, then calls. The agent knows what the chat was about (omni-channel doc). Thierry wrote on October 1: «the omni channel/connector concept will be important».

**What the Router does today.** The agent channel already joins voice and text:

1. `chatlog.Log` writes each phrase of a call into the agent channel ([`internal/chatlog/chatlog.go:1-17,43`](https://github.com/GetStream/Vision-Agents/blob/ead4a273f4d3623fff2a2286d5422725aa0af2e2/acceleration/internal/chatlog/chatlog.go#L1-L17)).
2. Athena gives the id of its own chat to `chatlog.Log`. Thus voice does not open a second agent channel ([`internal/chatlog/chatlog.go:109-111`](https://github.com/GetStream/Vision-Agents/blob/ead4a273f4d3623fff2a2286d5422725aa0af2e2/acceleration/internal/chatlog/chatlog.go#L109-L111)).
3. Stream Chat sends each new message in the agent channel to the message hook ([`internal/api/messagehooks.go:61`](https://github.com/GetStream/Vision-Agents/blob/ead4a273f4d3623fff2a2286d5422725aa0af2e2/acceleration/internal/api/messagehooks.go#L61)). The Router answers from the running `session.Session` or starts a new one (`Server.routeArrivingMessage`, [`:131`](https://github.com/GetStream/Vision-Agents/blob/ead4a273f4d3623fff2a2286d5422725aa0af2e2/acceleration/internal/api/messagehooks.go#L131)).

**What the Router does not do.** The Router has no Slack, WhatsApp or iMessage inbound channel. Stream Chat does not sync with these services. A channel bridge must receive the provider event, write the message to the thread channel, write the episode card to the omni-channel and send the reply back.

**Why the agent channel keeps the history.** The history stays after the call ends. Any Stream Chat client can read it. The Router does not need its own transcript API ([`internal/chatlog/chatlog.go:3-6`](https://github.com/GetStream/Vision-Agents/blob/ead4a273f4d3623fff2a2286d5422725aa0af2e2/acceleration/internal/chatlog/chatlog.go#L3-L6)).

**Open question:** how do we know that a WhatsApp number and a caller are the same person? The source documents do not answer this.

## Terms

This document uses these terms. Each term has one meaning. Code names are as in `acceleration/` on [`connectors/planning`](https://github.com/GetStream/Vision-Agents/tree/connectors/planning) @ [`ead4a273`](https://github.com/GetStream/Vision-Agents/commit/ead4a273f4d3623fff2a2286d5422725aa0af2e2).

| Term | Meaning | Where in code |
| --- | --- | --- |
| Router | The Go service that runs agents. | [`acceleration/cmd/router`](https://github.com/GetStream/Vision-Agents/tree/ead4a273f4d3623fff2a2286d5422725aa0af2e2/acceleration/cmd/router) |
| `session.Session` | One running conversation of one agent. | [`internal/session/session.go:101`](https://github.com/GetStream/Vision-Agents/blob/ead4a273f4d3623fff2a2286d5422725aa0af2e2/acceleration/internal/session/session.go#L101) |
| inbound channel | A way for a person to reach the agent: phone, Stream Chat app, Slack, WhatsApp, iMessage, SMS. | not a code type |
| agent channel | A Stream Chat channel of type `agent`. It keeps the history of one conversation. | `chatlog.ChannelType`, [`internal/chatlog/chatlog.go:43`](https://github.com/GetStream/Vision-Agents/blob/ead4a273f4d3623fff2a2286d5422725aa0af2e2/acceleration/internal/chatlog/chatlog.go#L43) |
| `chatlog.Log` | Writes what the person and the agent say into the agent channel. | [`internal/chatlog/chatlog.go:134`](https://github.com/GetStream/Vision-Agents/blob/ead4a273f4d3623fff2a2286d5422725aa0af2e2/acceleration/internal/chatlog/chatlog.go#L134) |
| message hook | The Router endpoint that receives `message.new` events from Stream Chat. | `Server.receiveMessageEvent`, [`internal/api/messagehooks.go:61`](https://github.com/GetStream/Vision-Agents/blob/ead4a273f4d3623fff2a2286d5422725aa0af2e2/acceleration/internal/api/messagehooks.go#L61) |
| call hook | The Router endpoint that receives call events from Stream Video. | `Server.receiveCallEvent`, [`internal/api/callhooks.go:58`](https://github.com/GetStream/Vision-Agents/blob/ead4a273f4d3623fff2a2286d5422725aa0af2e2/acceleration/internal/api/callhooks.go#L58) |
| tool | An action that the LLM calls during a conversation. Example: read a Slack thread. | `core.Binding`, `core.ToolGrant`, [`internal/connectors/core/source.go:47,62`](https://github.com/GetStream/Vision-Agents/blob/ead4a273f4d3623fff2a2286d5422725aa0af2e2/acceleration/internal/connectors/core/source.go#L47) |
| `store.ConnectorDefinition` | The description of one provider: endpoints, scopes, auth schemes. Its content is a `core.Manifest`. | [`internal/store/connectors.go:44`](https://github.com/GetStream/Vision-Agents/blob/ead4a273f4d3623fff2a2286d5422725aa0af2e2/acceleration/internal/store/connectors.go#L44), [`internal/connectors/core/manifest.go:28`](https://github.com/GetStream/Vision-Agents/blob/ead4a273f4d3623fff2a2286d5422725aa0af2e2/acceleration/internal/connectors/core/manifest.go#L28) |
| `store.ConnectorConnection` | One account at one provider and its sealed token. The owner is `app` or `user`. | [`internal/store/connections.go:72`](https://github.com/GetStream/Vision-Agents/blob/ead4a273f4d3623fff2a2286d5422725aa0af2e2/acceleration/internal/store/connections.go#L72) |
| `core.Scheme` | Gets a token from the provider and adds it to a request. | [`internal/connectors/core/scheme.go:17`](https://github.com/GetStream/Vision-Agents/blob/ead4a273f4d3623fff2a2286d5422725aa0af2e2/acceleration/internal/connectors/core/scheme.go#L17) |
| `core.Resolver` | Gives a valid token for one `store.ConnectorConnection`. Interface only, no implementation. | [`internal/connectors/core/resolver.go:17`](https://github.com/GetStream/Vision-Agents/blob/ead4a273f4d3623fff2a2286d5422725aa0af2e2/acceleration/internal/connectors/core/resolver.go#L17) |
| `core.Verifier` | Checks the signature of an inbound provider event. Interface only, no implementation. | [`internal/connectors/core/signal.go:7`](https://github.com/GetStream/Vision-Agents/blob/ead4a273f4d3623fff2a2286d5422725aa0af2e2/acceleration/internal/connectors/core/signal.go#L7) |
| channel bridge | Proposed code that moves messages between an inbound channel and its thread channel, and writes episode cards. | not in code |
| bot token, user token | The two Slack token types. The Slack app gets events and replies with the bot token. The Slack MCP tool uses the user token. | [`internal/connectors/providers/slack.yaml:1-4`](https://github.com/GetStream/Vision-Agents/blob/ead4a273f4d3623fff2a2286d5422725aa0af2e2/acceleration/internal/connectors/providers/slack.yaml#L1-L4) |
| omni-channel | The person's agent channel. It keeps one episode card for each episode of the person, from all inbound channels. | not a code type |
| episode | One call, or one run of messages on one external thread. A text episode ends after an idle period. | proposal, not in code |
| thread channel | The agent channel of one external thread: SMS, WhatsApp, Slack or iMessage. It keeps the messages word for word. The in-app Stream Chat channel is the thread channel of in-app chat. | proposal, not in code |
| call channel | The agent channel of one call, agent:\<call id>. It keeps the transcript. It is the thread channel of a call. | internal/chatlog/chatlog.go:193-196 |
| episode card | One message in the omni-channel for each episode. It has source, status, thread\_channel and, after the episode, the summary. | proposal, not in code |

## Sources

- [Connectors: architecture design](architecture.md) — one-way doors 4, 7 and 9, two-way door «Where a channel's transport lives», «Risks». Read on October 2, rev 30.
- [Linq, Chatbase and omni-channel: an explanation](https://claude.ai/code/artifact/859c6b80-c6cb-439e-a0e4-c0ccfba8a117) — the omni-channel definition, Thierry quotes, the table «What the Router already has». Read on October 2, rev 13.
- [Voice-agent connectors: competitor analysis](competitor-analysis.md) — «iMessage: a channel, not a connector», «Decide separately», «How Vercel does it: Eve and Connect». Read on October 2, rev 426.
- Code: [`connectors/planning`](https://github.com/GetStream/Vision-Agents/tree/connectors/planning) @ [`ead4a273`](https://github.com/GetStream/Vision-Agents/commit/ead4a273f4d3623fff2a2286d5422725aa0af2e2), which includes #727, #729 and #730 from `accelerate`. Paths are under `acceleration/`:
  - [`internal/chatlog/chatlog.go:1-17,43,62,109-111,134,227,239,262-270,298,469,551-556`](https://github.com/GetStream/Vision-Agents/blob/ead4a273f4d3623fff2a2286d5422725aa0af2e2/acceleration/internal/chatlog/chatlog.go#L1-L17)
  - [`internal/chat/hooks.go:1-6,26-32,80-116`](https://github.com/GetStream/Vision-Agents/blob/ead4a273f4d3623fff2a2286d5422725aa0af2e2/acceleration/internal/chat/hooks.go#L1-L6)
  - [`internal/api/messagehooks.go:61,113-128,131,292-296`](https://github.com/GetStream/Vision-Agents/blob/ead4a273f4d3623fff2a2286d5422725aa0af2e2/acceleration/internal/api/messagehooks.go#L61), [`internal/api/callhooks.go:58`](https://github.com/GetStream/Vision-Agents/blob/ead4a273f4d3623fff2a2286d5422725aa0af2e2/acceleration/internal/api/callhooks.go#L58), [`internal/api/connectors.go:169`](https://github.com/GetStream/Vision-Agents/blob/ead4a273f4d3623fff2a2286d5422725aa0af2e2/acceleration/internal/api/connectors.go#L169)
  - [`internal/phone/phone.go:14-17,47`](https://github.com/GetStream/Vision-Agents/blob/ead4a273f4d3623fff2a2286d5422725aa0af2e2/acceleration/internal/phone/phone.go#L14-L17), [`internal/phone/hooks.go:15`](https://github.com/GetStream/Vision-Agents/blob/ead4a273f4d3623fff2a2286d5422725aa0af2e2/acceleration/internal/phone/hooks.go#L15)
  - [`internal/session/session.go:101,491`](https://github.com/GetStream/Vision-Agents/blob/ead4a273f4d3623fff2a2286d5422725aa0af2e2/acceleration/internal/session/session.go#L101)
  - [`internal/store/connectors.go:44`](https://github.com/GetStream/Vision-Agents/blob/ead4a273f4d3623fff2a2286d5422725aa0af2e2/acceleration/internal/store/connectors.go#L44), [`internal/store/connections.go:72,137`](https://github.com/GetStream/Vision-Agents/blob/ead4a273f4d3623fff2a2286d5422725aa0af2e2/acceleration/internal/store/connections.go#L72)
  - [`internal/connectors/core/manifest.go:28-66`](https://github.com/GetStream/Vision-Agents/blob/ead4a273f4d3623fff2a2286d5422725aa0af2e2/acceleration/internal/connectors/core/manifest.go#L28-L66), [`scheme.go:17`](https://github.com/GetStream/Vision-Agents/blob/ead4a273f4d3623fff2a2286d5422725aa0af2e2/acceleration/internal/connectors/core/scheme.go#L17), [`resolver.go:17`](https://github.com/GetStream/Vision-Agents/blob/ead4a273f4d3623fff2a2286d5422725aa0af2e2/acceleration/internal/connectors/core/resolver.go#L17), [`signal.go:5-27`](https://github.com/GetStream/Vision-Agents/blob/ead4a273f4d3623fff2a2286d5422725aa0af2e2/acceleration/internal/connectors/core/signal.go#L5-L27), [`source.go:47,62`](https://github.com/GetStream/Vision-Agents/blob/ead4a273f4d3623fff2a2286d5422725aa0af2e2/acceleration/internal/connectors/core/source.go#L47)
  - [`internal/connectors/providers/slack.yaml:1-4,98`](https://github.com/GetStream/Vision-Agents/blob/ead4a273f4d3623fff2a2286d5422725aa0af2e2/acceleration/internal/connectors/providers/slack.yaml#L1-L4), [`internal/connectors/schemes/oauth2code/`](https://github.com/GetStream/Vision-Agents/tree/ead4a273f4d3623fff2a2286d5422725aa0af2e2/acceleration/internal/connectors/schemes/oauth2code)
  - [`migrations/20261002190000_connector_definitions.sql`](https://github.com/GetStream/Vision-Agents/blob/ead4a273f4d3623fff2a2286d5422725aa0af2e2/acceleration/migrations/20261002190000_connector_definitions.sql), [`20261002193000_connector_connections.sql:31,39,53-55,63,68-70`](https://github.com/GetStream/Vision-Agents/blob/ead4a273f4d3623fff2a2286d5422725aa0af2e2/acceleration/migrations/20261002193000_connector_connections.sql#L31), [`20261002193100_connector_authorization_attempts.sql`](https://github.com/GetStream/Vision-Agents/blob/ead4a273f4d3623fff2a2286d5422725aa0af2e2/acceleration/migrations/20261002193100_connector_authorization_attempts.sql)
- Code for «End to end» and «How the agent knows it is the same person» (same commit): [`internal/phone/stream.go:21-25`](https://github.com/GetStream/Vision-Agents/blob/ead4a273f4d3623fff2a2286d5422725aa0af2e2/acceleration/internal/phone/stream.go#L21-L25), [`internal/dispatch/dispatch.go:59-60`](https://github.com/GetStream/Vision-Agents/blob/ead4a273f4d3623fff2a2286d5422725aa0af2e2/acceleration/internal/dispatch/dispatch.go#L59-L60), [`internal/api/dispatchws.go:381`](https://github.com/GetStream/Vision-Agents/blob/ead4a273f4d3623fff2a2286d5422725aa0af2e2/acceleration/internal/api/dispatchws.go#L381), [`cmd/router/main.go:918-926`](https://github.com/GetStream/Vision-Agents/blob/ead4a273f4d3623fff2a2286d5422725aa0af2e2/acceleration/cmd/router/main.go#L918-L926), [`internal/chatlog/chatlog.go:193-196`](https://github.com/GetStream/Vision-Agents/blob/ead4a273f4d3623fff2a2286d5422725aa0af2e2/acceleration/internal/chatlog/chatlog.go#L193-L196), [`internal/session/manager.go:228-237`](https://github.com/GetStream/Vision-Agents/blob/ead4a273f4d3623fff2a2286d5422725aa0af2e2/acceleration/internal/session/manager.go#L228-L237), [`internal/api/sessions.go:513`](https://github.com/GetStream/Vision-Agents/blob/ead4a273f4d3623fff2a2286d5422725aa0af2e2/acceleration/internal/api/sessions.go#L513).

  Industry: public docs of Botpress, Twilio, Intercom, Chatwoot, Microsoft and Rasa, read on October 2. The links are in «How other companies do it».
