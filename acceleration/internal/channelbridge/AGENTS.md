# internal/channelbridge

The channel bridge (T57, AI-878). It moves messages between an external thread, such as a Slack thread, and the thread channel in Stream Chat that holds it. It is Router code above the connector layer: it uses `internal/connectors` (manifest, `core.Transports`) and Stream Chat, and the connector layer uses neither it nor Stream Chat. The design is «Who moves messages: the channel bridge» in `docs/connectors/channels.md` on `connectors/planning`.

## Why here

- Not under `internal/connectors/`: that tree is the connector layer, adapters that import `core`. The bridge writes to Stream Chat, which the connector layer must not know (channels.md, «Dependency direction»).
- Not under `internal/channels/`: that directory is the WhatsApp, SMS and iMessage lines of `internal/channels`, which move onto the bridge in T62 (AI-921). Until then they keep their own hook path and tables, and the bridge shares neither.

## Flow

```
Slack -> POST /v1/connectors/events/{connector}/{app}   api.receiveProviderAppEvent
  api.ProviderApp: the app's record and signing secret   unknown app: 404
  Verifier.Verify with that secret                       another app's secret: 401
  Bridge.Deliver(app, messages)              on the request, Postgres only
    connection   store.AppConnectionByAccount(customer, connector, provider unit)
    agent        store.AgentConfigsBindingConnection: exactly one, else dropped
    thread       store.LinkChannelThread: one thread channel per external thread
    claim        store.ClaimChannelThreadMessage: a retried delivery is dropped
    episode      omnichannel.Cards.Open: contact map row of the author, episode
                 opened on the thread's first message (episodeSources)
  200
  Bridge.write                                off the request
    author user, agent user (the channel's id), channel (agent_config_id,
    support_customer_id, support_agent_id), message without source -> Stream Chat
    omnichannel.Cards.Write: a new episode's card (source slack) -> the omni-channel
Stream Chat -> message.new -> api.receiveMessageEvent -> api.answerThread
  claim (thread channel, turn, Stream message id)
  lease the turn on channel_threads (one router per thread at a time)
  the session on the channel (ByAgentWhere), else a persistent text session from
  the channel's agent config, ConversationID agent:thread-<uuid>
  Session.FollowUp(text): the reply is written into the thread channel
conversation flush: final text stored (UpdateMessagePartial, no webhook)
  -> Service.OnFinishedReply -> Bridge.Reply        the one hand-off point
    claim (thread channel, reply, Stream message id)
    Bridge.send                               off the caller
      manifest reply template (ResolvedManifest.Reply)
      core.Transports: resolver credential, scheme Wrap, egress
      2xx and reply.accepted (Slack: ok true); no answer, 5xx, 429: again
      after 2 s, 10 s, 30 s, then unclaimed
      refused: scheme.Classify; invalid_grant -> Resolver.Invalidate
```

## Terms

| Term | What it is | In code |
| --- | --- | --- |
| thread channel | The agent channel that holds one external thread, `agent:thread-<uuid>` | `store.ChannelThread.ChannelID` |
| thread link | One external thread (customer, connector, provider unit, thread key) to its thread channel, with the connection replies use | table `channel_threads`, `store.LinkChannelThread` |
| author user | The Stream Chat user an external author writes as: one per customer, connector, provider unit and author | `authorUserID` |
| episode card | One message in the omni-channel of whoever started a thread, with `source`, for the thread's episode (T41) | `internal/omnichannel`, table `episodes` |

## Rules

- **Acknowledge fast, write later.** `Deliver` touches Postgres only, so the provider gets its answer within its limit (Slack: three seconds, https://docs.slack.dev/apis/events-api/). Stream Chat writes and replies run on goroutines; `Close` waits for them. A write that fails after the answer is logged; the provider does not retry it. `Deliver` then calls its `unanswered`, so the customer's event destinations of unhandled events get the delivery (AI-924).
- **A retried delivery is dropped by the provider's message id**, in the thread channel (`channel_thread_messages`). Example: Slack sends the same `message` event with `X-Slack-Retry-Num: 1`; the claim fails and nothing is written.
- **The person's message has no `source`.** That is how the message hook (`api.addressed`) knows a person wrote it, and how the conversation on the channel reads it back as a user turn (`conversation.messageFromThread`). The session is told it with `FollowUp`, which writes no second copy.
- **A thread channel is a conversation.** The session that answers holds its persistent conversation on the thread channel, so the reply is kept there: the Router's own session (`api.threadSession`) and one a caller opens through `POST /v1/agents/sessions` with `agent_id` naming the channel (`api.threadConversation`). The hook does not hand a thread channel's message to a dispatch worker.
- **One hand-off point for replies.** The conversation calls `Reply` once a reply's final text is stored; the webhook never carries it (`UpdateMessagePartial` sends none). `Reply` claims the reply by its Stream message id, so a reply written again is sent once. Example: a reply a login later marks is told twice and leaves once.
- **One turn per thread, across routers.** The hook leases the thread's turn on its `channel_threads` row (`store.TakeChannelThreadTurn`) before it tells the session, and lets go after the reply; a router that stops holds it at most `askTimeout` + 30 s. A session the hook opens is closed after the turn, so the next turn on any router reopens the conversation from the channel. Example: router A answers Alice; Bob's message reaches router B, which waits for A's lease, then answers with Alice's turn in its history.
- **A failed send is sent again.** No answer, a 5xx, a 429, or a refusal the scheme reads as transient or rate limited: the reply is sent again after each of `Options.RetryBackoff` (2 s, 10 s, 30 s). The claim stays while it is retried; a reply never sent is unclaimed, so its next hand-off sends it.
- **A refusal the transport cannot see still ends the grant.** Slack answers a revoked token with HTTP 200 `invalid_auth`; the scheme's `Classify` reads it and `Reply` calls `Resolver.Invalidate`.
- **One agent per connection.** The agent that answers is the one agent config of the customer that binds the connection as `fixed`. None or two: the message is dropped and logged.
- **Replies leave only through `core.Transports`**, so the credential, the scheme and the egress checks are the connector layer's. The bridge holds no token.
- **One card for each thread, in its starter's omni-channel.** The first message of a thread opens its episode; the later ones, anybody's, find it open. Example: Alice starts a thread, Bob replies; the card is in Alice's omni-channel only. Slack has no phone number, so a Slack user's omni-channel is keyed by workspace and user until account linking joins it to a phone's.
- **A connector gets cards once it has an episode source** (`episodeSources`): only `slack_bot` today. The manifest's `channel` block does not say what its authors are to the contact map; iMessage (T36), WhatsApp (T51) and SMS (T53) add theirs, keyed by the number.
- **No text and no author in logs.** They are a person's.

## Open

- **Missed messages are not recovered.** Slack retries an event three times over about five minutes (https://docs.slack.dev/apis/events-api/); an event every delivery of which failed is lost. The design for the recovery, not built: keep the newest provider ts of each thread on its `channel_threads` row; when the events route has failed deliveries or after a router restart, read `conversations.replies` (thread) and `conversations.history` (top-level) since that ts through `core.Transports`, and hand each message to `Deliver`, whose inbound claim drops what was already taken.

## Tests

From `acceleration/`:

```bash
go test ./internal/channelbridge
ROUTER_POSTGRES_DSN=... ROUTER_REDIS_ADDR=... \
  go test -tags integration -run 'TestSlackChannelSuite|TestConnectorEventsSuite|TestMessageHooksSuite' ./internal/api
ROUTER_POSTGRES_DSN=... go test -tags integration -run 'TestStoreSuite/(TestTheFirst|TestASecondMessage|TestARetried)' ./internal/store
```

`SlackChannelSuite` (`internal/api/provider_app_events_test.go`) runs the whole flow against Postgres, the suite's Stream Chat (`chattest`), real persistent sessions and `fakeprovider` with `SlackChannel`. `chattest` sends no webhooks, so the suite delivers the `message.new` Stream Chat would send, built from the message `chattest` stored. `go test -run TestThreadChannelSuite ./internal/conversation` covers the conversation side: history and the hand-off.
