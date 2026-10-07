# internal/channelbridge

The channel bridge (T57, AI-878). It moves messages between an external thread, such as a Slack thread, and the thread channel in Stream Chat that holds it. It is Router code above the connector layer: it uses `internal/connectors` (manifest, `core.Transports`) and Stream Chat, and the connector layer uses neither it nor Stream Chat. The design is «Who moves messages: the channel bridge» in `docs/connectors/channels.md` on `connectors/planning`.

## Why here

- Not under `internal/connectors/`: that tree is the connector layer, adapters that import `core`. The bridge writes to Stream Chat, which the connector layer must not know (channels.md, «Dependency direction»).
- Not under `internal/channels/`: that package is the older WhatsApp, SMS and iMessage lines on agent configs, with its own hook path. The bridge does not share its code or its tables.

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
  200
  Bridge.write                                off the request
    author user, agent user (the channel's id), channel (agent_config_id,
    support_customer_id, support_agent_id), message without source -> Stream Chat
Stream Chat -> message.new -> api.receiveMessageEvent -> api.answerThread
  claim (thread channel, turn, Stream message id)
  the session on the channel (ByAgentWhere), else a persistent text session from
  the channel's agent config, ConversationID agent:thread-<uuid>
  Session.FollowUp(text): the reply is written into the thread channel
conversation flush: final text stored (UpdateMessagePartial, no webhook)
  -> Service.OnFinishedReply -> Bridge.Reply        the one hand-off point
    claim (thread channel, reply, Stream message id)
    Bridge.send                               off the caller
      manifest reply template (ResolvedManifest.Reply)
      core.Transports: resolver credential, scheme Wrap, egress
      2xx and reply.accepted (Slack: ok true)
      refused: scheme.Classify; invalid_grant -> Resolver.Invalidate
```

## Terms

| Term | What it is | In code |
| --- | --- | --- |
| thread channel | The agent channel that holds one external thread, `agent:thread-<uuid>` | `store.ChannelThread.ChannelID` |
| thread link | One external thread (customer, connector, provider unit, thread key) to its thread channel, with the connection replies use | table `channel_threads`, `store.LinkChannelThread` |
| author user | The Stream Chat user an external author writes as: one per customer, connector, provider unit and author | `authorUserID` |

## Rules

- **Acknowledge fast, write later.** `Deliver` touches Postgres only, so the provider gets its answer within its limit (Slack: three seconds, https://docs.slack.dev/apis/events-api/). Stream Chat writes and replies run on goroutines; `Close` waits for them. A write that fails after the answer is logged; the provider does not retry it.
- **A retried delivery is dropped by the provider's message id**, in the thread channel (`channel_thread_messages`). Example: Slack sends the same `message` event with `X-Slack-Retry-Num: 1`; the claim fails and nothing is written.
- **The person's message has no `source`.** That is how the message hook (`api.addressed`) knows a person wrote it, and how the conversation on the channel reads it back as a user turn (`conversation.messageFromThread`). The session is told it with `FollowUp`, which writes no second copy.
- **A thread channel is a conversation.** The session that answers holds its persistent conversation on the thread channel, so the reply is kept there: the Router's own session (`api.threadSession`) and one a caller opens through `POST /v1/agents/sessions` with `agent_id` naming the channel (`api.threadConversation`). The hook does not hand a thread channel's message to a dispatch worker.
- **One hand-off point for replies.** The conversation calls `Reply` once a reply's final text is stored; the webhook never carries it (`UpdateMessagePartial` sends none). `Reply` claims the reply by its Stream message id, so a reply written again is sent once. Example: a reply a login later marks is told twice and leaves once.
- **A refusal the transport cannot see still ends the grant.** Slack answers a revoked token with HTTP 200 `invalid_auth`; the scheme's `Classify` reads it and `Reply` calls `Resolver.Invalidate`.
- **One agent per connection.** The agent that answers is the one agent config of the customer that binds the connection as `fixed`. None or two: the message is dropped and logged.
- **Replies leave only through `core.Transports`**, so the credential, the scheme and the egress checks are the connector layer's. The bridge holds no token.
- **No text and no author in logs.** They are a person's.

## Tests

From `acceleration/`:

```bash
go test ./internal/channelbridge
ROUTER_POSTGRES_DSN=... ROUTER_REDIS_ADDR=... \
  go test -tags integration -run 'TestSlackChannelSuite|TestConnectorEventsSuite|TestMessageHooksSuite' ./internal/api
ROUTER_POSTGRES_DSN=... go test -tags integration -run 'TestStoreSuite/(TestTheFirst|TestASecondMessage|TestARetried)' ./internal/store
```

`SlackChannelSuite` (`internal/api/provider_app_events_test.go`) runs the whole flow against Postgres, the suite's Stream Chat (`chattest`), real persistent sessions and `fakeprovider` with `SlackChannel`. `chattest` sends no webhooks, so the suite delivers the `message.new` Stream Chat would send, built from the message `chattest` stored. `go test -run TestThreadChannelSuite ./internal/conversation` covers the conversation side: history and the hand-off.
