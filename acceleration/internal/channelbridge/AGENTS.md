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
    addressed    a reply not to the bot on a thread nobody linked waits
                 (store.WaitChannelThreadMessage) for the message that links it
    thread       store.LinkChannelThread: one thread channel per external thread;
                 the message that starts it takes the replies that waited
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
      refused: scheme.Classify; invalid_grant -> Resolver.Invalidate; unclaimed,
      logged with the answer's error member (Slack: not_in_channel)
      2xx it cannot read: claim kept (the provider may have posted), logged
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
- **The agent gets the person's words without the mention of the account** (AI-990 F29). `take` writes `MessageRule.WithoutMention` of the connection's pinned revision: in Slack «<@U0000BOT> is the build green?» is written as «is the build green?». A mention of anybody else stays; a message that is only the mention is written as it is. A connector without `addressed.mention` is written as the provider sent it. Check: `go test -tags integration -run 'TestSlackChannelSuite/(TestAMessageIsWritten|TestAMentionInAThread)' ./internal/api`.
- **The person's message has no `source`.** That is how the message hook (`api.addressed`) knows a person wrote it, and how the conversation on the channel reads it back as a user turn (`conversation.messageFromThread`). The session is told it with `FollowUp`, which writes no second copy.
- **A thread channel is a conversation.** The session that answers holds its persistent conversation on the thread channel, so the reply is kept there: the Router's own session (`api.threadSession`) and one a caller opens through `POST /v1/agents/sessions` with `agent_id` naming the channel (`api.threadConversation`). The hook does not hand a thread channel's message to a dispatch worker.
- **One hand-off point for replies.** The conversation calls `Reply` once a reply's final text is stored; the webhook never carries it (`UpdateMessagePartial` sends none). `Reply` claims the reply by its Stream message id, so a reply written again is sent once. Example: a reply a login later marks is told twice and leaves once.
- **One turn per thread, across routers.** The hook leases the thread's turn on its `channel_threads` row (`store.TakeChannelThreadTurn`) before it tells the session, and lets go after the reply; a router that stops holds it at most `askTimeout` + 30 s. A session the hook opens is closed after the turn, so the next turn on any router reopens the conversation from the channel. Example: router A answers Alice; Bob's message reaches router B, which waits for A's lease, then answers with Alice's turn in its history.
- **A failed send is sent again.** No answer, a 5xx, a 429, or a refusal the scheme reads as transient or rate limited: the reply is sent again after each of `Options.RetryBackoff` (2 s, 10 s, 30 s). The claim stays while it is retried; a reply never sent is unclaimed, so its next hand-off sends it. A refusal is unclaimed too, so a refused reply leaves no `reply` row in `channel_thread_messages` (AI-990 F28), and the log line names the answer's `error` member. A 2xx answer the bridge cannot read keeps the claim, since the provider may have posted the reply. Example: Slack answers HTTP 200 `{"ok": false, "error": "not_in_channel"}`; the row is gone and the log says `error "not_in_channel"`.
- **A reply that arrives before its thread's link waits for it** (AI-990 F31a). Slack retries a mention whose delivery failed after the replies in its thread arrived (https://docs.slack.dev/apis/events-api/, «Retries»). A reply that is not to the bot, on a thread nobody linked, is kept in `channel_thread_waiting` for 10 minutes. When a message that starts its thread links it, it takes those replies, and they are answered after it, in the same delivery. Its retry takes them too when its first delivery linked the thread and failed before the take; each waiting row is taken once. A link made by a reply, such as a mention in a thread of people, takes none: some came before the bot was spoken to (AI-989). The reply looks for the link again after it is kept, so a link made meanwhile finds it or it finds the link. Check: `go test -tags integration -run 'TestSlackChannelSuite/TestAReply(ThatArrivesBefore|ThatWaited)' ./internal/api`.
- **A reply that waited was already forwarded as unhandled** (AI-1001). When it arrives, no agent answers it, so `Deliver` returns `answered` false and the events route sends its delivery to the customer's destinations of unhandled events. When the mention then links the thread, the agent answers the reply too. Nothing takes the forward back, and no event says that the agent answered it. If the reply's write fails after the take, `Deliver` does not call `unanswered`: that is the mention's delivery, which the destinations must not get as unhandled, and they already have the reply's own. Example: Bob's reply arrives at 10:00:01 and goes to the customer's URL; Slack retries Alice's mention at 10:01:00; the agent answers both. Check: `go test -tags integration -run 'TestEventForwardingSuite/TestAReplyThatWaited' ./internal/api`.
- **A refusal the transport cannot see still ends the grant.** Slack answers a revoked token with HTTP 200 `invalid_auth`; the scheme's `Classify` reads it and `Reply` calls `Resolver.Invalidate`.
- **One agent per connection.** The agent that answers is the one agent config of the customer that binds the connection as `fixed`. None or two: the message is dropped and logged.
- **Replies leave only through `core.Transports`**, so the credential, the scheme and the egress checks are the connector layer's. The bridge holds no token.
- **One card for each thread, in its starter's omni-channel.** The first message of a thread opens its episode; the later ones, anybody's, find it open. Example: Alice starts a thread, Bob replies; the card is in Alice's omni-channel only. Slack has no phone number, so a Slack user's omni-channel is keyed by workspace and user until account linking joins it to a phone's.
- **A connector gets cards once it has an episode source** (`episodeSources`): `slack_bot` (`slack`, keyed by workspace and user), `linq` (`imessage`, keyed by the sender's E.164 number with `omnichannel.Phone`; an author it refuses, such as an email handle, gets no card) `telnyx` (`sms`, keyed by the number the same way) and `whatsapp` (`whatsapp`, T51: Meta writes the author as digits with no `+`, so the person and their opt-outs are keyed by `+` and the digits, `recipient`; the thread key and the reply's `to` stay as Meta wrote them). The manifest's `channel` block does not say what its authors are to the contact map. Check: `go test -tags integration -run TestLinqChannelSuite/TestAChatIsOneThreadChannelAndOneIMessageCard ./internal/api`.
- **SMS answers the carriers' keywords before the agent** (`keywords.go`, T53, wave 3c Q9). A connector whose episode source names an opt-out channel (`optOuts`: `telnyx` `sms`, `whatsapp` `whatsapp`, `linq` `imessage`) has them; Meta answers none of them itself, so the bridge answers all three on WhatsApp. After the inbound claim and before the card, a message that is STOP (or STOPALL, END, UNSUBSCRIBE, CANCEL, QUIT, REVOKE, OPT OUT), START (UNSTOP) or HELP, read without case or punctuation, reaches no agent and is answered by the bridge with internal/channels' texts unless the provider answered it itself (`answered`: Telnyx always answers STOP, START and HELP and says so in `data.payload.autoresponse_type`, so the bridge answers only the rest, such as REVOKE and OPT OUT); STOP and START write the customer's `opt_outs` (`source: keyword`), the record `dlc.Gate` and the opt-out API read. A person with a live opt-out reaches no agent, and `Reply` sends them nothing, not even a reply the agent finished after their STOP or one it sends again after a failed send. A store that fails releases the claim and answers 500, so the provider's retry records it. Sources: CTIA Messaging Principles and Best Practices (May 2023) 5.1.3, FCC 24-24 (47 CFR 64.1200(a)(10)), Telnyx's default opt-in/out words. `internal/channels` keeps its own list (`internal/channels/keywords.go`) until T62. Check: `go test -tags integration -run 'TestTelnyxChannelSuite/(TestStop|TestHelp|TestAReply|TestAStop|TestEach|TestAMessage|TestKeywords|TestARevoke)' ./internal/api`.
- **iMessage has the keywords too** (T62a, AI-921): `linq` names `imessage` as its opt-out channel. Linq answers none of them itself; it refuses later sends to a person who texted STOP with 403 (error 2024, https://docs.linqapp.com/channel/imessage/error/codes/2xxx/2024/index.md), so the bridge's STOP confirmation there is refused unless a send sets `override_optout` (not set yet). A Linq thread is a chat, so the person its replies reach is the one who started it (`replyTo`: the contact map row of its episode). Check: `go test -tags integration -run 'TestLinqChannelSuite/(TestAStop|TestAReply)' ./internal/api`.
- **Keyword texts come from the line's use case** (T62a): `store.UseCaseForNumber` with the provider unit, so the use case the number is assigned to, else the customer's default; an empty text is the one before. WhatsApp's provider unit is a `phone_number_id`, which no number is stored under, so a WhatsApp line has the default use case's texts. Check: `go test -tags integration -run 'TestTelnyxChannelSuite/(TestKeywords|TestAKeyword)' ./internal/api`.
- **The sandbox gate on texting lines** (T62a): `Options.Gate` is the router's `dlc.Gate`, the one `internal/channels` uses. On a connector with an opt-out channel, a message passes `Gate.Allow` after the keywords and before the episode (refused: it reaches no agent and opens no card), each reply passes it before it is claimed and before each resend, and a reply sent counts with `Gate.Sent`. Keyword answers go out past it, as on `internal/channels`. Slack has no opt-out channel and no gate. A nil gate lets everything through. Check: `go test -tags integration -run TestTelnyxSandboxSuite ./internal/api`.
- **A reply's files go as links** (T62a): `conversation.FinishedReply.Files` are appended to the text, one URL a line after a blank line (`withFiles`), on every provider; a reply that is only files is handed over too. Native media per provider is a later ticket.
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
