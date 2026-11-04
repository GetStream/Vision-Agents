# internal/omnichannel

The contact map and the episode cards (T43 and T41, AI-883), and closing and summarizing them (T55, AI-884). The design is «The episode card» and «How the agent knows it is the same person» in `docs/connectors/channels.md` on `connectors/planning`.

## Flow

```
channel bridge, a thread's message        session manager, a call's session joined
  Person: SlackUser(team, user)             Person: Phone(SIP caller | PhoneSpec.To)
  source slack, thread agent:thread-<id>    source call, thread agent:<agent id>
              \                               /
              |   api.chatEpisode, an app chat's message
              |     api.routeArrivingMessage: /v1/chat/internal/messages or the message hook
              |     config: the running session's spec, else ownerOf's; episode_cards on
              |     not a thread channel, an incognito or a call's session
              |     Person: StreamUser(message.user.id); source chat, thread agent:<channel id>
              |  /
               Cards.Open           Postgres only
                 store.MapContact   (customer, agent config, kind, address)
                                    -> omni-channel agent:omni-<uuid>
                                    a new stream_user or phone row of a linked pair
                                    takes its partner's omni-channel (joinPartner)
                 store.OpenEpisode  open episode of the thread or the call session
               Cards.Write          only an episode Open opened
                 omni-channel       agent user = channel id, agent_config_id
                 card message       id episode-<episode id>, source, status,
                                    started_at, thread_channel, call_id
```

Linking (AI-2701), the customer's backend:

```
PUT /v1/agents/users/{user_id} {"phone": "+15551234567" | null}    api.putUser, server-side only
  app: the request's pin (streamapp.Clients.Pin), 0 for the deployment's own
  phone: omnichannel.Phone, else 400; null unlinks
  store.LinkUserPhone   user_phones (customer, app, user_id, phone), one pair per app;
                        a number another user of the app holds: 409 phone_taken
    joinLinked          each agent config of the customer: the stream_user row
                        («<app pk>:<user id>») and the phone row of the pair, pinned to that
                        app, point at the older omni-channel
                        (whose oldest row was made first) and take its pin; no card moves
```

Reading (T56 and T42, AI-885), only for an agent config with `episode_cards` on:

```
session.Manager.Create, before the history is restored       callCards.read
  text session on a thread channel     voice session on a call
    store.ThreadContact(own thread)      calledParty -> Phone -> store.Contact
    alone: every person's message in        (read only: never makes a row)
    the thread is by its creator
              \                        /
               Cards.Context
                 store.EpisodeCards   same customer, same agent, same omni-channel;
                                      own thread and own call left out; newest 5
                 for each card        summarized: the card's text (GetMessage), when the
                                      message is in an omni-channel, by its own agent
                                      otherwise: last 20 lines of thread_channel,
                                      before Until, and for a call not before it started;
                                      a call outside agent:<call id>, or whose window
                                      holds another speaker: none
                 render               system note + one user message, oldest first,
                                      at most maxCardRunes
```

Closing (T55, AI-884), `Closer`:

```
idle sweeper, every minute                     call.session_ended hook (api.endCallEpisodes)
  Closer.Start, from cmd/router                  Closer.EndCall(scope, call id)
  only with connectors on or a config            any call; one with no episode
  with episode_cards on (startEpisodeSweeper)    matches no row and reaches no Stream
              |                                           |
  store.CloseIdleEpisodes                        store.EndCallEpisodes
    thread and chat episodes, last message         call episodes of the call in the
    older than episodes.idle_after                 event's app, still in progress
              \                                          /
               status ended, ended_at, summary lease (5 min): one router wins each row
               card: status ended                      partial update, no new message
               summary                                 config's own LLM (default llm-fast)
                 linesOf(the episode's own window, last 100, 60,000 runes)
                 ok:   card text = summary, status summarized; row summarized
                 fail: card status summary_failed; row summary_failed; text and thread kept
                 router stopped: row stays ended; the next sweep takes it once the lease runs out
                                 (store.ClaimEpisodeSummaries)
```

### Close API (for T48 and later callers)

| Call | What it does | Returns |
| --- | --- | --- |
| `NewCloser(CloserOptions{Store, Stream, LLM, IdleAfter})` | builds a closer; runs nothing | error without a store, Stream clients, an LLM or a positive idle period |
| `Closer.Start()` | the idle sweeper until `Close`: a sweep at once, then every `Every` (1 min) | — |
| `Closer.Sweep(ctx)` | one sweep: closes idle thread episodes, takes expired summary leases, summarizes each; returns when they are done | store error |
| `Closer.EndCall(ctx, store.AppScope, callID)` | closes the call's episodes now, summarizes them off the caller | store error |
| `Closer.Close()` | stops the sweeper and the summaries, and waits | — |

A card closed after the idle period is updated in the app its episode is pinned to (`episodes.stream_app_pk`, through `streamapp.Clients.ForApp`), the same app the card was written in.

## Terms

| Term | What it is | In code |
| --- | --- | --- |
| omni-channel | One agent channel for each person and agent of a customer, holding one card for each episode | `store.ContactMapEntry.ConversationID` |
| contact map | How a person is known (an E.164 number, a Slack user, or a Stream user of the app chat) to their omni-channel | table `contact_map`, `store.MapContact` |
| episode | One call, or one run of messages on one external thread or in one agent channel of the app chat | table `episodes`, `store.OpenEpisode` |
| episode card | The episode's one message in the omni-channel | `Cards.Write` |
| link | A customer's word that a Stream user of its app and a number are one person | table `user_phones`, `store.LinkUserPhone` |

## Rules

- **A card has a source**, so the message hook (`api.addressed`) never answers it, though the omni-channel names its agent config.
- **One number is one person**, whatever it came in on: an SMS and a call from `+15550100100` are one contact map row and one omni-channel. `Phone` reads `+1 (555) 010-0100`, `0015550100100` and `+44 (0) 20 7946 0018` (as `+442079460018`), and refuses what does not say it is international: `5550100100`, `911`, `+44 (020) 7946 0018`. A refused number is no row and no card; the call goes on.
- **No channel id holds a number.** The omni-channel is `agent:omni-<uuid>`.
- **Call cards are opt-in.** Only a session under an agent config with `episode_cards` on writes one, so a call under any other config reads nothing of its call and writes nothing, as before the cards existed. Slack cards come from the channel bridge, which is new.
- **The raw text stays where it is.** A call keeps its transcript in its call channel; a thread in its thread channel. The card names that channel.
- **A Slack user is a person of their own** until account linking: keyed by workspace and user, so one person's Slack threads share an omni-channel.
- **The app chat's person is the Stream user who wrote**: `message.user.id`, which Stream signed through the customer's own token, never a custom field of the user (a user may update their own user). Example: Anya (`anya_123`) writes in `agent:chat-1`; her card goes to the omni-channel of `stream_user:anya_123`.
- **App chat cards are opt-in.** Only a message answered under an agent config with `episode_cards` on opens an episode: the running session's spec, else the config `ownerOf` finds. Off, nothing is read or written beyond the config itself. A message with a source (the agent's own reply), a thread channel's (the bridge's), an incognito session's and a call session's open none: a chat episode in a call's channel would end the call card's lines at its start (`until`).
- **One chat episode per run of messages.** Every message calls `Open`, which keeps the episode open; only the message that opened it writes a card. A delivery Stream makes again, or the queue tries again, finds the open episode (`episodes_open_thread`) and writes nothing. A card that fails costs the card, not the answer: logged at ERROR, and the message is answered. A hook's card is written off its request, as its answer is.
- **An app chat's channel is the customer's by the config it names.** The customer's app makes it with `agent_config_id` and none of the router's stamps, so `linesOf` reads it for a chat card of that config (`theirs`); a stamped channel is read only for the customer it names.
- **A link makes one omni-channel of two.** The customer's backend says a Stream user of its app and a number are one person (`PUT /v1/agents/users/{user_id}`); under each agent the user's `stream_user` row and the number's `phone` row then point at the older of their omni-channels, and a row made later takes its partner's (`store.MapContact`). One number per user and one user per number within a Stream app: the same user id in another app is another person, and a number another user holds is a 409. No message moves: the cards written before stay where they were, and are read by their message id from any omni-channel, as long as its own agent wrote them (`writtenAsCard`). A link and a first sighting at once each read after their own write, so one sees the other. Unlinking (`"phone": null`), or a new number, stops the joining; rows already joined stay so. Example: Anya's SMS made `omni-A` on Monday and her app chat `omni-B` on Tuesday; once linked, both rows point at `omni-A`, and her next SMS reads the chat's card from `omni-B`.
- **The deployment's own app is one app, whatever its pin.** A chat row from it carries its id (the message hook), in its pin and its address, an SMS row none, and a request acting in it 0 (`store.linkApp`, `streamUserAddresses`).
- **Reading cards is opt-in too.** Off, `callCards.read` returns before any read, and the session is handed what it was before. Example: an SMS thread session under a config without `episode_cards` reads its thread as before and nothing else.
- **A person's cards are their omni-channel's.** `store.EpisodeCards` takes the contact rows of the same customer and agent that point at the omni-channel, so a linked Slack row and a phone row are one person, and the same number under another agent or customer is another. Example: `+15550100199`'s cards are never read for `+15550100100`.
- **A shared thread reads no cards.** A thread's cards are of the one who started it. Example: Bob writes in Alice's Slack thread; the session reads none of Alice's cards.
- **A call's lines are its own.** A call channel named after the number rung (`phone-{{called_number}}`) holds every caller's calls, so a card's lines end at `Until` (the episode's end, its session's close, or the next episode in the channel; index `episodes_thread`) and a call's start at its own start. A call card whose window still holds a person's line by anybody but `sip-<the card's number>` gives no lines at all. Example: Alice's and Bob's calls share a call id; neither is handed the other's words.
- **A call's lines come only from its own channel.** Only a call card whose `thread_channel` is `agent:<call id>`, chatlog's default, gives lines. A channel the session named (`agent_id` or `conversation_id`) can hold another call at the same time, and an agent line names nobody it answers. Such a card waits for its summary (T55). Example: two calls with `agent_id: front-desk`; the agent's «Thanks Alice» is never handed to Bob. This is isolation by channel, not an «overlapped» flag from `episodes`: a call under a config without `episode_cards` writes no episode, so no flag could see it.
- **The last lines whatever Stream's order.** The query is bounded by the window's end only, as Stream's documented backward pagination is; the start is applied in Go, and the lines are sorted by `created_at` before the tail is taken.
- **No cards for a native speech-to-speech session.** Its history goes into its instructions as a transcript (`agent.openSpeech`), which drops the system note, so the cards would lose the note that they are not authority. A cascaded session that started with cards is refused a move onto one (`session.ErrCardedToNative`; the API answers `carded_session_to_native`, 400, from `setSessionSettings` and `updateSession`); a session without cards moves as before.
- **A call card names where its transcript is.** Its `thread_channel` is the channel the transcript is written into (`Session.transcribedInto`), read as the transcript factory reads it (`Spec.TranscriptChannel`, `Spec.ConversationChannel`). A device's call under a thread channel's agent id writes none there (`persistent.BarThread`), so it has no episode.
- **The budget covers the call read.** `callCards.read` puts `calledParty` under `ReadTimeout` too, so a slow Stream delays a call's join by at most 5 s.
- **A card with nothing to say is skipped, not a stop.** `render` passes over a card with neither a summary nor a line, so the older cards are still read. Example: a call in a named channel at -60 s does not hide an SMS at -120 s.
- **An episode closes once, whatever the number of routers.** Each close is one transaction whose row lock one router wins (`FOR UPDATE SKIP LOCKED`, then the status and the last message checked again), and the winner holds the summary for `summaryLease`. Example: two routers sweep at the same second; each idle thread gets one summary.
- **A message keeps its thread open.** `store.OpenEpisode` sets `episode_activity.last_message_at` with each message, in the statement that finds the episode open and holds it against a close; a message that arrives after the close opens the next episode and card.
- **A call ends with its call, never as idle.** The sweeper closes thread and chat episodes only (`session_id IS NULL`): a chat episode has no session, so it closes after `episodes.idle_after` as a thread's does.
- **A failed summary keeps the raw text.** `summary_failed` changes the card's status only; the thread channel is never written. A call in a channel its session named gives no lines (`linesOf`), so its summary fails rather than mix in another caller's words.
- **The summary is the config's own LLM**, the agent config's `llm`, or `llm-fast` (a session's default) when it names none. It is not written to memory (follow-up: memory's `Scope.UserID` is «the customer», not the person from the contact map).
- **Card text is data.** The cards come as one user message, behind `cardsAttribution`, as restored shared history comes behind conversation's note.

## Bounds

| Bound | Value | Source |
| --- | --- | --- |
| Cards read | 5, newest | a choice: each costs one Stream Chat read before a call is joined |
| Lines of a card without a summary | the last 20 | a choice |
| Characters handed over, note included | 15,000 | `conversation.MaxHistoryRunes / 4` |
| Time for the whole read, the call's `GetCall` included | 5 s (`ReadTimeout`) | a choice; past it, the cards read so far |
| Idle period of a text episode | 1 h (`episodes.idle_after`), under 24 h | Kanat, 2026-10-07 (D4), `unverified` against traffic; 24 h is WhatsApp's window (AI-884) |
| Sweep interval, and episodes per sweep | 1 min, 10 | a choice |
| Summary lease | 5 min | a choice: longer than the 1 min a summary may take |
| Lines a summary reads | the last 100, at most 60,000 runes | `conversation.MaxHistoryMessages`, `conversation.MaxHistoryRunes` |
| Summary length | 400 tokens, 2,000 runes | a choice |

## Open

- T62 moves `channel_identities` onto `contact_map`.
- The summary is not written to memory: it would change what `memory.Scope.UserID` means (`internal/memory/memory.go:31-32`).
- A call whose `call.session_ended` arrives before its card's episode is opened (the card is written off the session's start) stays in progress: the sweeper closes thread episodes only.
- The sweeper's gate is read once, at start: a config that turns `episode_cards` on later has its calls closed by the hook, and its expired summary leases taken from the next start.
- A text session on a channel other than a thread channel reads no cards yet: the app chat writes its cards (`stream_user`), and reading them is the next change.
- A Stream user is keyed by user id alone, under the customer and the agent config: a customer whose config answers in two Stream apps has one person for a user id in both. A link is kept per app (`user_phones`) and joins only rows pinned to its app, so such a row is joined by the link of the app that made it.
- A data move carries no link (`store.dataTables`): the contact rows it joins stay behind too, so the customer links again where it moved to.
- A card written before a link and still in progress is closed as the omni-channel its row points at now (`Closer.mark` names that channel's agent as the author of the partial update); that a server-side partial update by another user is accepted is unverified against Stream.
- A session a caller opens on a thread channel through the API and keeps open is checked for a shared thread once, when it opens; the Router's own sessions open for each turn (`api.answerThread`).

## Tests

```bash
go test ./internal/omnichannel
ROUTER_POSTGRES_DSN=... ROUTER_REDIS_ADDR=... \
  go test -tags integration -run 'TestEpisodeCardsSuite|TestSlackChannelSuite|TestChatEpisodesSuite|TestUserPhonesSuite' ./internal/api
ROUTER_POSTGRES_DSN=... go test -tags integration -run TestEpisodeReadingSuite ./internal/session
ROUTER_POSTGRES_DSN=... go test -tags integration -run TestCloserSuite ./internal/omnichannel
ROUTER_POSTGRES_DSN=... go test -tags integration -run TestEpisodeSweeperSuite ./cmd/router
```
