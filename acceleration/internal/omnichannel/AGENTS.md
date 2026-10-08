# internal/omnichannel

The contact map and the episode cards (T43 and T41, AI-883), and closing and summarizing them (T55, AI-884). The design is «The episode card» and «How the agent knows it is the same person» in `docs/connectors/channels.md` on `connectors/planning`.

## Flow

```
channel bridge, a thread's message        session manager, a call's session joined
  Person: SlackUser(team, user)             Person: Phone(SIP caller | PhoneSpec.To)
  source slack, thread agent:thread-<id>    source call, thread agent:<agent id>
              \                               /
               Cards.Open           Postgres only
                 store.MapContact   (customer, agent config, kind, address)
                                    -> omni-channel agent:omni-<uuid>
                 store.OpenEpisode  open episode of the thread or the call session
               Cards.Write          only an episode Open opened
                 omni-channel       agent user = channel id, agent_config_id
                 card message       id episode-<episode id>, source, status,
                                    started_at, thread_channel, call_id
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
                 for each card        summarized: the card's text (GetMessage)
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
    thread episodes, last message                  call episodes of the call in the
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
| contact map | How a person is known (an E.164 number, or a Slack user) to their omni-channel | table `contact_map`, `store.MapContact` |
| episode | One call, or one run of messages on one external thread | table `episodes`, `store.OpenEpisode` |
| episode card | The episode's one message in the omni-channel | `Cards.Write` |

## Rules

- **A card has a source**, so the message hook (`api.addressed`) never answers it, though the omni-channel names its agent config.
- **One number is one person**, whatever it came in on: an SMS and a call from `+15550100100` are one contact map row and one omni-channel. `Phone` reads `+1 (555) 010-0100`, `0015550100100` and `+44 (0) 20 7946 0018` (as `+442079460018`), and refuses what does not say it is international: `5550100100`, `911`, `+44 (020) 7946 0018`. A refused number is no row and no card; the call goes on.
- **No channel id holds a number.** The omni-channel is `agent:omni-<uuid>`.
- **Call cards are opt-in.** Only a session under an agent config with `episode_cards` on writes one, so a call under any other config reads nothing of its call and writes nothing, as before the cards existed. Slack cards come from the channel bridge, which is new.
- **The raw text stays where it is.** A call keeps its transcript in its call channel; a thread in its thread channel. The card names that channel.
- **A Slack user is a person of their own** until account linking: keyed by workspace and user, so one person's Slack threads share an omni-channel.
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
- **A call ends with its call, never as idle.** The sweeper closes thread episodes only (`session_id IS NULL`).
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
- A text session on a channel other than a thread channel reads no cards: the contact map keys no in-app user yet (`contact_map.user_id`, T62).
- A session a caller opens on a thread channel through the API and keeps open is checked for a shared thread once, when it opens; the Router's own sessions open for each turn (`api.answerThread`).

## Tests

```bash
go test ./internal/omnichannel
ROUTER_POSTGRES_DSN=... ROUTER_REDIS_ADDR=... \
  go test -tags integration -run 'TestEpisodeCardsSuite|TestSlackChannelSuite' ./internal/api
ROUTER_POSTGRES_DSN=... go test -tags integration -run TestEpisodeReadingSuite ./internal/session
ROUTER_POSTGRES_DSN=... go test -tags integration -run TestCloserSuite ./internal/omnichannel
ROUTER_POSTGRES_DSN=... go test -tags integration -run TestEpisodeSweeperSuite ./cmd/router
```
