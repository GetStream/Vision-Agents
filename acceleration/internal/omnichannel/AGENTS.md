# internal/omnichannel

The contact map and the episode cards (T43 and T41, AI-883). The design is «The episode card» and «How the agent knows it is the same person» in `docs/connectors/channels.md` on `connectors/planning`.

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
                                      before Until, and for a call not before it started
                 render               system note + one user message, oldest first,
                                      at most maxCardRunes
```

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
- **A call's lines are its own.** A call channel named after the number rung (`phone-{{called_number}}`) holds every caller's calls, so a card's lines end at `Until` (the episode's end, its session's close, or the next episode in the channel) and a call's start at its own start.
- **Card text is data.** The cards come as one user message, behind `cardsAttribution`, as restored shared history comes behind conversation's note.

## Bounds

| Bound | Value | Source |
| --- | --- | --- |
| Cards read | 5, newest | a choice: each costs one Stream Chat read before a call is joined |
| Lines of a card without a summary | the last 20 | a choice |
| Characters handed over, note included | 15,000 | `conversation.MaxHistoryRunes / 4` |
| Time for the whole read | 5 s | a choice; past it, the cards read so far |

## Open

- T55 closes and summarizes a card; T62 moves `channel_identities` onto `contact_map`.
- A text session on a channel other than a thread channel reads no cards: the contact map keys no in-app user yet (`contact_map.user_id`, T62).
- A session a caller opens on a thread channel through the API and keeps open is checked for a shared thread once, when it opens; the Router's own sessions open for each turn (`api.answerThread`).

## Tests

```bash
go test ./internal/omnichannel
ROUTER_POSTGRES_DSN=... ROUTER_REDIS_ADDR=... \
  go test -tags integration -run 'TestEpisodeCardsSuite|TestSlackChannelSuite' ./internal/api
ROUTER_POSTGRES_DSN=... go test -tags integration -run TestEpisodeReadingSuite ./internal/session
```
