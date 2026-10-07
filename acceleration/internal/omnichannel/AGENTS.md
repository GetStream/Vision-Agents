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

## Open

- T55 closes and summarizes a card; T56 and T42 read cards; T62 moves `channel_identities` onto `contact_map`.

## Tests

```bash
go test ./internal/omnichannel
ROUTER_POSTGRES_DSN=... ROUTER_REDIS_ADDR=... \
  go test -tags integration -run 'TestEpisodeCardsSuite|TestSlackChannelSuite' ./internal/api
```
