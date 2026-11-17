---
name: generative-ui
description: How an agent's interactive UI (OpenUI, A2UI) travels from the model to the apps - the agent.yaml `ui:` key and the prompt it compiles, the router's streaming splitter, the `ui_delta`/`ui_completed` socket frames, what is kept in Stream Chat, and LiveUI in the client SDKs. Read before changing any of them or debugging a card that is missing, stuck or jumps.
---

# Generative UI

An agent can answer with a card instead of text: a restaurant with its tables, a trip with a map.
The model writes the card as a program in a fenced block, the router splits it off its words as it
streams, and the apps render it with their own renderer (`stream-chat-react`'s `AIUIHost`,
`stream_chat_flutter_ai`, StreamChatAI). The router knows no UI format beyond OpenUI patches; it
carries the program as text.

```
model tokens
  -> agent: ui_stream.go splits UI from prose (before stage directions and TTS)
       -> prose: spoken on a call, written in chat
       -> UIDelta / UICompleted events
            -> session socket: ui_delta / ui_completed frames (no buffering)
            -> conversation: Stream Chat message custom.ai_ui (partial updates, then settled)
            -> chatlog: custom.ai_ui on a call's transcript message
  -> agent history keeps the UI block, so the model can patch it next turn
```

## The agent.yaml `ui:` key

`ui:` tells the model what it may draw. The SDK that loads the folder compiles it into a
`## Interactive UI Generation` section appended to `instructions.md`, so it is synced as part of
the instructions and the router sees only text. The output is deterministic, so the prompt
prefix stays cacheable across turns.

```yaml
# agents/concierge/agent.yaml
name: concierge
llm: llm-fast
ui:
  format: openui            # the default, or a2ui
  keep: [CardHeader, Form, Chips, FollowUpBlock]   # built-ins to offer; their children come along
  components: ui/components.json   # default path; read when the file exists
  blueprints: ui/blueprints.yaml   # default path; read when the file exists
  patch_mode: true          # teach openui-patch for refining the previous card
  voice_ui: true            # a card on calls too, after one spoken sentence
  rules:
    - Always pair RestaurantCard with AvailabilityGauge when the wait is known.
```

| Key | What it does |
| --- | --- |
| `format` | `openui`, the default, or `a2ui`. Anything else is refused. |
| `keep` | OpenUI: restricts the built-in catalog (`uiBuiltins` in `sdks/go/agents/ui.go`) to these names plus their required children (`Form` brings `FormControl`, `Buttons`, `Button`); `Card` is always in. A2UI: names the basic catalog's components and functions to offer, and nothing else comes along, as stream-chat-swift's own prompt generator does. Empty offers all of them. A name that is neither built in nor a custom component is refused. |
| `components` | The app's own components in the portable format every app reads (`UIComponents.json`): `name` (snake_case), `description`, `parameters` (JSON schema with `x-order`, the positional order), `events`, optional `slots`. Each becomes a PascalCase signature: `restaurant_card` -> `RestaurantCard(name: string, ...)`. |
| `blueprints` | A YAML list of `name`, `when`, `order`, `rules`, `example`. Each is a line telling the model which components, in which order, for which kind of ask. An `example` is a program in the format, checked at load so it renders as written: OpenUI for unknown components, too many arguments and undefined or orphan references, with the line of the problem; A2UI as stream-chat-swift's `A2UIProgram` checks it, for messages it can't read, components and properties the catalog doesn't have, missing required properties, children never defined, components nothing holds, cycles and nesting past 50. |
| `patch_mode` | OpenUI only: adds the rule to answer a refinement of the previous card with an `openui-patch` block of only the changed statements. Refused with `a2ui`, since only OpenUI patches are merged and an A2UI reply writes a new surface. |
| `voice_ui` | Changes the opening rule from "no UI block on a voice call" to "one spoken sentence, then the card". |
| `rules` | Appended as `### Brand & Design Rules`. |

Paths are relative to the agent directory and may not leave it (symlinks included); each file is
capped at 512 KiB.

Loaded by: Go (`sdks/go/agents/ui.go`, from `Load`) and JavaScript (`sdks/js/src/ui-catalog.ts`,
from `loadFolder` in `folder.ts`). Python, .NET, Ruby, Rust and PHP don't read `ui:` yet (see the
note at the end of the `sdk` skill), and the `getstream` CLI refuses it as an unknown key
(`cli/internal/agentrc/agentrc.go`, `KnownFields`), so `getstream agents sync` fails on it. The two compilers must produce the same bytes for the same
folder: `sdks/testdata/ui/concierge/` is compiled by a Go test and a JavaScript test, both held to
`sdks/testdata/ui/concierge.instructions.txt`, which Go writes. An SDK that adds `ui:` checks itself
against the same pair. `a2ui` is compiled by Go (`sdks/go/agents/ui_a2ui.go`) and JavaScript
(`sdks/js/src/ui-a2ui.ts`), both held to `sdks/testdata/ui/concierge-a2ui/` and
`concierge-a2ui.instructions.txt`. JavaScript parses this YAML by hand; it still differs from yaml.v3 on
escapes inside quoted strings (`'it''s'`), empty list items, and `ui:` written as a flow mapping.

## How the model writes UI

The compiled prompt asks for one short sentence, then the card in a ```` ```openui-lang ```` block,
`root = Card([...])` first so the shell streams at once. With `format: a2ui` the card is A2UI 1.0
messages in a ```` ```ui:a2ui@1.0 ```` block, one JSON object per line: a `createSurface` naming the
catalog `getstream.io:app-catalog`, the id the apps' A2UI renderers give the basic catalog with their
own components, then `updateComponents`, `root` first. The prompt is the one stream-chat-swift's
renderer writes for itself (`A2UIInstructions`): its protocol and guidelines, then the catalog as
JSON Schema, the basic catalog's components and functions that `keep` names beside the app's own,
each app component as the renderer has it (`PortableComponent`: its parameters as properties, its
events joining its description). `sdks/go/agents/a2ui/` holds the basic catalog and the protocol as
the renderer has them, which `a2ui/generate.sh` writes again from a stream-chat-swift checkout; a Go
test holds the catalog to the one in `swift_assistant/a2ui/instructions.md`, which the renderer
wrote, so regenerate both together. JavaScript carries the same three files as a module,
`sdks/js/src/generated/a2ui.ts`, which `npm run a2ui` in `sdks/js` writes from Go's copies; run it
after `generate.sh`, or a JavaScript test fails. The router's splitter
(`acceleration/internal/agent/ui_stream.go`) reads:

- ```` ```openui-lang ```` and ```` ```openui-patch ```` (OpenUI, replace or patch)
- ```` ```ui:<format>[@<version>][+patch] ````, such as `ui:a2ui@1.0`

A block opens only on a fence at the start of a line. The splitter runs on every agent's reply,
UI or not, so prose that mentions UI (`wrap the list in a <ui> element`) stays prose. Only the
first UI block of a reply is its UI. A fence split across tokens is held back until it is known,
never shown as prose for a moment.

A patch is statements that replace those with the same identifier, add new ones at the end, or
remove one with `name = null` (a `$variable` set to `null` stays). `MergeOpenUIStatements`
(`conversation/ui.go`) and its copies in `sdks/js/src/conversation.ts` and
`sdks/dart/.../core/conversation.dart` do the merge.

## In the router

- `agent.go` runs every chunk through `uiStream.Add` before stage directions and the TTS
  chunker, so `[header, map]` isn't stripped as a direction and a call never reads UI code
  aloud. It sends `UIDelta` as UI arrives and `UICompleted` when the block closes, and keeps the
  block in the assistant's history (`HistoryBlock`) so the model can patch it.
- `api/sessionws.go` writes them to the session socket with no timer:
  - `{"type":"ui_delta","turn_id","format","version"?,"mode"?,"delta"}`
  - `{"type":"ui_completed","turn_id","format","version"?,"mode"?,"content"}`

  `mode` is left out for `replace`. A delta is the next piece of the raw block; a patch's delta is
  the patch, not the merged program.
- `conversation/conversation.go` folds them into the reply's `custom.ai_ui`
  (`{v, format, version?, status: "streaming"|"complete", content}`), merging a patch onto the
  last completed UI of the conversation (seeded from history when it reopens). Stream Chat gets
  partial updates every `answerEvery` (150ms) while the reply is written and the settled UI when
  it finishes. `conversation_updated` broadcasts for text deltas are coalesced to the same 150ms;
  one held back is sent from the live loop's tick (`publishHeld`), so the last words before a
  pause reach watchers without waiting for the next event.
- A UI that would take the message's custom data over 48 KiB is left out of that write
  (`fitUI`): the reply keeps its words and has no card.
- `chatlog/chatlog.go` keeps `custom.ai_ui` on a call's transcript message, so a card drawn on a
  call is in the chat afterwards.

## In the client SDKs

- `Turn.ui` (JavaScript, Dart and Swift `TurnUI`) is the turn's UI from the socket:
  `format`, `version`, `status`, `content`, with patches merged onto the transcript's previous
  card. A patch with no card before it in this transcript, as after a reload, sets no `ui`:
  Stream Chat's copy, which the router merged, is the one to show.
- `LiveUI` lays the socket's UI over the Stream Chat reply while it is written, so the card draws
  at socket speed instead of at Chat's 150ms. JavaScript's `AgentChat` applies it to every
  message it hands out; in Dart, the app calls `liveUI.observe` and `liveUI.overlay` itself
  (`flutter_assistant/lib/agent_chat.dart`), and in Swift too (`swift_assistant`'s
  `AgentChat.swift`, which publishes each change inside an animation). Swift's and JavaScript's
  `TurnUI.content` also hold back the OpenUI statement no newline has ended yet while it streams,
  so a card grows a part at a time rather than reflowing with every delta, and `LiveUI` leaves a
  reply its text until one statement is finished (Dart doesn't yet). The rules, which keep a card
  from jumping:
  - A reply that names its turn (`support_runtime.turn_id`) shows only that turn's UI.
  - Before its turn id is in Chat, the newest generating reply shows `latest()`: the newest
    turn's UI, only when that turn is the agent's. The person's question (`said`) and a new
    `responding` turn clear it, so a new reply never shows the card before it.
  - Once Chat says the reply is written (`generating` false) or its `ai_ui` is complete, what
    Chat stored shows.

## Debugging

- Card never shows: is it in the reply's `custom.ai_ui` in Stream Chat? If not, did the model
  write a fence the splitter reads (router logs, or the `ui_delta` frames on the socket)? Was it
  over 48 KiB?
- Card shows UI code as text, or a call reads it aloud: the block didn't open as UI, usually a
  fence info string the splitter doesn't know.
- Card blanks or jumps between turns: check the `LiveUI` rules above against the message's
  `support_runtime.turn_id`, `generating` and `ai_ui.status`.
- A component the model places but the app can't render: compare the agent's `ui/components.json`
  and `keep` with what the app's renderer registers. Flutter's `OpenUIRenderer` doesn't render
  every built-in yet.

## Known gaps

- The agent, the conversation and the chatlog each keep their own "previous UI" for patches; a
  patch after a tool round can be merged onto a different base in each.
- An interrupted call reply can leave a half-written `ai_ui` with `status: "streaming"` in the
  transcript, and a refused one is saved with its UI.
- The UI block is in the assistant's history `Content`, which `lastAssistantSaid`, the echo
  filter and the flow controller read as speech.
- `ui_delta` and `ui_completed` aren't declared in `internal/api/handwritten.go` yet.
