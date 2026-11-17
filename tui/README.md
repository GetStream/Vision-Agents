# tui

A conversation with an agent, in a terminal.

It is the terminal counterpart of [the dashboard](../dashboard): the same conversation,
the same tool activity and the same timings, read where the work is being done rather
than in a browser. The agent is reached through a `Session`, which an
[`*agents.Session`](../sdks/go/agents) satisfies, so what is shown is whatever the
backend reports — the answer as it is written, the tools being run to find it and how
long each took, and whether what was said survived being saved.

```bash
go get github.com/GetStream/Vision-Agents/tui
```

## In writing

```go
err := tui.Run(ctx, tui.Options{
	Open: func(ctx context.Context, conversationID string) (tui.Session, error) {
		return agent.Chat(ctx, agents.ChatOptions{Persist: true, ConversationID: conversationID})
	},
	History:  tui.BackendHistory(stream.Backend{}, "jean"),
	Branding: tui.Branding{Title: "Jean", Subtitle: "answers questions about the weather"},
})
```

`Open` is the only thing required, and it is called again for every `/new` and `/resume`,
so where a conversation comes from stays the application's decision: a stored config, a
sandbox profile, an organization's memory filter. Whatever session it returns is closed
when it is replaced or left.

`History` reads messages from before the ones a session was given, which is what `/older`
pages through. Leaving it out is the honest thing for a conversation that is not
persisted, and `/older` says so rather than appearing to do nothing.

`Enter` sends, `alt-enter` writes another line, `esc` cancels the answer being worked on,
`PgUp`/`PgDn` scroll, and `ctrl-c` leaves. The commands are `/new`, `/resume <id>`,
`/older`, `/help` and `/quit`.

## What an application adds

```go
tui.Options{
	// Extra header lines, asked for again every frame.
	Header: func(state tui.State) []string {
		if state.Busy {
			return []string{"Researching " + state.Scope}
		}
		return []string{"Documentation and SDK source available"}
	},
	// Somewhere else to open the same conversation, as it becomes live.
	OnOpen: func(ctx context.Context, session tui.Session) error {
		return browse(dashboardURL(session.ID()))
	},
	// Slash commands of its own, run off the event loop.
	Commands: []tui.Command{{
		Name: "ticket", Args: "<id>", Help: "open a support ticket",
		Run: func(ctx context.Context, args []string) (string, error) {
			return "Opened ticket " + args[0], openTicket(ctx, args[0])
		},
	}},
}
```

`Branding` carries the banner, the title, the subtitle, the labels on the two sides of
the conversation, the composer's placeholder and the key reminder. `Theme` is seven
adaptive colours, defaulting to the dashboard's palette so the two read as one product;
setting one leaves the rest alone.

The product and SDK a tool reports are picked up on their own and shown in the header, so
an agent that researches a particular SDK says which one it is in without being asked to.

## The layout

The chrome measures itself. The header grows with what there is to say — a banner, a
subtitle, a scope, an application's own lines, a notice that the session was given less
than the whole history — and the conversation is given whatever is left. On a terminal too
short for all of it the banner goes first and the frames next, so the conversation is the
last thing to be given up rather than the first.

```
   ▄▄▄  STREAM
  ▀▄▄   SUPPORT
  ▄▄▀   source-backed answers
╭─────────────────────────────────────────────────────────────────────────────╮
│ Stream Support                              agent:support-27392eaf-1f43-4c1 │
│ source-backed answers                                                       │
│ chat / react                                                                │
│ Documentation and SDK source available                                      │
╰─────────────────────────────────────────────────────────────────────────────╯

▎ YOU
▎   For chat/react, what does useChatContext return?

▎ AGENT
▎   useChatContext returns ChatContextValue.
▎
▎   ✓ Search the documentation    24.9s  Verified 2 source citations
▎   ⠹ Read source                  4.4s  executing
▎   Completed · 30.2s · saved

╭─────────────────────────────────────────────────────────────────────────────╮
│ Ask about Chat, Video, Moderation or Feeds…                                 │
│                                                                             │
╰─────────────────────────────────────────────────────────────────────────────╯
  ⠹ Running tools · 30.2s total   enter send · alt-enter newline · esc cancel
```

## Try it

```bash
STREAM_ACCELERATION_URL=http://localhost:8080 \
STREAM_ACCELERATION_CUSTOMER_ID=acme \
go run ./cmd/chat -name jean
```

`cmd/chat` registers a `get_weather` function, so asking about the weather shows a tool
being run and timed. `-dir agents/jean` reads instructions, skills and knowledge from a
directory, `-conversation <id>` resumes a saved one, and `-log chat.log` keeps the logs
somewhere readable instead of throwing them away.

## Tests

```bash
go test -race ./...
```

No mocks. The conversation is driven against a real session on a real
`httptest.Server` with a real `gorilla/websocket` upgrader and a real history endpoint,
so what is tested is the exchange rather than a description of it. The layout is tested as
a property: across a range of terminal sizes the interface is exactly as tall as the
terminal and never wider, whatever the header has been asked to say.

## A note on the module

`sdks/go` carries no module tag of its own, so `go.mod` here replaces it with the checkout
beside it and the repository's `go.work` ties the two together. A release of this module
would pin a version instead.
