# sdks/go

The Go SDK for agents that run in the [acceleration backend](../../acceleration).

It is the Go counterpart of [plugins/stream](../../plugins/stream), not of
[sdks/python](../python). The backend already runs the whole pipeline: it joins the
call, transcribes, answers and speaks. What this does is create the session, hold the events
socket, run your functions and configure everything.

```bash
go get github.com/GetStream/Vision-Agents/sdks/go
```

```bash
export STREAM_ACCELERATION_URL=http://localhost:8080
export STREAM_ACCELERATION_CUSTOMER_ID=acme
```

## In writing

```go
llm := stream.Accelerated(stream.Config{Agent: "jean"})

agents.RegisterFunction(llm, "get_weather",
    "Get current weather for a location",
    func(ctx context.Context, in struct {
        Location string `json:"location" schema:"the city and state"`
    }) (any, error) {
        return weatherAt(ctx, in.Location)
    })

agent, err := agents.New(agents.Options{
    Name:         "jean",
    LLM:          llm,
    Harness:      agents.DefaultHarness(),
    CostTracking: map[string]string{"customer_id": "123"},
    MemoryFilter: map[string]string{"user_id": "123"},
})

session, err := agent.Chat(ctx)
defer session.Close(ctx)

session.Respond("What is the weather in Boulder?")
for event := range session.Events() {
    fmt.Println(event.Kind, event.Text)
}
```

`Config{Agent: "jean"}` starts from a stored agent config, named the way a person knows it
rather than by an id nobody chose, so the things worth deciding once are decided once.
Everything else in `Config` overrides what it says. A caller that was handed an id instead —
a dispatched message carries the id of the config it was routed to — names it as
`Config{ConfigID: id}`; a name that matches nothing stored is refused rather than quietly
starting an unconfigured agent.

Functions are declared by their argument struct: the `json` tags name the arguments and the
`schema` tags say what they mean, and the JSON Schema the model is offered is derived from
them. The model asks over the session's socket and the function runs here, in your process,
with whatever it can reach.

## Conversations somebody comes back to

`agents.New` spells an agent out here and holds one conversation at a time. The other way
round is an agent configured once in the backend and addressed by the name a person knows it
as, with its conversations kept so they can be found again:

```go
api, _ := client.New(stream.Backend{})
docs := api.Agent("docs")

session, _ := docs.Sessions.Create(ctx, client.SessionOptions{
    Title:   "Is Stream better than Sendbird?",
    Project: "docs",
    Custom:  map[string]any{"ticket": "4721"},
    Persist: true,
})
defer session.Close(ctx)

answer, _ := session.Responses.Create(ctx, "Is Stream better than Sendbird?")

items := answer.Items.Unwind(ctx, 0)
for item := range items.Items() {
    fmt.Println(item.Kind, item.Text)
}
if err := items.Err(); err != nil {
    return err
}
```

`Responses.Create` returns once the agent has started answering rather than when it has
finished, because a model takes seconds. `Items` is what the backend wrote down, so it reads
the same during the conversation and a week after it ended; `session.Events()` is still the
live view, and the two answer different questions.

Old conversations are found by filter or by phrase:

```go
recent, _ := docs.Sessions.Query(ctx, client.Query{Project: "docs", Limit: 20})
found, _ := docs.Sessions.Search(ctx, "sendbird comparison", client.Query{})
```

`Incognito: true` holds the conversation and keeps nothing: no row, no turns, no transcript,
whatever `Persist` says. It cannot be listed, searched or forked afterwards, which is the
point of it.

`Fork` continues a conversation as a new one — the same question asked of a harder model, or
of a different agent, with the parent left untouched and each writing its own transcript:

```go
harder, _ := session.Fork(ctx, client.ForkOptions{
    Title:           "Again, with reasoning",
    ModelOverwrites: &acceleration.ModelOverwrites{Llm: pointer("openai/gpt-5")},
})
```

`ModelOverwrites` is also a `SessionOptions` field, so one conversation can overrule the
config's models without a config of its own.

Somebody who has not signed up yet is a guest. `api.GuestUser` mints one with a token to
hold, `api.AsGuest` is a client acting for them, and `api.ClaimGuestUser` moves their
conversations onto the account they turn out to be — server side only, because only the
backend that just authenticated the account knows which guest it was.

`session.Chat()` is the Stream Chat channel the transcript is written into and
`session.Video()` is the call, both through `getstream-go`, which is already a dependency.
They need `STREAM_API_KEY` and `STREAM_API_SECRET`: the channel is Stream rather than this
router.

## Waiting to be written to

`agent.Chat` is a conversation this process started. A conversation somebody else starts —
a person writing in an agent channel — arrives the other way round: they wrote, the router
found out by webhook, and it has to reach an agent that runs here. So a worker connects out
and waits, and nothing has to be publicly reachable.

```go
dispatch, _ := agents.NewDispatch(agents.DispatchOptions{Capacity: 8})

dispatch.OnMessage(func(ctx context.Context, message agents.InboundMessage) error {
    conversation, err := dispatch.Conversation(ctx, message, build)
    if err != nil {
        return err
    }
    return conversation.Respond(message.Text)
})

dispatch.Run(ctx)
```

`Conversation` is the agent answering that channel, started if none is. A channel is one
conversation, so the second message on it goes to the agent that answered the first, which
is still open and knows what has been said; only a channel nothing is answering calls
`build`. Agents are kept until the worker stops waiting.

Nothing is waited for after `Respond`, because the answer is written into the channel by the
backend as it is generated: the person who wrote is already reading it. Questions on one
channel are answered one at a time, since `Session.Respond` interrupts, and two messages
written in quick succession would otherwise throw the first answer away half-written.

Several workers can wait at once, in which case the router shares the work between them.
`Capacity` is a promise about what this process can hold: a full worker is passed over
rather than queued behind. `OnCall` is the same thing for calls that arrive over SIP.

A message only arrives here when no session is running on its channel. One written to an
agent that *is* running is answered by the router from that session, because that agent is
the one that knows what has been said. That is what `ChatOptions.AgentID` is for: a session
opened without it joins under the agent's own user id, and the router cannot find it.

The router needs to know whose channel it is. A channel a conversation has already been held
in is claimed by that conversation. A channel created for somebody opening a support chat has
no such history, and names the agent config answering in it under its own
`agent_config_id` custom field.

Whatever else that channel was created with arrives in `message.Custom`, carried through
unread. It is where `build` finds what the conversation is for and the router has no
opinion about — the organization to scope memory to, the locale to answer in. Whoever
created the channel decided what is in it, so read it as a claim rather than a fact.

Two options are worth setting on a worker whose agent does more than answer from the model.
`TurnTimeout` is how long one answer is given before the conversation abandons it and takes
the next question, five minutes by default, which is short for an agent whose tools read a
source tree. `OnEvent` is told everything the backend says about every conversation the
worker holds, which is the only way to see any of it — the conversation reads its own
session, and a second reader would take events from it:

```go
agents.NewDispatch(agents.DispatchOptions{
    Capacity:    8,
    TurnTimeout: 15 * time.Minute,
    OnEvent: func(channelID string, event stream.Event) {
        metrics.Record(channelID, event)
    },
})
```

`sdks/go/examples/dispatch` is the whole of it, and
`examples/voice_agents/chat_support` is the Python one.

## In a call

```go
llm := stream.Accelerated(stream.Config{
    STT: "deepgram/flux-general-en", TTS: "cartesia/sonic-preview",
    LLM: "gemini/gemini-3.8-flash",
    Greeting: "Hey, I'm Jean. What can I do for you?",
})
agent, _ := agents.New(agents.Options{
    Name: "jean", LLM: llm,
    // A subagent turns the built-in think, recall and explain skills on: the fast model
    // hands the hard questions over and keeps talking while the slower one reasons.
    Harness: &agents.Harness{
        UseSkills: true,
        Subagents: map[string]string{"default": "openai/gpt-5.6-sol"},
    },
})

call, _ := agent.Join(ctx, edge.Call{})
defer call.Close(ctx)

fmt.Println(call.MonitorURL())
```

`Join` creates a Stream call and has the backend join it, and `MonitorURL` is a link a person
can open to talk to the agent from a browser. A modality is either a capability shortcut such
as `en-low-latency`, which the backend routes and fails over, or a concrete `provider/model`.
Leaving one empty takes the backend's default.

## On the phone

```go
llm := stream.Accelerated(stream.Config{
    TTS: "sonic_36", STT: "parakeet", LLM: "gemma-4-E2B-it", Subagent: "openai/gpt-5.6-sol",
})
agent, _ := agents.New(agents.Options{Name: "jean", LLM: llm})

number, _ := agent.PurchaseAnyNumber(ctx, agents.NumberSearch{Vendor: "twilio", Country: "US"})

call, _ := agent.WaitForCall(ctx, number)
defer call.Close(ctx)

call.Respond("Say hello and let them know you are a voice AI.")
fmt.Println(call.MonitorURL())
```

`MonitorURL` is a link a person can open to be on the other end of the call from a browser.
Ringing somebody instead is `agent.StartCall(ctx, number, "+15551234567")`, which tells the
backend it is navigating so recordings are let finish and menus are answered.

`PurchaseAnyNumber` starts a monthly charge. An agent that answers the same number every day
should buy it once and pass it to `WaitForCall`.

## One modality at a time

A client is where the router is and who is calling it, said once. Anything routed through one
reads the same whether that turned out to be localhost or the hosted proxy:

```go
client, _ := stream.NewClient(stream.Backend{})
```

The zero `Backend` reads the environment. A hosted router authenticates a Stream app rather
than naming a customer, which is `stream.Backend{URL: ..., Authenticate: true}` and the key
and secret in the environment.

`client.Router` then routes a modality on its own, live or from a recording, with the options
said once in a stored config:

```go
router := client.Router("healthcare")

yes := true
transcript, _ := router.STT().Recording(ctx,
    stream.Recorded{URL: "https://example.com/interview.mp4"},
    &acceleration.SttOptions{Diarize: &yes})
fmt.Println(transcript.Text, transcript.Speakers)

voice, _ := router.TTS().Realtime(ctx, &acceleration.TtsOptions{Voice: &id})
defer voice.Close()
voice.Speak("hello there")
for audio := range voice.Audio() {
    ...
}

found, _ := router.Search(ctx, "perioperative antibiotic guidance", nil)
```

`Realtime` opens the socket and yields on a channel; `Recording` is the non-realtime form,
served by the batch half of a vendor rather than the streaming one, which is cheaper and more
accurate. Anything named in the options overrides that field of the config, and an option no
provider behind the target can express is refused rather than dropped.

## An agent as a directory

```
agents/jean/
  instructions.md       the system prompt
  skills/think.md       frontmatter (name, description, deadline) and a body
  knowledge/*.md        what the agent may look things up in
  knowledge/urls.yaml   pages to keep that filled from, as urls or url/title/description
```

```go
agent, _ := agents.New(agents.Options{Dir: "agents/jean", LLM: llm})
agent.Sync(ctx)
```

`Sync` stores the skills, fills a knowledge base named after the agent from both its files
and its pages, and stores a config pointing at all of it. It finds each by name first, and a
page is keyed by its url, so running it twice edits and re-reads what is there rather than
storing another copy. What is written in code wins over what the directory says,
so a directory is a starting point rather than an override.

## Layout

| Path            | What is in it                                                     |
| --------------- | ----------------------------------------------------------------- |
| `agents/`       | `Agent`, its lifecycle, the harness, the directory loader and function registration |
| `client/`       | Agents by name, their sessions, responses and items, forking and guest users |
| `stream/`       | The remote pipeline, the backend it talks to, the socket and the phone endpoints |
| `edge/`         | Creating the Stream call and minting a link to listen in on it     |
| `tools/`        | The function registry and the JSON Schema derived from an argument struct |
| `acceleration/` | The generated client. Do not edit it                               |
| `examples/`     | A conversation in writing, and one out loud                        |

## Regenerating the client

`acceleration/generated.go` comes from
[acceleration/api/openapi.yaml](../../acceleration/api/openapi.yaml), which is the same file
the Go server and the Python client come from. It is committed, so installing this module
needs no code generation.

```bash
go generate ./...
```

The two sockets are excluded: OpenAPI stops at the upgrade, so `stream/socket.go` is written
by hand, the way the server side of them is.

## Tests

```bash
go test ./...
```

No mocks. The client and the socket are tested against an `httptest.Server` and a real
`gorilla/websocket` upgrader, and the schema reflection and the directory loader against
what they are given.

`e2e_test.go` drives a running router instead of a stand-in for one, so it is behind a tag.
Start Postgres, Redis and the router as
[acceleration/README.md](../../acceleration/README.md) describes, then:

```bash
STREAM_ACCELERATION_URL=http://localhost:8080 \
STREAM_ACCELERATION_CUSTOMER_ID=e2e \
go test -tags e2e -v .
```

Anything the deployment cannot do is skipped rather than failed: a router with no knowledge
provider is a valid router, and so is one with no telephony.
