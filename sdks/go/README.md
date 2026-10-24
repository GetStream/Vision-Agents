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
type GetWeather struct {
    Location string `json:"location" schema:"the city and state"`
}

func (GetWeather) Name() string        { return "get_weather" }
func (GetWeather) Description() string { return "Get current weather for a location" }
func (w GetWeather) Run(ctx context.Context) (any, error) {
    return weatherAt(ctx, w.Location)
}

agent, err := agents.New(agents.Options{
    Name:         "jean",
    LLM:          stream.Accelerated(stream.Config{Agent: "jean"}),
    Harness:      agents.DefaultHarness(),
    CostTracking: map[string]string{"customer_id": "123"},
    MemoryFilter: map[string]string{"user_id": "123"},
})
agent.Tools().Add(GetWeather{})

session, err := agent.Chat(ctx)
defer session.Close(ctx)

session.Responses.Create(ctx, "What is the weather in Boulder?")
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

A function shows the model pictures by returning a `tools.Output`, each picture an https URL or a
data URI:

```go
return tools.Output{Text: "the room, just now", Images: []tools.Image{{URL: "data:image/jpeg;base64," + frame}}}, nil
```

On a call the router hands them to the agent's `vision` skill. In a chat, which has no such
worker, a model that sees images is shown them itself, with its next reply alone; one that reads
only text is told the tool failed. A tool hosted on a dispatch worker sends its pictures the same
way.

A function registered with `tools.Register` takes options for how its calls are run:

```go
tools.Register(agent.Tools(), "search_flights", "Search for flights", searchFlights,
    tools.Interruptible(true),       // the caller talking over the agent cancels the call
    tools.Timeout(10*time.Second),   // in place of the session's 30 seconds
    tools.Concurrency(4))            // at most four at once; the registry allows 16
```

A function that changes something can ask a voice caller first:

```go
tools.Register(agent.Tools(), "book_flight", "Book a flight", bookFlight,
    tools.NeedsApproval("Book flight {flight_id}?"))
```

The router asks the question, with each `{argument}` filled in from the call, and runs the
function only once the caller says yes within 15 seconds of the question ending. The function
sees `tools.CallFrom(ctx).Approved` set, and its timeout starts then. A no or silence declines
the call, and it never runs. A speech-to-speech model asks in its own voice and may not say the
question word for word. Registering a question that names an argument the function does not
take is an error. Only voice calls ask: in writing, the question goes out as the tool's
`approval` for the app to show, and the call runs as it comes.

Its `ctx` is cancelled when the router cancels the call: its timeout passed, or, for an
interruptible tool, the caller interrupted. `tools.CallFrom(ctx)` is the call being run, whose
`ID` is the same when the router sends a call again, so a function that changes something can
key on it. A call sent again while it is still running joins it, and one that finished is answered
with what it returned, without running twice.

A call that takes a while can say how it is going, and protect a write from the caller
talking over the agent:

```go
func(ctx context.Context, args RebookFlight) (any, error) {
    call := tools.CallFrom(ctx)
    call.Update(ctx, "Checking seats on the new flight")

    // Read only: the caller interrupting cancels ctx, and nothing has changed.
    seat, err := airline.HoldSeat(ctx, args.NewFlightID)
    if err != nil {
        return nil, err
    }

    // From here on, the caller interrupting no longer cancels the call.
    ctx = call.DisallowInterruptions(ctx)
    call.Say(ctx, "Got a seat, moving your booking now.")
    return airline.Rebook(ctx, args.BookingID, seat, call.ID)
}
```

`call.Update` hands the agent a status, which it tells a voice caller in its own words once
nobody is talking, and which a conversation shows on the call's step. `call.Say` has it say
the words as written, or post them in text; a speech-to-speech model may paraphrase them.
`DisallowInterruptions` returns a context only the call's timeout and the socket ending cancel.
When the call was already stopped, that context comes back cancelled, so the write never starts.
Outside a call, all three do nothing.

Struct tags describe arguments beyond `schema`: `enum:"economy,business"`, `minimum:"1"`,
`maximum:"9"`, `pattern`, `format` and `example`. A type that describes itself implements
`JSONSchema() map[string]any`, and `tools.RegisterSchema` takes a schema as written and hands the
function the raw arguments.

## Conversations somebody comes back to

`agents.New` spells an agent out here and holds one conversation at a time. The other way
round is an agent configured once in the backend and addressed by the name a person knows it
as, with its conversations kept so they can be found again:

```go
api, _ := client.New(stream.Backend{})
docs := api.Agent("docs")

session, _ := docs.Sessions.Create(ctx, client.SessionOptions{
    Title:     "Is Stream better than Sendbird?",
    ProjectID: "docs",
    Custom:    map[string]any{"ticket": "4721"},
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

Old conversations are listed most recently updated first, or found by phrase, best match
first:

```go
recent, _ := docs.Sessions.Query(ctx, client.Query{ProjectID: "docs", Limit: 20})
found, _ := docs.Sessions.Search(ctx, "sendbird comparison", client.Query{})
```

Every text conversation is kept in a Stream Chat channel, and `session.ConversationID()` is
the one to pass back as `ConversationID` to resume it. `Incognito: true` holds the
conversation and keeps nothing: no row, no turns, no transcript. It cannot be listed,
searched or forked afterwards, which is the point of it.

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

`Update` changes a session in place and returns it as it now is: its title, description,
custom labels, instructions, models or voice. It is server side only. A nil field is left
alone, an empty `Sts` makes the session a cascade again, and an empty `Voice` returns to the
provider's default. The id, the call and incognito cannot change. Instructions, models and
voice take over from the next turn and need the session running; a conversation that ended
can still be renamed through `Sessions.Update` by id:

```go
_, err := session.Update(ctx, client.SessionUpdate{
    Title: pointer("Pricing"),
    Llm:   pointer("openai/gpt-5"),
})
_, err = agent.Sessions.Update(ctx, "session-id", client.SessionUpdate{Title: pointer("Pricing")})
```

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
channel are answered one at a time, so a second message written in quick succession waits
for the first answer rather than landing on top of it.

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

A worker can also host an agent's tools for every session opened under it, including one a
browser opened, where the session's own process cannot reach what the tool needs:

```go
agent := api.Agent("my-agent")
agent.Tools().Add(LookupOrder{})
dispatch.Host(agent, time.Minute)
```

The router offers them to each session naming the agent and sends every call here. `Run`
works with only hosted tools and no handler, and stops if the router refuses them.

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

call, _ := agent.Join(ctx)
defer call.Close(ctx)

fmt.Println(call.MonitorURL())
```

`Join` opens a session with voice on, and the backend joins the session's own call,
`agent:<session id>`. `MonitorURL` is a link a person can open to talk to the agent from a
browser. `session.Voice.Stop(ctx)` carries the conversation on in writing, and
`session.Voice.Start(ctx)` puts the agent back on the call. A modality is either a capability shortcut such
as `en-low-latency`, which the backend routes and fails over, or a concrete `provider/model`.
Leaving one empty takes the backend's default.

## On the phone

```go
llm := stream.Accelerated(stream.Config{
    Agent: "jean", TTS: "sonic_36", STT: "parakeet", LLM: "gemma-4-E2B-it",
})
agent, _ := agents.New(agents.Options{Name: "jean", LLM: llm, Harness: &agents.Harness{
    Subagents: map[string]string{"default": "openai/gpt-5.6-sol"},
}})
agent.Sync(ctx) // the harness is stored on the agent's config, which the call runs under

number, _ := agent.PurchaseAnyNumber(ctx, agents.NumberSearch{Vendor: "twilio", Country: "US"})

call, _ := agent.WaitForCall(ctx, number)
defer call.Close(ctx)

call.Responses.Create(ctx, "Say hello and let them know you are a voice AI.")
fmt.Println(call.MonitorURL())
```

`MonitorURL` is a link a person can open to be on the other end of the call from a browser.
Every caller gets a session of their own, which `WaitForCall` opens when the router hands the
call over; a worker answering many uses `agent.Answer(ctx, call)` from a dispatch `OnCall`
handler. Ringing somebody instead is `agent.StartCall(ctx, number, "+15551234567")`, which
tells the backend it is navigating so recordings are let finish and menus are answered.

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
  agent.yaml            required: the name and what it runs on (llm, stt, tts, tags, ...)
  instructions.md       the system prompt
  guardrail.md          the policy each turn is screened against
  skills/think.md       frontmatter (name, description, deadline) and a body
  knowledge/*.md        what the agent may look things up in
  knowledge/urls.yaml   pages to keep that filled from, as urls or url/title/description
  simulations/*.yaml    each a list of simulations (name, scenario, assertion, variations, ...)
  .agent_sync           written by Sync: the hash last synced and when
```

```go
agent, _ := agents.New(agents.Options{Dir: "agents/jean", LLM: llm})
agent.Sync(ctx)
```

`Sync` sends the whole directory in one request: the skills, the files and pages for a
knowledge base named after the agent, the simulations, and what `agent.yaml` declares. A key
`agent.yaml` or a simulation file does not know is refused. The simulations in `simulations/`
are the whole of the agent's: one taken out is deleted on the next sync, and a directory
without `simulations/` leaves the stored ones alone. The request carries a hash of the
directory, the same one the Python SDK takes for a directory without simulations, and
`.agent_sync` records it, so running `Sync` again on an unchanged
directory only reads the config back. What is written in code wins over what the directory
says, so a directory is a starting point rather than an override.

## When the router says no

A request or a socket the router refuses comes back as a `*stream.RouterError`, whichever of
`client`, `agents` or `stream` made it: `errors.As(err, &refused)` finds it, with the `Status`,
the `Type` and `Code` to branch on, the `Message` and the `DocURL` explaining it, and the
`RequestID` to quote to support, since a 500 says only "something went wrong". New codes are
added, so expect one this SDK does not know, and a failure that was not the router's own, such
as a proxy's error page, has its body as the `Message` and no type or code.

## What it cost, and when credit runs low

Every successful answer to a backend says what the request cost and what the organization has
left, and a refusal for money says what it has left.
`client.New` takes two options for it, and every resource method's answers go through them:

```go
api, err := client.New(stream.Backend{},
    client.WithUsage(func(usage client.Usage) {
        log.Printf("%s cost %d micros", usage.Operation, usage.CostMicros)
    }),
    client.WithBalanceWarning(func(warning client.BalanceWarning) {
        alert("credit is " + warning.Warning) // low, then exhausted
    }),
)
```

The balance warning fires once on going `low` and once more on going on to `exhausted`, and not
again until an answer has said there is credit enough, so a balance hovering near the line does
not fire on every answer. Both callbacks run on the goroutine that made the request: they must
not block, and requests made at once call them at once. With neither option the client logs one
WARN per crossing. A `stream.Router` built on `api.Backend()` is heard the same way, and its
sockets carry `CostMicros` on a final `Transcript`, a finished `Audio` and a finished `Answer`,
and a `Balance` when credit goes down into low or exhausted while the socket is open. That
`Balance` comes as an item of its own on the socket's channel, so a loop over the channel
should expect items whose only field set, beside the raw `Frame`, is `Balance`. An end user's
client is told none of it. To be told by webhook instead, add an event destination for the
connector `stream`, which is sent `account.balance_low` and `account.balance_exhausted`.

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
