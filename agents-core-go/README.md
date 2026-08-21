# agents-core-go

The Go SDK for agents that run in the [acceleration backend](../acceleration).

It is the Go counterpart of [plugins/stream](../plugins/stream), not of
[agents-core](../agents-core). The backend already runs the whole pipeline: it joins the
call, transcribes, answers and speaks. What this does is create the session, hold the events
socket, run your functions and configure everything.

```bash
go get github.com/GetStream/Vision-Agents/agents-core-go
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

`Config{Agent: "jean"}` starts from a stored agent config, so the things worth deciding once
are decided once. Everything else in `Config` overrides what it says.

Functions are declared by their argument struct: the `json` tags name the arguments and the
`schema` tags say what they mean, and the JSON Schema the model is offered is derived from
them. The model asks over the session's socket and the function runs here, in your process,
with whatever it can reach.

## In a call

```go
llm := stream.Accelerated(stream.Config{
    STT: "deepgram/flux-general-en", TTS: "cartesia/sonic-preview",
    LLM: "gemini/gemini-3.5-flash-lite",
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

## An agent as a directory

```
agents/jean/
  instructions.md       the system prompt
  skills/think.md       frontmatter (name, description, deadline) and a body
  knowledge/*.md        what the agent may look things up in
```

```go
agent, _ := agents.New(agents.Options{Dir: "agents/jean", LLM: llm})
agent.Sync(ctx)
```

`Sync` stores the skills, fills a knowledge base named after the agent, and stores a config
pointing at both. It finds each by name first, so running it twice edits what is there
rather than storing another copy. What is written in code wins over what the directory says,
so a directory is a starting point rather than an override.

## Layout

| Path            | What is in it                                                     |
| --------------- | ----------------------------------------------------------------- |
| `agents/`       | `Agent`, its lifecycle, the harness, the directory loader and function registration |
| `stream/`       | The remote pipeline, the backend it talks to, the socket and the phone endpoints |
| `edge/`         | Creating the Stream call and minting a link to listen in on it     |
| `tools/`        | The function registry and the JSON Schema derived from an argument struct |
| `acceleration/` | The generated client. Do not edit it                               |
| `examples/`     | A conversation in writing, and one out loud                        |

## Regenerating the client

`acceleration/generated.go` comes from
[acceleration/api/openapi.yaml](../acceleration/api/openapi.yaml), which is the same file
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
[acceleration/README.md](../acceleration/README.md) describes, then:

```bash
STREAM_ACCELERATION_URL=http://localhost:8080 \
STREAM_ACCELERATION_CUSTOMER_ID=e2e \
go test -tags e2e -v .
```

Anything the deployment cannot do is skipped rather than failed: a router with no knowledge
provider is a valid router, and so is one with no telephony.
