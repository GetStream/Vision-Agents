# Vision Agents Go SDK

The Go SDK for agents that run in Stream's [acceleration backend](https://github.com/GetStream/Vision-Agents/tree/accelerate/acceleration).

It is the Go counterpart of [plugins/stream](https://github.com/GetStream/Vision-Agents/tree/accelerate/plugins/stream), not of
[sdks/python](https://github.com/GetStream/Vision-Agents/tree/accelerate/sdks/python). The backend already runs the whole pipeline: it joins the
call, transcribes, answers and speaks. What this does is create the session, hold the events
socket, run your functions and configure everything.

It needs Go 1.25 or newer.

```bash
go get github.com/GetStream/Vision-Agents/sdks/go@latest
```

## Quickstart

The hosted router needs a Stream app's key and secret, and nothing else:

```bash
export STREAM_API_KEY=your_api_key
export STREAM_API_SECRET=your_api_secret
```

It is in us-east4. `STREAM_ACCELERATION_REGION=europe-west4`, or
`stream.Backend{Region: "europe-west4"}`, is the one in another region.

This is [examples/quickstart](examples/quickstart): an agent read from its directory,
`agents/myagent`, with one function the model can call, asked one question in writing. Run it
from that directory with `go run .`.

```go
package main

import (
	"context"
	"fmt"
	"log"

	"github.com/GetStream/Vision-Agents/sdks/go/agents"
	"github.com/GetStream/Vision-Agents/sdks/go/tools"
)

type LookupOrder struct {
	OrderID string `json:"order_id" schema:"the order number, e.g. 1042"`
}

func (LookupOrder) Name() string        { return "lookup_order" }
func (LookupOrder) Description() string { return "Look up an order by its number" }
func (l LookupOrder) Run(ctx context.Context) (any, error) {
	return fmt.Sprintf("Order %s shipped yesterday.", l.OrderID), nil
}

func main() {
	ctx := context.Background()

	agent, err := agents.New(agents.Options{
		Name:  "myagent",
		Tools: []tools.Tool{LookupOrder{}},
	})
	if err != nil {
		log.Fatal(err)
	}
	if _, err := agent.Sync(ctx); err != nil {
		log.Fatal(err)
	}

	session, err := agent.Sessions.Create(ctx, agents.SessionOptions{})
	if err != nil {
		log.Fatal(err)
	}
	defer session.Close(ctx)

	if _, err := session.Responses.Create(ctx, "Where is order 1042?"); err != nil {
		log.Fatal(err)
	}
	for event := range session.Events() {
		if event.Kind == "response_delta" {
			fmt.Print(event.Text)
		}
		if event.Kind == "responded" && !event.PendingWork {
			fmt.Println()
			break
		}
	}
}
```

`Sync` stores the directory in the backend, so the session runs on it. The model asks for
`lookup_order` over the session's socket, and the function runs here, in your process.

### Against a local router

A router you run yourself is named with its URL and the customer to bill:

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

A function that changes something can have a person allow each call first:

```go
tools.Register(agent.Tools(), "book_flight", "Book a flight", bookFlight,
    tools.NeedsApproval("Book flight {flight_id}?"))
```

On a voice call the router asks the question out loud, with each `{argument}` filled in from the
call, and sends the call only once the caller says yes within 15 seconds of the question ending.
The function sees `tools.ToolCallContextFrom(ctx).Approved` set, and its timeout starts then. A no
or silence declines the call, and it never runs. A speech-to-speech model asks in its own voice
and may not say the question word for word. Registering a question that names an argument the
function does not take is an error.

In writing nobody is asked out loud, so the session holds the call and hands it to you as a
`tool_call` event. It runs only once you answer for the person:

```go
for event := range session.Events() {
    if event.Kind == "tool_call" {
        // Show event.Frame.String("name") and event.Frame.String("arguments"), and ask.
        allowed := askThePerson(event.Frame)
        session.AnswerApproval(event.Frame.String("id"), event.Frame.String("request_id"), allowed, "")
    }
}
```

Allowed, the function runs and its result goes back as it would have. Declined, it never runs
and the model is told the person declined. A call nobody answers does not run: the router gives
up on it at the tool's timeout. A dispatch worker has no person to ask, so it refuses a hosted
call to such a function unless the router says the caller said yes, which it does on a voice
call only.

`tools.Authorize` decides whether a call may run at all, before its arguments are checked. It
is where to check who the call is for: the model asks for whatever the conversation talked it
into.

```go
tools.Register(agent.Tools(), "refund_order", "Refund an order", refundOrder,
    tools.Authorize(func(ctx context.Context, call tools.Call) error {
        if !staff.Has(call.User.ID) {
            return errors.New("refunds are for staff")
        }
        return nil
    }))
```

A refused call never runs, and the model is told only that it was refused.

When a function returns an error the model is told "the tool failed" and nothing more, and the
error is logged here: what a database or an API says when it fails can name hosts, accounts and
queries. Return a `tools.UserError` for a failure the model should read as written, such as
`tools.UserError("there is no order 1042")`. Arguments that do not fit the function's schema
are explained to the model too, so it can correct its call.

Its `ctx` is cancelled when the router cancels the call: its timeout passed, or, for an
interruptible tool, the caller interrupted. `tools.ToolCallContextFrom(ctx)` is the call being
run, whose `ID` is the same when the router sends a call again, so a function that changes
something can key on it. A call sent again while it is still running joins it, and one that
finished is answered with what it returned, without running twice.

A call that takes a while can say how it is going, and protect a write from the caller
talking over the agent:

```go
func(ctx context.Context, args RebookFlight) (any, error) {
    toolCallContext := tools.ToolCallContextFrom(ctx)
    toolCallContext.Update(ctx, "Checking seats on the new flight")

    // Read only: the caller interrupting cancels ctx, and nothing has changed.
    seat, err := airline.HoldSeat(ctx, args.NewFlightID)
    if err != nil {
        return nil, err
    }

    // From here on, the caller interrupting no longer cancels the toolCallContext.
    ctx = toolCallContext.DisallowInterruptions(ctx)
    toolCallContext.Say(ctx, "Got a seat, moving your booking now.")
    return airline.Rebook(ctx, args.BookingID, seat, toolCallContext.ID)
}
```

`toolCallContext.Update` hands the agent a status, which it tells a voice caller in its own
words once nobody is talking, and which a conversation shows on the call's step.
`toolCallContext.Say` has it say the words as written, or post them in text; a speech-to-speech
model may paraphrase them. `DisallowInterruptions` returns a context only the call's timeout and
the socket ending cancel. When the call was already stopped, that context comes back cancelled,
so the write never starts. Outside a call, all three do nothing.

Struct tags describe arguments beyond `schema`:

```go
type BookFlight struct {
	FlightID string `json:"flight_id" schema:"e.g. LH400" pattern:"^[A-Z]{2}[0-9]{1,4}$"`
	Seats    int    `json:"seats" minimum:"1" maximum:"9"`
	Cabin    string `json:"cabin,omitempty" enum:"economy,business,first" default:"economy"`
}
```

The tags are `enum`, `minimum`, `maximum`, `exclusiveMinimum`, `exclusiveMaximum`,
`multipleOf`, `minLength`, `maxLength`, `minItems`, `maxItems`, `pattern`, `format`, `title`,
`default` and `example`. A struct already written for another Go JSON Schema library can keep its
`jsonschema:"description=Full name,minLength=1,enum=a,enum=b"` tag, with `\,` for a comma in a
value. A type that describes itself implements `JSONSchema() map[string]any`.

Every call's arguments are held to the schema the model was told before the function runs.
Arguments it refuses fail the call with what was wrong, such as `/seats: maximum: got 12, want
9`, and the model can try again; the function never sees them.

`tools.RegisterSchema` takes a schema written by hand, for a shape no struct says, such as a
`oneOf`: a `map[string]any`, raw JSON, or another library's schema type, such as
`google/jsonschema-go`'s, taken as the JSON it marshals to. Its function is handed the
arguments as raw JSON. `tools.RegisterWithSchema` takes a schema the same way and decodes the
arguments into a struct:

```go
schema := []byte(`{
  "type": "object",
  "properties": {
    "flight_id": {"type": "string", "pattern": "^[A-Z]{2}[0-9]{1,4}$", "description": "e.g. LH400"},
    "seats":     {"type": "integer", "minimum": 1, "maximum": 9},
    "cabin":     {"type": "string", "enum": ["economy", "business", "first"], "default": "economy"}
  },
  "required": ["flight_id", "seats"]
}`)

tools.RegisterWithSchema(registry, "book_flight", "Book a flight for the caller", schema,
	func(ctx context.Context, in BookFlight) (any, error) {
		return airline.Book(ctx, in.FlightID, in.Seats, in.Cabin)
	})
```

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
    if item.Text != nil {
        fmt.Println(item.Kind, *item.Text)
    }
}
if err := items.Err(); err != nil {
    return err
}
```

`Responses.Create` returns once the agent has started answering rather than when it has
finished, because a model takes seconds. `Items` is what the backend wrote down, so it reads
the same during the conversation and a week after it ended; `session.Events()` is still the
live view, and the two answer different questions.

A text session's events survive the router restarting. When its socket drops, the SDK opens it
again, waiting longer between attempts for up to five minutes, and the router sends what it
missed. `Events()` stays open across it and carries a `resumed` event saying how it went:

```go
for event := range session.Events() {
    if event.Kind == "resumed" && !event.Resumed.Complete {
        // Some of what was said while the socket was away is gone: read it back from
        // session.Responses.Items. A question the router stopped on its way down is
        // interrupted and never answered, so ask it again.
        if command := event.Resumed.Command; command != nil && command.State == "interrupted" {
            session.Responses.Create(ctx, question)
        }
    }
}
```

`Command` is the last question `Responses.Create` asked. A call's events are not reconnected:
the call ends with the router holding it, and `Events()` closes as before.

Old conversations are listed by when the agent last answered, newest first, or found by
phrase, best match first. The filter is written in the shape the router reads, as a
`client.Filter` (a `map[string]any`), and is always narrowed to the agent asked:

```go
recent, _ := docs.Sessions.Query(ctx, client.Query{
	Filter: client.Filter{"project_id": "docs"},
	Limit:  20,
})
either, _ := docs.Sessions.Query(ctx, client.Query{
	Filter: client.Filter{"$or": []client.Filter{
		{"project_id": "docs"},
		{"project_id": "blog"},
	}},
})
pinnedFirst, _ := docs.Sessions.Query(ctx, client.Query{
	Sort: []client.Sort{{Field: client.SortPinned}, {Field: client.SortLastMessageAt, Direction: -1}},
})
poor, _ := docs.Sessions.Query(ctx, client.Query{
	Filter: client.Filter{"user_id": "jean", "review_score": client.Filter{"$lte": 2}},
	Sort:   []client.Sort{{Field: client.SortReviewScore, Direction: 1}},
})
found, _ := docs.Sessions.Search(ctx, "sendbird comparison", client.Query{
	Filter: client.Filter{"project_id": "docs"},
})
if recent.HasMore {
	next, _ := docs.Sessions.Query(ctx, client.Query{
		Filter: client.Filter{"project_id": "docs"}, Limit: 20, Cursor: *recent.NextCursor,
	})
}
```

The fields are `project_id` (a string, `{"$eq": s}` or `{"$in": [...]}`), `review_score` (a
number 1 to 5, or `$eq`, `$gt`, `$gte`, `$lt`, `$lte`), `user_id` and `text`, combined with
`$and` and `$or`; the router refuses any other with a 400 that names it. `Search` writes
`text` itself and takes no `Sort`. A sort is `client.SortLastMessageAt` either way,
`client.SortPinned` alone or followed by `client.SortLastMessageAt`, or
`client.SortReviewScore` either way; a cursor only continues the sort it came from.

A session is pinned and unpinned by id or on the session held, and says when in `PinnedAt`:

```go
pinned, _ := docs.Sessions.Pin(ctx, id) // pinned.PinnedAt is set
session.Unpin(ctx)
docs.Sessions.Update(ctx, id, client.SessionUpdate{Pinned: &yes}) // with other changes
```

Every text conversation is kept in a Stream Chat channel, `session.ConversationID()`.
`docs.Sessions.Resume(ctx, session.ID())` carries it on later, and one that ended is reopened
with what was said in it. `Incognito: true` holds the
conversation and keeps nothing: no row, no turns, no transcript. It cannot be listed,
searched or forked afterwards, which is the point of it.

`Fork` continues a conversation as a new one — the same question asked of a harder model, or
of a different agent, with the parent left untouched and each writing its own transcript:

```go
// pointer is a helper of your own: func pointer[T any](v T) *T { return &v }
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
_, err = docs.Sessions.Update(ctx, "session-id", client.SessionUpdate{Title: pointer("Pricing")})
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
    if message.SessionID != "" {
        return dispatch.Answer(ctx, message)
    }
    conversation, err := dispatch.Conversation(ctx, message, build)
    if err != nil {
        return err
    }
    return conversation.Respond(message.Text, message.Images...)
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

A message written to a running session whose agent leaves text to dispatch carries its
`SessionID`, and `Answer` has that session's model answer it, with the photos sent with it.
`message.Images` are those photos, as URLs to fetch, which `Respond` takes beside the text.
A photo sent with no text is handed over too, with an empty `Text`.

Several workers can wait at once, in which case the router shares the work between them.
`Capacity` is a promise about what this process can hold: a full worker is passed over
rather than queued behind. `OnCall` is the same thing for calls that arrive over SIP.

A message only arrives here when no session is running on its channel. One written to an
agent that *is* running is answered by the router from that session, because that agent is
the one that knows what has been said. That is what `agents.SessionOptions.AgentID` is for: a session
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
routers, _ := stream.NewClient(stream.Backend{})
```

The zero `Backend` reads the environment. A hosted router authenticates a Stream app rather
than naming a customer, which is `stream.Backend{URL: ..., Authenticate: true}` and the key
and secret in the environment.

`routers.Router` then routes a modality on its own, live or from a recording, with the options
said once in a stored config:

```go
router := routers.Router("healthcare")

yes := true
transcript, _ := router.STT().Recording(ctx,
    stream.Recorded{URL: "https://example.com/interview.mp4"},
    &acceleration.SttOptions{Diarize: &yes})
if transcript.Text != nil && transcript.Speakers != nil {
    fmt.Println(*transcript.Text, *transcript.Speakers)
}

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

`region: eu` (or `us`) in `agent.yaml` is where the agent's work is done: every model it runs,
its knowledge base and its sandbox are there and nowhere else. Left out, the stored region
stays, and a new agent is in `us`. The router refuses a sync naming a model with no instance in
the region, naming each, and one whose knowledge base is already kept in another region.

## When the router says no

A request or a socket the router refuses comes back as a `*stream.RouterError`, whichever of
`client`, `agents` or `stream` made it: `errors.As(err, &refused)` finds it, with the `Status`,
the `Type` and `Code` to branch on, the `Message` and the `DocURL` explaining it, and the
`RequestID` to quote to support, since a 500 says only "something went wrong". New codes are
added, so expect one this SDK does not know, and a failure that was not the router's own, such
as a proxy's error page, has its body as the `Message` and no type or code.

A 429 or a 503 that asks to be given time says how long in `RetryAfter`. The SDK retries no
request itself: it cannot know which of yours are safe to send twice. `Responses.Create` is,
because every question carries a request id the router answers once, and so is a tool call,
which the router may send again and is answered from what the first run returned. A text
session's socket is the one thing reconnected for you.

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
WARN per crossing. A router from `stream.NewClient(api.Backend())` is heard the same way, and so
is an agent whose pipeline is built on `api.Backend()` (`stream.Config{Backend: api.Backend()}`)
or a dispatch worker given it as `DispatchOptions.Backend`. A router's
sockets carry `CostMicros` on a final `Transcript`, a finished `Audio` and a finished `Answer`,
and a `Balance` when credit goes down into low or exhausted while the socket is open. That
`Balance` comes as an item of its own on the socket's channel, so a loop over the channel
should expect items whose only field set, beside the raw `Frame`, is `Balance`. An end user's
client is told none of it. To be told by webhook instead, add an event destination for the
connector `stream`, which is sent `account.balance_low` and `account.balance_exhausted`.

## Spend limits

A backend caps what the organization, its apps, keys, end users, teams and usage categories may
spend in each day, week, two weeks or month. Amounts are `client.USD`, written the way a
`time.Duration` is, and exact: the router counts in millionths of a dollar, and so does `USD`.

```go
limits := api.SpendLimits()
limits.Create(ctx, client.Limit{Subject: client.Organization, Interval: client.Monthly, Amount: 20_000 * client.Dollar})

// Every key of app 1234 may spend $50 a day, and the CI key $10.
limits.Create(ctx, client.Limit{Subject: client.APIKey, AppID: "1234", Interval: client.Daily, Amount: 50 * client.Dollar})
limits.Create(ctx, client.Limit{Subject: client.APIKey, AppID: "1234", APIKey: "mz7kq2xa9b", Interval: client.Daily, Amount: 10 * client.Dollar, Note: "CI key"})
keys, err := limits.List(ctx, client.LimitFilter{Subject: client.APIKey, AppID: "1234", APIKey: "mz7kq2xa9b"})
limits.SetAmount(ctx, keys[0].Id, 25*client.Dollar)
limits.Delete(ctx, keys[0].Id)

// Every user of app 1234 may spend $20 a day, and user-42 $50.
limits.Create(ctx, client.Limit{Subject: client.User, AppID: "1234", Interval: client.Daily, Amount: 20 * client.Dollar})
limits.Create(ctx, client.Limit{Subject: client.User, AppID: "1234", UserID: "user-42", Interval: client.Daily, Amount: 50 * client.Dollar})

// Created or changed, whichever the team needs.
api.Teams().SetLimit(ctx, client.TeamLimit{Team: "finance", Interval: client.Monthly, Amount: 8_000 * client.Dollar})
session, err := api.Agent("docs").Sessions.Create(ctx, client.SessionOptions{Team: "finance"})
finance, err := client.New(stream.Backend{Team: "finance"}) // all of its work is finance's

limits.Reset(ctx, client.Reset{Subject: client.User, AppID: "1234", UserID: "user-42",
    Intervals: []acceleration.LimitInterval{client.Weekly}, Reason: "refunded a failed run"})

amount, err := client.ParseUSD("12.50") // from a form or a config file
```

Work counts under a team, and so against its limits, three ways. Whichever comes first wins,
and a session is counted under one team at most:

1. The team signed into the token: the end user's, or the one a client built with
   `stream.Backend{Team: "finance"}` signs for itself, which a router with no proxy in front
   of it is sent in `X-Stream-Usage-Categories` instead. A session cannot be moved off it, so
   a `Team` naming another is refused with a 400.
2. `client.SessionOptions{Team: "finance"}`, from a backend. A device sending a team is
   refused with a 403.
3. The agent's tag, `tags: team: finance` in its `agent.yaml`. A tag that is no team name is
   refused when the agent is synced.

`stream.Backend.UsageCategories`, such as `project:onboarding`, are counted the same way as the
team, up to 10 with it. Neither can be set beside a `Token` handed in, which carries its own.

What the router answers is in micros too: `client.USD(limit.LimitMicros)` prints as `$20000.00`.
Work a limit refuses is a `*stream.RouterError` of 402 `spend_limit_reached`, or 429
`user_limit_reached` for one end user's.

## Layout

| Path            | What is in it                                                     |
| --------------- | ----------------------------------------------------------------- |
| `agents/`       | `Agent`, its lifecycle, the harness, the directory loader and function registration |
| `client/`       | Agents by name, their sessions, responses and items, forking and guest users, spend limits and teams |
| `stream/`       | The remote pipeline, the backend it talks to, the socket and the phone endpoints |
| `edge/`         | Creating the Stream call and minting a link to listen in on it     |
| `tools/`        | The function registry and the JSON Schema derived from an argument struct |
| `acceleration/` | The generated client. Do not edit it                               |
| `examples/`     | Runnable programs: `quickstart`, `text`, `voice`, `dispatch`, `caregiving` |
| `api/`          | The oapi-codegen config that generates `acceleration/`             |

## Versioning

Versions are tags of the form `sdks/go/vX.Y.Z`, which `go get` reads as `vX.Y.Z`. The SDK is
developed on the repository's `accelerate` branch and released from `main`, which holds a copy
of this directory taken from one `accelerate` commit. While the version is `0.x`, a minor
version may change a signature; a patch only fixes. A version found broken after release is
retracted in `go.mod` and followed by the next patch, never replaced.

## Generated types

`acceleration/` is generated from the router's OpenAPI spec. Its types are this SDK's models:
`Session`, `AgentConfig`, `SttOptions` and the rest appear in signatures, and you build and read
them like any struct. Its operation methods (`CreateSessionWithResponse` and the rest) are not
the SDK's API. They are regenerated as the router changes and carry no compatibility promise.
The types follow the spec: a field the router adds is added in the next version, and one it
renames or removes is a breaking change the changelog names.
`stream.Backend.Client()` and `agents.Agent.Client()` hand out the generated client for an
operation this SDK has no method for yet; prefer a method of `client`, `agents` or `stream`
wherever one exists.

## Regenerating the client

`acceleration/generated.go` comes from
[acceleration/api/openapi.yaml](https://github.com/GetStream/Vision-Agents/blob/accelerate/acceleration/api/openapi.yaml), which is the same file
the Go server and the Python client come from. It is committed, so installing this module
needs no code generation. Generating it needs the spec beside it, so it runs in a checkout of
`accelerate`:

```bash
go generate ./...
```

The four socket operations are excluded: OpenAPI stops at the upgrade, so the three this SDK
uses (a session's events, a router modality, a dispatch worker) are written by hand on
`stream/socket.go`, the way the server side of them is. `openSocketSession` has no Go client
yet.

## Tests

```bash
go test ./...
```

No mocks. The client and the socket are tested against an `httptest.Server` and a real
`gorilla/websocket` upgrader, and the schema reflection and the directory loader against
what they are given.

`e2e_test.go` drives a running router instead of a stand-in for one, so it is behind a tag.
Start Postgres, Redis and the router as
[acceleration/README.md](https://github.com/GetStream/Vision-Agents/blob/accelerate/acceleration/README.md) describes, then:

```bash
STREAM_ACCELERATION_URL=http://localhost:8080 \
STREAM_ACCELERATION_CUSTOMER_ID=e2e \
go test -tags e2e -v .
```

Anything the deployment cannot do is skipped rather than failed: a router with no knowledge
provider is a valid router, and so is one with no telephony. `E2E_REQUIRED=1` fails every one of
those skips instead, which is how a release is checked. Its models are real, so a run costs
money; `e2e_test.go`'s package comment names the release's smoke set.
