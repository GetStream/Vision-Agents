# Tool call improvements

Asked for after sprint 17. Not started.

## Asked for

A tool registered in Go knows nothing about the call it is running: not who asked, not which
conversation, not whether the router has already sent this call once. It cannot tell the
person anything while it works, cannot say how it wants to be treated when the caller talks
over the agent, and cannot be told to stop. Six problems found in review (M1 to M6 below) all
come from the same gap. Asked for, in one piece:

- **The call as context.** `tools.CallFrom(ctx)` hands the function the model's call id, the
  session, the turn, the person whose turn triggered it, and the conversation. That is what
  deduplication, a per-user check on a shared registry, and an audit line all need.
- **Talking while it works.** `call.Update` puts a status in the conversation and lets the
  model say it in its own words, on a cascade and on speech-to-speech alike. `call.Say` speaks
  exact words, built on the `say` frame that exists today.
- **Policies per tool**, chosen at registration: `NeedsApproval`, `Interruptible`, `Timeout`.
- **A tool that changes its mind about interruption.** A read phase that barge-in may cancel,
  then `call.DisallowInterruptions(ctx)` before the write.
- **The six bugs fixed.**

```go
type LookupOrder struct {
    OrderID string `json:"order_id" schema:"the order number the caller read out"`
}

err := tools.Register(agent.Tools(), "lookup_order", "Look up an order by its number",
    func(ctx context.Context, args LookupOrder) (any, error) {
        call := tools.CallFrom(ctx)
        // call.ID            the model's call ID, the same when the router replays the call
        // call.SessionID     the session that asked
        // call.TurnID        the turn the call belongs to
        // call.User          the person whose turn triggered it: ID, Name, Custom
        // call.Conversation  the Chat channel (CID) and the session's Custom labels

        call.Update(ctx, "Looking up order "+args.OrderID)

        order, err := orders.Find(ctx, args.OrderID) // ctx is cancelled on barge-in or timeout
        if err != nil {
            return nil, err
        }
        if order.CustomerID != call.User.ID {
            return "That order isn't on this account.", nil
        }
        audit.Log(ctx, call.Conversation.CID, call.User.ID, "lookup_order", order.ID)
        return order, nil
    },
    tools.Interruptible(true),
    tools.Timeout(10*time.Second),
)
```

```go
err = tools.Register(agent.Tools(), "book_flight", "Book the flight the user picked",
    func(ctx context.Context, args BookFlight) (any, error) {
        call := tools.CallFrom(ctx)
        booking, err := airline.Book(ctx, airline.BookRequest{
            FlightID:       args.FlightID,
            UserID:         call.User.ID,
            IdempotencyKey: call.ID,
        })
        if err != nil {
            return nil, err
        }
        return fmt.Sprintf("Booked. Confirmation number %s.", booking.ID), nil
    },
    tools.Interruptible(false), // barge-in doesn't cancel ctx; only the timeout does
    tools.NeedsApproval("Book flight {flight_id}?"),
    tools.Timeout(60*time.Second),
)
```

```go
err = tools.Register(agent.Tools(), "rebook_flight", "Move the user's booking to another flight",
    func(ctx context.Context, args RebookFlight) (any, error) {
        call := tools.CallFrom(ctx)
        call.Update(ctx, "Checking seats on the new flight")

        // Read only: a barge-in here cancels ctx and nothing has changed.
        seat, err := airline.HoldSeat(ctx, args.NewFlightID)
        if err != nil {
            return nil, err
        }

        // From here on, barge-in no longer cancels.
        ctx = call.DisallowInterruptions(ctx)
        call.Say(ctx, "Got a seat, moving your booking now.")

        booking, err := airline.Rebook(ctx, args.BookingID, seat, call.ID)
        if err != nil {
            return nil, err
        }
        return fmt.Sprintf("Moved to %s, seat %s.", booking.FlightID, seat.Number), nil
    },
    tools.Interruptible(true),
    tools.Timeout(45*time.Second),
)
```

The barge-in this is meant to support:

1. The model calls `search_flights`, and the router sends `tool_call` to the process.
2. The user starts talking. The router stops the agent's speech, closes the call with
   "interrupted by the user" and sends `tool_cancel`.
3. The SDK cancels that call's ctx, and `airline.Search` returns `context.Canceled`.
4. The model answers the new turn ("make it Osaka") and calls `search_flights` again with a
   new call id.

## What exists

### Two paths into a Go function

| Path | Where it runs | Frame in | Frame out | Timeout |
| --- | --- | --- | --- | --- |
| Session tool | `Pipeline.runTool`, [stream/accelerated.go](../../sdks/go/stream/accelerated.go) | `tool_call` on the session socket | `tool_result` with `tool_call_id` | 30 s, `defaultToolTimeout` |
| Hosted tool | `Dispatch.runHosted`, [stream/dispatch.go](../../sdks/go/stream/dispatch.go) | `tool_call` on the dispatch socket | `tool_result` with `id` | 2 min, `defaultHostedTimeout` |

Both end in `Registry.Call` in [tools/registry.go](../../sdks/go/tools/registry.go), which
decodes the arguments and runs the function on the context it was given. That context is
the socket's in both paths, so nothing about one call can reach it.

The session frame carries `id`, `name`, `arguments`, `request_id` and `turn_id`
(`frameOf` in [api/sessionws.go](../../acceleration/internal/api/sessionws.go)). The hosted
frame carries `id`, `session_id`, `name` and `arguments` (`toolCallFrame` in
[api/dispatchws.go](../../acceleration/internal/api/dispatchws.go)). Neither says who spoke
the turn or which channel the conversation is in, though the router knows both.

### What the router already does per tool

[agent.ToolPolicy](../../acceleration/internal/agent/telephony.go) has `PreSpeech`, `Waits`
and `Cancels`, filled today only from a connector binding's `BindingPolicy`
(`pre_speech`, `on_interrupt`, `cancellable`). On a voice call, every tool but the two
telephony tools and a binding with `on_interrupt: cancel` **outlives a barge-in**: the call
is answered "still running" and its result follows as a message of its own. A caller's
session tool cannot ask for anything else, because `SessionTool` has no field for it.

`SessionTool` already has `approval` (title, message, button labels, reason argument),
`display_title` and `executor`. Approval is shown on the `ai_tool_call` attachment of a
persistent conversation, and the JS SDK holds an approval-gated call in
`session.approvals` until the app answers. The Go SDK sets none of these fields.

The `say` command exists on the session socket and speaks exact words. Nothing carries a
status for the model to phrase.

When the bridge stops waiting on a session tool, by its timeout or by the turn's context
ending, it publishes a `ToolCall{Cancel: true}`, which goes out as `tool_cancel`
(`bridge.Run` in [session/tools.go](../../acceleration/internal/session/tools.go)). The
hosted path has no equivalent: `Pool.RunHosted` in
[dispatch/tools.go](../../acceleration/internal/dispatch/tools.go) forgets the call and the
worker is never told.

## The six problems, checked against the code

**M1: a second watcher can fail a call the owner would have answered.** A session's events
go to every watcher, `tool_call` included, and the first `tool_result` resolves the bridge;
later ones answer nothing (`bridge.Resolve`). The JS SDK answers a tool it does not have
with an error at once (`runTool` in [sdks/js/src/session.ts](../../sdks/js/src/session.ts)),
and Go does the same (`nothing is registered as …`). A backend that creates the session with
tools and a browser that watches it with the JS SDK race, and the browser wins, because its
answer costs nothing. The model sees a failure. The backend's function still runs, so the
side effect happens with nobody told. Every open tab is another racer, and a browser that
registers a tool of the same name runs it with the browser's credentials. Derived from the
code, not reproduced end to end.

**M2: Go ignores `tool_cancel`.** `Pipeline.watch` matches only `tool_call`; `tool_cancel`
passes through as an ordinary event, and the function goes on running on the socket's context.
Python, JS, C#, Kotlin, Swift, Dart, Ruby, PHP and Rust all handle the frame. The hosted
path is worse: the router never sends one.

**M3: voice calls can run twice.** `Session.watch(true)` re-sends every call still pending
to a process that attaches (`WatchPendingVoiceTools`), and `bridge.Pending` says hosts
"must fence repeated execution with durable receipts". The Go SDK has no fence, and the
function cannot see the id to build one.

**M4: unbounded concurrency.** Every `tool_call` starts a goroutine, on both paths, with no
limit or queue. A burst of parallel calls reaches the customer's database all at once. The
router's 64-call `toolQueue` per worker bounds only the socket, not the work.

**M5: the two paths differ.** 30 s against 2 min. Images work on the session path; the
hosted path refuses them in the SDK (`runHosted`) and could not carry them anyway, since
`dispatch.ToolResult.Output` is a string. The result names the call `tool_call_id` on one
socket and `id` on the other.

**M6: no escape hatch for the schema.** `Schema` in [tools/schema.go](../../sdks/go/tools/schema.go)
reflects types and reads one tag, `schema`, as a description. No `enum`, no bounds, no
examples, no format beyond `time.Time`, and no way to register a hand-written JSON Schema.

## What changes

### On the wire (router)

- **`tool_call` carries the call.** Both sockets gain `session_id` (the session socket lacks
  it), `turn_id` (the hosted one lacks it), `user` (`id`, `name`, `custom`: the participant
  who spoke the turn on a call, the command's sender in text) and `conversation` (`cid`, and
  the session's `custom`). A replayed call keeps its id, which is what makes `call.ID` a
  usable idempotency key.
- **`SessionTool` and `host_tools` gain `interruptible` and `timeout_ms`.**
  `interruptible: true` sets `ToolPolicy.Cancels` for that tool, so barge-in cancels it and
  `tool_cancel` goes out; false is today's behaviour, answered "still running" and finished
  later. `timeout_ms` replaces the session-wide timeout for that one tool. `host_tools`
  gains `approval` as well, so a hosted tool can ask too.
- **Calls are addressed (M1).** Every watcher still sees `tool_call`, since a browser shows
  calls and holds approvals. But the frame says whether this watcher is the one to run it:
  `run: true` goes only to watchers that declared the tool, and of those, a tool declared by
  a server-side caller runs only on a server-side watcher. A watcher declares what it runs
  when it creates the session (as today) or when it attaches. When nobody attached can run
  a call, the router answers it with a failure at once, rather than waiting out the timeout.
- **`tool_cancel` reaches hosted workers (M2).** `RunHosted` sends it when the deadline or
  the turn ends, as the bridge does.
- **The hosted result matches the session one (M5).** It accepts `tool_call_id` (and keeps
  `id` for old workers) and content parts with images. `dispatch.ToolResult` carries
  `[]llm.ContentPart` rather than a string.
- **`tool_update` (new command).** `{tool_call_id, text}` while the call is running. On a
  cascade it feeds the line the agent says while a tool runs (the same slot `PreSpeech` and
  `holdPurpose` fill), so the model phrases it. On speech-to-speech it is added to the
  provider's conversation as context for what the model says next. An update for a call
  that has finished is dropped.
- **`tool_hold` (new command).** `{tool_call_id}`: from now on, barge-in answers this call
  "still running" instead of cancelling it. The router answers `tool_held`, or `tool_cancel`
  if it had already cancelled it, so the SDK knows which happened first.
- **Approval on a call.** `approval.title` may name arguments in braces, filled from the
  call. In text this is the attachment flow that exists. On a voice call the agent asks the
  question aloud and the call is held until the caller agrees or declines; declined, it is
  answered as declined and never reaches the function.

### In the Go SDK

- **`tools.Register(registry, name, description, run, options...)`** and
  `Registry.Add(tool, options...)`, with `Interruptible`, `Timeout`, `NeedsApproval`
  (and `Concurrency`, below). Options go out on `SessionTool` and `host_tools`.
- **`tools.CallFrom(ctx) *Call`**, holding `ID`, `SessionID`, `TurnID`, `User`,
  `Conversation`, and the methods `Update(ctx, text) error`, `Say(ctx, text) error` and
  `DisallowInterruptions(ctx) context.Context`. Outside a tool call it returns a `Call`
  whose methods do nothing, so a function can be unit-tested without a socket.
- **One runner for both paths (M5).** `runTool` and `runHosted` both become
  `tools.Runner`: decode, apply the policy, run on a context of the call's own, write the
  result with the field names that socket takes. Pictures work on both.
- **Each call gets its own context (M2).** Derived from the socket's, cancelled on
  `tool_cancel`, on its own `Timeout`, or when the socket ends.
  `DisallowInterruptions` sends `tool_hold`, waits for the answer, and returns a context
  that only the timeout and the socket ending can cancel. If `tool_cancel` won, the context
  it returns is already cancelled, so the write after it fails before it starts.
- **A fence for replays (M3).** Calls are keyed on `call.ID`. A replay of a call still
  running in this process joins the running one rather than starting another. A finished
  call's result is kept for a while and sent again. `tools.Receipts` is an interface
  (`Load`, `Store`) a caller can back with Redis or Postgres so the fence holds across
  processes; the default is in memory.
- **A limit (M4).** `Concurrency(n)` per tool, and a ceiling for the registry, default 16.
  A call over the limit waits, and its timeout keeps counting while it does.
- **Never answer what you do not run (M1).** A call not addressed to this process, or for a
  tool it does not hold, is left alone. The same rule goes into every other SDK.
- **The schema (M6).** `tools.RegisterSchema(registry, name, description, schema, run)`
  takes a JSON Schema as written and hands the function `json.RawMessage`. A type can
  implement `JSONSchema() map[string]any` to describe itself. Struct tags gain `enum`,
  `minimum`, `maximum`, `pattern`, `format` and `example`, beside `schema` for the
  description.

## Order of work

Each step ships on its own.

1. **Go only, no router change.** Handle `tool_cancel` on the session socket with a context
   per call; the concurrency limit; the in-process replay fence; stop answering unknown
   tools; `RegisterSchema` and the new tags; `CallFrom` with what the frames carry already
   (`ID`, `SessionID`, `TurnID`).
2. **Every other SDK stops answering unknown tools.** JS first, since it is the one that
   races a backend.
3. **Router: the call on the frame and the policies.** `user` and `conversation` on
   `tool_call`; `interruptible` and `timeout_ms` on `SessionTool` and `host_tools`;
   `tool_cancel` to hosted workers; hosted results with `tool_call_id` and images.
   OpenAPI first, then regenerate the clients.
4. **Router: addressed calls (M1).** `run` on the frame, declaring tools on attach, the
   server-side rule, and the immediate failure when nobody can run a call.
5. **Router and Go: `tool_update`, `tool_hold`, `call.Update`, `call.Say`,
   `DisallowInterruptions`.**
6. **Approval on a voice call**, and `NeedsApproval` in Go.
7. **The rest of the SDKs** catch up on `CallFrom`, the policies and updates, Python first.

## How it is tested

- **The race (M1).** A suite in `acceleration/internal/api` with two watchers on one
  session, one holding the tool and one not: the holder's result is the one the model sees,
  whichever answers first. Then a browser-credential watcher that holds a tool of the same
  name: it is never addressed.
- **Cancel (M2).** A fake router over a real socket: `tool_cancel` cancels the function's
  context on the session path and the hosted path; an `Interruptible(false)` tool is not
  cancelled by barge-in and is by its timeout.
- **Replay (M3).** Attach, detach mid-call, attach again: the function runs once and both
  attachments get its result.
- **The limit (M4).** Fifty parallel calls against `Concurrency(4)`: never more than four
  running, and the queued ones time out on their own deadline.
- **Hold.** `tool_hold` racing a barge-in, both orders: the function's context ends exactly
  when the router says it cancelled.
- **A voice eval**, beside [telephony-latency](../evals/telephony-latency.md), for
  `call.Update` on a cascade and on speech-to-speech: the status is said, once, in the
  model's words, and does not delay the result.

## Not decided

- **The default for `Interruptible`.** The router keeps every caller tool running through a
  barge-in today. Defaulting to true would match the example above and would also start
  cancelling every existing tool the first time somebody talks over the agent. The plan
  above keeps false as the default.
- **One default timeout or two.** With a per-tool timeout the defaults matter less; 30 s
  and 2 min could stay, or both become 30 s with hosted workers asking for more.
- **Where an update goes on speech-to-speech.** Adding context mid-call works at OpenAI and
  Gemini differently, and a provider that cannot take it may have to say the update as
  exact words instead.
- **Spoken approval.** Whether the agent's question is the template as written or a line
  the model writes from it, and how long a call waits for an answer before it is declined.
