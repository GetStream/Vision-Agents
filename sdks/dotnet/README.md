# getstream-vision-agents-net

The .NET server-side SDK for the Stream acceleration backend. The backend joins the call,
hears the caller, answers and speaks; this creates the session, holds the events socket,
runs your functions and configures everything.

```bash
dotnet add package getstream-vision-agents-net
```

.NET 10. It depends on [getstream-net](https://www.nuget.org/packages/getstream-net) for
Stream calls and tokens, and on YamlDotNet for `agent.yaml`.

## Who is calling

```csharp
using GetStream.VisionAgents;

// A router with nothing in front of it, which is a laptop.
using var api = new VisionAgentsClient(new VisionAgentsOptions { Url = "http://localhost:8080", CustomerId = "examples" });

// Stream's hosted router, through its authenticating proxy. The default when no Url is set.
using var hosted = new VisionAgentsClient(new VisionAgentsOptions { ApiKey = key, ApiSecret = secret });

// From IHttpClientFactory, which owns the handler.
var pooled = new VisionAgentsClient(factory.CreateClient("vision-agents"));
```

Every option left null falls back to the environment the other SDKs read:
`STREAM_ACCELERATION_URL`, `STREAM_ACCELERATION_CUSTOMER_ID`, `STREAM_API_KEY`,
`STREAM_API_SECRET` and `STREAM_ACCELERATION_AUTHENTICATE`. With no URL at all the client goes
to Stream's hosted router. A failure raises
`RouterException`, carrying the status, the operation and what the router said.

## An agent

```csharp
var agent = new Agent(new AgentOptions
{
    Name = "jean",
    Instructions = "You are a friendly assistant. Keep replies short.",
    Pipeline = new Pipeline { Llm = "llm-fast", Stt = "en-low-latency", Tts = "sonic_36" },
    Harness = new Harness { Subagents = new Dictionary<string, string> { ["default"] = "openai/gpt-5.6-sol" }, Sandbox = Sandbox.Daytona() },
    CostTracking = new Dictionary<string, string> { ["customer_id"] = "123" },
    MemoryFilter = new Dictionary<string, string> { [Agent.UserKey] = "123" },
});

agent.Tools.Register<Weather, string>("get_weather", "Get the current weather for a city",
    (weather, cancellationToken) => WeatherAsync(weather.City, cancellationToken));

await using var session = await agent.JoinAsync();
Console.WriteLine(agent.MonitorUrl(session));

await agent.Responses.CreateAsync("Say hello.");
await foreach (var happened in session.EventsAsync())
{
    Console.WriteLine($"{happened.Kind} {happened.Text}");
}

record Weather(string City);
```

`JoinAsync` creates the Stream call through getstream-net and has the backend join it.
`ChatAsync` holds the same conversation in writing. `Config = "docs"` starts from a config
stored under that name, and `ConfigId` from one by id. The harness is written onto the config
when it syncs, and every session under that config runs with it. The model asks for a tool over
the session socket, this process runs it, and the answer or the exception goes back the same way.

A stored config is changed in place, and only what is set is sent:

```csharp
await api.Agent("support").UpdateConfigAsync(new AgentConfigPatch { Guardrail = "No refunds.", VisibleTools = ["athena_*"] });
```

On the phone:

```csharp
var answered = await agent.WaitForCallAsync("+15551234567");
var placed = await agent.OutboundCallAsync(from: "+15551234567", to: "+15557654321");
```

## Dispatch

A caller reached a Stream call over SIP, or somebody wrote in a channel, and the router
found out by webhook. The worker connects out and waits, so nothing you run has to be
publicly reachable.

```csharp
var dispatch = new Dispatch(new DispatchOptions { Capacity = 4 });
dispatch
    .WaitForCall(async call =>
    {
        await using var agent = new Agent(new AgentOptions { Config = "support" });
        var session = await agent.JoinAsync(call);
        await session.WaitAsync();
    })
    .WaitForMessage(async message =>
    {
        if (message.SessionId != "")
        {
            await dispatch.AnswerAsync(message);
            return;
        }
        var agent = await dispatch.GetOrCreateAgentAsync(message, () => new Agent(new AgentOptions { ConfigId = message.ConfigId }));
        await agent.Responses.CreateAsync(message.Text);
    });

await dispatch.RunAsync(stopping);
```

`GetOrCreateAgentAsync` keeps one agent per channel, so the second message on it goes to
the agent that answered the first. An agent whose agent.yaml says `dispatch: {text: enabled}`
hands what end users write to the worker with the running `SessionId` (and `CommandId`);
`AnswerAsync` has the model answer it with the worker's credential acting for the writer, and
`GetOrCreateAgentAsync` refuses such a message. The router is told each call and message is
`done`, with the exception's message if the handler threw. `RunAsync` returns when the token
is cancelled, after the work in hand has finished. A connection the router drops is opened
again, telling the router how much is still being handled.

A worker can also host tools for every session opened under an agent id, including
sessions opened from a browser. The router sends each call here. Handlers are optional:

```csharp
var tools = new Tools().Register<Weather, string>("get_weather", "Get the current weather for a city",
    (weather, cancellationToken) => WeatherAsync(weather.City, cancellationToken));
await new Dispatch().Host("my-agent", tools, timeout: TimeSpan.FromMinutes(1)).RunAsync(stopping);
```

If the router refuses the tools, `RunAsync` throws `InvalidOperationException` with the
reason.

## An agent written down as a directory

```
agents/jean/
  agent.yaml            required: what it runs on (llm, stt, tts, speed, harness, tags, dispatch, ...)
  instructions.md
  guardrail.md
  skills/think.md
  knowledge/pricing.md
  knowledge/urls.yaml   urls, or {url, title, description, refresh_hours}
  simulations/lunch.yaml
  .agent_sync           written by SyncAsync: the fingerprint last synced and when
```

```csharp
var agent = new Agent(new AgentOptions { Config = "agents/jean" });
await agent.SyncAsync();
await agent.Knowledge.AddUrlAsync("https://example.com/pricing", title: "Pricing", refreshHours: 24);
```

`SyncAsync` sends the directory in one request. The fingerprint is the one the Go and
Python SDKs take, so `.agent_sync` written by any of them is read back by this one, and an
unchanged directory is only read back. A key `agent.yaml` does not know is refused. Joining
syncs first. Each file in `simulations/` is a list of `{name, scenario, assertion, mode, ...}`;
a directory with the folder replaces the agent's simulations, and no directory leaves them alone.

## Simulations

```csharp
var simulation = await api.Simulations.CreateAsync(new SimulationRequest
{
    ConfigId = config.Id, Name = "lunch", Scenario = "Order a club wrap.", Assertion = "One club wrap was ordered.",
});
var run = await api.Simulations.RunAsync(simulation.Id);
run = await api.Simulations.Runs.GetAsync(run.Id);
```

## Conversations, going back and branching off

```csharp
var session = await api.Agent("docs").ChatAsync(new SessionOptions { Title = "Pricing" });
var items = await session.Responses.Items.AllAsync();
var branch = await session.ForkAsync(new ForkOptions { ResponseId = items[2].ResponseId, Title = "asked again" });

var found = await api.Sessions.SearchAsync("pricing", new SessionQuery { Limit = 20 });
var live = await api.Sessions.QueryAsync(new SessionQuery { Agent = "docs", State = "live", Cursor = found.NextCursor });
```

Lists come a page at a time: `Items`, and `NextCursor` while `HasMore`. A text conversation
is kept in Stream Chat unless it is `Incognito`, which keeps nothing. One kept in Stream Chat
cannot be rewound, since the channel still holds the later turns: fork it at the response instead.

`CloseAsync` stops a session and keeps what it recorded. `DeleteAsync` deletes it with its turns
and memories, `DeleteMemoriesAsync` only what it remembered, and
`api.Memories.TruncateAsync(userId)` everything remembered about a user.

## Changing a session mid-conversation

```csharp
await session.UpdateAsync(new UpdateSessionRequest { Llm = "llm-thinking", Thinking = "high", Title = "Pricing, thought through" });
```

Models and instructions apply from the next turn; a property left null is left as it is.

Somebody who has not signed up is a guest. `GuestUserAsync` mints one, `AsGuest` is a client
acting for them, and `ClaimGuestUserAsync` moves their conversations onto the account they
turn out to be.

## One modality at a time

```csharp
// "healthcare" is a router config: it holds the target each modality runs on.
var router = api.Router("healthcare");

var transcript = await router.Stt.RecordingAsync(new Recorded { Url = "https://example.com/interview.mp4" });
var speech = await router.Tts.RecordingAsync("Hello there.");
var answer = await router.SearchAsync("perioperative antibiotic guidance");

await using var llm = await router.Llm.RealtimeAsync();
await llm.AskAsync(new Question { Messages = [new Said("user", "Hello")] });
await foreach (var delta in llm.ReadAllAsync()) { if (delta.Done) break; }
```

`RealtimeAsync` opens the socket; `RecordingAsync` is the batch form, which is cheaper and
more accurate. A router comes from the client and its models from its config; the options
carry settings such as language, never the target.

## Working on this package

Everything runs in Docker; nothing is installed on the host.

```bash
docker run --rm -v "$PWD/../..":/repo -v va-nuget:/root/.nuget -w /repo/sdks/dotnet \
  mcr.microsoft.com/dotnet/sdk:10.0 dotnet test --project tests/GetStream.VisionAgents.Tests

# regenerate src/GetStream.VisionAgents/Generated/Models.cs from the spec; --check in CI
docker run --rm -v "$PWD/../..":/repo -v va-nuget:/root/.nuget -w /repo/sdks/dotnet \
  mcr.microsoft.com/dotnet/sdk:10.0 dotnet run --project tools/Generate
```

The tests run against a real Kestrel server and a real WebSocket upgrade, with no mocks.
`LiveTests` talk to a running router and skip unless `VISION_AGENTS_LIVE_URL` names one
(`http://host.docker.internal:8091` from inside Docker); `VISION_AGENTS_LIVE_CUSTOMER`
defaults to `examples`. They hold real conversations, so they spend tokens.
