---
name: sdk-dotnet
description: How to build and extend the .NET SDK in sdks/dotnet. Read this before changing the client, the socket, the agent, dispatch or the router, before regenerating the models, or before reaching for a NuGet package.
---

# .NET SDK conventions

The per-language half of [sdk](../sdk/SKILL.md). It records the decisions
[`sdks/dotnet`](../../../sdks/dotnet) already follows, so a change lands consistently rather
than re-litigating them.

Assume .NET 10 (the LTS), C# 14, `net10.0`, nullable reference types on, warnings as errors,
central package management (`Directory.Packages.props`). Server side only.

## Toolchain: Docker, never the host

Nothing is installed on the host. Every command runs in `mcr.microsoft.com/dotnet/sdk:10.0`
with the repo mounted and the NuGet cache in the `va-nuget` volume:

```bash
D="docker run --rm -v /path/to/Vision-Agents:/repo -v va-nuget:/root/.nuget -w /repo/sdks/dotnet mcr.microsoft.com/dotnet/sdk:10.0"
$D dotnet build GetStream.VisionAgents.slnx
$D dotnet test --project tests/GetStream.VisionAgents.Tests
$D dotnet run --project tools/Generate              # regenerate the models
$D dotnet run --project tools/Generate -- --check   # what CI runs
docker run ... -e VISION_AGENTS_LIVE_URL=http://host.docker.internal:8091 ... \
  dotnet test --project tests/GetStream.VisionAgents.Tests -- --filter-class GetStream.VisionAgents.Tests.LiveTests
```

Build output (`bin/`, `obj/`, `TestResults/`) is ignored by `sdks/dotnet/.gitignore`, nowhere
else.

## Naming

Stream's .NET packages are `getstream-net` (the current generated server SDK, namespace
`GetStream`), `stream-chat-net` and `stream-feed-net`. So the package id is
**`getstream-vision-agents-net`** and the namespace **`GetStream.VisionAgents`**, with the
generated DTOs under `GetStream.VisionAgents.Models`. Do not rename to `StreamIO.*`; nothing
Stream ships to NuGet is called that.

Public types are named after what they are to a caller (`Agent`, `Session`, `Dispatch`,
`Router`, `Folder`), async methods end in `Async` and take a `CancellationToken` last,
defaulted. Options are `sealed record`s with `init` properties.

## The generator: NSwag, DTOs only

`tools/Generate` uses NSwag 14.7.1 **as a library**, writing
`src/GetStream.VisionAgents/Generated/Models.cs`. The output is committed, so a restore needs no
code generation, and `--check` fails CI when the spec has moved without it.

Why NSwag, and why DTOs only:

- **The request layer has to be ours.** A token is minted per request, a local router takes
  `X-Customer-Id`, a hosted one wants `api_key` and `stream-auth-type`, and sockets carry the
  same credential. A generated client brings a runtime that decides those things, and it
  decides them differently. So the generator emits POCOs and `VisionAgentsClient` has one
  method per HTTP verb (`GetAsync<T>`, `PostAsync<T>`, ...), the way sdk-js does.
- **Kiota** was the other serious option. It generates request builders over its own
  abstractions (`Microsoft.Kiota.*` packages at runtime), its models are not plain
  `System.Text.Json` types, and it cannot be told to emit models alone. That is a runtime
  dependency and an auth pipeline to fight for types we only need the shapes of.
- **openapi-generator** needs a JVM, emits an `ApiClient`/`Configuration` runtime, and its
  `System.Text.Json` path is the less-travelled one.
- **Refitter** generates Refit interfaces, so Refit at runtime, for the same reason no.
- NSwag emits `System.Text.Json` POCOs with nullable reference types, runs on the same .NET
  image, and is pinned in `Directory.Packages.props`. The library rather than the CLI because
  the CLI cannot be told two things that make the output usable: PascalCase property names
  (`PascalCaseNames`) and enums as strings (`EnumsAsStrings`). A C# enum has nowhere to put a
  value the router adds after this ships, so an enum would fail the whole response; the
  values stay in the doc comment.
- `GenerateDefaultValues = false` and optional properties nullable: **never copy a schema
  default into a request.** A default sent because the type had one is how a caller silently
  loses what their config named. A field left null is left out, so the config or the router
  decides.

Never hand-edit `Generated/Models.cs`. Never re-declare a schema by hand; use the model.

## Dependencies

- **getstream-net 16.0.1** for what is Stream rather than this router: `Edge` creates the call
  (`VideoClient.GetOrCreateCallAsync`) and mints user tokens (`CreateUserToken`). It refuses a
  secret shorter than 256 bits, which a real Stream secret never is; test fixtures use a
  64-character one. Its exceptions are wrapped in `RouterException`.
- **YamlDotNet 18.1.0** for `agent.yaml`, `urls.yaml` and skill frontmatter, with duplicate
  keys refused. A key `agent.yaml` does not know is refused, not dropped.
- Tokens for the router itself are HS256 signed with `System.Security.Cryptography`, not a JWT
  package: a dozen lines in `Backend.SignToken`.
- Nothing else at runtime. A new package is a design question; ask first.

## HttpClient

`new VisionAgentsClient(options)` owns its `HttpClient`; `new VisionAgentsClient(httpClient,
options)` borrows one, which is how `IHttpClientFactory` plugs in, and never disposes it.
`Agent` disposes a client only if it made it. Do not add a DI extension package.

## Sockets

Hand-written in `Socket.cs`, because OpenAPI stops at the upgrade. One `Socket` over
`ClientWebSocket` for the session events, dispatch and the modality streams.

- Server side, so **credentials go in headers**, never the query string.
- Frames are loose (`Frame`, read through `Text`, `Flag`, `Number`, `Nested`), so a frame type
  the router learns later reaches a caller rather than being dropped.
- The modality start frame carries `target` at the top level as well as inside the option
  block, like Go and the Python plugin: the router refuses a frame whose target is only in the
  block (see `readStart` in `acceleration/internal/api/streamws.go`), although the spec
  describes only the block.
- No automatic reconnection: `respond` and `tool_result` are not idempotent.

## Sessions and tools

- The watch loop starts when the session opens and handles `tool_call` and `tool_cancel`
  itself, so a tool runs whether or not anybody iterates `EventsAsync`. At most 16 run at
  once; a tool that throws is answered with its message, because the model is mid-sentence
  waiting.
- `EventsAsync` has one reader over a bounded 256-slot channel that drops the oldest.
- `CloseAsync` sends `close` over an open socket, else DELETEs the session; a 404 is fine.
- Rewinding a conversation kept in Stream Chat is refused by the router (400, "fork it at the
  response"); `ForkAsync(new ForkOptions { ResponseId = ... })` is the way.

## Agent and folder

- `JoinAsync` creates the Stream call then opens the session; `JoinAsync(InboundCall)` joins
  one dispatch already made; `ChatAsync` is `text: true` and no call; `OutboundCallAsync`
  places the call first, then joins with `navigating`; `WaitForCallAsync` attaches the number,
  joins and returns once somebody speaks.
- `Config` is a directory holding `agent.yaml` or a stored config's name; `ConfigId` is a
  stored config by id, which is what a dispatched message carries. Naming both is refused.
- What code sets wins over what the directory says.
- `SyncAsync` is one `POST /v1/agents/sync`. The fingerprint is **MD5, ported byte for byte
  from Go** (`Folder.Fingerprint`, `Folder.GoMap` for Go's `%v` of a map), so `.agent_sync`
  written by Go or Python is read back here. The fixture hash `02a7b2c8428f31e3a2b93ca2f5a6ec70`
  is pinned in `FolderTests`; if it changes, Go changed first or this is wrong. The stamp is
  written with relaxed escaping, since `System.Text.Json` would otherwise write the `+` of the
  offset as `\u002B`.

## Dispatch

- Handlers run off the read loop in `Task.Run`, tracked by id in a `ConcurrentDictionary`
  whose entry is removed before the tracking task completes, so `Active` is exact once
  `RunAsync` has drained.
- `RunAsync` returns quietly when cancelled, even during connect, then drains work in flight
  and closes the agents it made. Dropping work would hang up on whoever is talking.
- `GetOrCreateAgentAsync` keeps one agent per channel while its session is live, and opens
  chats with `Persist` and the message's `AgentId`, as Go and Python do.

## Tests

xUnit v3 (4.0.1) on the Microsoft Testing Platform (`global.json` sets the runner), because
.NET 10's `dotnet test` no longer drives VSTest here. Never mock.

- `TestRouter` is a real Kestrel server on loopback port 0 with real WebSocket upgrades. It
  records what arrived and lets a test script frames. Assert on what reached the far end.
- Getstream-net is pointed at the same server through `EdgeOptions.BaseUrl`, so the Stream call
  request is checked on the wire too.
- `LiveTests` skip unless `VISION_AGENTS_LIVE_URL` is set. They hold real conversations and
  wait for what the router writes behind a request instead of sleeping.
