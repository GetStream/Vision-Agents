---
name: sdk
description: Build or debug applications with the Stream acceleration SDKs. Use for agent configuration, sessions, responses, tools, dispatch, voice, video and SDK authentication. For changes to SDK implementations or generated clients, use sdk-maintenance.
---

# Build an application with the SDK

## Start with the application's version

1. Identify the language, installed SDK version and runtime: server, browser or mobile.
2. Read that version's public API and examples. This repository's `accelerate` branch can
   contain APIs that are not released yet. Check the application's manifest and lockfile
   before copying an example or changing a dependency.
3. Choose the backend environment and caller identity before creating a client.
4. Implement one conversation, then add the tools or media the application needs.

Use public resource methods. Check signatures in the selected SDK; method names, option
names and supported features differ between languages. If an API is missing from the
installed version, report the gap and select a supported version or implementation path.
Do not invent a wrapper based on another language's example.

## Choose the starting point

Paths below are relative to this repository. Read only the language and feature needed.

| Task | Source to start from |
| --- | --- |
| Go agent with a local tool | [Go quickstart](../../../sdks/go/examples/quickstart/main.go) and its adjacent `agents/myagent` folder |
| Go application using stored configs | [Go SDK](../../../sdks/go/README.md), [client](../../../sdks/go/client/client.go) and [sessions](../../../sdks/go/client/sessions.go) |
| JavaScript or TypeScript, server or browser | [JS SDK](../../../sdks/js/README.md), [client](../../../sdks/js/src/client.ts) and [sessions](../../../sdks/js/src/session.ts); package `@stream-io/calypso` |
| Inbound dispatch worker | [Go dispatch](../../../sdks/go/examples/dispatch/main.go) or [JS worker](../../../sdks/js/examples/worker.ts) |
| Voice and video | [Go voice](../../../sdks/go/examples/voice/main.go) or [JS voice](../../../sdks/js/examples/voice.ts) |

For Python, Swift, Kotlin, Dart, .NET, Ruby, Rust or PHP, start in `sdks/<language>` and
read its manifest, README and examples. The `sdk-<language>` skills describe SDK
maintenance; they are not application tutorials. Do not assume parity with Go or JS.

The Go quickstart creates an agent from a folder, syncs it, registers `lookup_order` and
asks one question. Run it from `sdks/go/examples/quickstart` so its relative config path
resolves. Use the environment below; running it against a hosted backend makes live calls.

## Connect and authenticate

Read [environment setup](../../../DEVELOPMENT.md#pointing-an-sdk-at-a-router) for the full
configuration. For Go and Node, the common setups are:

| Environment | Endpoint selection | Identity |
| --- | --- | --- |
| Production, us-east4 | Default, or `STREAM_ACCELERATION_REGION=us-east4` | `STREAM_API_KEY` and `STREAM_API_SECRET` on the server |
| Staging | `STREAM_ACCELERATION_URL=https://accelerate-staging.gcp.stream-io-api.com` | Server key and secret; unset `STREAM_ACCELERATION_CUSTOMER_ID`. Node also sets `STREAM_ACCELERATION_AUTHENTICATE=1` |
| Local router without an auth proxy | `STREAM_ACCELERATION_URL=http://localhost:8080` | `STREAM_ACCELERATION_CUSTOMER_ID` for the local tenant |

Choose a URL or region. An explicit constructor choice wins over environment variables;
the environment URL wins over the environment region. Check for leftover overrides when
requests reach the wrong backend. A valid region name does not mean that region is
deployed. Confirm availability before selecting another production region.

Go does **not** read `STREAM_ACCELERATION_AUTHENTICATE`. With no customer ID it
authenticates. For staging with a customer ID still set, use
`stream.Backend{Authenticate: true}` or remove that ID. Regional production endpoints
authenticate even if a local customer ID remains set. See [Go backend](../../../sdks/go/stream/backend.go).

Browser and mobile clients receive a public API key and a user token from a trusted
server. Keep the API secret on the server. In JS, await `api.setUser(user, token)` before
opening the user's session. Browsers do not read the server's environment variables;
pass the endpoint explicitly when overriding production. Reuse the same user identity
for the agent, Stream Chat and Stream Video.

## Configure the agent, then open a conversation

Creating a handle such as `api.agent("support")` does not create a stored agent config.
Use an existing config or sync one from a server process first. The Go quickstart's
folder shows `agent.yaml`, `instructions.md`, `guardrail.md`, `skills/` and `knowledge/`.
Use the selected SDK's sync API; do not hand-edit its `.agent_sync` state.

For Go and JS, the conversation flow is:

1. Create a session through the agent's `Sessions.Create` / `sessions.create` resource,
   or resume an existing one by its session ID. Save that ID if the user will return.
2. Send input with `session.Responses.Create(ctx, text)` / `session.responses.create(text)`.
   Its return value identifies the response; answer text arrives through session events.
3. Consume `session.Events()` / `session.events()`. Display `response_delta` text, handle
   errors and wait for `responded` with `PendingWork` / `pendingWork` false when collecting
   a complete turn. Keep a single JS event consumer; multiple iterators split events.
4. Close the session when the application is done with it. `Close` / `close` stops it and
   keeps its history. `Delete` / `delete` also removes its turns and learned memory;
   deletion is not connection cleanup. Release client-owned media connections separately.

Use the SDK's typed session update request for editable fields. Do not copy config fields
into a session update: for example, Go's `UpdateSessionRequest` has no `instructions` or
`subagent` field. User tokens can reach fewer operations than server credentials. Keep
agent configuration and dispatch on the server, and scope session access to the caller.

## Tools and media

In accelerated mode, the backend runs the model loop. A tool registered in your worker
runs in that worker's process: the backend sends the call over its socket and receives
the result. Keep the worker running for conversations that need those tools. Starting
a client or syncing a folder does not deploy your tool implementation to Stream.

Register tools through the SDK with a name, description, argument schema and handler.
Validate arguments and authorize access using the authenticated caller, not an ID the
model supplied. For a side effect, use the SDK's approval mechanism and an idempotency key
where the destination supports one. Return failures through the tool API so the pending
model turn can finish. The [caregiving example](../../../sdks/go/examples/caregiving/main.go)
shows tools, approvals and memory scoped to a family.

Add voice or video from the language's example after text works. Check its Stream media
dependencies and teardown flow. Keep text, tools and media attached to the same session.
For automated browser tests, use fake media devices and synthetic input.

## Verify the application

- Compile or typecheck against the installed SDK version.
- Exercise a complete response and tool result, not just client construction. Handle
  errors and clean up the session when the application exits.
- Check user ownership when resuming sessions or accessing memory.
- Use local test services or the intended test environment. State which live operations
  were exercised and which were only checked offline.

For SDK implementation changes, generators or release work, continue with
[sdk-maintenance](../sdk-maintenance/SKILL.md). Keep feature backlogs and update history
out of this application guide.
