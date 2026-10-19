# Kotlin SDK

One Android library for talking to an agent from an Android app: `io.getstream:vision-agents-core`,
package `io.getstream.visionagents.android.core`. It holds sessions, responses, the event socket
and the conversation as `StateFlow`; the Stream Chat channel a text conversation is kept in; and
joining the call, so the conversation can be spoken. It depends on Stream Chat (7.15.0) and
Stream Video (1.34.0), and needs minSdk 24 and compileSdk 37, which is what Stream's SDKs require.

It ships no views. Show a conversation with Stream Chat's AI components,
`io.getstream:stream-chat-android-ai-compose`, which the app adds itself;
[`examples/voice_agents/android_demo`](../../examples/voice_agents/android_demo) does.

Stream Chat 7.15.0 is published to Stream's Maven repository rather than Maven Central:

```kotlin
dependencyResolutionManagement {
    repositories {
        google()
        mavenCentral()
        maven("https://stream-io-repo.com")
    }
}
```

## Using them

```kotlin
val agents = VisionAgents(apiKey = "your_api_key")
agents.setUser(User(id = "jlahey")) { yourBackend.agentToken() }

// In writing. No call is joined, nothing is transcribed or spoken.
val chat = agents.agent("docs").chat()
chat.responses.create("What are your opening hours?")
// chat.conversation is a StateFlow: the question shows at once, the turns grow as the reply
// streams in, and its state says what the agent is doing.

// The Stream Chat channel the conversation is kept in.
val channel: ChannelClient = chat.chat(context)

// Out loud. The agent joins a call and so does this device.
val voice = VoiceSession.start(context, agents, options = SessionOptions(agent = "docs"))
voice.join()

// Signing out: closes the Stream Chat and Video clients the SDK built.
agents.disconnect()
```

`VisionAgents(apiKey = ...)` reaches Stream's hosted router; pass `url` for another. A router
running locally with nothing in front of it is reached by customer id instead:
`VisionAgents(url = "http://localhost:8080", customerId = "acme")`. Pass `apiKey` beside it and
the key is Stream's alone, so chat and video connect on a local router too.

An app targeting SDK 37 reaches a router on the local network (the emulator's `10.0.2.2`, or a
Mac on the same Wi-Fi) only with `android.permission.ACCESS_LOCAL_NETWORK`, declared in the
manifest and asked for at runtime. Without it the connection times out rather than failing.

Stream is set up once. The key, and the user and token `setUser` is given, are what the router,
Stream Chat and Stream Video all connect with. Nothing connects when the user is set: the first
session to ask for `chat()` or `join()` builds the client, and every session after it shares
that one, so the user is one connection and one presence to Stream however many conversations
are open. Leaving a call or closing a session disconnects nothing; `agents.disconnect()` does.

An app that already has a `ChatClient` or a `StreamVideo` hands it over, connected as the same
user, and it is used instead of a second one and never disconnected here:

```kotlin
agents.use(ChatClient.instance())
agents.use(StreamVideo.instance())
```

Both are one per process on Android. One the app built without handing it over is used if it is
connected as the user `setUser` named, and refused otherwise, rather than replaced.

The agent is named rather than configured, and nothing here mints a token. Both are the same
fact: writing configs and minting tokens are server-side only, so the app is told which agent it
talks to and is handed the user's token by its own backend.

`AgentSession` owns a coroutine scope and `close()` ends it, so the usual home is a `ViewModel`
that closes it in `onCleared`. Closing stops the session and keeps what it recorded and
remembered; `delete()` (or `agents.sessions.delete(id)`) deletes it, turns and memories
included. The channel from `chat(context)` is what Stream Chat's AI components show: the reply
streaming in, its reasoning and tool calls, and the approvals waiting on the person. On a call,
`voice.call` is the Stream `Call` that Stream Video's Compose views render.

### A tool that runs on the phone

The agent runs in the backend; a tool you give it runs here. That is the point: it can read
what only the device knows, and the agent only ever sees the answer.

```kotlin
val lookup = AgentTool(
    name = "lookup_order",
    description = "Look up one of the caller's orders by its order number.",
    parameters = AgentTool.strings(mapOf("order_id" to "the order number"), required = listOf("order_id")),
) { arguments -> orders.find(arguments["order_id"]?.jsonPrimitive?.content.orEmpty()) }

val chat = agents.agent("docs").chat(lookup)
```

Tools are answered inside `AgentSession` whether or not anybody collects `events()`, and a
`tool_cancel` cancels the coroutine running one. `displayTitle` is what the people in the
conversation see while it runs, and `executor = AgentTool.Executor.Client` shows the call as
waiting on the person's device.

### Finding a conversation again

Lists come a page at a time. Pass a page's `nextCursor` back as `cursor`, with the same filters,
for the next one:

```kotlin
val sessions = agents.agent("docs").sessions
val live = SessionQuery(state = Session.State.Live, limit = 20)
val first = sessions.query(live)
val next = first.nextCursor?.let { sessions.query(live.copy(cursor = it)) }
val found = sessions.search("billing")
```

`update` renames a conversation or relabels it, `chat.update(title = "Billing")` on an open one
and `sessions.update(id, title = "Billing")` on one that ended. A device may change `title`,
`description` and `custom` and nothing else; models and voice are your backend's.

### Looking something up

`search` is the one routed modality a device may reach, because a question and its answer are
one round trip and the answer is for whoever asked:

```kotlin
val found = agents.router(config = "healthcare").search("perioperative antibiotic guidance")
```

The router comes from the client, never from a constructor of its own, and which model answers
is set in the `healthcare` router config rather than in the call.

### Going back, and branching off

`rewind` takes back every turn after the one kept, so the next message carries on from there.
`fork` branches a new session off one instead, leaving the original as it was. A turn's id is
the `id` of an `AgentResponse`, not the `turnId` a socket event carries:

```kotlin
val turns = chat.responses.list().items
chat.responses.rewind(turns[0])
val branch = agents.attach(chat.fork(ForkOptions(responseId = turns[0].id)).id)
```

A text session is kept in Stream Chat unless it is `incognito`, and a conversation kept there
cannot be rewound, since its transcript would bring the turns back: the router answers 400, so
fork it at the response instead. An open `AgentSession` keeps the transcript it already showed, so reload it from
`responses` after a rewind.

### Guests, and who this device is

```kotlin
val guest = agents.guestUser(GuestOptions(name = "Ada"), FileGuestStore(File(filesDir, "guest.json")))
agents.setUser(guest)
```

The stored guest is reused on the next launch rather than minting a second one with an empty
history. With no user set the guest is asked for anonymously; a router that admits no anonymous
callers, Stream's hosted one for now, throws `AgentsException.Configuration`, and your backend
mints the guest instead. Against a router with an api key, `setUser(user, TokenProvider)` is asked for a token
when one is needed and again after a 401, never on every request.

## What is deliberately not here

The router is server-side only by default: a handful of operations are marked
`x-client-accessible` in the spec and everything else answers a device 403. This SDK uses thirteen
of them plus the event socket, and `generate.py` fails if the list names anything the spec does
not open.

| Not here | Ask your backend for |
| --- | --- |
| Writing a config, syncing a folder, dispatching agents | The agent's name, which is all the app needs |
| A token for the user | The token `setUser` is given, which the router, Chat and Video all take |
| Moving a guest's history onto the account they signed up with | Nothing: the backend that authenticated them does it |
| Changing a session's models or voice | Your backend, which has the Go or Node SDK's `update` |
| Interrupting over HTTP | Nothing: `interrupt()` goes over the socket |

Every request and socket handshake sends `Stream-Auth-Type: jwt`, which is what declares this
caller a device. Credentials go in handshake headers, never in the socket's query string.

## Regenerating the models

The spec at [`acceleration/api/openapi.yaml`](../../acceleration/api/openapi.yaml) is the source
of truth. The generated models are committed and `internal`; every public type is hand-written
in `core`, so a field the spec adds is invisible until it is wrapped.

```bash
uv run sdks/kotlin/generate.py --check   # does the list still agree with the spec?
uv run sdks/kotlin/generate.py           # regenerate, with openapi-generator in Docker
```

## Building and testing

Everything runs in Docker; nothing is installed on the host. The Gradle cache is a host
directory rather than a volume, so a second run starts warm.

```bash
cd <repo>

# unit tests against a real loopback router, no mocks, and the release build
docker run --rm --platform linux/amd64 -u root -v "$PWD":/repo \
  -v ~/.cache/vision-agents-kotlin/gradle-android:/gradle-home -e GRADLE_USER_HOME=/gradle-home \
  -w /repo/sdks/kotlin cimg/android:2026.08.1 \
  ./gradlew :core:testDebugUnitTest :core:assembleRelease

# the same, plus the live tests against a running router
docker run --rm --platform linux/amd64 -u root -v "$PWD":/repo \
  -v ~/.cache/vision-agents-kotlin/gradle-android:/gradle-home -e GRADLE_USER_HOME=/gradle-home \
  -e VISION_AGENTS_URL=http://host.docker.internal:8091 -e VISION_AGENTS_CUSTOMER_ID=examples \
  -w /repo/sdks/kotlin cimg/android:2026.08.1 ./gradlew :core:testDebugUnitTest

# the same, plus a device token's 403s against a router in api_key mode
docker run --rm --platform linux/amd64 -u root -v "$PWD":/repo \
  -v ~/.cache/vision-agents-kotlin/gradle-android:/gradle-home -e GRADLE_USER_HOME=/gradle-home \
  -e INTEGRATION_ROUTER_URL=http://host.docker.internal:8099 \
  -e INTEGRATION_ROUTER_KEY=vak_live_… -e INTEGRATION_ROUTER_SECRET=vas_live_… \
  -w /repo/sdks/kotlin cimg/android:2026.08.1 ./gradlew :core:testDebugUnitTest
```

`StreamClientsTest` is what keeps a second Stream connection from being opened: sessions asking
at once share one client, the app's own is used and left connected, and `disconnect` closes only
what was built. `PostureTest` drives every operation the SDK has against the loopback router and
fails if any is one the spec does not mark `x-client-accessible`. `KeyedRouterPostureTest` runs
only with `INTEGRATION_ROUTER_KEY` set; the router it needs is set up as in the
[JavaScript README](../js/README.md#working-on-this-package).

The unit tests are plain JVM tests, run by the Android Gradle plugin with no emulator.
`cimg/android` publishes no arm64 image, hence `--platform linux/amd64`.
