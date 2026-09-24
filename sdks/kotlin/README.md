# Kotlin SDKs

Three Gradle modules for talking to an agent from an Android app.

| Module | Artifact | Depends on | What it is |
| --- | --- | --- | --- |
| `core/` | `io.getstream:vision-agents-core` | coroutines, kotlinx.serialization, Ktor on OkHttp | Sessions, responses, the event socket and the conversation as `StateFlow`. Plain JVM |
| `ui/` | `io.getstream:vision-agents-ui` | `core`, Compose | Compose views over that state |
| `rtc/` | `io.getstream:vision-agents-rtc` | `core`, Stream Video 1.34.0, Stream Chat 7.12.0 | Joining the call, so the conversation can be spoken |

`core` has nothing Android in it, so an app that only holds a text conversation does not pull
in WebRTC, and its tests run on a plain JDK. `ui` and `rtc` need minSdk 24 and compileSdk 37,
which is what Stream's SDKs require.

## Using them

```kotlin
val agents = VisionAgents(url = "https://your-router", customerId = "acme")

// In writing. No call is joined, nothing is transcribed or spoken.
val chat = agents.agent("docs").chat()
chat.send("What are your opening hours?")
// chat.conversation is a StateFlow: the turns grow as the reply streams in, and its state says
// what the agent is doing.

// Out loud. The agent joins a call and so does this device.
val voice = VoiceSession.start(context, agents, options = SessionOptions(agent = "docs"))
voice.join(credentials = yourBackend::callCredentials)
```

The agent is named rather than configured, and `join` is handed a function rather than minting
its own token. Both are the same fact: writing configs and minting a call token are server-side
only, so the app is told which agent it talks to and is handed the token to join with.

`AgentSession` owns a coroutine scope and `close()` ends it, so the usual home is a `ViewModel`
that closes it in `onCleared`. With `vision-agents-ui` a whole conversation is one composable:

```kotlin
ConversationView(session = chat)
```

`TranscriptView`, `Composer` and `AgentStatusView` are public and work on their own, so a host
that wants a different arrangement takes them apart rather than fighting `ConversationView`.
`VoiceCallView` and `AgentVideoView` in `rtc` are the call's controls and its video.

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
`tool_cancel` cancels the coroutine running one.

### Looking something up

`search` is the one routed modality a device may reach, because a question and its answer are
one round trip and the answer is for whoever asked:

```kotlin
val found = agents.router(config = "healthcare").search("perioperative antibiotic guidance")
```

### Going back, and branching off

`rewind` takes back every turn after the one kept, so the next message carries on from there.
`fork` branches a new session off one instead, leaving the original as it was. A turn's id is
the `id` of an `AgentResponse`, not the `turnId` a socket event carries:

```kotlin
val turns = chat.responses.list()
chat.responses.rewind(turns[0])
val branch = agents.attach(chat.fork(ForkOptions(responseId = turns[0].id)).id)
```

A conversation kept in Stream Chat (`persistConversation`) cannot be rewound, since its
transcript would bring the turns back: the router answers 400, so fork it at the response
instead. An open `AgentSession` keeps the transcript it already showed, so reload it from
`responses` after a rewind.

### Guests, and who this device is

```kotlin
val guest = agents.guestUser(GuestOptions(name = "Ada"), FileGuestStore(File(filesDir, "guest.json")))
agents.setUser(guest)
```

The stored guest is reused on the next launch rather than minting a second one with an empty
history. Against a router with an api key, `setUser(user, TokenProvider)` is asked for a token
when one is needed and again after a 401, never on every request.

## What is deliberately not here

The router is server-side only by default: a handful of operations are marked
`x-client-accessible` in the spec and everything else answers a device 403. This SDK uses twelve
of them plus the event socket, and `generate.py` fails if the list names anything the spec does
not open.

| Not here | Ask your backend for |
| --- | --- |
| Writing a config, syncing a folder, dispatching agents | The agent's name, which is all the app needs |
| A token to join the agent's call | `CallCredentials`, which `join` is handed |
| Moving a guest's history onto the account they signed up with | Nothing: the backend that authenticated them does it |
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

# core: unit tests against a real loopback router, no mocks
docker run --rm -v "$PWD":/repo -v ~/.cache/vision-agents-kotlin/gradle:/gradle-home \
  -e GRADLE_USER_HOME=/gradle-home -w /repo/sdks/kotlin gradle:jdk21 \
  ./gradlew -PcoreOnly :core:test

# the same, plus the live tests against a running router
docker run --rm -v "$PWD":/repo -v ~/.cache/vision-agents-kotlin/gradle:/gradle-home \
  -e GRADLE_USER_HOME=/gradle-home -e VISION_AGENTS_URL=http://host.docker.internal:8091 \
  -e VISION_AGENTS_CUSTOMER_ID=examples -w /repo/sdks/kotlin gradle:jdk21 \
  ./gradlew -PcoreOnly :core:test

# ui and rtc: built rather than tested, since they are views and a WebRTC wrapper
docker run --rm --platform linux/amd64 -u root -v "$PWD":/repo \
  -v ~/.cache/vision-agents-kotlin/gradle-android:/gradle-home -e GRADLE_USER_HOME=/gradle-home \
  -w /repo/sdks/kotlin cimg/android:2026.08.1 \
  ./gradlew :ui:assembleRelease :rtc:assembleRelease
```

`-PcoreOnly` leaves the Android modules out of the build, so the JDK image needs no Android
SDK. `cimg/android` publishes no arm64 image, hence `--platform linux/amd64`.
