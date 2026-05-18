# v0.6.0

## Breaking Changes

### Agent inference pipeline rewrite (#501)

Large internal refactor of the agent's audio/LLM/TTS pipeline (~10k LOC across ~195 files). The orchestration that used to live inside `Agent` is now an explicit
`InferenceFlow` abstraction with two implementations — `TranscribingInferenceFlow` (STT → LLM → TTS) and `RealtimeInferenceFlow` (speech-to-speech).

Data flow between components is driven by a new generic `Stream[T]` primitive (with backpressure, `clear()` and `close()`) instead of plugin events:
`LLM.simple_response` now returns an `AsyncIterator[LLMResponseDelta | LLMResponseFinal]`, and `STT`, `TTS`, `TurnDetector`, and `Realtime` expose
`.output` streams that consumers iterate. The `MetricsCollector` was rewritten to a direct
`on_*()` API with parent/child merging so plugin metrics fan in to the agent root and OpenTelemetry is written exactly once. Turn lifecycle is now managed by a dedicated
`LLMTurn` class with explicit `confirm` / `cancel` / `finalize` steps.

#### Migration

`agent.llm.simple_response(...)` → `agent.simple_response(...)`:

```python
# Before
await agent.llm.simple_response("greet the user")

# After
await agent.simple_response("greet the user")
```

`LLM.simple_response` is now an async iterator, not event-driven:

```python
# Before — listen for chunks/completed events
@agent.events.subscribe
async def on_chunk(event: LLMResponseChunkEvent): ...


@agent.events.subscribe
async def on_done(event: LLMResponseCompletedEvent): ...


# After — iterate the response stream
async for item in llm.simple_response("hello"):
    if isinstance(item, LLMResponseDelta):
        print(item.delta, end="")
    elif isinstance(item, LLMResponseFinal):
        print("\n[done]", item.text)
```

`streaming_tts` is gone — TTS always streams sentence-by-sentence via the new `TTSSentenceTokenizer`:

```python
# Before
agent = Agent(..., streaming_tts=True)

# After — remove the flag
agent = Agent(...)
```

Turn-detection events on the bus are replaced by `TurnStarted` / `TurnEnded` items on the STT / `TurnDetector` output stream (`TurnEvent` enum removed).

#### Other breakages

- `Agent.simple_audio_response()` removed; speech-to-speech is now handled by `RealtimeInferenceFlow`.
- `TTSErrorEvent` and `TTSConnectionEvent` removed; TTS errors are surfaced through metrics (`MetricsCollector.on_tts_error`).
- `MetricsCollector` no longer subscribes to plugin events — plugins call `metrics.on_*()` directly. Custom metric consumers that listened for
  `LLMResponseCompletedEvent` etc. must read from `agent.metrics` or merge into the collector tree.
- Plugin author API: `TTS.send()` removed, replaced by `TTS.send_iter(stream)` which consumes a `Stream[TTSInput | TTSInputEnd]` and produces a `Stream[TTSOutputChunk]`. `set_output_format()` is also gone — output format is fixed by the inference flow.
- Plugin-specific event types removed from `aws`, `gemini`, `huggingface`, `nvidia`, `openai`, and `xai`. Data that used to ride on those events now flows through the standard `LLMResponseDelta` / `LLMResponseFinal` stream items.

### Avatar plugin API (#534)

Avatars are now a first-class plugin type with their own kwarg on `Agent(...)` rather than being passed through the generic
`processors=[...]` list. The existing publisher classes have been removed in favour of an `Avatar` symbol exported from each plugin.

- `anam.AnamAvatarPublisher` removed. Use `anam.Avatar`.
- `lemonslice.LemonSliceAvatarPublisher` removed. Use `lemonslice.Avatar`.
- `liveavatar.Avatar` is new in this release and uses the new API (see New Features below).
- `heygen.AvatarPublisher` is unchanged — the `heygen` plugin keeps the legacy `processors=[...]` wiring for this release and is scheduled for removal (#553). New code should use `liveavatar.Avatar` instead.
- `vision_agents.core.utils.av_synchronizer` moved to `vision_agents.core.avatars.av_synchronizer` (also re-exported from `vision_agents.core.avatars`).

Migration:

```python
# Before
from vision_agents.plugins.anam import AnamAvatarPublisher

agent = Agent(..., processors=[AnamAvatarPublisher(...)])

# After
from vision_agents.plugins import anam

agent = Agent(..., avatar=anam.Avatar(...))
```

### EventManager (#555)

`core.events.EventManager` was simplified: events now dispatch immediately on `send()` (one
`asyncio.Task` per matching handler) instead of going through an internal deque and a background
`_process_events_loop`. This eliminates the "task was destroyed but is pending" warnings that appeared when an `EventManager` was garbage-collected.

- `EventManager.stop()` removed. Use `await EventManager.shutdown()` for teardown.
- `send()` calls after `shutdown()` are silently dropped.
- Tests that relied on observing the internal queue should switch to `await events.wait()`.

### ElevenLabs STT class rename (#559)

The internal class `STT` was renamed to `ElevenlabsSTT` to make it searchable. The public import path is unchanged —
`from vision_agents.plugins.elevenlabs import STT` still works via re-export.

## New Features

### Public API on `Agent` (#501)

| Member                                                                | Description                                                                                                                                   |
|-----------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------|
| `await agent.simple_response(text, participant=None, interrupt=True)` | Inject an instruction for the LLM to respond to. Replaces `agent.llm.simple_response(...)`. Routed through the active `InferenceFlow`.        |
| `await agent.say(text, interrupt=False)`                              | Speak text directly through TTS, bypassing the LLM. New `interrupt` flag preempts an in-flight turn and clears the TTS pipeline.              |
| `agent.resolve_inference_flow() -> InferenceFlow`                     | Extension point: picks `RealtimeInferenceFlow` for `Realtime` LLMs, otherwise `TranscribingInferenceFlow`. Override to plug in a custom flow. |

### Public utilities under `vision_agents.core` (#501)

- `core.utils.stream.Stream[T]` — generic async producer/consumer queue with `send` / `get` / `clear` / `close` / `collect(timeout)` / `peek()`.
- `core.utils.tokenizer.TTSSentenceTokenizer` — sentence chunker used to feed TTS incrementally.
- `core.agents.inference` — `InferenceFlow`, `TranscribingInferenceFlow`, `RealtimeInferenceFlow`, `AudioInputStream`, `AudioOutputStream`, `AudioInputChunk`, `AudioOutputChunk`,
  `AudioOutputFlush`.
- `core.observability.MetricsCollector.merge(child)` — hierarchical metrics merging so child collectors forward to a parent without duplicating OTel writes.

### LLM interruption API (#501)

`await llm.interrupt()` is now part of the public `LLM` contract. The active inference flow calls it to preempt an in-flight response (e.g. on a barge-in); plugin authors must implement it.

### Dedicated Avatar plugin type (#534)

`Agent(...)` accepts a new
`avatar=` keyword argument. The avatar owns the agent's outbound video/audio tracks, and the inference flow's audio output is routed through the avatar provider for lipsync. Plugin authors can build new integrations by subclassing
`vision_agents.core.avatars.Avatar`.

```python
from vision_agents.plugins import anam

agent = Agent(..., avatar=anam.Avatar(...))
```

### LiveAvatar plugin (#534)

New first-party avatar plugin:
`vision_agents.plugins.liveavatar.Avatar` — LiveAvatar (by HeyGen) integration over WebRTC + websocket. Recommended replacement for the now-deprecated `heygen` plugin.

Install with: `uv add "vision-agents[liveavatar]"`

### HeyGen plugin scheduled for removal (#553)

The `heygen` plugin will be removed in a future release in favour of `liveavatar`.

### Tencent TRTC plugin (#386)

New
`vision_agents.plugins.tencent` edge transport for Tencent TRTC (WebRTC). Wraps Tencent's manylinux-only
`liteav` SDK and is gated by `sys_platform == 'linux'` in both `vision-agents[tencent]` and `vision-agents[all-plugins]` — non-Linux installs silently skip it.

Install with: `uv add "vision-agents[tencent]"`

Required env vars: `TENCENT_SDK_APP_ID`, `TENCENT_SDK_SECRET_KEY`. Optional: `TENCENT_TRTC_SCENE`, `TENCENT_LITEAV_DEBUG`.

## Bug Fixes

- **Agent**: fix audio output queue falling behind when the main thread blocks longer than 20 ms.
  `_poll_audio_queues` now drains all available audio per cycle instead of consuming at a fixed 20 ms cadence. (#558)
- **ElevenLabs STT**: fix duplicated `TurnEnded` event. (#559)
- **ElevenLabs STT**: fix websocket keep-alive — the connection used to close after a few seconds of silence; now sends 2s of silence every 5s to keep it open. (#559)

# v0.4.0

## Breaking Changes

### Built-in HTTP server

#### API Endpoints

All session endpoints now include `call_id` as a path parameter:

| Before                               | After                                                |
|--------------------------------------|------------------------------------------------------|
| `POST /sessions`                     | `POST /calls/{call_id}/sessions`                     |
| `DELETE /sessions/{session_id}`      | `DELETE /calls/{call_id}/sessions/{session_id}`      |
| `POST /sessions/{session_id}/close`  | `POST /calls/{call_id}/sessions/{session_id}/close`  |
| `GET /sessions/{session_id}`         | `GET /calls/{call_id}/sessions/{session_id}`         |
| `GET /sessions/{session_id}/metrics` | `GET /calls/{call_id}/sessions/{session_id}/metrics` |

#### Request Body

- `call_id` removed from `POST /sessions` request body — now a URL path parameter

#### Response Codes

- `DELETE /calls/{call_id}/sessions/{session_id}` now returns **202 Accepted** (was 204)
- `POST /calls/{call_id}/sessions/{session_id}/close` now returns **202 Accepted** (was 200)
- Session closure is now asynchronous — the owning node processes the close request on its next maintenance cycle

#### ServeOptions

- `get_current_user` option removed
- Permission callbacks (`can_start_session`, `can_close_session`, `can_view_session`, `can_view_metrics`) now receive `call_id: str` as a parameter

#### Removed Dependencies

- `vision_agents.core.runner.http.dependencies.get_session` removed
- `vision_agents.core.runner.http.dependencies.get_current_user` removed

### FunctionRegistry

- `register_function()` now rejects synchronous functions with `ValueError` — all registered functions must be async (#373)
- `FunctionRegistry.call_function()` and `LLM.call_function()` are now `async` (#373)

### Agent authentication

- `Agent.create_user()` renamed to `Agent.authenticate()` (#380)
- `EdgeTransport.create_user()` renamed to `EdgeTransport.authenticate()` (#380)
- Authentication is now called automatically during `Agent.start()` — manual calls are no longer needed (#380)

### Testing

- `mock_tools()` and standalone `mock_functions()` removed from `vision_agents.testing` (#376)
- Use `TestSession.mock_functions()` instead

### AgentLauncher

- `cleanup_interval` parameter renamed to `maintenance_interval` (#374)
- `created_by` parameter removed from `start_session()` (#374)
- `AgentSession.created_by` field removed (#374)
- `call_id` values must match `^[a-z0-9_-]+$` (raises `InvalidCallId`) (#374)
- New parameter: `registry: SessionRegistry | None = None` (#374)

## New Features

### Inworld Realtime plugin (WebRTC)

Adds
`inworld.Realtime` for low-latency speech-to-speech over Inworld's Realtime API (WebRTC transport). Protocol-compatible with OpenAI Realtime — supports function calling, turn detection, and multiple upstream models via the
`<provider>/<model>` ID format (e.g. `"openai/gpt-4o-mini"`, `"google-ai-studio/gemini-2.5-flash"`). (#502)

### Redis-backed Agent session registry for horizontal scaling

Sessions are shared across nodes via Redis, enabling cross-node session queries and closure without sticky sessions. (#374)

```python
from vision_agents.core import AgentLauncher, Runner
from vision_agents.core.agents.session_registry import RedisSessionKVStore, SessionRegistry

store = RedisSessionKVStore(url="redis://localhost:6379")
registry = SessionRegistry(store=store)
runner = Runner(AgentLauncher(create_agent=create_agent, join_call=join_call, registry=registry))
```

Install with: `uv add "vision-agents[redis]"`

**New public API on AgentLauncher:**

| Method                                       | Description                            |
|----------------------------------------------|----------------------------------------|
| `get_session_info(call_id, session_id)`      | Query session info from shared storage |
| `request_close_session(call_id, session_id)` | Request closure from any node          |

**Custom store backends:** `SessionKVStore` is an abstract class that can be subclassed to support any TTL key-value store (DynamoDB, Memcached, etcd, etc.).

### PEP 561 compliance

`py.typed` markers added to `vision_agents.core` and `vision_agents.testing` for downstream type checking support. (#378)

### Inworld TTS v2

`inworld-tts-2` added to the model `Literal` and used as the default for `inworld.TTS()`. (#531)

## Bug Fixes

- **EventManager**: fix crash when event handlers have return type annotations (#381)
- **RedisSessionKVStore**: fix import error when `redis` package is not installed (#384)
- **Agent metrics**: fix metrics storage and serialization in session registry (#387)
- **Inworld TTS**: fix garbled / failed playback for replies that span multiple stream chunks by forcing `LINEAR16` audio encoding (#531)
- **MCPServerRemote
  **: fix cancel-scope leak in which closing an MCP session left a half-cancelled anyio scope that pegged the event loop. The transport lifecycle now runs inside a dedicated supervisor task so
  `__aenter__` / `__aexit__` task-identity holds regardless of which caller drives `connect()` and `disconnect()`. (#529)
