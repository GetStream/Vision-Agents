# Unreleased

## New Features

### `vision-agents` CLI with `init` and `app` commands (#533)

Adds a console script with two subcommands. `uvx vision-agents init <name>` scaffolds a new agent project (`pyproject.toml`, `agent.py`, `vision-agents.toml`, `.env.example`, `.gitignore`, `README.md`) and runs `uv sync` to provision a venv (skip with `--no-install`). `vision-agents app run|serve` reads `vision-agents.toml` from the project root and forwards args to the project's `Runner.cli()`, so users have a stable entry point even if the agent file is renamed. Templates are rendered with Jinja2.

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

# v0.5.9

## Bug Fixes

- **Anam**: fix avatar publisher settings. (#556)

# v0.5.8

## Breaking Changes

- **OpenAI Realtime**: default model bumped to `gpt-realtime-2`. (#537)

## Bug Fixes

- **Inworld TTS**: stop poisoning the receive stream with stale-context errors after >60s of inactivity. (#539)

# v0.5.7

## New Features

- **AWS**: new `aws.STT` (Amazon Transcribe streaming) plugin. (#528)
- **Inworld TTS**: switch from AV/HTTP to bidirectional WebSocket streaming (replaces the `av` dependency with `websockets`). (#532)

# v0.5.6

## Bug Fixes

- **Inworld TTS**: force `LINEAR16` audio encoding — MP3 default broke replies that span multiple stream chunks; also makes `inworld-tts-2` the default model. (#531)
- **MCP**: own MCP transport lifecycle in a single supervisor task to fix event-loop peg on `_deliver_cancellation`. (#529)

# v0.5.5

## New Features

- **Inworld**: new `inworld.Realtime` (WebRTC) plugin — low-latency speech-to-speech with function calling, turn detection, and multi-provider upstream models via `<provider>/<model>` IDs. (#502)
- **xAI**: new `xai.TTS` (Grok TTS) with `Voice` and `VOICE_DESCRIPTIONS` exports. (#433)
- **LLM**: max tool-calling rounds is now configurable across LLM plugins. (#500)

## Bug Fixes

- **xAI Realtime**: default model updated to `grok-voice-think-fast-1.0`; model now passed as `?model=` query param; voice IDs lowercased to match docs. (#515)
- **Gemini**: route MCP tool schemas through `parameters_json_schema` so `$schema` and `additionalProperties` round-trip correctly. (#513)

# v0.5.4

## Bug Fixes

- **Gemini Realtime**: fix infinite error loop on network disconnection. (#490)

# v0.5.3

## New Features

- **Sarvam**: new `sarvam.LLM`, `sarvam.STT`, `sarvam.TTS` plugin. (#488)

## Bug Fixes

- **Sarvam STT**: fix transcript ordering; support new WebSocket URL structures for translation/regular endpoints; speaker compatibility checks in TTS. (#489)

# v0.5.2

## New Features

- **Decart**: update to SDK 0.0.29 with default model Lucy 2. `RestylingProcessor` now accepts an initial reference image and supports atomic `update_state` of prompt + image for virtual try-on. (#446)

## Bug Fixes

- **Agent**: fix "dict changed size during iteration" in `_poll_audio_queues`. (#487)
- **AWS Realtime**: fix reconnect. (#486)
- **StreamEdge**: fix track identification when multiple anonymous candidates exist. (#446)

# v0.5.1

## New Features

- **OpenAI LLM**: `model` parameter is now optional and defaults to `gpt-5.4`. (#470)
- **Core**: new `AVSynchronizer` to sync audio/video playback for avatars (used by `anam` and `lemonslice`). (#466)
- **HuggingFace**: refactor tool calling — enforce max function-calling rounds, fix silently skipped tool calls with missing IDs, respect non-streaming mode. (#455)

## Bug Fixes

- **Agent**: fix `Task exception was never retrieved` in `_poll_audio_queues` on close. (#473)
- **Agent**: prevent OOM from unconsumed video frames in voice-only agents. (#458)
- **Gemini / OpenAI Realtime**: fix turn detection, support interruptions, add `interrupted` field to `RealtimeAudioOutputDoneEvent`. (#470)

# v0.5.0

## New Features

- **Anam**: new `anam.Avatar` plugin. (#445)

## Bug Fixes

- **Resources**: close `AsyncStream`, ElevenLabs TTS httpx client, and Deepgram STT httpx client on session end to fix per-session TCP connection leaks. (#457)

# v0.4.7

## Bug Fixes

- **Deepgram**: pin `deepgram-sdk` to `<6.1.0`. (#456)

# v0.4.6

## Bug Fixes

- **Deepgram STT**: reduce latency. (#451)

# v0.4.5

## New Features

- **AWS**: `aws_profile` parameter on `BedrockLLM` and `aws.Realtime` for profile / SSO / instance-profile auth. (#415)
- **CLI**: splash screen with current core version; hidden in non-interactive terminals or via `--no-splash`. (#447, #449)

## Bug Fixes

- **Agent**: fix memory leak from stringifying numpy arrays in event logging. (#444)
- **Gemini Realtime**: non-blocking tool execution. (#437)
- **AWS Realtime**: track background tool tasks. (#438)
- **OpenAI / xAI Realtime**: non-blocking tool execution. (#439)
- **OpenRouter**: fix function calling. (#442)

# v0.4.4

## New Features

- **Local**: new `local.LocalEdge`, `local.LocalCall`, and device classes for running agents against local audio/video devices. (#347)

## Bug Fixes

- **AWS Realtime**: emit transcription events and handle barge-in for Nova Sonic. (#408)
- **EventManager**: fix `shutdown()` hanging forever. (#421)
- **Gemini Realtime**: refactor event handling — fix skipping parts in multipart replies, add VAD config to reduce latency. (#436)
- **ElevenLabs STT**: switch to VAD mode instead of manual commits. (#435)
- **TTS**: use epoch tracking to skip stale TTS events on turn change. (#430)
- **Inworld TTS**: add `X-User-Agent` and `X-Request-Id` headers. (#428)

# v0.4.3

## New Features

- **Fish Audio**: update to S2-Pro model. (#405)

## Bug Fixes

- **StreamEdge**: fix connection race condition. (#412)
- **Agent**: fix memory leak by cleaning up handler tasks and closures on close. (#407)

# v0.4.2

## New Features

- **AssemblyAI**: add diarisation support for `assemblyai.STT`. (#394)
- **HuggingFace**: new Transformers-based detection processor; LLM/VLM refactor. (#377)

## Bug Fixes

- **Agent**: buffer realtime transcripts into single chat messages. (#383)
- **Agent**: fix eager turn detection updating transcripts out of order. (#401)
- **LLMJudge**: set instructions once at init. (#399)

# v0.4.1

## New Features

- **AssemblyAI**: add support for streaming STT. (#389)

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

## Bug Fixes

- **EventManager**: fix crash when event handlers have return type annotations (#381)
- **RedisSessionKVStore**: fix import error when `redis` package is not installed (#384)
- **Agent metrics**: fix metrics storage and serialization in session registry (#387)
