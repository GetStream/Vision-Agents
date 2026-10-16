# Pipecat to accelerate

Sources checked 2026-10-16: Pipecat docs (`docs.pipecat.ai/pipecat/learn`) and
`pipecat-ai/pipecat` examples (Pipecat 1.12 import paths verified).

## Source artifacts to ask for

- The bot file: `Pipeline([...])` order, every service constructor, transport, `PipelineWorker`.
- `LLMContext` setup: system instruction, `tools=[...]`, context aggregators, VAD analyzer.
- Every function in the tools list and any `register_function` / direct-function handlers.
- RTVI, observers, metrics, custom `FrameProcessor` subclasses, `ParallelPipeline` branches.
- Transport config: Daily, LiveKit, WebSocket, telephony (Twilio/Telnyx serializers).

Pipecat is code, not config: the migration reads behaviour out of the pipeline and
re-expresses it. Nothing of the frame plumbing carries across.

## Core config mapping

| Pipecat | Accelerate |
| --- | --- |
| `OpenAILLMService.Settings(system_instruction=...)` (or a system message in `LLMContext`) | `instructions.md` |
| `LLMContext(messages=[...])` seeded turns | Durable facts into `instructions.md`; the rest is dropped and reported |
| First-message / greeting logic in `on_client_connected` | `greeting: {text: ..., mode: exact}` in `agent.yaml` |
| `PipelineParams(enable_metrics...)`, idle timeouts | No equivalent keys; report |

## Voice and model pipeline

Read the pipeline list in order; the services in it are the whole mapping:

| Pipecat | Accelerate |
| --- | --- |
| `DeepgramSTTService(model=...)`, `AssemblyAI...`, `Whisper...` | `stt: <provider>/<model>` in `agent.yaml` |
| `OpenAILLMService(model=...)`, Anthropic/Gemini LLM services | `llm: <provider>/<model>` |
| `CartesiaTTSService(Settings(voice=...))`, ElevenLabs, etc. | `tts: <provider>/<model>` plus `voice: <id>` |
| A realtime service (OpenAI Realtime, Gemini Live) in place of STT+LLM+TTS | `sts: <provider>/<model>`; drop STT/TTS keys |
| `SileroVADAnalyzer`, turn strategies, interruption config | No keys; the backend owns this. Report custom strategies |

A provider Pipecat supports but the router does not serve has no equivalent: leave the
field out (backend default) and report the substitution, never guess a model spelling.

## Tool calling

- Functions listed in `LLMContext(tools=[fn])` become
  `@agent.llm.register_function(description=...)` functions. Pipecat derives the schema
  from the signature and docstring; keep both identical so the model sees the same tool.
- `params.result_callback(...)` becomes the function's return value.
- Tool-call filler speech (`TTSSpeakFrame` on `on_function_calls_started`) becomes an
  instruction ("say you are checking before a slow lookup") or nothing; report it.
- `DirectFunction` / `FunctionSchema` definitions become typed Python functions.

## Knowledge base

Pipecat has no built-in KB; RAG is a processor, a context source, or a tool. Export the
underlying documents into `knowledge/` (or URLs into `knowledge/urls.yaml`) and remove
the retrieval machinery. A retrieval *tool* disappears: lookup is built in, and
`instructions.md` should say to look things up before answering.

## Integrations and telephony

- The transport is replaced: Stream call/chat is the transport. Daily/LiveKit room
  settings, WebRTC details, and phone serializers do not carry across.
- Telephony connections (Twilio/Telnyx via Pipecat transports) move to the router's
  phone setup (`acceleration/cmd/phone`).
- HTTP calls inside functions stay as HTTP calls inside the migrated Python functions,
  or become connectors/`mcp_servers:` when the system is catalogued.

## Memory and state

Context aggregators, conversation state, and processor-held state have no counterpart.
User-keyed backend memory plus tool lookups replace them. Anything a custom processor
remembered mid-call must be re-expressed as a tool or reported as lost.

## No equivalent: do not invent

Custom `FrameProcessor`s, `ParallelPipeline` branches, observers/metrics plumbing, RTVI
client messages, frame-level interruption handling, and pipeline ordering effects. Each
custom processor is either a tool, a skill, or a report line; never a new `agent.yaml` key.

## Mini example

Source:

```python
stt = DeepgramSTTService(api_key=...)
llm = OpenAILLMService(settings=OpenAILLMService.Settings(system_instruction="You are Northwind support..."))
tts = CartesiaTTSService(settings=CartesiaTTSService.Settings(voice="71a7..."))
context = LLMContext(tools=[lookup_order_status])
pipeline = Pipeline([transport.input(), stt, user_aggregator, llm, tts, transport.output(), assistant_aggregator])
```

Migrated: the system instruction becomes `instructions.md`, and:

```yaml
name: northwind_support
description: Northwind support.
stt: deepgram/nova-3
llm: openai/gpt-4.1-mini
tts: cartesia/sonic-3
voice: 71a7ad14-091c-4e8e-a314-022ece01c121
```

```python
@agent.llm.register_function(description="Look up the status of an order")
async def lookup_order_status(order_id: str) -> str:
    ...
```

The transport, aggregators, VAD, and worker are deleted, not translated.
