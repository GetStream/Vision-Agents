# LiveKit Agents to accelerate

Sources checked 2026-10-16: LiveKit Agents docs (`docs.livekit.io/agents`,
`docs.livekit.io/reference/python/livekit/agents`) and the `livekit/agents` examples.

## Source artifacts to ask for

- The worker code: `Agent` subclasses, `AgentSession(...)`, the entrypoint, `AgentServer`.
- Every `@function_tool` method, MCP servers, and any `llm_node`/`stt_node`/`tts_node` overrides.
- Plugins used for STT/LLM/TTS/realtime, VAD, turn detection, noise cancellation.
- Room/telephony setup: SIP trunks, dispatch rules, phone numbers, room metadata, `userdata`.
- RAG: vector store, embedding model, query tool, uploaded documents.

## Core config mapping

| LiveKit | Accelerate |
| --- | --- |
| `Agent(instructions=...)` | `instructions.md` |
| `session.generate_reply(instructions="Greet...")` on start | `greeting: {text: "Greet...", mode: variation}`; a fixed greeting is `mode: exact` |
| Agent class per role / handoff target | One folder; each role becomes `skills/<role>.md` |
| `userdata`, room metadata | Session tags / what a tool returns; not prompt variables |
| `ChatContext` seeded history | No equivalent seed; put durable facts in `instructions.md` |

## Voice and model pipeline

| LiveKit | Accelerate |
| --- | --- |
| `stt=inference.STT("deepgram/nova-3")` or a plugin STT | The served Deepgram target (`stt: deepgram/flux-general-en`) or `stt-fast`; `nova-3` is not served |
| `llm=inference.LLM("openai/gpt-4.1-mini")` or a plugin LLM | `llm: llm-fast` (gpt-4.1-mini is not a served target) |
| `tts=inference.TTS("cartesia/sonic-3", voice=...)` | `tts: cartesia/sonic-preview` plus `voice: <id>` when that voice exists there, else `tts-fast` |
| `openai.realtime.RealtimeModel(...)` / Gemini Live plugin | `sts: openai/gpt-realtime-2` (a served candidate under `sts-vision`), no STT/TTS keys |
| `vad=silero.VAD.load()`, turn detection, noise cancellation | No keys. The backend owns turn detection and noise handling; report, do not approximate |
| `preemptive_generation`, `max_tool_steps` | No equivalent keys. Leave out and report |

Inference strings name the vendor model, not a router target: check each against
`ListRoutes` (see SKILL.md). `deepgram/nova-3` is not served; the served Deepgram
spelling is `deepgram/flux-general-en`. Plugin class names never carry across. A custom plugin (your own STT/TTS) has no equivalent unless the router serves that
provider; report it instead of substituting a different vendor silently.

## Tool calling

- `@function_tool` methods become `@agent.llm.register_function(description=...)` methods
  in the Python entry point. Keep the name, typed parameters, and return text; drop the
  `RunContext` parameter.
- Tools on the `AgentSession` shared by all agents work the same way: register once.
- A tool returning a new `Agent` (handoff) becomes a skill, not a function. The function
  disappears; `instructions.md` tells the agent to hand that kind of work to the skill.
- `mcp_servers=[mcp.MCPServerHTTP(url)]` becomes an `mcp_servers:` entry
  `{name, url, tools}` in `agent.yaml`, or a catalog `connectors:` entry when the server
  is a known product.
- Raw JSON-schema tools (`raw_schema`) become typed Python functions with the schema
  expressed as type hints and docstring.

## Knowledge base

LiveKit has no built-in KB; agents use a query tool over a vector store. Move the
documents (not the embeddings) into `knowledge/`, and delete the query tool: lookup is
built in. If the content lives at URLs, use `knowledge/urls.yaml`. The vector store,
embedding model, chunking, and `top_k` have no equivalent; report them.

## Integrations and telephony

- SIP inbound/outbound and phone numbers move to the router's phone setup
  (`acceleration/cmd/phone`); the number is not in `agent.yaml`.
- LiveKit room join/leave, participant events, and data packets have no equivalent in the
  folder. Code listening to them stays in the Python entry point if it is ported at all;
  flag it in the report.
- External webhooks the agent code calls become Python functions making the same call,
  or connectors when the system is catalogued.

## Memory and state

`session.userdata` and per-room state do not persist across accelerate sessions the same
way. User-keyed memory exists on the backend; anything the agent must remember should be
fetched by a tool at conversation start or stated in `instructions.md`. Report state that
was load-bearing.

## No equivalent: do not invent

Custom inference plugins, `llm_node`/`tts_node` pipeline overrides, false-interruption and
resume settings, room I/O options, agent dispatch rules, and LiveKit Cloud deployment
settings. List each in the report.

## Mini example

Source:

```python
class NorthwindAgent(Agent):
    def __init__(self):
        super().__init__(instructions="You are Northwind support...", tools=[])

    @function_tool
    async def lookup_order_status(self, context: RunContext, order_id: str) -> str:
        """Look up the status of an order."""
        ...

session = AgentSession(
    stt=inference.STT("deepgram/nova-3"),
    llm=inference.LLM("openai/gpt-4.1-mini"),
    tts=inference.TTS("cartesia/sonic-3", voice="71a7..."),
)
```

Migrated `agent.yaml`, `instructions.md` (`You are Northwind support...`), and entry point:

```yaml
name: northwind_support
description: Northwind support.
stt: deepgram/flux-general-en
llm: llm-fast
tts: cartesia/sonic-preview
voice: 71a7ad14-091c-4e8e-a314-022ece01c121
greeting:
  text: Hello, this is Northwind support. How can I help?
  mode: variation
```

```python
agent = Agent(config="northwind_support", llm=stream.Accelerated(config="northwind_support"))

@agent.llm.register_function(description="Look up the status of an order")
async def lookup_order_status(order_id: str) -> str:
    ...
```
