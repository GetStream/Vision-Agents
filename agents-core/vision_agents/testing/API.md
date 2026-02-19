# Vision-Agents Testing API

Text-only testing for Vision-Agents. Test agent responses, tool calls, and intent — no audio, video, or edge connection required.

## What You Can Test

- **Agent responses** — verify the agent replied (`agent_responds()`), check content via LLM judge (`agent_responds(intent="...")`)
- **Grounding** — verify the agent doesn't hallucinate (judge with intent like "Does NOT claim to know...")
- **Tool calls** — verify the agent called the right function (`agent_calls("get_weather")`), with the right arguments (`arguments={"location": "Tokyo"}`), partial argument matching supported
- **Tool outputs** — verify tool results (`agent_calls_output(output={...})`), error handling (`is_error=True`)
- **Event order** — strict sequential checks (`agent_calls` → `agent_responds` → `no_more_events()`), auto-skips `FunctionCallOutputEvent` after `agent_calls`
- **Tool mocking** — swap tool implementations while keeping schemas intact (`mock_tools`), simulate errors
- **Multi-turn conversations** — multiple `user_says()` calls share conversation history, test context retention across turns

### Architecture: Why We Test the LLM, Not the Agent

The testing framework operates at the **LLM level**, not the full Agent level. This is by design.

`Agent` is an orchestrator that combines Edge (transport), STT, LLM, TTS, turn detection, and processors into a complete audio/video pipeline. Its constructor validates that at least one processing capability (audio or video) is present, making it impossible to instantiate without real infrastructure dependencies.

However, all agent **behavior** — what it says, which tools it calls, how it follows instructions — happens inside the LLM. By testing the LLM directly, we cover:

- Response content and intent
- Function/tool calling with correct arguments
- Grounding (not hallucinating)
- Multi-turn context retention
- Tool error handling

What remains at the Agent level is **infrastructure**: audio-to-text (STT), text-to-audio (TTS), turn detection, media transport (Edge), and processor pipelines. These are candidates for future testing capabilities.

### Not Yet Supported

**Requires full Agent testing (infrastructure level):**
- Audio/video stream testing (STT, TTS, Edge transport)
- Latency/performance benchmarks (full pipeline measurement)

**Can be added at LLM level (behavioral):**
- Parallel multi-agent testing
- MCP server testing
- Snapshot testing (comparison against saved reference outputs)

## Quick Start

```python
from vision_agents.plugins import gemini
from vision_agents.testing import TestEval

async def test_greeting():
    llm = gemini.LLM("gemini-2.5-flash-lite")
    judge_llm = gemini.LLM("gemini-2.5-flash-lite")

    async with TestEval(llm=llm, judge=judge_llm, instructions="Be friendly") as session:
        await session.user_says("Hello")
        await session.agent_responds(intent="Friendly greeting")
        session.no_more_events()
```

## Running Tests

```bash
# Run integration tests
uv run py.test path/to/test_file.py -m integration

# With verbose event output
VISION_AGENTS_EVALS_VERBOSE=1 uv run py.test path/to/test_file.py -m integration

# With debugger support
uv run py.test path/to/test_file.py -m integration -s --timeout=0 --pdb --skip-bb
```

---

## TestEval

The main entry point. Wraps an LLM instance, sends text input, captures events, and provides scenario-style assertion methods.

### Constructor

```python
TestEval(llm: LLM, instructions: str = "You are a helpful assistant.", judge: LLM | None = None)
```

| Parameter | Type | Description |
|---|---|---|
| `llm` | `LLM` | The LLM instance with tools already registered. |
| `instructions` | `str` | System instructions for the agent. |
| `judge` | `LLM \| None` | Optional separate LLM for intent evaluation. Required if `agent_responds(intent=...)` is used. |

> **Important:** Use a separate LLM instance for `judge`. Using the agent's LLM would pollute its conversation history.

### Properties

| Property | Type | Description |
|---|---|---|
| `events` | `list[RunEvent]` | Current turn's event list. |
| `llm` | `LLM` | The LLM instance (useful for `mock_tools(session.llm, {...})`). |

### Usage as Context Manager

```python
async with TestEval(llm=llm, judge=judge_llm, instructions="...") as session:
    await session.user_says("Hello")
    await session.agent_responds(intent="Friendly greeting")
```

### Methods

#### `await session.user_says(text) -> RunResult`

Send user text to the LLM and capture the response events. Resets the assertion cursor for the new turn. Conversation history accumulates across successive calls.

```python
await session.user_says("What's the weather in Tokyo?")
```

#### `session.agent_calls(name=None, *, arguments=None) -> FunctionCallEvent`

Assert the next event is a `FunctionCallEvent`. Checks name and arguments (partial match — only specified keys are checked). **Auto-skips** the following `FunctionCallOutputEvent`.

```python
# Check name only
session.agent_calls("get_weather")

# Check name and specific arguments
session.agent_calls("get_weather", arguments={"location": "Tokyo"})

# Partial match — extra arguments are ignored
session.agent_calls("search", arguments={"query": "hello"})
# passes even if arguments also has "limit" and "offset"

# No name check — just advance to next function call
session.agent_calls()
```

Returns the matched `FunctionCallEvent` for further inspection:

```python
event = session.agent_calls("get_weather")
print(event.name)        # "get_weather"
print(event.arguments)   # {"location": "Tokyo"}
```

#### `session.agent_calls_output(*, output=..., is_error=None) -> FunctionCallOutputEvent`

Assert the next event is a `FunctionCallOutputEvent`. Use this when you need to inspect the tool output explicitly (normally auto-skipped by `agent_calls`).

```python
session.agent_calls_output(output={"temp": 70, "condition": "sunny"})
session.agent_calls_output(is_error=True)
```

#### `await session.agent_responds(*, intent=None) -> ChatMessageEvent`

Assert the next event is a `ChatMessageEvent`. If `intent` is given and a judge LLM was provided, evaluates whether the message fulfils the intent.

```python
# Just check a message exists
await session.agent_responds()

# Check intent with LLM judge
await session.agent_responds(intent="Reports weather for Tokyo including temperature")
```

Returns the matched `ChatMessageEvent`:

```python
event = await session.agent_responds()
print(event.content)  # "The weather in Tokyo is sunny, 70F."
print(event.role)     # "assistant"
```

#### `session.no_more_events()`

Assert that no events remain after the cursor. Raises `AssertionError` if events are left.

```python
session.agent_calls("get_weather")
await session.agent_responds(intent="Reports weather")
session.no_more_events()
```

#### `await session.start()` / `await session.close()`

Manually manage the session lifecycle. Prefer using `async with` instead.

---

## RunResult

Returned by `user_says()`. Holds captured events from a single conversation turn. Mostly useful for raw event access.

### Properties

| Property | Type | Description |
|---|---|---|
| `events` | `list[RunEvent]` | All captured events in chronological order. |
| `user_input` | `str \| None` | The user input that produced these events. |

---

## mock_tools

Context manager that temporarily replaces tool implementations. The tool schemas (name, description, parameters) remain visible to the LLM — only the callable is swapped.

```python
from vision_agents.testing import mock_tools

with mock_tools(session.llm, {"get_weather": lambda location: {"temp": 0, "condition": "snow"}}):
    await session.user_says("What's the weather?")
```

| Parameter | Type | Description |
|---|---|---|
| `llm` | `LLM` | The LLM instance whose tools to mock. |
| `mocks` | `dict[str, Callable]` | Tool name to mock callable mapping. |

Raises `KeyError` if a tool name is not registered.

Originals are always restored, even if an exception occurs inside the block.

```python
# Simulate a tool error
with mock_tools(session.llm, {"get_weather": lambda location: (_ for _ in ()).throw(RuntimeError("down"))}):
    await session.user_says("What's the weather?")
    # Agent should handle the error gracefully
```

---

## Event Types

### `ChatMessageEvent`

| Field | Type | Description |
|---|---|---|
| `role` | `str` | `"assistant"` |
| `content` | `str` | Message text |
| `type` | `"message"` | Always `"message"` |

### `FunctionCallEvent`

| Field | Type | Description |
|---|---|---|
| `name` | `str` | Function name |
| `arguments` | `dict[str, Any]` | Call arguments |
| `tool_call_id` | `str \| None` | Optional tool call ID |
| `type` | `"function_call"` | Always `"function_call"` |

### `FunctionCallOutputEvent`

| Field | Type | Description |
|---|---|---|
| `name` | `str` | Function name |
| `output` | `Any` | Return value or error dict |
| `is_error` | `bool` | `True` if the call failed |
| `tool_call_id` | `str \| None` | Optional tool call ID |
| `execution_time_ms` | `float \| None` | Execution time |
| `type` | `"function_call_output"` | Always `"function_call_output"` |

### `RunEvent`

Union type: `ChatMessageEvent | FunctionCallEvent | FunctionCallOutputEvent`

---

## Backwards Compatibility

`TestSession` is available as an alias for `TestEval`:

```python
from vision_agents.testing import TestSession  # same as TestEval
```

---

## Recommended Pattern for Examples

Separate LLM setup from Agent construction so tests can reuse the LLM without requiring Edge/STT/TTS infrastructure:

```python
# simple_agent_example.py

INSTRUCTIONS = "You're a helpful voice assistant. Be concise."

def setup_llm(model: str = "gemini-2.5-flash-lite") -> gemini.LLM:
    llm = gemini.LLM(model)

    @llm.register_function(description="Get current weather")
    async def get_weather(location: str) -> dict:
        return await get_weather_by_location(location)

    return llm

async def create_agent(**kwargs) -> Agent:
    llm = setup_llm()
    return Agent(
        edge=getstream.Edge(),
        llm=llm,
        instructions=INSTRUCTIONS,
        ...
    )
```

```python
# test_simple_agent.py

from .simple_agent_example import INSTRUCTIONS, setup_llm

async def test_weather():
    llm = setup_llm()
    judge_llm = gemini.LLM("gemini-2.5-flash-lite")

    async with TestEval(llm=llm, judge=judge_llm, instructions=INSTRUCTIONS) as session:
        await session.user_says("What's the weather in Berlin?")
        session.agent_calls("get_weather", arguments={"location": "Berlin"})
        await session.agent_responds(intent="Reports weather for Berlin")
        session.no_more_events()
```

---

## Environment Variables

| Variable | Description |
|---|---|
| `GOOGLE_API_KEY` | API key for Gemini LLM |
| `VISION_AGENTS_TEST_MODEL` | Override the model used in tests (default: `gemini-2.5-flash-lite`) |
| `VISION_AGENTS_EVALS_VERBOSE` | Set to `1` to print events and judge results during test runs |
