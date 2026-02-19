# Vision-Agents Testing API

Text-only testing for Vision-Agents. Test agent responses, tool calls, and intent — no audio, video, or edge connection required.

## What You Can Test

- **Agent responses** — verify the agent replied (`response.judge()`), check content via LLM judge (`response.judge(intent="...")`)
- **Grounding** — verify the agent doesn't hallucinate (judge with intent like "Does NOT claim to know...")
- **Tool calls** — verify the agent called the right function (`response.function_called("get_weather")`), with the right arguments (`arguments={"location": "Tokyo"}`), partial argument matching supported
- **Tool outputs** — verify tool results (`response.function_output(output={...})`), error handling (`is_error=True`)
- **Event order** — strict sequential checks (`function_called` → `judge` → `no_more_events()`), auto-skips `FunctionCallOutputEvent` after `function_called`
- **Tool mocking** — swap tool implementations while keeping schemas intact (`mock_tools`), simulate errors
- **Multi-turn conversations** — multiple `simple_response()` calls share conversation history, test context retention across turns

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
        response = await session.simple_response("Hello")
        await response.judge(intent="Friendly greeting")
        response.no_more_events()
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

The main entry point. Manages the LLM session lifecycle, sends text input, and returns `TestResponse` objects.

### Constructor

```python
TestEval(llm: LLM, instructions: str = "You are a helpful assistant.", judge: LLM | None = None)
```

| Parameter | Type | Description |
|---|---|---|
| `llm` | `LLM` | The LLM instance with tools already registered. |
| `instructions` | `str` | System instructions for the agent. |
| `judge` | `LLM \| None` | Optional separate LLM for intent evaluation. Required if `response.judge(intent=...)` is used. |

> **Important:** Use a separate LLM instance for `judge`. Using the agent's LLM would pollute its conversation history.

### Properties

| Property | Type | Description |
|---|---|---|
| `llm` | `LLM` | The LLM instance (useful for `mock_tools(session.llm, {...})`). |

### Usage as Context Manager

```python
async with TestEval(llm=llm, judge=judge_llm, instructions="...") as session:
    response = await session.simple_response("Hello")
    await response.judge(intent="Friendly greeting")
```

### Methods

#### `await session.simple_response(text) -> TestResponse`

Send user text to the LLM and capture the response events. Conversation history accumulates across successive calls.

```python
response = await session.simple_response("What's the weather in Tokyo?")
```

#### `await session.start()` / `await session.close()`

Manually manage the session lifecycle. Prefer using `async with` instead.

---

## TestResponse

Returned by `simple_response()`. Holds captured data from a single conversation turn and provides assertion methods.

### Data Fields

| Field | Type | Description |
|---|---|---|
| `input` | `str` | The user input that produced this response. |
| `output` | `str \| None` | The final assistant message text, or `None` if no message. |
| `events` | `list[RunEvent]` | All captured events in chronological order. |
| `function_calls` | `list[FunctionCallEvent]` | Filtered list of function call events. |
| `duration_ms` | `float` | Wall-clock time for the turn in milliseconds. |

```python
response = await session.simple_response("What's the weather?")
print(response.output)           # "The weather in Tokyo is sunny, 22°C"
print(response.function_calls)   # [FunctionCallEvent(name='get_weather', ...)]
print(response.duration_ms)      # 1234.5
```

### Assertion Methods

#### `response.function_called(name=None, *, arguments=None) -> FunctionCallEvent`

Assert the next event is a `FunctionCallEvent`. Checks name and arguments (partial match — only specified keys are checked). **Auto-skips** the following `FunctionCallOutputEvent`.

```python
# Check name only
response.function_called("get_weather")

# Check name and specific arguments
response.function_called("get_weather", arguments={"location": "Tokyo"})

# Partial match — extra arguments are ignored
response.function_called("search", arguments={"query": "hello"})
# passes even if arguments also has "limit" and "offset"

# No name check — just advance to next function call
response.function_called()
```

Returns the matched `FunctionCallEvent` for further inspection:

```python
event = response.function_called("get_weather")
print(event.name)        # "get_weather"
print(event.arguments)   # {"location": "Tokyo"}
```

#### `response.function_output(*, output=..., is_error=None) -> FunctionCallOutputEvent`

Assert the next event is a `FunctionCallOutputEvent`. Use this when you need to inspect the tool output explicitly (normally auto-skipped by `function_called`).

```python
response.function_output(output={"temp": 70, "condition": "sunny"})
response.function_output(is_error=True)
```

#### `await response.judge(*, intent=None) -> ChatMessageEvent`

Assert the next event is a `ChatMessageEvent`. If `intent` is given and a judge LLM was provided to `TestEval`, evaluates whether the message fulfils the intent.

```python
# Just check a message exists
await response.judge()

# Check intent with LLM judge
await response.judge(intent="Reports weather for Tokyo including temperature")
```

Returns the matched `ChatMessageEvent`:

```python
event = await response.judge()
print(event.content)  # "The weather in Tokyo is sunny, 70F."
print(event.role)     # "assistant"
```

#### `response.no_more_events()`

Assert that no events remain after the cursor. Raises `AssertionError` if events are left.

```python
response.function_called("get_weather")
await response.judge(intent="Reports weather")
response.no_more_events()
```

---

## mock_tools

Context manager that temporarily replaces tool implementations. The tool schemas (name, description, parameters) remain visible to the LLM — only the callable is swapped.

```python
from vision_agents.testing import mock_tools

with mock_tools(session.llm, {"get_weather": lambda location: {"temp": 0, "condition": "snow"}}):
    response = await session.simple_response("What's the weather?")
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
    response = await session.simple_response("What's the weather?")
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
        response = await session.simple_response("What's the weather in Berlin?")
        response.function_called("get_weather", arguments={"location": "Berlin"})
        await response.judge(intent="Reports weather for Berlin")
        response.no_more_events()
```

---

## Environment Variables

| Variable | Description |
|---|---|
| `GOOGLE_API_KEY` | API key for Gemini LLM |
| `VISION_AGENTS_TEST_MODEL` | Override the model used in tests (default: `gemini-2.5-flash-lite`) |
| `VISION_AGENTS_EVALS_VERBOSE` | Set to `1` to print events and judge results during test runs |
