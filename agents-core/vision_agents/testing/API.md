# Vision-Agents Testing API

> Text-only testing for Vision-Agents.
> Test agent responses, tool calls, and intent — no audio, video, or edge connection required.

## Table of Contents

- [Quick Start](#quick-start)
- [TestEval](#testeval) — session lifecycle, sends input
- [TestResponse](#testresponse) — response data + assertions
- [mock_tools](#mock_tools) — swap tool implementations
- [Event Types](#event-types) — ChatMessageEvent, FunctionCallEvent, FunctionCallOutputEvent
- [Recommended Pattern](#recommended-pattern)
- [Environment Variables](#environment-variables)

---

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

```python
async def test_tool_call():
    # ...
    async with TestEval(llm=llm, judge=judge_llm, instructions="...") as session:
        response = await session.simple_response("What's the weather in Tokyo?")
        response.function_called("get_weather", arguments={"location": "Tokyo"})
        await response.judge(intent="Reports weather for Tokyo")
        response.no_more_events()
```

### Running Tests

```bash
uv run py.test path/to/test_file.py -m integration                        # integration tests
VISION_AGENTS_EVALS_VERBOSE=1 uv run py.test path/to/test_file.py -m integration  # verbose output
uv run py.test path/to/test_file.py -m integration -s --timeout=0 --pdb   # with debugger
```

---

## TestEval

Manages the LLM session lifecycle, sends text input, returns [`TestResponse`](#testresponse) objects.

### Constructor

```python
TestEval(
    llm: LLM,
    instructions: str = "You are a helpful assistant.",
    judge: LLM | None = None,
)
```

| Parameter      | Type           | Description                                                                 |
|----------------|----------------|-----------------------------------------------------------------------------|
| `llm`          | `LLM`          | The LLM instance with tools already registered.                             |
| `instructions` | `str`          | System instructions for the agent.                                          |
| `judge`        | `LLM \| None`  | Separate LLM for intent evaluation. Required for `response.judge(intent=...)`. |

> **Note:** Always use a **separate** LLM instance for `judge` — using the agent's LLM would pollute its conversation history.

### `await session.simple_response(text) -> TestResponse`

Send user text to the LLM and capture the response. Conversation history accumulates across calls.

```python
response = await session.simple_response("What's the weather in Tokyo?")
```

### Properties

| Property | Type  | Description                                           |
|----------|-------|-------------------------------------------------------|
| `llm`    | `LLM` | The LLM instance (useful for `mock_tools(session.llm, ...)`). |

### Lifecycle

```python
# Preferred — context manager
async with TestEval(llm=llm, judge=judge_llm) as session:
    ...

# Manual
session = TestEval(llm=llm, judge=judge_llm)
await session.start()
try:
    ...
finally:
    await session.close()
```

---

## TestResponse

Returned by [`simple_response()`](#await-sessionsimple_responsetext---testresponse). Contains both the response data and assertion methods.

### Data Fields

| Field            | Type                       | Description                                     |
|------------------|----------------------------|-------------------------------------------------|
| `input`          | `str`                      | The user input that produced this response.      |
| `output`         | `str \| None`              | Final assistant message text (`None` if absent). |
| `events`         | `list[RunEvent]`           | All captured events in chronological order.      |
| `function_calls` | `list[FunctionCallEvent]`  | Filtered list of function call events.           |
| `duration_ms`    | `float`                    | Wall-clock time for the turn (ms).               |

```python
response = await session.simple_response("What's the weather?")

response.output           # "The weather in Tokyo is sunny, 22°C"
response.function_calls   # [FunctionCallEvent(name='get_weather', ...)]
response.duration_ms      # 1234.5
response.events           # full event list
```

### Assertions

#### `response.function_called(name?, *, arguments?) -> FunctionCallEvent`

Assert the next event is a `FunctionCallEvent`. Auto-skips the following `FunctionCallOutputEvent`.

```python
response.function_called("get_weather")                              # name only
response.function_called("get_weather", arguments={"location": "Tokyo"})  # name + args (partial match)
response.function_called()                                           # any function call
```

**Partial matching** — only specified keys are checked:

```python
# passes even if arguments also has "limit" and "offset"
response.function_called("search", arguments={"query": "hello"})
```

Returns the matched event for inspection:

```python
event = response.function_called("get_weather")
event.name        # "get_weather"
event.arguments   # {"location": "Tokyo"}
```

#### `response.function_output(*, output?, is_error?) -> FunctionCallOutputEvent`

Assert the next event is a `FunctionCallOutputEvent`. Use when you need to inspect the tool output explicitly (normally auto-skipped by `function_called`).

```python
response.function_output(output={"temp": 70, "condition": "sunny"})
response.function_output(is_error=True)
```

#### `await response.judge(*, intent?) -> ChatMessageEvent`

Assert the next event is a `ChatMessageEvent`. If `intent` is given, evaluates whether the message fulfils it using the judge LLM.

```python
await response.judge()                                                # message exists
await response.judge(intent="Reports weather for Tokyo with temperature")  # intent check
```

Returns the matched event:

```python
event = await response.judge()
event.content   # "The weather in Tokyo is sunny, 70F."
event.role      # "assistant"
```

#### `response.no_more_events()`

Assert no events remain. Raises `AssertionError` if there are unconsumed events.

```python
response.function_called("get_weather")
await response.judge(intent="Reports weather")
response.no_more_events()
```

---

## mock_tools

Context manager that temporarily swaps tool implementations. Schemas (name, description, parameters) stay intact — only the callable changes.

```python
from vision_agents.testing import mock_tools

with mock_tools(session.llm, {"get_weather": lambda location: {"temp": 0, "condition": "snow"}}):
    response = await session.simple_response("What's the weather?")
```

| Parameter | Type                  | Description                         |
|-----------|-----------------------|-------------------------------------|
| `llm`     | `LLM`                 | The LLM instance whose tools to mock. |
| `mocks`   | `dict[str, Callable]` | Tool name to mock callable mapping.   |

- Raises `KeyError` if a tool name is not registered.
- Originals are always restored, even on exception.

```python
# Simulate a tool error
with mock_tools(session.llm, {
    "get_weather": lambda location: (_ for _ in ()).throw(RuntimeError("Service down"))
}):
    response = await session.simple_response("What's the weather?")
```

---

## Event Types

### `ChatMessageEvent`

| Field     | Type          | Description          |
|-----------|---------------|----------------------|
| `role`    | `str`         | `"assistant"`        |
| `content` | `str`         | Message text         |
| `type`    | `"message"`   | Always `"message"`   |

### `FunctionCallEvent`

| Field          | Type              | Description            |
|----------------|-------------------|------------------------|
| `name`         | `str`             | Function name          |
| `arguments`    | `dict[str, Any]`  | Call arguments         |
| `tool_call_id` | `str \| None`     | Optional tool call ID  |
| `type`         | `"function_call"` | Always `"function_call"` |

### `FunctionCallOutputEvent`

| Field              | Type                      | Description              |
|--------------------|---------------------------|--------------------------|
| `name`             | `str`                     | Function name            |
| `output`           | `Any`                     | Return value or error    |
| `is_error`         | `bool`                    | `True` if the call failed |
| `tool_call_id`     | `str \| None`             | Optional tool call ID    |
| `execution_time_ms`| `float \| None`           | Execution time (ms)      |
| `type`             | `"function_call_output"`  | Always `"function_call_output"` |

### `RunEvent`

```python
RunEvent = ChatMessageEvent | FunctionCallEvent | FunctionCallOutputEvent
```

---

## Recommended Pattern

Separate LLM setup from Agent construction so tests can reuse the LLM without Edge/STT/TTS:

```python
# simple_agent_example.py

INSTRUCTIONS = "You're a helpful voice assistant. Be concise."

def setup_llm(model: str = "gemini-2.5-flash-lite") -> gemini.LLM:
    llm = gemini.LLM(model)

    @llm.register_function(description="Get current weather")
    async def get_weather(location: str) -> dict:
        return await get_weather_by_location(location)

    return llm
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

## Backwards Compatibility

```python
from vision_agents.testing import TestSession  # alias for TestEval
```

---

## Environment Variables

| Variable                       | Description                                                     |
|--------------------------------|-----------------------------------------------------------------|
| `GOOGLE_API_KEY`               | API key for Gemini LLM                                          |
| `VISION_AGENTS_TEST_MODEL`     | Override test model (default: `gemini-2.5-flash-lite`)          |
| `VISION_AGENTS_EVALS_VERBOSE`  | Set to `1` for detailed event and judge output during test runs |

---

## Architecture: Why We Test the LLM, Not the Agent

`Agent` combines Edge (transport), STT, LLM, TTS, turn detection, and processors into a full audio/video pipeline. It requires real infrastructure to instantiate.

All agent **behavior** — responses, tool calls, instruction following — happens inside the **LLM**. By testing the LLM directly we cover everything behavioral without infrastructure dependencies:

| Covered (LLM level)                | Not yet supported (Agent level)         |
|-------------------------------------|-----------------------------------------|
| Response content and intent         | Audio/video stream testing              |
| Function calling with correct args  | STT/TTS pipeline testing                |
| Grounding (no hallucination)        | Latency/performance benchmarks          |
| Multi-turn context retention        | Parallel multi-agent testing            |
| Tool error handling                 | MCP server testing                      |
| Tool mocking                        | Snapshot testing                        |
