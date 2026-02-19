# Vision-Agents Testing API

Text-only testing for Vision-Agents. Test agent responses, tool calls, and intent — no audio, video, or edge connection required.

## What You Can Test

- **Agent responses** — verify the agent replied with a message (`is_message`), check role, check content via LLM judge (`judge(llm, intent="...")`)
- **Grounding** — verify the agent doesn't hallucinate (judge with intent like "Does NOT claim to know...")
- **Tool calls** — verify the agent called the right function (`is_function_call(name="get_weather")`), with the right arguments (`arguments={"location": "Tokyo"}`), partial argument matching supported
- **Tool outputs** — verify tool results (`is_function_call_output(output={...})`), error handling (`is_error=True`)
- **Event order** — strict sequential checks (`next_event()` chain + `no_more_events()`), or order-agnostic search (`contains_function_call(...)`)
- **Tool mocking** — swap tool implementations while keeping schemas intact (`mock_tools`), simulate errors
- **Multi-turn conversations** — multiple `session.run()` calls share conversation history, test context retention across turns

### Architecture: Why We Test the LLM, Not the Agent

The testing framework operates at the **LLM level**, not the full Agent level. This is by design.

`Agent` is an orchestrator that combines Edge (transport), STT, LLM, TTS, turn detection, and processors into a complete audio/video pipeline. Its constructor validates that at least one processing capability (audio or video) is present, making it impossible to instantiate without real infrastructure dependencies.

However, all agent **behavior** — what it says, which tools it calls, how it follows instructions — happens inside the LLM. `Agent.simple_response()` simply delegates to `llm.simple_response()`. By testing the LLM directly, we cover:

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
from vision_agents.testing import TestSession

async def test_greeting():
    llm = gemini.LLM("gemini-2.5-flash-lite")
    judge_llm = gemini.LLM("gemini-2.5-flash-lite")

    async with TestSession(llm=llm, instructions="Be friendly") as session:
        result = await session.run("Hello")
        await (
            result.expect
            .next_event()
            .is_message(role="assistant")
            .judge(judge_llm, intent="Friendly greeting")
        )
        result.expect.no_more_events()
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

## TestSession

The main entry point. Wraps an LLM instance, sends text input, captures events, and returns a `RunResult`.

### Constructor

```python
TestSession(llm: LLM, instructions: str = "You are a helpful assistant.")
```

| Parameter | Type | Description |
|---|---|---|
| `llm` | `LLM` | The LLM instance with tools already registered. |
| `instructions` | `str` | System instructions for the agent. |

### Class Methods

#### `TestSession.from_agent(agent)`

Create a session from an existing `Agent`. Extracts the LLM and instructions — the LLM's registered tools are preserved.

```python
agent = await create_agent()
async with TestSession.from_agent(agent) as session:
    result = await session.run("Hello")
```

> **Note:** The agent must be constructable without requiring live infrastructure (Edge, STT, TTS). Consider extracting your LLM setup into a separate function (see [Recommended Pattern](#recommended-pattern-for-examples)).

### Methods

#### `await session.run(user_input) -> RunResult`

Execute a single conversation turn. Sends `user_input` to the LLM, captures all tool-call events and the final assistant response.

Conversation history accumulates across successive `run()` calls, enabling multi-turn testing.

```python
result = await session.run("What's the weather in Tokyo?")
```

#### `await session.start()` / `await session.close()`

Manually manage the session lifecycle. Prefer using `async with` instead.

### Usage as Context Manager

```python
async with TestSession(llm=llm, instructions="...") as session:
    result = await session.run("Hello")
```

---

## RunResult

Returned by `session.run()`. Holds captured events from a single conversation turn.

### Properties

| Property | Type | Description |
|---|---|---|
| `events` | `list[RunEvent]` | All captured events in chronological order. |
| `expect` | `RunAssert` | Fluent assertion interface. |

---

## RunAssert

Cursor-based assertion navigator. Accessed via `result.expect`.

### Sequential Navigation

#### `next_event(type=None)`

Advance the cursor to the next event. Optionally filter by type — non-matching events are skipped.

```python
# Get the next event (any type)
result.expect.next_event()

# Skip to the next message event
result.expect.next_event(type="message")

# Skip to the next function call
result.expect.next_event(type="function_call")

# Skip to the next function call output
result.expect.next_event(type="function_call_output")
```

**Returns:**
- No type filter: `EventAssert`
- `type="message"`: `ChatMessageAssert`
- `type="function_call"`: `FunctionCallAssert`
- `type="function_call_output"`: `FunctionCallOutputAssert`

#### `skip_next(count=1)`

Skip `count` events without asserting. Returns `self` for chaining.

```python
result.expect.skip_next(2)
result.expect.next_event().is_message(role="assistant")
```

#### `no_more_events()`

Assert that no events remain after the cursor. Raises `AssertionError` if events are left.

```python
result.expect.next_event().is_message()
result.expect.no_more_events()
```

### Indexed Access

```python
# By positive index
result.expect[0].is_function_call(name="get_weather")

# By negative index
result.expect[-1].is_message(role="assistant")
```

### Sliced Access

Returns an `EventRangeAssert` for searching within a range.

```python
# Search all events
result.expect[:].contains_message(role="assistant")

# Search a range
result.expect[0:2].contains_function_call(name="get_weather")
```

### Search (Order-Agnostic)

Shorthand for `result.expect[:].<method>()` — searches all events.

```python
result.expect.contains_message(role="assistant")
result.expect.contains_function_call(name="get_weather")
result.expect.contains_function_call_output()
```

---

## EventAssert

Assertion helper for a single event. Returned by `next_event()` or indexed access.

### Methods

#### `is_message(role=None) -> ChatMessageAssert`

Assert the event is a `ChatMessageEvent`. Optionally check `role`.

```python
result.expect.next_event().is_message(role="assistant")
```

#### `is_function_call(name=None, arguments=None) -> FunctionCallAssert`

Assert the event is a `FunctionCallEvent`. Optionally check `name` and `arguments` (partial match — only specified keys are checked).

```python
# Check name only
result.expect.next_event().is_function_call(name="get_weather")

# Check name and specific arguments
result.expect.next_event().is_function_call(
    name="get_weather", arguments={"location": "Tokyo"}
)

# Partial match — extra arguments are ignored
result.expect.next_event().is_function_call(
    name="search", arguments={"query": "hello"}
)  # passes even if arguments also has "limit" and "offset"
```

#### `is_function_call_output(output=None, is_error=None) -> FunctionCallOutputAssert`

Assert the event is a `FunctionCallOutputEvent`. Optionally check `output` (exact match) and `is_error`.

```python
result.expect.next_event().is_function_call_output()
result.expect.next_event().is_function_call_output(is_error=True)
result.expect.next_event().is_function_call_output(
    output={"temp": 70, "condition": "sunny"}
)
```

#### `event() -> RunEvent`

Access the underlying raw event.

```python
ev = result.expect.next_event()
print(ev.event().type)  # "function_call", "message", etc.
```

---

## ChatMessageAssert

Assertion for a `ChatMessageEvent`. Returned by `is_message()` or `next_event(type="message")`.

### Methods

#### `await judge(llm, *, intent) -> ChatMessageAssert`

Evaluate whether the message fulfils the given `intent` using a separate LLM as judge. Returns `self` for chaining.

```python
judge_llm = gemini.LLM("gemini-2.5-flash-lite")

await (
    result.expect
    .next_event()
    .is_message(role="assistant")
    .judge(judge_llm, intent="Friendly greeting that mentions the user's name")
)
```

| Parameter | Type | Description |
|---|---|---|
| `llm` | `LLM` | **Separate** LLM instance for judging. Must not be the agent's LLM. |
| `intent` | `str` | What the message should accomplish. Be specific. |

> **Important:** Use a separate LLM instance for judging. Using the agent's LLM would pollute its conversation history.

#### `event() -> ChatMessageEvent`

Access the underlying event.

```python
msg = result.expect.next_event().is_message()
print(msg.event().content)   # "Hello! How can I help?"
print(msg.event().role)      # "assistant"
```

---

## FunctionCallAssert

Assertion for a `FunctionCallEvent`. Returned by `is_function_call()`.

### Methods

#### `event() -> FunctionCallEvent`

```python
fc = result.expect.next_event().is_function_call()
print(fc.event().name)        # "get_weather"
print(fc.event().arguments)   # {"location": "Tokyo"}
```

---

## FunctionCallOutputAssert

Assertion for a `FunctionCallOutputEvent`. Returned by `is_function_call_output()`.

### Methods

#### `event() -> FunctionCallOutputEvent`

```python
fco = result.expect.next_event().is_function_call_output()
print(fco.event().name)      # "get_weather"
print(fco.event().output)    # {"temperature": 22, ...}
print(fco.event().is_error)  # False
```

---

## EventRangeAssert

Assertion helper for a range of events. Returned by slice access (`result.expect[:]`).

### Methods

#### `contains_message(role=None) -> ChatMessageAssert`

Find the first matching `ChatMessageEvent` in the range.

#### `contains_function_call(name=None, arguments=None) -> FunctionCallAssert`

Find the first matching `FunctionCallEvent` in the range.

#### `contains_function_call_output(output=None, is_error=None) -> FunctionCallOutputAssert`

Find the first matching `FunctionCallOutputEvent` in the range.

---

## mock_tools

Context manager that temporarily replaces tool implementations. The tool schemas (name, description, parameters) remain visible to the LLM — only the callable is swapped.

```python
from vision_agents.testing import mock_tools

with mock_tools(llm, {"get_weather": lambda location: {"temp": 0, "condition": "snow"}}):
    result = await session.run("What's the weather?")
```

| Parameter | Type | Description |
|---|---|---|
| `llm` | `LLM` | The LLM instance whose tools to mock. |
| `mocks` | `dict[str, Callable]` | Tool name to mock callable mapping. |

Raises `KeyError` if a tool name is not registered.

Originals are always restored, even if an exception occurs inside the block.

```python
# Simulate a tool error
with mock_tools(llm, {"get_weather": lambda location: (_ for _ in ()).throw(RuntimeError("down"))}):
    result = await session.run("What's the weather?")
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

    async with TestSession(llm=llm, instructions=INSTRUCTIONS) as session:
        result = await session.run("What's the weather in Berlin?")

        result.expect.next_event().is_function_call(
            name="get_weather", arguments={"location": "Berlin"}
        )
        result.expect.next_event().is_function_call_output()
        await (
            result.expect.next_event()
            .is_message(role="assistant")
            .judge(judge_llm, intent="Reports weather for Berlin")
        )
        result.expect.no_more_events()
```

---

## Environment Variables

| Variable | Description |
|---|---|
| `GOOGLE_API_KEY` | API key for Gemini LLM |
| `VISION_AGENTS_TEST_MODEL` | Override the model used in tests (default: `gemini-2.5-flash-lite`) |
| `VISION_AGENTS_EVALS_VERBOSE` | Set to `1` to print events and judge results during test runs |
